from __future__ import annotations

import os
import warnings
from dataclasses import asdict
from enum import Enum
from typing import Any

import bitsandbytes as bnb
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from ..utils import print_rank_0
from .config import SQALoraConfig
from .layer import Linear, SQALoraLayer


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def _get_module_name(model, target_module):
    for name, module in model.named_modules():
        if module is target_module:
            return name
    return None


def _get_module(root: nn.Module, path: str) -> nn.Module:
    if hasattr(root, "get_submodule"):
        return root.get_submodule(path)
    for p in path.split("."):
        root = getattr(root, p)
    return root


class SQALoraModel(nn.Module):
    prefix: str = "lora_"

    def __init__(self, model, sqalora_config) -> None:
        super().__init__()
        self.model = model
        self.sqalora_config = sqalora_config
        self.dynamic_quantization_config = sqalora_config.dynamic_quantization_config
        self.targeted_module_names: list[str] = []
        self.inject_adapter(self.model, sqalora_config)
        if self.sqalora_config.quantization:
            self.quantize()
        self.modules_to_save = (nn.Linear, nn.Embedding, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)

    def forward(self, *args: Any, **kwargs: Any):
        return self.model.forward(*args, **kwargs)

    def inject_adapter(self, model: nn.Module, sqalora_config: SQALoraConfig) -> None:
        target_suffixes = set(sqalora_config.target_modules)

        for full_name, module in model.named_modules():
            if full_name == "":
                continue

            if not any(full_name == s or full_name.endswith(f".{s}") for s in target_suffixes):
                continue

            self.targeted_module_names.append(full_name)
            parent, _, child_name = _get_submodules(model, full_name)
            self._create_and_replace(sqalora_config, module, child_name, parent)

        self._mark_only_adapters_as_trainable(model)

    def _create_and_replace(
        self,
        sqalora_config,
        target,
        target_name,
        parent,
    ):
        r = sqalora_config.r
        quant_method = next(
            (cfg for k, cfg in (self.dynamic_quantization_config or {}).items() if target_name in k),
            sqalora_config.quant_method
        )
        kwargs = {
            "r": r,
            "lora_alpha": sqalora_config.lora_alpha,
            "lora_dropout": sqalora_config.lora_dropout,
            "fan_in_fan_out": sqalora_config.fan_in_fan_out,
            "init_lora_weights": sqalora_config.init_lora_weights,
            "use_rslora": sqalora_config.use_rslora,
            "lora_bias": sqalora_config.lora_bias,
            "sparse_preserve_mode": sqalora_config.sparse_preserve_mode,
            "quant_method": quant_method,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        if isinstance(target, Linear):
            target.update_layer(
                r,
                sqalora_config.lora_alpha,
                sqalora_config.lora_dropout,
                sqalora_config.init_lora_weights,
                sqalora_config.use_rslora,
                sqalora_config.lora_bias,
            )
        else:
            new_module = self._create_new_module(target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        meta = torch.device("meta")
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if (self.prefix in name) or ("ranknum" in name):
                if hasattr(child, "qweight"):
                    weight = child.qweight
                elif hasattr(child, "W_q"):
                    weight = child.W_q
                elif hasattr(child, "weight"):
                    weight = child.weight
                elif getattr(child, "in_proj_weight", None) is not None:  # MHA
                    weight = child.in_proj_weight
                else:
                    weight = next(child.parameters())
                if not any(p.device == meta for p in module.parameters()):
                    module.to(weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

    @staticmethod
    def _create_new_module(target, **kwargs):
        loaded_in_8bit = kwargs.get("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.get("loaded_in_4bit", False)

        if isinstance(target, SQALoraLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )
        new_module = Linear(
            target,
            kwargs["r"],
            kwargs["lora_alpha"],
            kwargs["lora_dropout"],
            kwargs["fan_in_fan_out"],
            kwargs["init_lora_weights"],
            kwargs["use_rslora"],
            kwargs["lora_bias"],
            kwargs["sparse_preserve_mode"],
            kwargs["quant_method"],
        )

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def prune(self, sparsity_ratio: float = 0.5, prune_n=0, prune_m=0, offload=True, sparse_prune_largest=False):
        for name, module in self.model.named_modules():
            if isinstance(module, Linear):
                print_rank_0(f"Pruning layer - {name}, sparsity ratio = {sparsity_ratio}")
                module.prune(
                    sparsity_ratio=sparsity_ratio,
                    prune_n=prune_n,
                    prune_m=prune_m,
                    offload=offload,
                    sparse_prune_largest=sparse_prune_largest,
                )

    def quantize(self):
        # TODO: check if the model is on GPU
        if self.device == "cpu" or self.model.device == "cpu":
            print_rank_0(
                "You are tring to quantize the model on cpu, which may cause errors. We recommend to quantize the model on GPU."
            )
        for name, module in self.model.named_modules():
            if isinstance(module, Linear):
                print_rank_0(f"Quantizing layer - {name} to {module.quant_method}")
                module.quantize()
        self.sqalora_config.quantization = True

    def dequantize(self):
        for name, module in self.model.named_modules():
            if isinstance(module, Linear):
                print_rank_0(f"Dequantizing layer - {name} from {module.quant_method}")
                module.dequantize()
        self.sqalora_config.quantization = False

    @torch.no_grad()
    def calculate_sparsity(self, eps=1e-4) -> float:
        rates = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.data
                num_zeros = torch.sum(torch.abs(weight) < eps).item()
                total = weight.numel()
                rates.append(num_zeros / total)
        if not rates:
            return 0.0
        return sum(rates) / len(rates)

    def collect_tensors_to_save(self, prefix: str = "") -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        seen_ptr: set[int] = set()

        for module_name, module in self.named_modules():
            if isinstance(module, self.modules_to_save):
                for param_name, param_tensor in module.state_dict().items():
                    key = f"{prefix}{module_name}.{param_name}" if module_name else f"{prefix}{param_name}"
                    tensors[key] = param_tensor.detach().cpu()
        for name in tensors.keys():
            print_rank_0(f"Collecting {name} to save.")

        return tensors

    @staticmethod
    def handle_weight(model: SQALoraModel, state_dict: dict[str, torch.Tensor]):
        for name, module in model.named_modules():
            module_weight_name = f"{name}.weight"
            if isinstance(module, bnb.nn.Linear4bit):
                print_rank_0(f"Loading weight to {module_weight_name}")
                weight = state_dict[module_weight_name]
                if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
                    module.weight = bnb.nn.Params4bit(
                        weight,
                        requires_grad=False,
                        quant_type=module.quant_state.quant_type,
                    ).to(module.weight.device)
                elif weight.dtype in (torch.uint8, torch.int8):
                    quantized_stats = {}
                    for k, v in state_dict.items():
                        if module_weight_name + "." in k:
                            quantized_stats[k] = v

                    new_value = bnb.nn.Params4bit.from_prequantized(
                        data=weight,
                        quantized_stats=quantized_stats,
                        requires_grad=False,
                        device=module.weight.device,
                    )
                    module.weight = new_value
            elif isinstance(module, bnb.nn.Linear8bitLt):
                print_rank_0(f"Loading weight to {module_weight_name}")
                weight = state_dict[module_weight_name]
                if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
                    module.weight = bnb.nn.Int8Params(
                        weight,
                        requires_grad=False,
                    ).to(module.weight.device)
                elif weight.dtype is torch.int8:
                    module.weight = bnb.nn.Int8Params(
                        data = weight,
                        SCB = state_dict[name + ".SCB"],
                        requires_grad=False,
                    ).to(module.weight.device)
            elif isinstance(module, (nn.Linear, nn.Embedding)):
                try:
                    module.weight.copy_(state_dict[module_weight_name])
                    print_rank_0(f"Loading weight to {module_weight_name}")
                except KeyError:
                    print_rank_0(f"Cannot find {module_weight_name} in state_dict.")
                continue
            else:
                # since the module is not nn.Linear or nn.Embedding or bnb.nn.Linear4bit, we don't need to handle it
                continue

    @torch.no_grad()
    def save_pretrained(self, save_directory: str) -> None:
        """
        sqalora_model.safetensors should only contains the weights of the all linears.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, "sqalora_model.safetensors")
        tensors = self.collect_tensors_to_save()
        save_file(tensors, model_path)
        self.sqalora_config.save_pretrained(save_directory)

    @classmethod
    @torch.no_grad()
    def from_pretrained(
        cls,
        model: nn.Module,
        sqalora_model_name_or_path: str,
        **kwargs,
    ) -> SQALoraModel:
        """
        Load a SQALoraModel from path.
        1. Load the SQALoraConfig.
        2. construct the SQALoraModel based on SQALoraConfig and model.
        3. should only replace the xx modules from ckpt to the model.
        """
        sqalora_config = SQALoraConfig.from_pretrained(sqalora_model_name_or_path)
        sqalora_model = cls(model, sqalora_config)

        safetensors_path = os.path.join(sqalora_model_name_or_path, "sqalora_model.safetensors")
        if not os.path.isfile(safetensors_path):
            raise FileNotFoundError(f"Cannot find sqalora_model.safetensors in {safetensors_path}")
        state_dict = load_file(safetensors_path)

        # TODO: refactor this part, construct the WL and WR when init the sqalora_model
        # process WL and WR
        for full_key in list(state_dict.keys()):
            if ".WL." not in full_key and ".WR." not in full_key:
                continue

            parts = full_key.split(".")
            if "WL" in parts:
                lr_type = "WL"
            else:
                lr_type = "WR"
            idx = parts.index(lr_type)

            module_path = ".".join(parts[:idx])
            step_name, param_name = parts[idx + 1], parts[idx + 2]  # e.g. sparse_step_0, weight

            try:
                linear_module: Linear = _get_module(sqalora_model.model, module_path)
            except AttributeError as e:
                raise RuntimeError(f"Cannot find submodule in {module_path}") from e

            lr_dict: nn.ModuleDict = getattr(linear_module, lr_type)  # WL 或 WR
            if step_name not in lr_dict:
                tensor = state_dict[full_key]
                out_dim, in_dim = tensor.shape
                new_linear = nn.Linear(
                    in_dim,
                    out_dim,
                    bias=False,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                lr_dict.update({step_name: new_linear})

        # missing, unexpected = sqalora_model.load_state_dict(state_dict, strict=False)
        # missing, unexpected = load_model(sqalora_model, safetensors_path, strict=False)
        # if len(unexpected) > 0:
        #     warnings.warn(f"there are {len(unexpected)} parameters that unexpected: {list(unexpected)[:10]} ...")
        # if len(missing) > 0:
        #     warnings.warn(f"there are {len(missing)} parameters that missing: {list(missing)[:10]} ...")
        # TODO: check here, the expect output is:
        # unexpected only contains the quant_state, such as .absmax, .nested_absmax, .nested_quant_map, .quant_map, .quant_state.bitsandbytes__nf4
        # missing is empty

        # quant_method = sqalora_config.quant_method.lower()
        # _quant_load_handlers = {
        #     "nf4": cls._load_nf4_weights,
        # }
        # handler_fn = _quant_load_handlers[quant_method]
        # handler_fn(sqalora_model, state_dict)

        cls.handle_weight(sqalora_model, state_dict)

        return sqalora_model

    @staticmethod
    def _load_nf4_weights(model: SQALoraModel, state_dict: dict[str, torch.Tensor]):
        for name, module in model.named_modules():
            if isinstance(module, Linear):
                base_layer = module.get_base_layer()
                base_layer_name = _get_module_name(model, base_layer)
                base_layer_name = f"{base_layer_name}.weight"
                if base_layer_name in state_dict:
                    print_rank_0(f"Loading weight to {base_layer_name}")
                    param_value = state_dict[base_layer_name]
                    param_dtype = param_value.dtype
                    target_device = base_layer.weight.device
                    target_dtype = base_layer.weight.dtype

                    if target_dtype == torch.bfloat16 and param_dtype == torch.bfloat16:
                        base_layer.weight.copy_(param_value)
                    elif target_dtype == torch.bfloat16 and param_dtype == torch.uint8:
                        module.quantize()
                        quantized_stats = {}
                        for k, v in state_dict.items():
                            if base_layer_name + "." in k:
                                quantized_stats[k] = v
                        new_value = bnb.nn.Params4bit.from_prequantized(
                            data=param_value,
                            quantized_stats=quantized_stats,
                            requires_grad=False,
                            device=target_device,
                        )
                        base_layer.weight = new_value
                    elif target_dtype == torch.uint8 and param_dtype == torch.bfloat16:
                        base_layer.weight = bnb.nn.Params4bit(
                            param_value,
                            requires_grad=False,
                            quant_type="nf4",
                        ).to(target_device)
                    elif target_dtype == torch.uint8 and param_dtype == torch.uint8:
                        quantized_stats = {}
                        for k, v in state_dict.items():
                            if base_layer_name + "." in k:
                                quantized_stats[k] = v

                        new_value = bnb.nn.Params4bit.from_prequantized(
                            data=param_value,
                            quantized_stats=quantized_stats,
                            requires_grad=False,
                            device=target_device,
                        )
                        base_layer.weight = new_value
                    else:
                        raise ValueError(
                            "The dtype of base layer and it's state in state_dict must in [torch.bfloat16, torch.uint8]."
                        )

    # TODO
    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
    ):
        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue

            if hasattr(target, "base_layer"):
                if merge:
                    target.merge(safe_merge=safe_merge)

                self._replace_module(parent, target_name, target.get_base_layer(), target)

        return self.model

    # TODO
    def merge_and_unload(self, progressbar: bool = False, safe_merge: bool = False):
        return self._unload_and_optionally_merge(progressbar=progressbar, safe_merge=safe_merge)

    # TODO
    def unload(self):
        return self._unload_and_optionally_merge(merge=False)
