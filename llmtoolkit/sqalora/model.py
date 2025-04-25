from __future__ import annotations

import os
import math
import warnings
from dataclasses import asdict
from enum import Enum
from typing import Any, Optional, Union
import json

import torch
import torch.nn as nn
from torch.nn.init import _calculate_correct_fan
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D
from safetensors.torch import load_file, save_file

from peft.import_utils import is_bnb_4bit_available, is_bnb_available

import bitsandbytes as bnb

from .config import SQALoraConfig
from .layer import SQALoraLayer, Linear
from .bnb import Linear8bitLt, Linear4bit

from ..utils import print_rank_0

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


class SQALoraModel(nn.Module):
    prefix: str = "lora_"

    def __init__(self, model, config) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.targeted_module_names: list[str] = []
        self.inject_adapter(self.model, config)

    def forward(self, *args: Any, **kwargs: Any):
        return self.model.forward(*args, **kwargs)

    def inject_adapter(self, model: nn.Module, config: SQALoraConfig) -> None:
        target_suffixes = set(config.target_modules)

        for full_name, module in model.named_modules():
            # 跳过根模块
            if full_name == "":
                continue

            # 名称匹配
            if not any(full_name == s or full_name.endswith(f".{s}") for s in target_suffixes):
                continue

            # 命中：记录并替换
            self.targeted_module_names.append(full_name)
            parent, _, child_name = _get_submodules(model, full_name)
            self._create_and_replace(config, module, child_name, parent)

        # 只保留 LoRA 相关参数可训练
        self._mark_only_adapters_as_trainable(model)

    def _create_and_replace(
        self,
        sqalora_config,
        target,
        target_name,
        parent,
    ):
        r = sqalora_config.r
        kwargs = {
            "r": r,
            "lora_alpha": sqalora_config.lora_alpha,
            "lora_dropout": sqalora_config.lora_dropout,
            "fan_in_fan_out": sqalora_config.fan_in_fan_out,
            "init_lora_weights": sqalora_config.init_lora_weights,
            "use_rslora": sqalora_config.use_rslora,
            "lora_bias": sqalora_config.lora_bias,
            "sparse_preserve_mode": sqalora_config.sparse_preserve_mode,
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

        if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
            eightbit_kwargs = {}
            return Linear8bitLt(
                target,
                kwargs["r"],
                kwargs["lora_alpha"],
                kwargs["lora_dropout"],
                kwargs["fan_in_fan_out"],
                kwargs["init_lora_weights"],
                kwargs["use_rslora"],
                kwargs["lora_bias"],
                kwargs["sparse_preserve_mode"],
            )
        elif loaded_in_4bit and isinstance(target_base_layer, bnb.nn.Linear4bit):
            fourbit_kwargs = {}
            return Linear4bit(
                target,
                kwargs["r"],
                kwargs["lora_alpha"],
                kwargs["lora_dropout"],
                kwargs["fan_in_fan_out"],
                kwargs["init_lora_weights"],
                kwargs["use_rslora"],
                kwargs["lora_bias"],
                kwargs["sparse_preserve_mode"],
            )
        elif isinstance(target_base_layer, torch.nn.Linear):
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
            if isinstance(module, (Linear, Linear8bitLt, Linear4bit)):
                print_rank_0(f"Pruning layer - {name}, sparsity ratio = {sparsity_ratio}")
                module.prune(
                    sparsity_ratio=sparsity_ratio,
                    prune_n=prune_n,
                    prune_m=prune_m,
                    offload=offload,
                    sparse_prune_largest=sparse_prune_largest,
                )
    def quantize(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (Linear, Linear8bitLt, Linear4bit)):
                print_rank_0(f"Quantizing layer - {name}")
                module.quantize()

    def dequantize(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (Linear, Linear8bitLt, Linear4bit)):
                print_rank_0(f"Dequantizing layer - {name}")
                module.dequantize()

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

    @torch.no_grad()
    def save_pretrained(self, save_directory: str) -> None:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        tensor_dict: dict[str, torch.Tensor] = {}

        for module_name, module in self.model.named_modules():
            if not isinstance(module, (Linear, Linear8bitLt, Linear4bit)):
                continue

            prefix = module_name

            # lora_A / lora_B
            if module.lora_A is not None:
                tensor_dict[f"{prefix}.lora_A.weight"] = module.lora_A.weight.cpu()
            if module.lora_B is not None:
                tensor_dict[f"{prefix}.lora_B.weight"] = module.lora_B.weight.cpu()
                if module.lora_bias and module.lora_B.bias is not None:
                    tensor_dict[f"{prefix}.lora_B.bias"] = module.lora_B.bias.cpu()

            for wl_key, wl_layer in module.WL.items():
                tensor_dict[f"{prefix}.WL.{wl_key}.weight"] = wl_layer.weight.cpu()
            for wr_key, wr_layer in module.WR.items():
                tensor_dict[f"{prefix}.WR.{wr_key}.weight"] = wr_layer.weight.cpu()

            if module.sparse_mask is not None:
                tensor_dict[f"{prefix}.sparse_mask"] = module.sparse_mask.to(torch.uint8).cpu()

        model_path = os.path.join(save_directory, "adapter_model.safetensors")
        save_file(tensor_dict, model_path)
        if hasattr(self.config, "peft_type"):
            delattr(self.config, "peft_type")
        self.config.save_pretrained(save_directory)

    @classmethod
    @torch.no_grad()
    def from_pretrained(
        cls,
        model: nn.Module,
        peft_model_name_or_path: str,
        **kwargs,
    ) -> SQALoraModel:

        config = SQALoraConfig.from_pretrained(peft_model_name_or_path)
        sqalora_model = cls(model, config)

        weight_path = os.path.join(peft_model_name_or_path, "adapter_model.safetensors")
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Cannot find adapter weight file: {weight_path}")

        sd: dict[str, torch.Tensor] = load_file(weight_path, device="cpu")

        for module_name, module in sqalora_model.model.named_modules():
            if not isinstance(module, (Linear, Linear8bitLt, Linear4bit)):
                continue

            prefix = module_name
            dev_A   = module.lora_A.weight.device
            dev_B   = module.lora_B.weight.device
            dtype_A = module.lora_A.weight.dtype
            dtype_B = module.lora_B.weight.dtype

            key = f"{prefix}.lora_A.weight"
            if key in sd:
                module.lora_A.weight.copy_(sd[key].to(device=dev_A, dtype=dtype_A))

            key = f"{prefix}.lora_B.weight"
            if key in sd:
                module.lora_B.weight.copy_(sd[key].to(device=dev_B, dtype=dtype_B))

            if module.lora_bias:
                key = f"{prefix}.lora_B.bias"
                if key in sd and module.lora_B.bias is not None:
                    module.lora_B.bias.copy_(sd[key].to(device=dev_B, dtype=dtype_B))

            wl_prefix = f"{prefix}.WL."
            wr_prefix = f"{prefix}.WR."

            steps = set()
            for k in sd.keys():
                if k.startswith(wl_prefix):
                    steps.add(k.split(".")[-2])
                if k.startswith(wr_prefix):
                    steps.add(k.split(".")[-2])

            for step in sorted(steps, key=lambda s: int(s.split("_")[-1])):
                wl_key = f"{wl_prefix}{step}.weight"
                wr_key = f"{wr_prefix}{step}.weight"
                if wl_key not in sd or wr_key not in sd:
                    continue

                WL = sd[wl_key].to(device=dev_A, dtype=dtype_A)
                WR = sd[wr_key].to(device=dev_B, dtype=dtype_B)

                mode = 1 if step.endswith("0") else 2
                module.update_WL_WR(WL, WR, mode)

            mask_key = f"{prefix}.sparse_mask"
            if mask_key in sd:
                mask = sd[mask_key].to(torch.bool)
                module.apply_sparse_mask(mask)

        return sqalora_model

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

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False):

        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge
        )

    def unload(self):
        return self._unload_and_optionally_merge(merge=False)
