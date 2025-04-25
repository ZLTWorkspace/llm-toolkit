import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.utils.other import transpose
import bitsandbytes as bnb

from .utils import (
    decomposeW2LinearWeightLR,
    mergeW2AB,
    _get_mask_prune_magnitude,
)


"""
sparse_preserve_mode
0: no sparse preserve
1: (A+∑WL) * (B+∑WR)
2: ∑(WL*WR) + AB
"""


class SQALoraLayer(nn.Module):
    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.r = -1
        self.lora_alpha = -1
        self.scaling = -1
        self.lora_dropout = None
        self.lora_A = None
        self.lora_B = None
        self.sparse_mask = None
        self.WL = nn.ModuleDict({})
        self.WR = nn.ModuleDict({})
        self.merged = False
        self._disable_adapters = False
        self.merged_adapters = []
        self.lora_bias: bool = False
        self._caches: dict[str, Any] = {}
        self.cast_input_dtype_enabled: bool = True
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv1d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        else:
            raise NotImplementedError("SQALora only supports Linear and Conv1D layers.")

        self.in_features = in_features
        self.out_features = out_features

    def get_base_layer(self) -> nn.Module:
        base_layer = self
        while hasattr(base_layer, "base_layer"):
            base_layer = base_layer.base_layer
        return base_layer

    def update_layer(
        self,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        lora_bias: bool = False,
    ):
        # collect the kwargs
        kwargs = locals().copy()
        del kwargs["self"]

        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout = lora_dropout_layer
        # Actual trainable parameters
        # TODO: pass the dtype from top class model, or auto detect
        self.lora_A = nn.Linear(self.in_features, r, bias=False, dtype=torch.bfloat16)
        self.lora_B = nn.Linear(r, self.out_features, bias=lora_bias, dtype=torch.bfloat16)
        self.lora_bias = lora_bias

        if use_rslora:
            self.scaling = lora_alpha / math.sqrt(r)
        else:
            self.scaling = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(init_lora_weights)

        self.lora_A.to(self.base_layer.weight.device)
        self.lora_B.to(self.base_layer.weight.device)
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)

    def reset_lora_parameters(self, init_lora_weights):
        if init_lora_weights is False:
            return
        elif init_lora_weights is True:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        elif init_lora_weights.lower() == "gaussian":
            nn.init.normal_(self.lora_A.weight, std=1 / self.r)
        else:
            raise ValueError(f"Unknown initialization {init_lora_weights=}")
        nn.init.zeros_(self.lora_B.weight)
        if self.lora_bias:
            nn.init.zeros_(self.lora_B.bias)

    @torch.no_grad()
    def update_WL_WR(self, WL: torch.Tensor, WR: torch.Tensor, mode: int = 0):
        dtype_A = self.lora_A.weight.dtype
        device_A = self.lora_A.weight.device
        dtype_B = self.lora_B.weight.dtype
        device_B = self.lora_B.weight.device

        if mode == 0:
            return
        elif mode == 1:
            self.WL.update({"sparse_step_0": nn.Linear(WL.shape[1], WL.shape[0], bias=False, dtype=dtype_A)})
            self.WR.update({"sparse_step_0": nn.Linear(WR.shape[1], WR.shape[0], bias=False, dtype=dtype_B)})
            self.WL["sparse_step_0"].weight.data = WL
            self.WR["sparse_step_0"].weight.data = WR
            self.WL["sparse_step_0"].requires_grad_(False)
            self.WR["sparse_step_0"].requires_grad_(False)
            self.WL["sparse_step_0"].to(device_A)
            self.WR["sparse_step_0"].to(device_B)
        elif mode == 2:
            self.WL.update(
                {f"sparse_step_{len(self.WL)}": nn.Linear(WL.shape[1], WL.shape[0], bias=False, dtype=dtype_A)}
            )
            self.WR.update(
                {f"sparse_step_{len(self.WR)}": nn.Linear(WR.shape[1], WR.shape[0], bias=False, dtype=dtype_B)}
            )
            self.WL[f"sparse_step_{len(self.WL) - 1}"].weight.data = WL
            self.WR[f"sparse_step_{len(self.WR) - 1}"].weight.data = WR
            self.WL[f"sparse_step_{len(self.WL) - 1}"].requires_grad_(False)
            self.WR[f"sparse_step_{len(self.WR) - 1}"].requires_grad_(False)
            self.WL[f"sparse_step_{len(self.WL) - 1}"].to(device_A)
            self.WR[f"sparse_step_{len(self.WR) - 1}"].to(device_B)
        else:
            raise ValueError(f"Unknown mode {mode}")


class Linear(SQALoraLayer):
    # SQALora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        lora_bias: bool = False,
        sparse_preserve_mode: int = 0,
        quant_method: str = "nf4",
        **kwargs,
    ) -> None:
        super().__init__(base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self.sparse_preserve_mode = sparse_preserve_mode
        self.quant_method = quant_method

        self.update_layer(r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, lora_bias)

    @torch.no_grad()
    def merge(self, safe_merge: bool = False) -> None:
        """
        Merge the active adapter weights into the base weights
        Not recommend for SQALora, since it will change the base weights from sparse into dense.

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        base_layer = self.get_base_layer()
        if safe_merge:
            orig_weight = base_layer.weight.data.clone()
            orig_dtype = orig_weight.dtype
            delta_weight = self.get_delta_weight()
            orig_weight += delta_weight.to(orig_dtype)
            if not torch.isfinite(orig_weight).all():
                raise ValueError("NaNs detected in the merged weights. The adapter seems to be broken")
            base_layer.weight.data = orig_weight
            if self.lora_bias:
                new_bias = base_layer.bias + self.lora_B.bias * self.scaling
                if not torch.isfinite(new_bias).all():
                    raise ValueError("NaNs detected in the merged weights. The adapter seems to be broken")
                base_layer.bias.data = new_bias.to(orig_dtype)
        else:
            delta_weight = self.get_delta_weight()
            base_layer.weight.data += delta_weight
            if self.lora_bias:
                base_layer.bias.data += self.lora_B.bias * self.scaling

        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        self.get_base_layer().weight.data -= self.get_delta_weight()
        self.merged = False

    @torch.no_grad()
    def get_delta_weight(self) -> torch.Tensor:
        weight_A = self.lora_A.weight
        weight_B = self.lora_B.weight

        if self.sparse_preserve_mode == 0:
            output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling
        elif self.sparse_preserve_mode == 1:
            weight_B_sum = weight_B + sum(module.weight for module in self.WR.values())
            weight_A_sum = weight_A + sum(module.weight for module in self.WL.values())

            output_tensor = transpose(weight_B_sum @ weight_A_sum, self.fan_in_fan_out) * self.scaling
        elif self.sparse_preserve_mode == 2:
            output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling
            for wlr_name in self.WL:
                output_tensor = output_tensor + transpose(
                    self.WR[wlr_name].weight @ self.WL[wlr_name].weight, self.fan_in_fan_out
                )
        else:
            raise ValueError(f"Unknown sparse preserve mode {self.sparse_preserve_mode}")

        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            lora_A = self.lora_A
            lora_B = self.lora_B
            dropout = self.lora_dropout
            scaling = self.scaling
            x = self._cast_input_dtype(x, self.lora_A.weight.dtype)
            if self.sparse_preserve_mode == 0:
                result = result + lora_B(lora_A(dropout(x))) * scaling
            elif self.sparse_preserve_mode == 1:
                result_A = lora_A(dropout(x))
                for wlr_name in self.WL:
                    result_A = result_A + self.WL[wlr_name](x)
                result_B = lora_B(result_A)
                for wlr_name in self.WL:
                    result_B = result_B + self.WR[wlr_name](result_A)
                result = result + result_B * scaling
            elif self.sparse_preserve_mode == 2:
                result = result + lora_B(lora_A(dropout(x))) * scaling
                for wlr_name in self.WL:
                    result = result + self.WR[wlr_name](self.WL[wlr_name](x))
            else:
                raise ValueError(f"Unknown sparse preserve mode {self.sparse_preserve_mode}")
            result = result.to(torch_result_dtype)
        return result

    @torch.no_grad()
    def prune(
        self,
        sparsity_ratio: float,
        prune_n=0,
        prune_m=0,
        offload=True,
        sparse_prune_largest=False,
    ):
        """
        Prune the weights of the base layer to make them sparse.
        For now we only support magnitude pruning.
        The prune func should be in custome Linear layer, since it needs to handle when base layer is quantized.
        The prune func should:
        1. sparse the base layer weights
        2. return sparse mask (no need to return name of the layer)
        3. update the WL and WR weights if sparse_preserve is True
        """
        sparse_preserve_mode = self.sparse_preserve_mode
        if sparsity_ratio > 1 or sparsity_ratio < 0:
            raise ValueError("sparsity_ratio should be in (0,1).")
        if (prune_n, prune_m) not in [(0, 0), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64)]:
            raise ValueError("structured pruning only support (2,4), (4,8), (8,16), (16,32),(32,64) for now.")
        if sparse_preserve_mode not in [0, 1, 2]:
            raise ValueError("sparse_preserve_mode should be in (0,1,2).")
        self.merge()
        base_layer = self.get_base_layer()
        sparse_mask = _get_mask_prune_magnitude(
            base_layer.weight.data,
            sparsity_ratio,
            prune_n,
            prune_m,
            sparse_prune_largest,
            offload,
        )
        self.unmerge()
        if sparse_preserve_mode == 0:
            self.apply_sparse_mask(sparse_mask)
        elif sparse_preserve_mode == 1:
            tmp_W = base_layer.weight.data.detach().clone()
            self.apply_sparse_mask(sparse_mask)
            tmp_W = tmp_W - base_layer.weight.data
            WL, WR = mergeW2AB(
                tmp_W,
                self.lora_A.weight,
                self.lora_B.weight,
                self.scaling,
            )
            self.update_WL_WR(WL, WR, 1)
            pass
        elif sparse_preserve_mode == 2:
            tmp_W = base_layer.weight.data.detach().clone()
            self.apply_sparse_mask(sparse_mask)
            tmp_W = tmp_W - base_layer.weight.data
            WL, WR = decomposeW2LinearWeightLR(
                tmp_W,
                self.r,
            )
            self.update_WL_WR(WL, WR, 2)
            pass
        else:
            raise ValueError(f"Unknown sparse preserve mode {sparse_preserve_mode}")

    @torch.no_grad()
    def quantize(
        self,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        if self.quant_method == "nf4":
            if isinstance(self.base_layer, bnb.nn.Linear4bit):
                return
            device = self.base_layer.weight.device
            weight_bf16 = self.base_layer.weight.detach().to(compute_dtype).contiguous()
            bias = None if self.base_layer.bias is None else self.base_layer.bias.detach().clone()

            qlinear = bnb.nn.Linear4bit(
                self.in_features,
                self.out_features,
                bias=bias is not None,
                compute_dtype=compute_dtype,
                quant_type="nf4",
            ).to(device)

            # 3. 利用 bitsandbytes 的 Params4bit 保存量化后权重
            qlinear.weight = bnb.nn.Params4bit(
                weight_bf16,
                requires_grad=False,
                quant_type="nf4",
            ).to(device)
            if bias is not None:
                qlinear.bias = nn.Parameter(bias)
            self.base_layer = qlinear

    @torch.no_grad()
    def dequantize(
        self,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        if self.quant_method == "nf4":
            if not isinstance(self.base_layer, bnb.nn.Linear4bit):
                return

            weight_bf16 = bnb.functional.dequantize_4bit(
                self.base_layer.weight.data, self.base_layer.weight.quant_state
            )
            # assert weight_bf16.dtype == compute_dtype
            bias = None if self.base_layer.bias is None else self.base_layer.bias.detach().clone()

            dense = nn.Linear(
                self.in_features,
                self.out_features,
                bias=bias is not None,
                dtype=compute_dtype,
                device=weight_bf16.device,
            )
            dense.weight.data.copy_(weight_bf16)
            if bias is not None:
                dense.bias.data.copy_(bias)

            self.base_layer = dense

    @torch.no_grad()
    def apply_sparse_mask(self, sparse_mask: torch.Tensor):
        if sparse_mask is not None:
            base_layer = self.get_base_layer()
            base_layer.weight.data[sparse_mask.cuda()] = 0
            self.sparse_mask = sparse_mask.cpu()

    @torch.no_grad()
    def sparsity(self, eps=1e-8) -> float:
        base_layer = self.get_base_layer()
        num_zeros = torch.sum(torch.abs(base_layer.weight.data) < eps).item()
        total = base_layer.weight.data.numel()
        return num_zeros / total

    def _cast_input_dtype(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if x.dtype == dtype:
            return x
        else:
            return x.to(dtype)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "sqalora." + rep
