from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import bitsandbytes as bnb
import torch

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils.integrations import dequantize_bnb_weight
from peft.utils.other import transpose

from .layer import SQALoraLayer
from .utils import (
    decomposeW2LinearWeightLR,
    mergeW2AB,
    _get_mask_prune_magnitude,
)


if is_bnb_available():

    class Linear8bitLt(SQALoraLayer):
        # SQALora implemented in a dense layer for bnb 8bit
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
            **kwargs,
        ) -> None:
            super().__init__(base_layer, **kwargs)
            self.fan_in_fan_out = fan_in_fan_out
            self.sparse_preserve_mode = sparse_preserve_mode
            self.update_layer(r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, lora_bias)

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

        def unmerge(self) -> None:
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return
            self.get_base_layer().weight.data -= self.get_delta_weight()
            self.merged = False

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
                for wlr_name in self.WLR_names:
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
                    for wlr_name in self.WLR_names:
                        result_A = result_A + self.WL[wlr_name](x)
                    result_B = lora_B(result_A)
                    for wlr_name in self.WLR_names:
                        result_B = result_B + self.WR[wlr_name](result_A)
                    result = result + result_B * scaling
                elif self.sparse_preserve_mode == 2:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                    for wlr_name in self.WLR_names:
                        result = result + self.WR[wlr_name](self.WL[wlr_name](x))
                else:
                    raise ValueError(f"Unknown sparse preserve mode {self.sparse_preserve_mode}")
                result = result.to(torch_result_dtype)
            return result

        def prune(
            self,
            sparsity_ratio: float = 0.5,
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
            if (prune_n, prune_m) not in [(0,0), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64)]:
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
                base_layer.weight.data[sparse_mask.cuda()] = 0
            elif sparse_preserve_mode == 1:
                tmp_W = base_layer.weight.data.detach().clone()
                base_layer.weight.data[sparse_mask.cuda()] = 0
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
                base_layer.weight.data[sparse_mask.cuda()] = 0
                tmp_W = tmp_W - base_layer.weight.data
                WL, WR = decomposeW2LinearWeightLR(
                    tmp_W,
                    self.r,
                )
                self.update_WL_WR(WL, WR, 2)
                pass
            else:
                raise ValueError(f"Unknown sparse preserve mode {sparse_preserve_mode}")
            return sparse_mask

        def _cast_input_dtype(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            if x.dtype == dtype:
                return x
            else:
                return x.to(dtype)

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "sqalora." + rep



if is_bnb_4bit_available():

    class Linear4bit(SQALoraLayer):
        # SQALora implemented in a dense layer for bnb 4bit
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
            **kwargs,
        ) -> None:
            super().__init__(base_layer, **kwargs)
            self.fan_in_fan_out = fan_in_fan_out
            self.sparse_preserve_mode = sparse_preserve_mode
            self.update_layer(r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, lora_bias)

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

        def unmerge(self) -> None:
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return
            self.get_base_layer().weight.data -= self.get_delta_weight()
            self.merged = False

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
                for wlr_name in self.WLR_names:
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
                    for wlr_name in self.WLR_names:
                        result_A = result_A + self.WL[wlr_name](x)
                    result_B = lora_B(result_A)
                    for wlr_name in self.WLR_names:
                        result_B = result_B + self.WR[wlr_name](result_A)
                    result = result + result_B * scaling
                elif self.sparse_preserve_mode == 2:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                    for wlr_name in self.WLR_names:
                        result = result + self.WR[wlr_name](self.WL[wlr_name](x))
                else:
                    raise ValueError(f"Unknown sparse preserve mode {self.sparse_preserve_mode}")
                result = result.to(torch_result_dtype)
            return result

        def prune(
            self,
            sparsity_ratio: float = 0.5,
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
            if (prune_n, prune_m) not in [(0,0), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64)]:
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
                base_layer.weight.data[sparse_mask.cuda()] = 0
            elif sparse_preserve_mode == 1:
                tmp_W = base_layer.weight.data.detach().clone()
                base_layer.weight.data[sparse_mask.cuda()] = 0
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
                base_layer.weight.data[sparse_mask.cuda()] = 0
                tmp_W = tmp_W - base_layer.weight.data
                WL, WR = decomposeW2LinearWeightLR(
                    tmp_W,
                    self.r,
                )
                self.update_WL_WR(WL, WR, 2)
                pass
            else:
                raise ValueError(f"Unknown sparse preserve mode {sparse_preserve_mode}")
            return sparse_mask

        def _cast_input_dtype(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            if x.dtype == dtype:
                return x
            else:
                return x.to(dtype)

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "sqalora." + rep
