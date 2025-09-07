from typing import Union

import bitsandbytes as bnb
import torch
import torch.nn as nn

from .utils import (
    _get_mask_prune_magnitude,
    cast_input_dtype,
)


class CompressedLinear(nn.Module):
    """
    Linear layer with pseudo compression (quantization and/or pruning).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        compute_type = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.compute_type = compute_type if compute_type is not None else torch.bfloat16

    @staticmethod
    def validate_params(quant, prune):
        if quant is not None:
            quant_option = quant.lower()
            if quant_option not in ["nf4", "fp4"]:
                raise ValueError("Only 'nf4' and 'fp4' quantization supported currently.")
        if prune is not None:
            if not (isinstance(prune, float) or isinstance(prune, str)):
                raise ValueError("Prune must be a float ratio or a structured pruning string like '2:4'.")
            if isinstance(prune, float):
                if prune < 0.0 or prune >= 1.0:
                    raise ValueError("Prune ratio must be in [0.0, 1.0).")
            elif isinstance(prune, str):
                nm = prune.split(":")
                if len(nm) != 2:
                    raise ValueError("Structured pruning string must be like '2:4'.")
                try:
                    n = int(nm[0])
                    m = int(nm[1])
                except Exception:
                    raise ValueError("Structured pruning string must be like '2:4'.")
                if n <= 0 or m <= 0 or n > m:
                    raise ValueError("Structured pruning string must be like '2:4' with 0 < n <= m.")

    @classmethod
    def from_linear(self, linear, quant, prune, compute_dtype):
        self.validate_params(quant, prune)
        in_features = linear.in_features
        out_features = linear.out_features
        device = linear.weight.device
        CL = CompressedLinear(
            in_features,
            out_features,
            linear.bias is not None,
            compute_dtype,
        )
        W = linear.weight.detach().clone().to(compute_dtype).contiguous()

        if prune:
            if isinstance(prune, float):
                prune_mask = _get_mask_prune_magnitude(W, prune, 0, 0, False, True)
            else:
                nm = prune.split(":")
                prune_mask = _get_mask_prune_magnitude(W, 0, nm[0], nm[1], False, True)
            W[prune_mask.cuda()] = 0

        if quant:
            quant_option = quant.lower()
            if quant_option == "nf4":
                W_quant, W_quant_state = bnb.functional.quantize_4bit(W, quant_type="nf4")
                W = bnb.functional.dequantize_4bit(W_quant, W_quant_state, quant_type="nf4")
            elif quant_option == "fp4":
                W_quant, W_quant_state = bnb.functional.quantize_4bit(W, quant_type="nf4")
                W = bnb.functional.dequantize_4bit(W_quant, W_quant_state, quant_type="nf4")
            else:
                raise ValueError("Only 'nf4' and 'fp4' quantization supported currently.")

        clinear = nn.Linear(in_features, out_features, bias=linear.bias is not None, dtype = compute_dtype).to(device)
        clinear.weight = nn.Parameter(W.to(device), requires_grad=False)
        CL.clinear = clinear
        CL.quant = quant
        CL.prune = prune
        for p in CL.parameters():
            p.requires_grad_(False)
        return CL

    def get_decompressed_weight(self):
        return self.clinear.weight.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, ..., in_features]
        Returns:
          y = clinear(x)
        """
        x = cast_input_dtype(x, self.compute_type)
        return self.clinear(x)

class CCLinear(nn.Module):
    """
    Compressed-Compensated Linear:
      y = C(x) + (x @ residual.T) + bias
    where:
      - C is a Compressed linear (quantization or prune).
      - residual = W_full - W_c(de-compressed to fp16), stored in fp16/bf16.
      - bias is copied from the original layer.
    """

    def __init__(
        self,
        linear: nn.Linear,
        quant: str = None,
        prune: Union[float, str] = None,
        residual_dtype: torch.dtype = torch.float16,
        compute_type = torch.bfloat16,
    ):
        super().__init__()

        self.quant = quant
        self.prune = prune
        self.residual_dtype = residual_dtype
        self.device = linear.weight.device
        self.compute_type = compute_type if compute_type is not None else torch.bfloat16

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = None

        self.compressed_linear = CompressedLinear.from_linear(linear, quant, prune, compute_type)
        residual = linear.weight.detach() - self.compressed_linear.get_decompressed_weight()
        self.residual_linear = nn.Linear(self.in_features, self.out_features, bias=False)
        self.residual_linear.weight = nn.Parameter(residual.to(self.residual_dtype), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, ..., in_features]
        Returns:
          y = c(x) + residual(x)
        """
        x = cast_input_dtype(x, self.compute_type)
        return self.compressed_linear(x) + self.residual_linear(x)


if __name__ == "__main__":
    torch.manual_seed(0)

    lin = nn.Linear(1024, 256, bias=False)
    cc = CCLinear(lin, prune=0.5, residual_dtype=torch.bfloat16)

    for name, p in cc.named_parameters():
        print(name, p.shape, p.requires_grad)

    x = torch.randn(8, 1024)
    y = cc(x)
    print(y.shape)  # [8, 256]
