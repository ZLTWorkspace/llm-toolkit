from typing import Union

import bitsandbytes as bnb
import torch
import torch.nn as nn

from ..sqalora.utils import (
    _get_mask_prune_magnitude,
    cast_input_dtype,
)


def decompose_and_reconstruct(residual, method="svd", rank=32):
    # residual: [out_features, in_features]
    if method == "svd":
        U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
        r = min(rank, S.size(0))
        rec = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
    elif method == "qr":
        Q, R = torch.linalg.qr(residual)
        rec = Q[:, :rank] @ R[:rank, :]
    elif method == "evd":
        C = residual @ residual.T
        eigvals, eigvecs = torch.linalg.eigh(C)
        top_idx = torch.argsort(eigvals, descending=True)[:rank]
        eigvecs_r = eigvecs[:, top_idx]
        eigvals_r = eigvals[top_idx]
        eigvals_r = torch.clamp(eigvals_r, min=1e-8)
        rec = eigvecs_r @ torch.diag(eigvals_r.sqrt()) @ eigvecs_r.T @ residual
    elif method == "nmf":
        import numpy as np
        from sklearn.decomposition import NMF
        W = residual.cpu().detach().float().numpy()
        model = NMF(n_components=rank, init='random', random_state=0, max_iter=200)
        W_pos = np.abs(W)
        W1 = model.fit_transform(W_pos)
        W2 = model.components_
        rec = torch.tensor(W1 @ W2, dtype=residual.dtype, device=residual.device)
    elif method == "pca":
        import numpy as np
        from sklearn.decomposition import PCA
        W = residual.cpu().detach().float().numpy()
        pca = PCA(n_components=rank)
        W_pca = pca.fit_transform(W)
        W_rec = pca.inverse_transform(W_pca)
        rec = torch.tensor(W_rec, dtype=residual.dtype, device=residual.device)
    else:
        raise ValueError(f"Unknown decompose method: {method}")

    diff = residual - rec
    fro_err = torch.norm(diff, p='fro')
    fro_orig = torch.norm(residual, p='fro')
    rel_err = fro_err / (fro_orig + 1e-8)
    print(f"[{method}] best-rank-{rank} relative fro-error: {rel_err.item():.6f}")
    return rec

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
        self.clinear = None
        self.quant = None
        self.prune = None

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
    def from_linear(cls, linear, quant, prune, compute_dtype):
        cls.validate_params(quant, prune)
        in_features = linear.in_features
        out_features = linear.out_features
        device = linear.weight.device
        obj = cls(
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
                prune_mask = _get_mask_prune_magnitude(W, 0, int(nm[0]), int(nm[1]), False, True)
            prune_mask = prune_mask.to(W.device)
            W[prune_mask] = 0

        if quant:
            quant_option = quant.lower()
            if quant_option == "nf4":
                W_quant, W_quant_state = bnb.functional.quantize_4bit(W, quant_type="nf4")
                W = bnb.functional.dequantize_4bit(W_quant, W_quant_state, quant_type="nf4")
            elif quant_option == "fp4":
                W_quant, W_quant_state = bnb.functional.quantize_4bit(W, quant_type="fp4")
                W = bnb.functional.dequantize_4bit(W_quant, W_quant_state, quant_type="fp4")
            else:
                raise ValueError("Only 'nf4' and 'fp4' quantization supported currently.")

        clinear = nn.Linear(in_features, out_features, bias=linear.bias is not None, dtype=compute_dtype).to(device)
        clinear.weight = nn.Parameter(W.to(device), requires_grad=False)
        if linear.bias is not None:
            clinear.bias = nn.Parameter(linear.bias.detach().clone().to(device), requires_grad=False)
        obj.clinear = clinear
        obj.quant = quant
        obj.prune = prune
        for p in obj.parameters():
            p.requires_grad_(False)
        return obj

    def get_decompressed_weight(self):
        return self.clinear.weight.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cast_input_dtype(x, self.compute_type)
        return self.clinear(x)

class CCLinear(nn.Module):
    """
    Compressed-Compensated Linear:
      y = C(x) + (x @ residual.T) + bias
    """

    def __init__(
        self,
        linear: nn.Linear,
        quant: str = None,
        prune: Union[float, str] = None,
        residual_dtype: torch.dtype = torch.float16,
        compute_type = torch.bfloat16,
        implementation: int = 0,
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

        if implementation == 0:
            self.compressed_linear = CompressedLinear.from_linear(linear, quant, prune, compute_type)
            residual = linear.weight - self.compressed_linear.get_decompressed_weight()

            residual = decompose_and_reconstruct(
                residual.to(torch.float32),
                method="svd",
                rank=32
            ).to(self.residual_dtype)

            self.residual_linear = nn.Linear(self.in_features, self.out_features, bias=False)
            self.residual_linear.weight = nn.Parameter(residual.to(self.residual_dtype), requires_grad=False)

        elif implementation == 1:
            W = linear.weight.detach().clone().to(self.compute_type).contiguous()
            W = decompose_and_reconstruct(W.to(torch.float32), method="svd", rank=32).to(self.compute_type)
            linear.weight.data = linear.weight.data - W.to(linear.weight.dtype)
            self.compressed_linear = CompressedLinear.from_linear(linear, quant, prune, compute_type)
            self.residual_linear = nn.Linear(self.in_features, self.out_features, bias=False)
            self.residual_linear.weight = nn.Parameter(W.to(self.residual_dtype), requires_grad=False)

        else:
            raise ValueError(f"Unsupported implementation: {implementation}")

        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone().to(self.device), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cast_input_dtype(x, self.compute_type)
        # our method is compressed_linear + compressed_linear
        out = self.compressed_linear(x) + self.residual_linear(x)
        # for ablation study, you can try solely compressed_linear, without compensation
        # out = self.compressed_linear(x)
        if self.bias is not None:
            out = out + self.bias
        return out

def replace_linear_with_cclinear(
    module: nn.Module,
    quant: str = None,
    prune: Union[float, str] = None,
    compute_type = torch.bfloat16,
    residual_dtype: torch.dtype = torch.float16,
    prefix: str = "",
    implementation: int = 0,
):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if name == "lm_head":
            continue

        if isinstance(child, nn.Linear):
            new_layer = CCLinear(
                child,
                quant=quant,
                prune=prune,
                residual_dtype=residual_dtype,
                compute_type=compute_type,
                implementation=implementation,
            )
            setattr(module, name, new_layer)
            print(f"[replace_linear_with_cclinear] Replaced {full_name} ({child.in_features}→{child.out_features})")
        else:
            replace_linear_with_cclinear(
                child,
                quant=quant,
                prune=prune,
                compute_type=compute_type,
                residual_dtype=residual_dtype,
                prefix=full_name,
                implementation=implementation,
            )
    return module
