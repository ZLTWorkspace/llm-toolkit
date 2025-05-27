import torch


def transpose(weight, fan_in_fan_out):
    if not fan_in_fan_out:
        return weight

    if isinstance(weight, torch.nn.Parameter):
        return torch.nn.Parameter(weight.T)
    return weight.T

@torch.no_grad()
def _get_mask_prune_magnitude(
    W,
    sparsity_ratio: float,
    prune_n: int,
    prune_m: int,
    largest: bool,
    offload: bool,
) -> torch.tensor:
    """
    get mask for pruning based on magnitude.
    largest: if True, prune the largest weights, otherwise prune the smallest weights.
    """
    W_metric = torch.abs(W)
    if prune_n != 0:
        W_mask = torch.zeros_like(W, dtype=torch.bool)
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:, ii : (ii + prune_m)].float()
                idx_to_keep = torch.topk(tmp, prune_n, dim=1, largest=largest)[1]
                W_mask.scatter_(1, ii + idx_to_keep, True)
    else:
        if largest:
            thresh = torch.sort(W_metric.flatten(), descending=True)[0][int(W.numel() * sparsity_ratio)].cpu()
            W_mask = W_metric >= thresh
        else:
            thresh = torch.sort(W_metric.flatten())[0][int(W.numel() * sparsity_ratio)].cpu()
            W_mask = W_metric < thresh
    if offload:
        return W_mask.cpu()
    else:
        return W_mask


@torch.no_grad()
def mergeW2AB(W, A, B, lora_scaling):
    """
    Given:
      - W: a (d x d) base weight matrix
      - A: a (d x r) "LoRA A" matrix
      - B: a (r x d) "LoRA B" matrix
      - lora_scaling: a scalar to apply to (A @ B)

    We solve:
        min_{A1,B1} || W + (A@B)*lora_scaling - ( (A + A1)(B + B1) ) ||_F^2

    Let r = A.shape[1].  The best solution follows from
    taking the best rank-r approximation to M = W + (A @ B)*lora_scaling.
    Then we factor the rank-r approximation with shapes (d x r) and (r x d).
    Finally, we solve for A1 and B1 that yield that factorization.

    Returns:
      A + A1        (of size d x r)
      B + B1        (of size r x d)

    So that (A + A1, B + B1) further factorizes M_approx with rank <= r.
    """
    if len(W.shape) != 2 or len(A.shape) != 2 or len(B.shape) != 2:
        raise ValueError()
    if min(A.shape) != min(B.shape):
        raise ValueError()

    # TODO check r is in [1,2,4,8,16,32,64,128,256,512]
    r = min(A.shape)

    M = W + (B @ A) * lora_scaling
    M = M / lora_scaling
    M = M.to(dtype=torch.float32)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)

    # Extract the top r singular values and corresponding singular vectors:
    # U_r -> d x r matrix (top r left singular vectors)
    # S_r -> r singular values (top r)
    # Vh_r -> r x d matrix (top r right singular vectors transposed)
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]

    # Compute Sigma_r^(1/2) as a diagonal matrix with the square roots of S_r.
    # Since S_r contains nonnegative singular values, taking the square root works elementwise.
    sqrt_S_r = torch.sqrt(S_r)
    Sigma_r_half = torch.diag(sqrt_S_r)

    # Form the optimal factors:
    # X_opt = B + B1 = U_r * Sigma_r^(1/2)
    # Y_opt = A + A1 = Sigma_r^(1/2) * V_r^T. Note: V_r^T is given by Vh_r.
    X_opt = U_r @ Sigma_r_half
    Y_opt = Sigma_r_half @ Vh_r
    WR = X_opt - B
    WL = Y_opt - A

    return WL.to(dtype=A.dtype), WR.to(dtype=B.dtype)


@torch.no_grad()
def decomposeW2LinearWeightLR(W: torch.Tensor, r: int):
    assert len(W.shape) == 2
    assert r in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    M = W.to(dtype=torch.float32)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)

    # Extract the top r singular values and corresponding singular vectors:
    # U_r -> d x r matrix (top r left singular vectors)
    # S_r -> r singular values (top r)
    # Vh_r -> r x d matrix (top r right singular vectors transposed)
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]

    # Compute Sigma_r^(1/2) as a diagonal matrix with the square roots of S_r.
    # Since S_r contains nonnegative singular values, taking the square root works elementwise.
    sqrt_S_r = torch.sqrt(S_r)
    Sigma_r_half = torch.diag(sqrt_S_r)

    # Form the optimal factors:
    # X_opt = U_r * Sigma_r^(1/2)
    # Y_opt = Sigma_r^(1/2) * V_r^T. Note: V_r^T is given by Vh_r.
    R = U_r @ Sigma_r_half
    L = Sigma_r_half @ Vh_r

    return L.to(dtype=W.dtype), R.to(dtype=W.dtype)


def relativeError(a: torch.Tensor, b: torch.Tensor):
    return torch.norm(a - b, p="fro") / torch.norm(a, p="fro")
