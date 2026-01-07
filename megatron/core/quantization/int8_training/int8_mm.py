# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from TorchAO: https://github.com/pytorch/ao
# Modified for Megatron-LM integration.

"""
INT8 matrix multiplication kernels using Triton.

This module provides optimized INT8 matmul with row/column scaling,
which is the core computation for INT8 mixed-precision training.
"""

import torch
from torch import Tensor

# Check for Triton availability
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    # Triton autotune configurations
    # (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
    _CONFIGS = [
        (128, 256, 64, 3, 8),
        (64, 256, 32, 4, 4),
        (128, 128, 32, 4, 4),
        (128, 64, 32, 4, 4),
        (64, 128, 32, 4, 4),
        (128, 32, 32, 4, 4),
        (64, 32, 32, 5, 2),
        (32, 64, 32, 5, 2),
        # Good config for fp8 inputs
        (128, 256, 128, 3, 8),
        (256, 128, 128, 3, 8),
        (256, 64, 128, 4, 4),
        (64, 256, 128, 4, 4),
        (128, 128, 128, 4, 4),
        (128, 64, 64, 4, 4),
        (64, 128, 64, 4, 4),
        (128, 32, 64, 4, 4),
        # From PyTorch inductor
        (64, 64, 32, 2, 4),
        (64, 128, 32, 3, 4),
        (128, 64, 32, 3, 4),
        (64, 128, 32, 4, 8),
        (128, 64, 32, 4, 8),
        (64, 32, 32, 5, 8),
        (32, 64, 32, 5, 8),
        (128, 128, 32, 2, 8),
        (64, 64, 64, 3, 8),
        (128, 256, 128, 3, 8),
        (256, 128, 128, 3, 8),
    ]

    _triton_configs = [
        triton.Config(
            dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K),
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps in _CONFIGS
    ]

    @triton.autotune(configs=_triton_configs, key=["M", "N", "K", "stride_ak", "stride_bk"])
    @triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0})
    @triton.jit
    def _scaled_int8_mm_kernel(
        A_ptr,
        B_ptr,
        C_ptr,
        row_scale_ptr,
        col_scale_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr = 8,
        EVEN_K: tl.constexpr = True,
        COL_SCALE_SCALAR: tl.constexpr = False,
    ):
        """Triton kernel for scaled INT8 matrix multiplication."""
        pid = tl.program_id(0)
        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_n = (N + BLOCK_N - 1) // BLOCK_N

        # Re-order program ID for better L2 performance
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // (group_size)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        A = A_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
        B = B_ptr + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k in range(K, 0, -BLOCK_K):
            if EVEN_K:
                a = tl.load(A)
                b = tl.load(B)
            else:
                a = tl.load(A, mask=rk[None, :] < k, other=0.0)
                b = tl.load(B, mask=rk[:, None] < k, other=0.0)
            acc += tl.dot(a, b)
            A += BLOCK_K * stride_ak
            B += BLOCK_K * stride_bk

        # Rematerialize rm and rn to save registers
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)

        row_scale = tl.load(row_scale_ptr + idx_m, mask=idx_m < M).to(tl.float32)
        if COL_SCALE_SCALAR:
            col_scale = tl.load(col_scale_ptr).to(tl.float32)
        else:
            col_scale = tl.load(col_scale_ptr + idx_n, mask=idx_n < N).to(tl.float32)
        acc = acc.to(tl.float32) * row_scale * col_scale

        xindex = idx_m * stride_cm + idx_n * stride_cn
        tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)


def scaled_int8_mm(
    A: Tensor, B: Tensor, row_scale: Tensor, col_scale: Tensor
) -> Tensor:
    """Compute `(A @ B) * row_scale * col_scale`, where `A` and `B` are INT8.
    
    This leverages INT8 tensor cores for faster computation.
    
    Args:
        A: INT8 tensor of shape (M, K)
        B: INT8 tensor of shape (K, N)
        row_scale: Scale tensor of shape (M,) or (M, 1)
        col_scale: Scale tensor of shape (N,) or (1, N), or scalar
        
    Returns:
        Result tensor of shape (M, N) with dtype matching scale tensors
    """
    if not HAS_TRITON:
        # Fallback: less performant but works without Triton
        return torch._int_mm(A, B) * col_scale.view(-1) * row_scale.view(-1, 1)
    
    assert A.dtype is torch.int8 and B.dtype is torch.int8
    assert row_scale.dtype is col_scale.dtype
    assert A.shape[1] == B.shape[0]
    assert row_scale.squeeze().shape == (A.shape[0],)
    assert col_scale.squeeze().shape in ((B.shape[1],), ())
    assert row_scale.is_contiguous()
    assert col_scale.is_contiguous()
    
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device=A.device, dtype=row_scale.dtype)
    
    grid = lambda meta: (
        triton.cdiv(meta["M"], meta["BLOCK_M"])
        * triton.cdiv(meta["N"], meta["BLOCK_N"]),
    )
    
    _scaled_int8_mm_kernel[grid](
        A,
        B,
        C,
        row_scale,
        col_scale,
        M,
        N,
        K,
        *A.stride(),
        *B.stride(),
        *C.stride(),
        COL_SCALE_SCALAR=col_scale.numel() == 1,
    )
    return C


@torch.no_grad()
def quantize_int8_rowwise(
    tensor: Tensor, stochastic_rounding: bool = False, eps: float = 1e-8
):
    """Quantize a tensor to INT8 with row-wise scaling.
    
    Uses symmetric quantization with absmax scaling [-127, 127].
    
    Args:
        tensor: Input tensor to quantize
        stochastic_rounding: If True, use stochastic rounding (helps with small updates)
        eps: Small value to prevent division by zero
        
    Returns:
        Tuple of (int8_tensor, scale_tensor)
    """
    # Absmax symmetric quantization using [-127, 127] range
    # This ensures symmetric quantization: -max_val maps to -127, +max_val maps to +127
    scale = tensor.abs().amax(1) / 127  # same dtype as tensor
    # Increased eps from 1e-12 to 1e-8 for better numerical stability
    inv_scale = 1.0 / scale.float().clip(min=eps)
    tensor = tensor.float() * inv_scale.view(-1, 1)

    if stochastic_rounding:
        tensor = (tensor + torch.rand_like(tensor)).floor()
    else:
        tensor = tensor.round()

    # Use symmetric range [-127, 127] to match the scale calculation
    # This avoids numerical asymmetry when dequantizing
    tensor = tensor.clip(-127, 127).to(torch.int8)
    return tensor, scale

