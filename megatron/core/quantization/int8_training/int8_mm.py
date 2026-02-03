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
    
    # Group-wise kernel configs
    # IMPORTANT: BLOCK_K must equal GROUP_SIZE for correct scaling!
    # We create separate config sets for different group sizes
    
    def _make_groupwise_configs(block_k):
        """Create autotune configs for a specific BLOCK_K (= GROUP_SIZE).
        
        Note: For smaller BLOCK_K (e.g., 16), we use fewer warps to avoid
        issues with shared memory and register pressure.
        """
        if block_k >= 64:
            configs = [
                (128, 256, block_k, 3, 8),
                (64, 256, block_k, 4, 4),
                (128, 128, block_k, 4, 4),
                (128, 64, block_k, 4, 4),
                (64, 128, block_k, 4, 4),
                (64, 64, block_k, 3, 8),
                (32, 64, block_k, 3, 4),
                (64, 32, block_k, 3, 4),
                (32, 32, block_k, 2, 4),
            ]
        elif block_k >= 32:
            configs = [
                (128, 128, block_k, 3, 4),
                (64, 128, block_k, 3, 4),
                (128, 64, block_k, 3, 4),
                (64, 64, block_k, 3, 4),
                (32, 64, block_k, 2, 4),
                (64, 32, block_k, 2, 4),
                (32, 32, block_k, 2, 4),
            ]
        else:  # block_k == 16
            # For small BLOCK_K, use conservative settings
            configs = [
                (64, 64, block_k, 2, 4),
                (32, 64, block_k, 2, 4),
                (64, 32, block_k, 2, 4),
                (32, 32, block_k, 2, 4),
                (32, 32, block_k, 2, 2),
            ]
        return [
            triton.Config(
                dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K),
                num_stages=num_stages,
                num_warps=num_warps,
            )
            for BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps in configs
        ]
    
    # Pre-built configs for common group sizes
    _groupwise_configs_16 = _make_groupwise_configs(16)
    _groupwise_configs_32 = _make_groupwise_configs(32)
    _groupwise_configs_64 = _make_groupwise_configs(64)
    
    # Default configs (for group_size=64)
    _groupwise_triton_configs = _groupwise_configs_64

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

    # Use a factory function to create kernels for different group sizes
    # This avoids issues with nested JIT calls while still sharing the logic
    
    def _make_groupwise_kernel(group_size, configs):
        """Factory function to create a groupwise matmul kernel for a specific group size."""
        
        @triton.autotune(configs=configs, key=["M", "N", "K"])
        @triton.jit
        def kernel(
            A_ptr, B_ptr, C_ptr, A_scale_ptr, B_scale_ptr,
            M, N, K, num_groups,
            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
            stride_as_m, stride_as_g, stride_bs_n, stride_bs_g,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr = 8,
        ):
            """Triton kernel for group-wise scaled INT8 matrix multiplication.
            
            IMPORTANT: BLOCK_K must equal GROUP_SIZE for correct scaling!
            Each group of GROUP_SIZE elements along K has its own scale factor.
            """
            pid = tl.program_id(0)
            grid_m = (M + BLOCK_M - 1) // BLOCK_M
            grid_n = (N + BLOCK_N - 1) // BLOCK_N

            # Re-order program ID for better L2 performance
            width = GROUP_M * grid_n
            group_id = pid // width
            group_sz = min(grid_m - group_id * GROUP_M, GROUP_M)
            pid_m = group_id * GROUP_M + (pid % group_sz)
            pid_n = (pid % width) // group_sz

            rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
            rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
            
            # Float32 accumulator for group-wise scaling
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            
            # Iterate over K dimension in blocks of BLOCK_K (= GROUP_SIZE)
            for k_block in range(0, K, BLOCK_K):
                rk = tl.arange(0, BLOCK_K)
                
                # Load A and B blocks
                A = A_ptr + (ram[:, None] * stride_am + (k_block + rk[None, :]) * stride_ak)
                B = B_ptr + ((k_block + rk[:, None]) * stride_bk + rbn[None, :] * stride_bn)
                
                # Mask for K boundary
                k_mask = (k_block + rk) < K
                a = tl.load(A, mask=k_mask[None, :], other=0)
                b = tl.load(B, mask=k_mask[:, None], other=0)
                
                # INT8 dot product -> int32
                partial = tl.dot(a, b).to(tl.float32)
                
                # BLOCK_K == GROUP_SIZE, so group_idx = k_block // BLOCK_K
                group_idx = k_block // BLOCK_K
                
                # Load scales for this group
                idx_m = rm[:, None]
                idx_n = rn[None, :]
                
                a_scale = tl.load(
                    A_scale_ptr + idx_m * stride_as_m + group_idx * stride_as_g,
                    mask=idx_m < M
                ).to(tl.float32)
                b_scale = tl.load(
                    B_scale_ptr + idx_n * stride_bs_n + group_idx * stride_bs_g,
                    mask=idx_n < N
                ).to(tl.float32)
                
                # Apply per-group scaling and accumulate
                acc += partial * a_scale * b_scale

            # Store result
            rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            idx_m = rm[:, None]
            idx_n = rn[None, :]
            mask = (idx_m < M) & (idx_n < N)
            
            xindex = idx_m * stride_cm + idx_n * stride_cn
            tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)
        
        return kernel
    
    # Pre-build kernels for each supported group size
    _scaled_int8_mm_groupwise_kernel_g16 = _make_groupwise_kernel(16, _groupwise_configs_16)
    _scaled_int8_mm_groupwise_kernel_g32 = _make_groupwise_kernel(32, _groupwise_configs_32)
    _scaled_int8_mm_groupwise_kernel_g64 = _make_groupwise_kernel(64, _groupwise_configs_64)
    
    
    def _make_two_stage_kernel(group_size, configs):
        """Factory function to create a two-stage quantization kernel for a specific group size.
        
        Two-stage quantization: top-k outliers and remaining elements have separate scales.
        Uses INT8 tensor cores by splitting into two matmuls: topk and others.
        
        OPTIMIZED: Accepts pre-split A_topk and A_others to avoid mask operations in kernel.
        C = (A_topk @ B) * scale_topk * scale_B + (A_others @ B) * scale_others * scale_B
        """
        
        @triton.autotune(configs=configs, key=["M", "N", "K"])
        @triton.jit
        def kernel(
            A_topk_ptr, A_others_ptr, B_ptr, C_ptr, 
            A_scales_ptr, B_scales_ptr,
            M, N, K, num_groups,
            stride_am, stride_ak, 
            stride_bk, stride_bn, 
            stride_cm, stride_cn,
            stride_as_m, stride_as_g, stride_as_scale,
            stride_bs_n, stride_bs_g,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr = 8,
        ):
            """Triton kernel for two-stage group-wise scaled INT8 matmul.
            
            OPTIMIZED: A_topk and A_others are pre-computed outside kernel.
            """
            pid = tl.program_id(0)
            grid_m = (M + BLOCK_M - 1) // BLOCK_M
            grid_n = (N + BLOCK_N - 1) // BLOCK_N

            # Re-order program ID for better L2 performance
            width = GROUP_M * grid_n
            group_id = pid // width
            group_sz = min(grid_m - group_id * GROUP_M, GROUP_M)
            pid_m = group_id * GROUP_M + (pid % group_sz)
            pid_n = (pid % width) // group_sz

            rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
            rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
            
            # Float32 accumulator
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            
            # Iterate over K dimension in blocks of BLOCK_K (= GROUP_SIZE)
            for k_block in range(0, K, BLOCK_K):
                rk = tl.arange(0, BLOCK_K)
                
                # Mask for K boundary
                k_mask = (k_block + rk) < K
                m_mask = ram[:, None] < M
                n_mask = rbn[None, :] < N
                
                # Load pre-split A_topk and A_others
                A_topk = A_topk_ptr + (ram[:, None] * stride_am + (k_block + rk[None, :]) * stride_ak)
                A_others = A_others_ptr + (ram[:, None] * stride_am + (k_block + rk[None, :]) * stride_ak)
                B = B_ptr + ((k_block + rk[:, None]) * stride_bk + rbn[None, :] * stride_bn)
                
                a_topk_i8 = tl.load(A_topk, mask=m_mask & k_mask[None, :], other=0)
                a_others_i8 = tl.load(A_others, mask=m_mask & k_mask[None, :], other=0)
                b_i8 = tl.load(B, mask=k_mask[:, None] & n_mask, other=0)
                
                # INT8 matmul using tensor cores (int8 @ int8 -> int32)
                acc_topk_i32 = tl.dot(a_topk_i8, b_i8)
                acc_others_i32 = tl.dot(a_others_i8, b_i8)
                
                # Load scales
                group_idx = k_block // BLOCK_K
                idx_m = ram[:, None]
                idx_n = rbn[None, :]
                
                a_scale_topk = tl.load(
                    A_scales_ptr + idx_m * stride_as_m + group_idx * stride_as_g + 0 * stride_as_scale,
                    mask=idx_m < M
                ).to(tl.float32)
                a_scale_others = tl.load(
                    A_scales_ptr + idx_m * stride_as_m + group_idx * stride_as_g + 1 * stride_as_scale,
                    mask=idx_m < M
                ).to(tl.float32)
                b_scale = tl.load(
                    B_scales_ptr + idx_n * stride_bs_n + group_idx * stride_bs_g,
                    mask=idx_n < N
                ).to(tl.float32)
                
                # Combine scale multiplications
                acc += (acc_topk_i32 * a_scale_topk + acc_others_i32 * a_scale_others) * b_scale

            # Store result
            rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            idx_m = rm[:, None]
            idx_n = rn[None, :]
            mask = (idx_m < M) & (idx_n < N)
            
            xindex = idx_m * stride_cm + idx_n * stride_cn
            tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)
        
        return kernel
    
    # Pre-build two-stage kernels for common group sizes
    _scaled_int8_mm_two_stage_kernel_g16 = _make_two_stage_kernel(16, _groupwise_configs_16)
    _scaled_int8_mm_two_stage_kernel_g32 = _make_two_stage_kernel(32, _groupwise_configs_32)
    _scaled_int8_mm_two_stage_kernel_g64 = _make_two_stage_kernel(64, _groupwise_configs_64)


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


def scaled_int8_mm_groupwise(
    A: Tensor, B: Tensor, A_scale: Tensor, B_scale: Tensor, group_size: int = 64
) -> Tensor:
    """Compute group-wise scaled INT8 matmul: A @ B with per-group scaling.
    
    Each group of `group_size` elements along K dimension has its own scale.
    This is useful for Tensor Parallelism compatibility.
    
    Args:
        A: INT8 tensor of shape (M, K)
        B: INT8 tensor of shape (K, N)
        A_scale: Scale tensor of shape (M, num_groups) where num_groups = ceil(K/group_size)
        B_scale: Scale tensor of shape (N, num_groups)
        group_size: Number of elements per group (default: 64)
        
    Returns:
        Result tensor of shape (M, N) with dtype matching scale tensors
    """
    if not HAS_TRITON:
        # Fallback: loop over groups
        M, K = A.shape
        _, N = B.shape
        num_groups = (K + group_size - 1) // group_size
        result = torch.zeros(M, N, device=A.device, dtype=A_scale.dtype)
        
        for g in range(num_groups):
            k_start = g * group_size
            k_end = min((g + 1) * group_size, K)
            A_g = A[:, k_start:k_end]
            B_g = B[k_start:k_end, :]
            # Dequantize and accumulate
            A_dq = A_g.float() * A_scale[:, g:g+1]
            B_dq = B_g.float() * B_scale[:, g:g+1].T
            result += A_dq @ B_dq
        return result.to(A_scale.dtype)
    
    assert A.dtype is torch.int8 and B.dtype is torch.int8
    assert A_scale.dtype is B_scale.dtype
    assert A.shape[1] == B.shape[0]
    
    M, K = A.shape
    _, N = B.shape
    num_groups = (K + group_size - 1) // group_size
    
    assert A_scale.shape == (M, num_groups), f"A_scale shape {A_scale.shape} != ({M}, {num_groups})"
    assert B_scale.shape == (N, num_groups), f"B_scale shape {B_scale.shape} != ({N}, {num_groups})"
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    A_scale = A_scale.contiguous()
    B_scale = B_scale.contiguous()
    
    C = torch.empty(M, N, device=A.device, dtype=A_scale.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    
    # Select the correct kernel based on group_size
    # Each kernel has BLOCK_K == group_size to ensure correct scaling
    if group_size == 16:
        kernel = _scaled_int8_mm_groupwise_kernel_g16
    elif group_size == 32:
        kernel = _scaled_int8_mm_groupwise_kernel_g32
    elif group_size == 64:
        kernel = _scaled_int8_mm_groupwise_kernel_g64
    else:
        raise ValueError(f"Unsupported group_size={group_size}. Must be 16, 32, or 64.")
    
    kernel[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        M,
        N,
        K,
        num_groups,
        *A.stride(),
        *B.stride(),
        *C.stride(),
        *A_scale.stride(),
        *B_scale.stride(),
    )
    return C


def scaled_int8_mm_two_stage(
    A: Tensor, B: Tensor, 
    A_scales: Tensor, B_scales: Tensor,
    A_mask: Tensor,
    group_size: int = 64
) -> Tensor:
    """Compute two-stage group-wise scaled INT8 matmul: A @ B.
    
    A uses two-stage quantization (top-k outliers + others with separate scales).
    B uses standard group-wise quantization (one scale per group).
    
    Args:
        A: INT8 tensor of shape (M, K)
        B: INT8 tensor of shape (K, N)
        A_scales: Scale tensor of shape (M, num_groups, 2) - [scale_topk, scale_others]
        B_scales: Scale tensor of shape (N, num_groups) - standard groupwise scales
        A_mask: Bool mask of shape (M, K) indicating top-k positions in A
        group_size: Number of elements per group (default: 64)
        
    Returns:
        Result tensor of shape (M, N) with dtype matching scale tensors
    """
    if not HAS_TRITON:
        # Fallback: element-wise scaling for A, standard for B
        M, K = A.shape
        _, N = B.shape
        num_groups = (K + group_size - 1) // group_size
        result = torch.zeros(M, N, device=A.device, dtype=A_scales.dtype)
        
        for g in range(num_groups):
            k_start = g * group_size
            k_end = min((g + 1) * group_size, K)
            A_g = A[:, k_start:k_end].float()
            B_g = B[k_start:k_end, :].float()
            A_mask_g = A_mask[:, k_start:k_end]
            
            # Two-stage scaling for A
            A_scale_topk = A_scales[:, g, 0:1]
            A_scale_others = A_scales[:, g, 1:2]
            A_scale_g = torch.where(A_mask_g, A_scale_topk, A_scale_others)
            
            # Standard groupwise scaling for B
            B_scale_g = B_scales[:, g:g+1]  # [N, 1]
            
            # Dequantize and accumulate
            A_dq = A_g * A_scale_g  # [M, group_size]
            B_dq = B_g * B_scale_g.T  # [group_size, N]
            result += A_dq @ B_dq
        
        return result.to(A_scales.dtype)
    
    assert A.dtype is torch.int8 and B.dtype is torch.int8
    assert A_scales.dtype is B_scales.dtype
    assert A.shape[1] == B.shape[0]
    
    M, K = A.shape
    _, N = B.shape
    num_groups = (K + group_size - 1) // group_size
    
    assert A_scales.shape == (M, num_groups, 2), f"A_scales shape {A_scales.shape} != ({M}, {num_groups}, 2)"
    assert B_scales.shape == (N, num_groups), f"B_scales shape {B_scales.shape} != ({N}, {num_groups})"
    assert A_mask.shape == (M, K), f"A_mask shape {A_mask.shape} != ({M}, {K})"
    
    # Pre-split A into topk and others
    A_topk = torch.where(A_mask, A, torch.zeros_like(A))
    A_others = torch.where(A_mask, torch.zeros_like(A), A)
    
    # Ensure contiguous
    A_topk = A_topk.contiguous()
    A_others = A_others.contiguous()
    B = B.contiguous()
    A_scales = A_scales.contiguous()
    B_scales = B_scales.contiguous()
    
    C = torch.empty(M, N, device=A.device, dtype=A_scales.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    
    # Select the correct kernel based on group_size
    if group_size == 16:
        kernel = _scaled_int8_mm_two_stage_kernel_g16
    elif group_size == 32:
        kernel = _scaled_int8_mm_two_stage_kernel_g32
    elif group_size == 64:
        kernel = _scaled_int8_mm_two_stage_kernel_g64
    else:
        raise ValueError(f"Unsupported group_size={group_size}. Must be 16, 32, or 64.")
    
    kernel[grid](
        A_topk,
        A_others,
        B,
        C,
        A_scales,
        B_scales,
        M,
        N,
        K,
        num_groups,
        A_topk.stride(0), A_topk.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        A_scales.stride(0), A_scales.stride(1), A_scales.stride(2),
        B_scales.stride(0), B_scales.stride(1),
    )
    return C


@torch.no_grad()
def quantize_int8_rowwise(
    tensor: Tensor, stochastic_rounding: bool = False, eps: float = 1e-20
):
    """Quantize a tensor to INT8 with row-wise scaling.
    
    Uses symmetric quantization with absmax scaling [-127, 127].
    
    Args:
        tensor: Input tensor to quantize
        stochastic_rounding: If True, use stochastic rounding (helps with small updates)
        eps: Small value to prevent division by zero (default 1e-20 for small gradients)
        
    Returns:
        Tuple of (int8_tensor, scale_tensor)
    """
    # Absmax symmetric quantization using [-127, 127] range
    # This ensures symmetric quantization: -max_val maps to -127, +max_val maps to +127
    scale = tensor.abs().amax(1) / 127  # same dtype as tensor
    # Use small eps (1e-20) to handle very small gradients (e.g., 1e-14)
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


@torch.no_grad()
def quantize_int8_groupwise(
    tensor: Tensor, group_size: int = 64, stochastic_rounding: bool = False, eps: float = 1e-20
):
    """Quantize a tensor to INT8 with group-wise scaling along K dimension.
    
    Each group of `group_size` elements shares a scale factor.
    This is useful for Tensor Parallelism where K dimension may be split.
    
    Args:
        tensor: Input tensor of shape [M, K]
        group_size: Number of elements per group (default: 64)
        stochastic_rounding: If True, use stochastic rounding
        eps: Small value to prevent division by zero (default 1e-20 for small gradients)
        
    Returns:
        Tuple of (int8_tensor [M, K], scales [M, num_groups])
    """
    M, K = tensor.shape
    
    # Handle case where K is not divisible by group_size
    if K % group_size != 0:
        # Pad K to be divisible by group_size
        pad_size = group_size - (K % group_size)
        tensor = torch.nn.functional.pad(tensor, (0, pad_size), value=0)
        K_padded = K + pad_size
    else:
        K_padded = K
        pad_size = 0
    
    num_groups = K_padded // group_size
    
    # Reshape to [M, num_groups, group_size]
    tensor_grouped = tensor.reshape(M, num_groups, group_size)
    
    # Compute scale per group: [M, num_groups]
    scales = tensor_grouped.abs().amax(dim=2) / 127
    
    # Quantize
    inv_scales = 1.0 / scales.float().clip(min=eps)
    tensor_grouped = tensor_grouped.float() * inv_scales.unsqueeze(-1)
    
    if stochastic_rounding:
        tensor_grouped = (tensor_grouped + torch.rand_like(tensor_grouped)).floor()
    else:
        tensor_grouped = tensor_grouped.round()
    
    tensor_grouped = tensor_grouped.clip(-127, 127).to(torch.int8)
    
    # Reshape back to [M, K_padded]
    int8_tensor = tensor_grouped.reshape(M, K_padded)
    
    # Remove padding if applied
    if pad_size > 0:
        int8_tensor = int8_tensor[:, :K]
    
    return int8_tensor, scales


# ============================================================================
# Triton kernel for groupwise quantization along K (first dimension)
# ============================================================================

if HAS_TRITON:
    @triton.jit
    def _quantize_int8_groupwise_along_k_kernel(
        # Input/Output pointers
        input_ptr,      # [K, N] input tensor
        output_ptr,     # [K, N] int8 output tensor
        scales_ptr,     # [N, num_groups] scales output
        # Dimensions
        K, N, num_groups,
        # Strides
        stride_input_k, stride_input_n,
        stride_output_k, stride_output_n,
        stride_scales_n, stride_scales_g,
        # Parameters
        eps: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Triton kernel for groupwise INT8 quantization along K dimension.
        
        Input: [K, N], quantize along K (groups of GROUP_SIZE rows)
        Output: [K, N] int8, scales [N, num_groups]
        
        Grid: (num_groups, cdiv(N, BLOCK_N))
        """
        pid_g = tl.program_id(0)  # group index along K
        pid_n = tl.program_id(1)  # block index along N
        
        # K indices for this group
        k_start = pid_g * GROUP_SIZE
        rk = tl.arange(0, GROUP_SIZE)
        k_indices = k_start + rk
        k_mask = k_indices < K
        
        # N indices for this block
        n_start = pid_n * BLOCK_N
        rn = tl.arange(0, BLOCK_N)
        n_indices = n_start + rn
        n_mask = n_indices < N
        
        # Load input data: [GROUP_SIZE, BLOCK_N]
        input_ptrs = input_ptr + k_indices[:, None] * stride_input_k + n_indices[None, :] * stride_input_n
        load_mask = k_mask[:, None] & n_mask[None, :]
        data = tl.load(input_ptrs, mask=load_mask, other=0.0)
        
        # Compute abs max per column (per N position): [BLOCK_N]
        abs_data = tl.abs(data)
        max_abs = tl.max(abs_data, axis=0)  # [BLOCK_N]
        
        # Compute scales in fp32
        scale = (max_abs / 127.0).to(tl.float32)  # [BLOCK_N]
        
        # Compute inverse scale
        inv_scale = 1.0 / tl.maximum(scale, eps)  # [BLOCK_N]
        
        # Quantize
        scaled_data = data.to(tl.float32) * inv_scale[None, :]  # [GROUP_SIZE, BLOCK_N]
        rounded = tl.math.floor(scaled_data + 0.5)
        clamped = tl.minimum(tl.maximum(rounded, -127.0), 127.0)
        int8_data = clamped.to(tl.int8)
        
        # Store int8 output
        output_ptrs = output_ptr + k_indices[:, None] * stride_output_k + n_indices[None, :] * stride_output_n
        tl.store(output_ptrs, int8_data, mask=load_mask)
        
        # Store scales: [N, num_groups]
        scale_ptrs = scales_ptr + n_indices * stride_scales_n + pid_g * stride_scales_g
        tl.store(scale_ptrs, scale, mask=n_mask)


@torch.no_grad()
def quantize_int8_groupwise_along_k(
    tensor: Tensor, group_size: int = 64, eps: float = 1e-20
):
    """Quantize a [K, N] tensor to INT8 with group-wise scaling along K dimension.
    
    This avoids the need for transpose when quantizing weight matrices.
    Each group of `group_size` elements along K shares a scale factor per N column.
    
    Args:
        tensor: Input tensor of shape [K, N]
        group_size: Number of elements per group along K (default: 64)
        eps: Small value to prevent division by zero
        
    Returns:
        Tuple of (int8_tensor [K, N], scales [N, num_groups] in fp32)
    """
    K, N = tensor.shape
    orig_K = K
    
    # Handle case where K is not divisible by group_size
    if K % group_size != 0:
        pad_size = group_size - (K % group_size)
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_size), value=0)
        K = K + pad_size
    else:
        pad_size = 0
    
    num_groups = K // group_size
    
    # Ensure tensor is contiguous
    tensor = tensor.contiguous()
    
    # Check if we can use Triton
    use_triton = (
        HAS_TRITON 
        and tensor.is_cuda
        and (group_size & (group_size - 1)) == 0  # Power of 2 check
    )
    
    if use_triton:
        # Allocate output tensors
        int8_output = torch.empty(K, N, dtype=torch.int8, device=tensor.device)
        scales = torch.empty(N, num_groups, dtype=torch.float32, device=tensor.device)
        
        # Launch kernel
        BLOCK_N = min(128, triton.next_power_of_2(N))
        grid = (num_groups, triton.cdiv(N, BLOCK_N))
        
        _quantize_int8_groupwise_along_k_kernel[grid](
            tensor, int8_output, scales,
            K, N, num_groups,
            tensor.stride(0), tensor.stride(1),
            int8_output.stride(0), int8_output.stride(1),
            scales.stride(0), scales.stride(1),
            eps=eps,
            GROUP_SIZE=group_size,
            BLOCK_N=BLOCK_N,
        )
        
        # Remove padding if applied
        if pad_size > 0:
            int8_output = int8_output[:orig_K, :]
        
        return int8_output, scales
    else:
        # Fallback to PyTorch: transpose, quantize, transpose back
        tensor_t = tensor.T.contiguous()  # [N, K]
        int8_t, scales = quantize_int8_groupwise(tensor_t, group_size, eps=eps)
        int8_output = int8_t.T.contiguous()  # [K, N]
        scales = scales.float()
        
        # Remove padding if applied
        if pad_size > 0:
            int8_output = int8_output[:orig_K, :]
        
        return int8_output, scales


# ============================================================================
# Triton kernel for two-stage quantization (fused for performance)
# ============================================================================

if HAS_TRITON:
    @triton.jit
    def _quantize_int8_two_stage_kernel(
        # Input/Output pointers
        input_ptr,      # [M, K] input tensor (f16 or f32)
        output_ptr,     # [M, K] int8 output tensor
        scales_ptr,     # [M, num_groups, 2] scales output (same dtype as input)
        mask_ptr,       # [M, num_groups, group_size] bool mask output
        # Dimensions
        M, K, num_groups,
        # Strides
        stride_input_m, stride_input_k,
        stride_output_m, stride_output_k,
        stride_scales_m, stride_scales_g, stride_scales_s,
        stride_mask_m, stride_mask_g, stride_mask_k,
        # Parameters
        eps: tl.constexpr,
        topk_k: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Optimized Triton kernel for two-stage INT8 quantization.
        
        Processes BLOCK_M rows at once for better parallelism.
        Uses sort-based top-k selection with proper tie-breaking to match torch.topk.
        Uses banker's rounding (round-half-to-even) to match PyTorch.
        
        Grid: (cdiv(M, BLOCK_M), num_groups)
        """
        pid_m = tl.program_id(0)
        pid_g = tl.program_id(1)
        
        # Row indices for this block
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rm < M
        
        # Column indices for this group
        group_start_k = pid_g * GROUP_SIZE
        rk = tl.arange(0, GROUP_SIZE)
        k_indices = group_start_k + rk
        col_mask = k_indices < K
        
        # Combined mask: [BLOCK_M, GROUP_SIZE]
        load_mask = row_mask[:, None] & col_mask[None, :]
        
        # Load input data: [BLOCK_M, GROUP_SIZE]
        input_ptrs = input_ptr + rm[:, None] * stride_input_m + k_indices[None, :] * stride_input_k
        data = tl.load(input_ptrs, mask=load_mask, other=0.0)
        
        # Compute absolute values in native dtype
        abs_data = tl.abs(data)
        
        # ================================================================
        # Sort-based Top-K Selection (per row)
        # ================================================================
        
        # For sorting, replace invalid elements with -inf
        abs_for_sort = tl.where(col_mask[None, :], abs_data, float('-inf'))
        
        # Sort descending along K dimension
        sorted_abs = tl.sort(abs_for_sort, dim=1, descending=True)
        
        # Find threshold (k-th largest value per row)
        position = tl.arange(0, GROUP_SIZE)[None, :]
        is_in_topk_sorted = position < topk_k
        
        # Threshold = minimum of top-k elements in sorted array
        topk_vals_sorted = tl.where(is_in_topk_sorted, sorted_abs, float('inf'))
        threshold = tl.min(topk_vals_sorted, axis=1)  # [BLOCK_M]
        
        # Max of top-k elements per row
        max_topk_vals = tl.where(is_in_topk_sorted, sorted_abs, float('-inf'))
        max_topk = tl.max(max_topk_vals, axis=1)  # [BLOCK_M]
        
        # ================================================================
        # Create top-k mask (matching torch.topk tie-breaking: smallest indices)
        # ================================================================
        
        # Elements strictly above threshold are definitely in top-k
        above_threshold = abs_data > threshold[:, None]
        
        # Count elements above threshold per row
        count_above = tl.sum(above_threshold.to(tl.int32), axis=1)  # [BLOCK_M]
        
        # Elements at threshold need tie-breaking
        at_threshold = (abs_data == threshold[:, None]) & col_mask[None, :]
        need_from_ties = topk_k - count_above  # [BLOCK_M]
        
        # Tie-breaking: select elements at threshold with smallest indices
        # Use cumulative sum to count ties up to each position
        at_threshold_int = at_threshold.to(tl.int32)
        cumsum_ties = tl.cumsum(at_threshold_int, axis=1)  # [BLOCK_M, GROUP_SIZE]
        selected_ties = at_threshold & (cumsum_ties <= need_from_ties[:, None])
        
        # Final top-k mask
        topk_mask = above_threshold | selected_ties
        
        # ================================================================
        # Compute scales in fp32 for full precision
        # ================================================================
        
        # Convert to fp32 for scale computation
        max_topk_f32 = max_topk.to(tl.float32)
        scale_topk = max_topk_f32 / 127.0  # [BLOCK_M], fp32
        
        # scale_others = max(non-topk elements) / 127
        abs_others = tl.where(topk_mask | (~col_mask[None, :]), 0.0, abs_data)
        max_others = tl.max(abs_others, axis=1)  # [BLOCK_M]
        max_others_f32 = max_others.to(tl.float32)
        scale_others = max_others_f32 / 127.0  # [BLOCK_M], fp32
        
        # ================================================================
        # Quantize with banker's rounding (round-half-to-even)
        # ================================================================
        
        # Clamp scales to avoid division by zero, compute inverse (all in fp32)
        inv_scale_topk = 1.0 / tl.maximum(scale_topk, eps)  # [BLOCK_M]
        inv_scale_others = 1.0 / tl.maximum(scale_others, eps)  # [BLOCK_M]
        
        # Select scale based on mask
        inv_scale = tl.where(topk_mask, inv_scale_topk[:, None], inv_scale_others[:, None])
        
        # Scale data
        scaled_data = data.to(tl.float32) * inv_scale
        
        # Simple round: floor(x + 0.5)
        rounded = tl.math.floor(scaled_data + 0.5)
        
        # Clamp to int8 range and convert
        clamped = tl.minimum(tl.maximum(rounded, -127.0), 127.0)
        int8_data = clamped.to(tl.int8)
        
        # ================================================================
        # Store results
        # ================================================================
        
        # Store int8 output
        output_ptrs = output_ptr + rm[:, None] * stride_output_m + k_indices[None, :] * stride_output_k
        tl.store(output_ptrs, int8_data, mask=load_mask)
        
        # Store scales [M, num_groups, 2] in fp32
        scale_topk_ptrs = scales_ptr + rm * stride_scales_m + pid_g * stride_scales_g + 0 * stride_scales_s
        scale_others_ptrs = scales_ptr + rm * stride_scales_m + pid_g * stride_scales_g + 1 * stride_scales_s
        tl.store(scale_topk_ptrs, scale_topk, mask=row_mask)
        tl.store(scale_others_ptrs, scale_others, mask=row_mask)
        
        # Store mask [M, num_groups, group_size]
        mask_ptrs = mask_ptr + rm[:, None] * stride_mask_m + pid_g * stride_mask_g + rk[None, :] * stride_mask_k
        tl.store(mask_ptrs, topk_mask, mask=load_mask)


@torch.no_grad()
def _quantize_int8_two_stage_groupwise_pytorch(
    tensor: Tensor, 
    group_size: int = 64, 
    topk_elements: int = 16,
    stochastic_rounding: bool = False, 
    eps: float = 1e-20
):
    """PyTorch fallback implementation for two-stage groupwise quantization."""
    M, K = tensor.shape
    orig_K = K
    
    # Handle case where K is not divisible by group_size
    if K % group_size != 0:
        pad_size = group_size - (K % group_size)
        tensor = torch.nn.functional.pad(tensor, (0, pad_size), value=0)
        K = K + pad_size
    else:
        pad_size = 0
    
    num_groups = K // group_size
    topk_elements = min(topk_elements, group_size)
    
    # Reshape to [M, num_groups, group_size]
    tensor_grouped = tensor.reshape(M, num_groups, group_size)
    
    # Compute absolute values once
    abs_grouped = tensor_grouped.abs()
    
    # Find top-k indices per group: [M, num_groups, topk_elements]
    topk_vals, topk_indices = torch.topk(abs_grouped, k=topk_elements, dim=2, sorted=False)
    
    # Create mask for top-k elements: [M, num_groups, group_size]
    topk_mask = torch.zeros_like(tensor_grouped, dtype=torch.bool)
    topk_mask.scatter_(2, topk_indices, True)
    
    # Compute two scales per group in fp32 for precision
    scale_topk = (topk_vals.float().amax(dim=2) / 127.0)  # [M, num_groups], fp32
    
    # For scale_others, use masked operations
    abs_others = abs_grouped.masked_fill(topk_mask, 0)
    scale_others = (abs_others.float().amax(dim=2) / 127.0)  # [M, num_groups], fp32
    
    # Compute inverse scales (vectorized), all in fp32
    inv_scale_topk = (1.0 / scale_topk.clamp(min=eps)).unsqueeze(-1)
    inv_scale_others = (1.0 / scale_others.clamp(min=eps)).unsqueeze(-1)
    
    # Apply appropriate scale based on mask (vectorized)
    inv_scales = torch.where(topk_mask, inv_scale_topk, inv_scale_others)
    tensor_scaled = tensor_grouped.float() * inv_scales
    
    # Round (vectorized)
    if stochastic_rounding:
        tensor_scaled = (tensor_scaled + torch.rand_like(tensor_scaled)).floor()
    else:
        tensor_scaled = tensor_scaled.round()
    
    # Clip and convert to int8 (vectorized)
    tensor_i8 = tensor_scaled.clamp(-127, 127).to(torch.int8)
    
    # Reshape back to [M, K]
    int8_tensor = tensor_i8.reshape(M, K)
    
    # Stack scales: [M, num_groups, 2]
    scales = torch.stack([scale_topk, scale_others], dim=2)
    
    # Remove padding from int8_tensor if applied
    if pad_size > 0:
        int8_tensor = int8_tensor[:, :orig_K]
    
    return int8_tensor, scales, topk_mask


@torch.no_grad()
def _quantize_int8_two_stage_groupwise_triton(
    tensor: Tensor, 
    group_size: int = 64, 
    topk_elements: int = 16,
    eps: float = 1e-20
):
    """Triton-accelerated two-stage groupwise INT8 quantization.
    
    Args:
        tensor: Input tensor of shape [M, K]
        group_size: Number of elements per group (must be power of 2, default: 64)
        topk_elements: Number of top-k elements per group (default: 16)
        eps: Small value to prevent division by zero
        
    Returns:
        Tuple of:
        - int8_tensor: [M, K] quantized tensor
        - scales: [M, num_groups, 2] where scales[:,:,0] is scale_topk, scales[:,:,1] is scale_others
        - topk_mask: [M, num_groups, group_size] bool mask indicating top-k positions
    """
    M, K = tensor.shape
    orig_K = K
    
    # Handle case where K is not divisible by group_size
    if K % group_size != 0:
        pad_size = group_size - (K % group_size)
        tensor = torch.nn.functional.pad(tensor, (0, pad_size), value=0)
        K = K + pad_size
    else:
        pad_size = 0
    
    num_groups = K // group_size
    topk_elements = min(topk_elements, group_size)
    
    # Ensure tensor is contiguous
    tensor = tensor.contiguous()
    
    # Allocate output tensors (scales always in fp32 for precision)
    int8_output = torch.empty(M, K, dtype=torch.int8, device=tensor.device)
    scales = torch.empty(M, num_groups, 2, dtype=torch.float32, device=tensor.device)
    topk_mask = torch.empty(M, num_groups, group_size, dtype=torch.bool, device=tensor.device)
    
    # Choose BLOCK_M based on M for good occupancy
    BLOCK_M = 32 if M >= 32 else (16 if M >= 16 else 8)
    
    # Launch kernel: process BLOCK_M rows per program
    grid = (triton.cdiv(M, BLOCK_M), num_groups)
    
    _quantize_int8_two_stage_kernel[grid](
        # Pointers
        tensor, int8_output, scales, topk_mask,
        # Dimensions
        M, K, num_groups,
        # Input strides
        tensor.stride(0), tensor.stride(1),
        # Output strides
        int8_output.stride(0), int8_output.stride(1),
        # Scales strides
        scales.stride(0), scales.stride(1), scales.stride(2),
        # Mask strides
        topk_mask.stride(0), topk_mask.stride(1), topk_mask.stride(2),
        # Parameters
        eps=eps,
        topk_k=topk_elements,
        GROUP_SIZE=group_size,
        BLOCK_M=BLOCK_M,
    )
    
    # Remove padding from int8_output if applied
    if pad_size > 0:
        int8_output = int8_output[:, :orig_K]
    
    return int8_output, scales, topk_mask


@torch.no_grad()
def quantize_int8_two_stage_groupwise(
    tensor: Tensor, 
    group_size: int = 64, 
    topk_elements: int = 16,
    stochastic_rounding: bool = False, 
    eps: float = 1e-20
):
    """Quantize a tensor to INT8 with two-stage group-wise scaling.
    
    Uses Triton kernel for acceleration when available. Falls back to PyTorch
    implementation for stochastic rounding or when Triton is not available.
    
    For each group, top-k elements by absolute value get their own scale (scale_topk),
    while remaining elements get a different scale (scale_others). This allows better
    precision by avoiding small values being "drowned out" by large outliers.
    
    Args:
        tensor: Input tensor of shape [M, K]
        group_size: Number of elements per group (default: 64)
        topk_elements: Number of top-k elements per group (default: 16)
        stochastic_rounding: If True, use stochastic rounding (uses PyTorch fallback)
        eps: Small value to prevent division by zero
        
    Returns:
        Tuple of:
        - int8_tensor: [M, K] quantized tensor
        - scales: [M, num_groups, 2] where scales[:,:,0] is scale_topk, scales[:,:,1] is scale_others
        - topk_mask: [M, num_groups, group_size] bool mask indicating top-k positions
    """
    # Use Triton kernel when available and stochastic_rounding is not needed
    # Also ensure group_size is a power of 2 (Triton requirement)
    use_triton = (
        HAS_TRITON 
        and not stochastic_rounding 
        and tensor.is_cuda
        and (group_size & (group_size - 1)) == 0  # Power of 2 check
    )
    
    if use_triton:
        return _quantize_int8_two_stage_groupwise_triton(
            tensor, group_size, topk_elements, eps
        )
    else:
        return _quantize_int8_two_stage_groupwise_pytorch(
            tensor, group_size, topk_elements, stochastic_rounding, eps
        )


# ============================================================================
# Mixed Precision Two-Stage Quantization: INT8 (top-k) + INT4 (others)
# ============================================================================

if HAS_TRITON:
    @triton.jit
    def _quantize_int8_int4_two_stage_kernel(
        # Input/Output pointers
        input_ptr,      # [M, K] input tensor
        output_ptr,     # [M, K] int8 output tensor  
        scales_ptr,     # [M, num_groups, 2] scales output
        mask_ptr,       # [M, num_groups, group_size] bool mask output
        # Dimensions
        M, K, num_groups,
        # Strides
        stride_input_m, stride_input_k,
        stride_output_m, stride_output_k,
        stride_scales_m, stride_scales_g, stride_scales_s,
        stride_mask_m, stride_mask_g, stride_mask_k,
        # Parameters
        eps: tl.constexpr,
        topk_k: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Triton kernel for mixed precision INT8+INT4 two-stage quantization.
        
        Top-k elements -> INT8 [-127, 127], scale = max/127
        Others elements -> INT4 [-7, 7], scale = max/7
        Both stored in same INT8 tensor (combined).
        """
        pid_m = tl.program_id(0)
        pid_g = tl.program_id(1)
        
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rm < M
        
        group_start_k = pid_g * GROUP_SIZE
        rk = tl.arange(0, GROUP_SIZE)
        k_indices = group_start_k + rk
        col_mask = k_indices < K
        
        load_mask = row_mask[:, None] & col_mask[None, :]
        
        input_ptrs = input_ptr + rm[:, None] * stride_input_m + k_indices[None, :] * stride_input_k
        data = tl.load(input_ptrs, mask=load_mask, other=0.0)
        abs_data = tl.abs(data)
        
        # Sort-based Top-K Selection
        abs_for_sort = tl.where(col_mask[None, :], abs_data, float('-inf'))
        sorted_abs = tl.sort(abs_for_sort, dim=1, descending=True)
        
        position = tl.arange(0, GROUP_SIZE)[None, :]
        is_in_topk_sorted = position < topk_k
        
        topk_vals_sorted = tl.where(is_in_topk_sorted, sorted_abs, float('inf'))
        threshold = tl.min(topk_vals_sorted, axis=1)
        
        max_topk_vals = tl.where(is_in_topk_sorted, sorted_abs, float('-inf'))
        max_topk = tl.max(max_topk_vals, axis=1)
        
        # Create top-k mask
        above_threshold = abs_data > threshold[:, None]
        count_above = tl.sum(above_threshold.to(tl.int32), axis=1)
        at_threshold = (abs_data == threshold[:, None]) & col_mask[None, :]
        need_from_ties = topk_k - count_above
        at_threshold_int = at_threshold.to(tl.int32)
        cumsum_ties = tl.cumsum(at_threshold_int, axis=1)
        selected_ties = at_threshold & (cumsum_ties <= need_from_ties[:, None])
        topk_mask = above_threshold | selected_ties
        
        # Compute scales: topk/127, others/7
        max_topk_f32 = max_topk.to(tl.float32)
        scale_topk = max_topk_f32 / 127.0
        
        abs_others = tl.where(topk_mask | (~col_mask[None, :]), 0.0, abs_data)
        max_others = tl.max(abs_others, axis=1)
        max_others_f32 = max_others.to(tl.float32)
        scale_others = max_others_f32 / 7.0  # INT4 scale
        
        # Quantize with different ranges
        inv_scale_topk = 1.0 / tl.maximum(scale_topk, eps)
        inv_scale_others = 1.0 / tl.maximum(scale_others, eps)
        inv_scale = tl.where(topk_mask, inv_scale_topk[:, None], inv_scale_others[:, None])
        
        scaled_data = data.to(tl.float32) * inv_scale
        rounded = tl.math.floor(scaled_data + 0.5)
        
        # Clamp: topk to [-127, 127], others to [-7, 7]
        clamped_topk = tl.minimum(tl.maximum(rounded, -127.0), 127.0)
        clamped_others = tl.minimum(tl.maximum(rounded, -7.0), 7.0)
        clamped = tl.where(topk_mask, clamped_topk, clamped_others)
        int8_data = clamped.to(tl.int8)
        
        # Store outputs
        output_ptrs = output_ptr + rm[:, None] * stride_output_m + k_indices[None, :] * stride_output_k
        tl.store(output_ptrs, int8_data, mask=load_mask)
        
        scale_topk_ptrs = scales_ptr + rm * stride_scales_m + pid_g * stride_scales_g + 0 * stride_scales_s
        scale_others_ptrs = scales_ptr + rm * stride_scales_m + pid_g * stride_scales_g + 1 * stride_scales_s
        tl.store(scale_topk_ptrs, scale_topk, mask=row_mask)
        tl.store(scale_others_ptrs, scale_others, mask=row_mask)
        
        mask_ptrs = mask_ptr + rm[:, None] * stride_mask_m + pid_g * stride_mask_g + rk[None, :] * stride_mask_k
        tl.store(mask_ptrs, topk_mask, mask=load_mask)


@torch.no_grad()
def _quantize_int8_int4_two_stage_triton(
    tensor: Tensor, 
    group_size: int = 64, 
    topk_elements: int = 16,
    eps: float = 1e-20
):
    """Triton-accelerated mixed precision INT8+INT4 quantization."""
    M, K = tensor.shape
    orig_K = K
    
    if K % group_size != 0:
        pad_size = group_size - (K % group_size)
        tensor = torch.nn.functional.pad(tensor, (0, pad_size), value=0)
        K = K + pad_size
    else:
        pad_size = 0
    
    num_groups = K // group_size
    topk_elements = min(topk_elements, group_size)
    tensor = tensor.contiguous()
    
    int8_output = torch.empty(M, K, dtype=torch.int8, device=tensor.device)
    scales = torch.empty(M, num_groups, 2, dtype=torch.float32, device=tensor.device)
    topk_mask = torch.empty(M, num_groups, group_size, dtype=torch.bool, device=tensor.device)
    
    BLOCK_M = 32 if M >= 32 else (16 if M >= 16 else 8)
    grid = (triton.cdiv(M, BLOCK_M), num_groups)
    
    _quantize_int8_int4_two_stage_kernel[grid](
        tensor, int8_output, scales, topk_mask,
        M, K, num_groups,
        tensor.stride(0), tensor.stride(1),
        int8_output.stride(0), int8_output.stride(1),
        scales.stride(0), scales.stride(1), scales.stride(2),
        topk_mask.stride(0), topk_mask.stride(1), topk_mask.stride(2),
        eps=eps,
        topk_k=topk_elements,
        GROUP_SIZE=group_size,
        BLOCK_M=BLOCK_M,
    )
    
    if pad_size > 0:
        int8_output = int8_output[:, :orig_K]
    
    return int8_output, scales, topk_mask


@torch.no_grad()
def quantize_int8_int4_two_stage_groupwise(
    tensor: Tensor, 
    group_size: int = 64, 
    topk_elements: int = 16,
    stochastic_rounding: bool = False, 
    eps: float = 1e-20
):
    """Quantize tensor with mixed precision: INT8 for top-k, INT4 for others.
    
    This two-stage quantization uses higher precision (INT8) for the largest
    values (top-k) and lower precision (INT4 range [-7,7]) for remaining smaller values.
    INT4 values are stored as INT8 for compute efficiency.
    
    Use with scaled_int8_mm_two_stage for fast matrix multiplication.
    
    Args:
        tensor: Input tensor of shape [M, K]
        group_size: Number of elements per group (default: 64)
        topk_elements: Number of top-k elements per group using INT8 (default: 16)
        stochastic_rounding: If True, use stochastic rounding
        eps: Small value to prevent division by zero
        
    Returns:
        Tuple of:
        - A_combined: [M, K] INT8 tensor (topk in [-127,127] + others in [-7,7] combined)
        - scales: [M, num_groups, 2] where scales[:,:,0] is scale_topk, scales[:,:,1] is scale_others
        - topk_mask: [M, num_groups, group_size] bool mask indicating top-k positions
        
    Example:
        >>> A_i8, A_scales, A_mask = quantize_int8_int4_two_stage_groupwise(A, 64, 16)
        >>> B_i8, B_scales = quantize_int8_groupwise_along_k(B, 64)
        >>> C = scaled_int8_mm_two_stage(A_i8, B_i8, A_scales, B_scales, A_mask.reshape(M,-1)[:,:K], 64)
    """
    use_triton = (
        HAS_TRITON 
        and not stochastic_rounding 
        and tensor.is_cuda
        and (group_size & (group_size - 1)) == 0
    )
    
    if use_triton:
        return _quantize_int8_int4_two_stage_triton(tensor, group_size, topk_elements, eps)
    else:
        return _quantize_int8_int4_two_stage_unpacked(
            tensor, group_size, topk_elements, stochastic_rounding, eps
        )


@torch.no_grad()
def _quantize_int8_int4_two_stage_unpacked(
    tensor: Tensor, 
    group_size: int = 64, 
    topk_elements: int = 16,
    stochastic_rounding: bool = False, 
    eps: float = 1e-20
):
    """Fast unpacked version: INT4 stored as INT8 for compute efficiency.
    
    Returns combined tensor instead of separate topk and others.
    Use with scaled_int8_mm_two_stage for faster matmul.
    """
    M, K = tensor.shape
    orig_K = K
    
    if K % group_size != 0:
        pad_size = group_size - (K % group_size)
        tensor = torch.nn.functional.pad(tensor, (0, pad_size), value=0)
        K = K + pad_size
    else:
        pad_size = 0
    
    num_groups = K // group_size
    topk_elements = min(topk_elements, group_size)
    
    tensor_grouped = tensor.reshape(M, num_groups, group_size)
    abs_grouped = tensor_grouped.abs()
    
    # Find top-k
    topk_vals, topk_indices = torch.topk(abs_grouped, k=topk_elements, dim=2, sorted=False)
    topk_mask = torch.zeros_like(tensor_grouped, dtype=torch.bool)
    topk_mask.scatter_(2, topk_indices, True)
    
    # Compute scales
    scale_topk = (topk_vals.float().amax(dim=2) / 127.0)
    abs_others = abs_grouped.masked_fill(topk_mask, 0)
    scale_others = (abs_others.float().amax(dim=2) / 7.0)
    
    inv_scale_topk = (1.0 / scale_topk.clamp(min=eps)).unsqueeze(-1)
    inv_scale_others = (1.0 / scale_others.clamp(min=eps)).unsqueeze(-1)
    
    # Quantize topk to INT8 [-127, 127]
    tensor_topk = tensor_grouped.float() * inv_scale_topk
    tensor_topk = torch.where(topk_mask, tensor_topk, torch.zeros_like(tensor_topk))
    if stochastic_rounding:
        tensor_topk = (tensor_topk + torch.rand_like(tensor_topk)).floor()
    else:
        tensor_topk = tensor_topk.round()
    int8_topk = tensor_topk.clamp(-127, 127)
    
    # Quantize others to INT4 range [-7, 7] but store as INT8
    tensor_others = tensor_grouped.float() * inv_scale_others
    tensor_others = torch.where(~topk_mask, tensor_others, torch.zeros_like(tensor_others))
    if stochastic_rounding:
        tensor_others = (tensor_others + torch.rand_like(tensor_others)).floor()
    else:
        tensor_others = tensor_others.round()
    int8_others = tensor_others.clamp(-7, 7)
    
    # Combine: since topk and others are mutually exclusive, we can add them
    A_combined = (int8_topk + int8_others).to(torch.int8).reshape(M, K)
    
    scales = torch.stack([scale_topk, scale_others], dim=2)
    
    if pad_size > 0:
        A_combined = A_combined[:, :orig_K]
    
    return A_combined, scales, topk_mask


