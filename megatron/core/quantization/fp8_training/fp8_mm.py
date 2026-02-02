# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from TorchAO: https://github.com/pytorch/ao
# Modified for Megatron-LM integration with FP8 support.

"""
FP8 matrix multiplication kernels using Triton.

This module provides optimized FP8 matmul with group-wise scaling,
which is the core computation for FP8 mixed-precision training.

FP8 Formats:
- E4M3 (float8_e4m3fn): Range [-448, 448], 4 exp bits, 3 mantissa bits
- E5M2 (float8_e5m2): Range [-57344, 57344], 5 exp bits, 2 mantissa bits

Note: On GPUs that don't support native FP8 (pre-Hopper), we use "simulated FP8"
which quantizes using FP8 ranges but stores values in BF16.
"""

import torch
from torch import Tensor

# FP8 format constants
FP8_E4M3_MAX = 448.0  # Max representable value for E4M3
FP8_E5M2_MAX = 57344.0  # Max representable value for E5M2

# Check for native FP8 support (requires Hopper H100+)
def _check_native_fp8_support():
    """Check if the current GPU supports native FP8 types."""
    try:
        if not torch.cuda.is_available():
            return False
        # Try to create a native FP8 tensor
        test_tensor = torch.zeros(1, device='cuda', dtype=torch.float8_e4m3fn)
        # Try a simple operation to verify it works
        _ = test_tensor.float()
        return True
    except (RuntimeError, TypeError, AttributeError):
        return False

# Lazy initialization of native FP8 support flag
_NATIVE_FP8_SUPPORT = None

def has_native_fp8_support():
    """Check if native FP8 is supported (cached)."""
    global _NATIVE_FP8_SUPPORT
    if _NATIVE_FP8_SUPPORT is None:
        _NATIVE_FP8_SUPPORT = _check_native_fp8_support()
    return _NATIVE_FP8_SUPPORT

# Check for Triton availability
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    # Triton autotune configurations for FP8 matmul
    def _make_fp8_groupwise_configs(block_k):
        """Create autotune configs for a specific BLOCK_K (= GROUP_SIZE).
        
        Note: For FP8, we can use larger block sizes since FP8 has better
        numerical properties than INT8.
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
    _fp8_groupwise_configs_16 = _make_fp8_groupwise_configs(16)
    _fp8_groupwise_configs_32 = _make_fp8_groupwise_configs(32)
    _fp8_groupwise_configs_64 = _make_fp8_groupwise_configs(64)
    
    def _make_fp8_groupwise_kernel(group_size, configs):
        """Factory function to create FP8 groupwise matmul kernel for a specific group size.
        
        This kernel works with simulated FP8 (BF16 storage) for compatibility with all GPUs.
        FP8 matmul: (A_fp8 * A_scale) @ (B_fp8 * B_scale)
        Each group of GROUP_SIZE elements along K has its own scale factor.
        """
        
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
            """Triton kernel for group-wise scaled FP8 matrix multiplication.
            
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
                
                # Load A and B blocks (stored as BF16 for simulated FP8)
                A = A_ptr + (ram[:, None] * stride_am + (k_block + rk[None, :]) * stride_ak)
                B = B_ptr + ((k_block + rk[:, None]) * stride_bk + rbn[None, :] * stride_bn)
                
                # Mask for K boundary
                k_mask = (k_block + rk) < K
                a = tl.load(A, mask=k_mask[None, :], other=0.0)
                b = tl.load(B, mask=k_mask[:, None], other=0.0)
                
                # Convert to float32 for computation
                a_f32 = a.to(tl.float32)
                b_f32 = b.to(tl.float32)
                
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
                
                # Dequantize: apply scales element-wise
                a_dequant = a_f32 * a_scale  # [BLOCK_M, BLOCK_K]
                b_dequant = b_f32 * b_scale  # [BLOCK_K, BLOCK_N]
                
                # Matmul and accumulate
                acc += tl.dot(a_dequant, b_dequant)

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
    _scaled_fp8_mm_groupwise_kernel_g16 = _make_fp8_groupwise_kernel(16, _fp8_groupwise_configs_16)
    _scaled_fp8_mm_groupwise_kernel_g32 = _make_fp8_groupwise_kernel(32, _fp8_groupwise_configs_32)
    _scaled_fp8_mm_groupwise_kernel_g64 = _make_fp8_groupwise_kernel(64, _fp8_groupwise_configs_64)


def get_fp8_max(fp8_format: str) -> float:
    """Get the maximum representable value for a given FP8 format.
    
    Args:
        fp8_format: 'e4m3' or 'e5m2'
        
    Returns:
        Maximum representable value
    """
    if fp8_format == 'e4m3':
        return FP8_E4M3_MAX
    elif fp8_format == 'e5m2':
        return FP8_E5M2_MAX
    else:
        raise ValueError(f"Unknown FP8 format: {fp8_format}. Use 'e4m3' or 'e5m2'.")


def scaled_fp8_mm_groupwise(
    A: Tensor, B: Tensor, A_scale: Tensor, B_scale: Tensor, group_size: int = 64
) -> Tensor:
    """Compute group-wise scaled FP8 matmul: A @ B with per-group scaling.
    
    Each group of `group_size` elements along K dimension has its own scale.
    This is useful for Tensor Parallelism compatibility.
    
    Note: This function works with "simulated FP8" tensors stored as BF16.
    The quantization to FP8 ranges happens in quantize_fp8_groupwise().
    
    Args:
        A: Tensor of shape (M, K) - quantized values stored as BF16
        B: Tensor of shape (K, N) - quantized values stored as BF16
        A_scale: Scale tensor of shape (M, num_groups) where num_groups = ceil(K/group_size)
        B_scale: Scale tensor of shape (N, num_groups)
        group_size: Number of elements per group (default: 64)
        
    Returns:
        Result tensor of shape (M, N) with dtype matching scale tensors
    """
    M, K = A.shape
    _, N = B.shape
    num_groups = (K + group_size - 1) // group_size
    
    if not HAS_TRITON:
        # Fallback: loop over groups
        result = torch.zeros(M, N, device=A.device, dtype=A_scale.dtype)
        
        for g in range(num_groups):
            k_start = g * group_size
            k_end = min((g + 1) * group_size, K)
            A_g = A[:, k_start:k_end].float()
            B_g = B[k_start:k_end, :].float()
            # Dequantize and accumulate
            A_dq = A_g * A_scale[:, g:g+1]
            B_dq = B_g * B_scale[:, g:g+1].T
            result += A_dq @ B_dq
        return result.to(A_scale.dtype)
    
    assert A_scale.dtype == B_scale.dtype
    assert A.shape[1] == B.shape[0]
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
    if group_size == 16:
        kernel = _scaled_fp8_mm_groupwise_kernel_g16
    elif group_size == 32:
        kernel = _scaled_fp8_mm_groupwise_kernel_g32
    elif group_size == 64:
        kernel = _scaled_fp8_mm_groupwise_kernel_g64
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


@torch.no_grad()
def quantize_fp8_rowwise(
    tensor: Tensor, 
    fp8_format: str = 'e4m3',
    eps: float = 1e-12
):
    """Quantize a tensor to FP8 with row-wise scaling (simulated FP8 in BF16).
    
    This uses "simulated FP8" - values are quantized to FP8 range but stored as BF16
    for compatibility with GPUs that don't support native FP8.
    
    Args:
        tensor: Input tensor to quantize
        fp8_format: 'e4m3' or 'e5m2' (default: 'e4m3')
        eps: Small value to prevent division by zero
        
    Returns:
        Tuple of (quantized_tensor in BF16, scale_tensor)
    """
    fp8_max = get_fp8_max(fp8_format)
    
    # Compute row-wise scale: scale = amax / fp8_max
    amax = tensor.abs().amax(dim=1, keepdim=True)
    scale = amax / fp8_max
    scale = scale.clamp(min=eps)
    
    # Quantize: tensor_quantized = tensor / scale
    # Values are now in [-fp8_max, fp8_max] range
    tensor_quantized = tensor / scale
    
    # Clamp to FP8 range and round to simulate FP8 quantization
    tensor_quantized = tensor_quantized.clamp(-fp8_max, fp8_max)
    
    # For E4M3, round to nearest representable value (3 mantissa bits = 8 levels)
    # For E5M2, round to nearest representable value (2 mantissa bits = 4 levels)
    if fp8_format == 'e4m3':
        # Simulate E4M3 rounding (not exact but approximate)
        tensor_quantized = (tensor_quantized * 8).round() / 8
    else:  # e5m2
        # Simulate E5M2 rounding
        tensor_quantized = (tensor_quantized * 4).round() / 4
    
    # Store as BF16 (simulated FP8)
    tensor_quantized = tensor_quantized.to(tensor.dtype)
    
    return tensor_quantized, scale.squeeze(1)


@torch.no_grad()
def quantize_fp8_groupwise(
    tensor: Tensor, 
    group_size: int = 64, 
    fp8_format: str = 'e4m3',
    eps: float = 1e-12
):
    """Quantize a tensor to FP8 with group-wise scaling (simulated FP8 in BF16).
    
    This uses "simulated FP8" - values are quantized to FP8 range but stored as BF16
    for compatibility with GPUs that don't support native FP8.
    
    Each group of `group_size` elements shares a scale factor.
    This is useful for Tensor Parallelism where K dimension may be split.
    
    Args:
        tensor: Input tensor of shape [M, K]
        group_size: Number of elements per group (default: 64)
        fp8_format: 'e4m3' or 'e5m2' (default: 'e4m3')
        eps: Small value to prevent division by zero
        
    Returns:
        Tuple of (quantized_tensor [M, K] in BF16, scales [M, num_groups])
    """
    fp8_max = get_fp8_max(fp8_format)
    
    M, K = tensor.shape
    orig_dtype = tensor.dtype
    
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
    amax = tensor_grouped.abs().amax(dim=2)
    scales = amax / fp8_max
    scales = scales.clamp(min=eps)
    
    # Quantize: tensor_quantized = tensor / scale
    tensor_quantized = tensor_grouped / scales.unsqueeze(-1)
    
    # Clamp to FP8 range
    tensor_quantized = tensor_quantized.clamp(-fp8_max, fp8_max)
    
    # Simulate FP8 rounding
    if fp8_format == 'e4m3':
        tensor_quantized = (tensor_quantized * 8).round() / 8
    else:  # e5m2
        tensor_quantized = (tensor_quantized * 4).round() / 4
    
    # Reshape back to [M, K_padded]
    quantized_tensor = tensor_quantized.reshape(M, K_padded)
    
    # Remove padding if applied
    if pad_size > 0:
        quantized_tensor = quantized_tensor[:, :K]
    
    # Store as original dtype (simulated FP8)
    quantized_tensor = quantized_tensor.to(orig_dtype)
    
    return quantized_tensor, scales


@torch.no_grad()
def dequantize_fp8_groupwise(
    quantized_tensor: Tensor,
    scales: Tensor,
    group_size: int = 64,
    output_dtype: torch.dtype = None
) -> Tensor:
    """Dequantize a FP8 tensor with group-wise scaling.
    
    Args:
        quantized_tensor: Quantized tensor of shape [M, K] (BF16 simulated FP8)
        scales: Scale tensor of shape [M, num_groups]
        group_size: Number of elements per group
        output_dtype: Output dtype (default: same as input)
        
    Returns:
        Dequantized tensor of shape [M, K]
    """
    if output_dtype is None:
        output_dtype = quantized_tensor.dtype
    
    M, K = quantized_tensor.shape
    num_groups = scales.shape[1]
    
    # Handle padding
    K_padded = num_groups * group_size
    if K < K_padded:
        quantized_tensor = torch.nn.functional.pad(quantized_tensor, (0, K_padded - K), value=0)
    
    # Reshape for group-wise scaling
    tensor_grouped = quantized_tensor.reshape(M, num_groups, group_size)
    
    # Dequantize: tensor = quantized_tensor * scale
    tensor_dequant = tensor_grouped.float() * scales.unsqueeze(-1)
    
    # Reshape back and remove padding
    tensor_out = tensor_dequant.reshape(M, -1)[:, :K]
    
    return tensor_out.to(output_dtype)
