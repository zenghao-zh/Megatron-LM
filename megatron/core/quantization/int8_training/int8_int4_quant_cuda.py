"""
Python wrapper for CUDA-accelerated INT8+INT4 two-stage quantization & dequant.

Quantise:
    quantize_int8_int4_cuda_fused   → (topk_out, others_out, scales)
    quantize_int8_int4_cuda_compat  → (combined, scales, mask)

Dequant (for the "CUDA quant → dequant → cuBLAS BF16 mm" fast-path):
    dequant_a_cuda       → BF16 tensor  (from pre-split A_topk, A_others, scales)
    dequant_b_cuda       → BF16 tensor  (from B_int8, B_scales)

Fused quant+dequant (training fast-path):
    quant_dequant_a_cuda → BF16 tensor  (A two-stage round-trip)
    quant_dequant_b_cuda → BF16 tensor  (B groupwise round-trip, single kernel)
"""

import os
import torch
from torch import Tensor
from torch.utils.cpp_extension import load

_DIR = os.path.dirname(os.path.abspath(__file__))
_EXT = None


def _get_ext():
    global _EXT
    if _EXT is None:
        _EXT = load(
            name="quant_int8_int4_cuda_ext",
            sources=[os.path.join(_DIR, "csrc", "quant_int8_int4.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _EXT


@torch.no_grad()
def quantize_int8_int4_cuda_fused(
    tensor: Tensor,
    group_size: int = 64,
    topk_elements: int = 16,
    eps: float = 1e-20,
):
    """Quantize with CUDA kernel, returning pre-split tensors for matmul.

    Args:
        tensor:       [M, K] input (bf16/fp16/fp32, CUDA)
        group_size:   Must be 64
        topk_elements: 8 or 16
        eps:          Division guard

    Returns:
        topk_out:  [M, K] INT8 (topk values in place, others = 0)
        others_out:[M, K] INT8 (others values in place, topk = 0)
        scales:    [M, num_groups, 2] FP32  ([:,:,0]=scale_topk, [:,:,1]=scale_others)
    """
    assert group_size == 64, "CUDA kernel is specialised for group_size=64"
    M, K = tensor.shape

    pad_size = 0
    if K % group_size != 0:
        pad_size = group_size - (K % group_size)
        tensor = torch.nn.functional.pad(tensor, (0, pad_size), value=0)
        K = K + pad_size

    ext = _get_ext()
    topk, others, scales = ext.quantize_fused(tensor, topk_elements, eps)

    if pad_size > 0:
        topk = topk[:, :K - pad_size]
        others = others[:, :K - pad_size]

    return topk, others, scales


@torch.no_grad()
def quantize_int8_int4_cuda_compat(
    tensor: Tensor,
    group_size: int = 64,
    topk_elements: int = 16,
    eps: float = 1e-20,
):
    """Quantize with CUDA kernel, API-compatible with existing function.

    Returns same format as quantize_int8_int4_two_stage_groupwise:
        combined: [M, K] INT8
        scales:   [M, num_groups, 2] FP32
        mask:     [M, num_groups, group_size] bool
    """
    assert group_size == 64, "CUDA kernel is specialised for group_size=64"
    M, K = tensor.shape
    orig_K = K

    pad_size = 0
    if K % group_size != 0:
        pad_size = group_size - (K % group_size)
        tensor = torch.nn.functional.pad(tensor, (0, pad_size), value=0)
        K = K + pad_size

    ext = _get_ext()
    combined, scales, mask = ext.quantize_compat(tensor, topk_elements, eps)

    if pad_size > 0:
        combined = combined[:, :orig_K]

    return combined, scales, mask


# ── Combined quant → dequant (for training fast-path) ────────────────────

@torch.no_grad()
def quant_dequant_a_cuda(
    tensor: Tensor,
    group_size: int = 64,
    topk_elements: int = 16,
    eps: float = 1e-20,
) -> Tensor:
    """Quantize A (two-stage INT8/INT4) then dequant back to BF16 in one shot.

    The round-trip introduces quantization noise identical to the Triton path
    but is ~6× faster end-to-end when paired with cuBLAS BF16 matmul.
    """
    assert group_size == 64, "CUDA kernel is specialised for group_size=64"
    M, K = tensor.shape
    orig_K = K

    pad_size = (group_size - K % group_size) % group_size
    if pad_size > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_size), value=0)

    ext = _get_ext()
    topk, others, scales = ext.quantize_fused(tensor, topk_elements, eps)
    out = ext.dequant_a(topk, others, scales.float().contiguous(), group_size)

    if pad_size > 0:
        out = out[:, :orig_K]
    return out


# ── Dequant wrappers ─────────────────────────────────────────────────────

@torch.no_grad()
def dequant_a_cuda(
    A_topk: Tensor,
    A_others: Tensor,
    A_scales: Tensor,
    group_size: int = 64,
) -> Tensor:
    """Single-pass dequant: (A_topk * scale_topk + A_others * scale_others) → BF16.

    ~6× faster than the equivalent multi-pass PyTorch ops.
    """
    ext = _get_ext()
    sc = A_scales.to(torch.float32).contiguous()
    return ext.dequant_a(A_topk.contiguous(), A_others.contiguous(), sc, group_size)


@torch.no_grad()
def dequant_b_cuda(
    B: Tensor,
    B_scales: Tensor,
    group_size: int = 64,
) -> Tensor:
    """Single-pass dequant: (B * b_scale_per_group) → BF16."""
    ext = _get_ext()
    sc = B_scales.to(torch.float32).contiguous()
    return ext.dequant_b(B.contiguous(), sc, group_size)


# ── Fused quant+dequant for B (single kernel) ────────────────────────────

@torch.no_grad()
def quant_dequant_b_cuda(
    tensor: Tensor,
    group_size: int = 64,
) -> Tensor:
    """Fused groupwise INT8 quant+dequant for B in a single CUDA kernel.

    Equivalent to quantize_int8_groupwise_along_k followed by dequant_b_cuda,
    but avoids the intermediate INT8 tensor / scales allocation and the extra
    kernel launch.  The kernel reads B twice (max-abs pass + quant-dequant pass)
    with the second read hitting L2 cache.
    """
    K, N = tensor.shape
    pad_size = (group_size - K % group_size) % group_size
    if pad_size > 0:
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_size), value=0)

    ext = _get_ext()
    out = ext.quant_dequant_b(tensor, group_size)

    if pad_size > 0:
        out = out[:K, :]
    return out
