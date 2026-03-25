#!/usr/bin/env python3
"""
Benchmark: CUDA vs Triton vs PyTorch quantization, and INT8 two-stage vs BF16 matmul.

Usage:
    python benchmark_quant_cuda.py              # full benchmark
    python benchmark_quant_cuda.py --quick      # quick smoke test
    python benchmark_quant_cuda.py --correctness-only   # correctness only
"""

import argparse
import sys
import time

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# imports from the project
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")
from megatron.core.quantization.int8_training.int8_mm import (
    quantize_int8_int4_two_stage_groupwise,
    quantize_int8_groupwise_along_k,
    scaled_int8_mm_two_stage,
)
from megatron.core.quantization.int8_training.int8_int4_quant_cuda import (
    quantize_int8_int4_cuda_fused,
    quantize_int8_int4_cuda_compat,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _cuda_timer(fn, warmup=10, repeat=50):
    """Time a CUDA function using CUDA events (microseconds)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms → us
    times.sort()
    mid = len(times) // 2
    return times[mid]  # median µs


def _cosine_sim(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def _rmse(a, b):
    return (a.float() - b.float()).pow(2).mean().sqrt().item()


# ---------------------------------------------------------------------------
# 1. Correctness: CUDA vs Triton / PyTorch quantize
# ---------------------------------------------------------------------------
def test_correctness(topk_k=16, group_size=64, device="cuda"):
    print(f"\n{'='*70}")
    print(f"  Correctness test   topk_k={topk_k}  group_size={group_size}")
    print(f"{'='*70}")

    for M, K in [(128, 256), (512, 1024), (2048, 4096)]:
        A = torch.randn(M, K, dtype=torch.bfloat16, device=device)

        # reference (Triton / PyTorch)
        ref_combined, ref_scales, ref_mask = quantize_int8_int4_two_stage_groupwise(
            A, group_size, topk_k
        )
        # CUDA compat
        cuda_combined, cuda_scales, cuda_mask = quantize_int8_int4_cuda_compat(
            A, group_size, topk_k
        )

        # compare scales (should be very close)
        scale_diff = (ref_scales - cuda_scales).abs().max().item()

        # compare combined int8 values
        int8_match = (ref_combined == cuda_combined).float().mean().item() * 100

        # compare masks
        mask_match = (ref_mask == cuda_mask).float().mean().item() * 100

        # dequantize both and compare
        num_groups = K // group_size
        ref_mask_2d = ref_mask.reshape(M, -1)[:, :K]
        cuda_mask_2d = cuda_mask.reshape(M, -1)[:, :K]
        ref_deq = _dequantize(ref_combined, ref_scales, ref_mask_2d, group_size)
        cuda_deq = _dequantize(cuda_combined, cuda_scales, cuda_mask_2d, group_size)
        cos = _cosine_sim(ref_deq, cuda_deq)
        rmse = _rmse(ref_deq, cuda_deq)

        status = "PASS" if int8_match > 95 and cos > 0.999 else "WARN"
        print(f"  [{status}] M={M:5d} K={K:5d}  "
              f"int8_match={int8_match:6.2f}%  mask_match={mask_match:6.2f}%  "
              f"scale_maxΔ={scale_diff:.2e}  cos={cos:.6f}  rmse={rmse:.2e}")

    # CUDA fused output check
    A = torch.randn(1024, 2048, dtype=torch.bfloat16, device=device)
    ref_combined, ref_scales, ref_mask = quantize_int8_int4_two_stage_groupwise(
        A, group_size, topk_k
    )
    cuda_topk, cuda_others, cuda_scales = quantize_int8_int4_cuda_fused(
        A, group_size, topk_k
    )
    recon = cuda_topk.int() + cuda_others.int()
    comb_match = (recon.to(torch.int8) == cuda_topk + cuda_others).all().item()
    print(f"\n  Fused topk+others consistency: {'PASS' if comb_match else 'FAIL'}")
    print(f"  Fused vs ref scales cos: {_cosine_sim(ref_scales, cuda_scales):.6f}")


def _dequantize(combined, scales, mask_2d, group_size):
    """Simple dequantization for error measurement."""
    M, K = combined.shape
    num_groups = K // group_size
    out = torch.zeros(M, K, dtype=torch.float32, device=combined.device)
    for g in range(num_groups):
        s = g * group_size
        e = s + group_size
        vals = combined[:, s:e].float()
        m = mask_2d[:, s:e]
        sc_tk = scales[:, g, 0:1]
        sc_ot = scales[:, g, 1:2]
        sc = torch.where(m, sc_tk, sc_ot)
        out[:, s:e] = vals * sc
    return out


# ---------------------------------------------------------------------------
# 2. Quantization latency benchmark
# ---------------------------------------------------------------------------
def bench_quantize(topk_k=16, group_size=64, device="cuda",
                   warmup=10, repeat=50):
    print(f"\n{'='*70}")
    print(f"  Quantization latency (µs, median of {repeat})   topk_k={topk_k}")
    print(f"{'='*70}")
    print(f"  {'M':>6s} {'K':>6s}  {'PyTorch/Triton':>14s}  {'CUDA compat':>12s}  "
          f"{'CUDA fused':>11s}  {'speedup(f)':>10s}")
    print(f"  {'-'*6} {'-'*6}  {'-'*14}  {'-'*12}  {'-'*11}  {'-'*10}")

    for M, K in [(512, 2048), (1024, 4096), (2048, 4096),
                 (4096, 4096), (4096, 8192), (8192, 8192)]:
        A = torch.randn(M, K, dtype=torch.bfloat16, device=device)

        t_ref = _cuda_timer(
            lambda: quantize_int8_int4_two_stage_groupwise(A, group_size, topk_k),
            warmup, repeat)

        t_compat = _cuda_timer(
            lambda: quantize_int8_int4_cuda_compat(A, group_size, topk_k),
            warmup, repeat)

        t_fused = _cuda_timer(
            lambda: quantize_int8_int4_cuda_fused(A, group_size, topk_k),
            warmup, repeat)

        speedup = t_ref / max(t_fused, 0.1)
        print(f"  {M:6d} {K:6d}  {t_ref:14.1f}  {t_compat:12.1f}  "
              f"{t_fused:11.1f}  {speedup:9.2f}x")


# ---------------------------------------------------------------------------
# 3. End-to-end: quantize + matmul vs BF16 matmul
# ---------------------------------------------------------------------------
def bench_e2e(topk_k=16, group_size=64, device="cuda", warmup=10, repeat=50):
    print(f"\n{'='*70}")
    print(f"  End-to-end: quantize A + matmul  vs  BF16 matmul   topk_k={topk_k}")
    print(f"{'='*70}")
    print(f"  {'M':>5s} {'K':>5s} {'N':>5s}  {'BF16 mm':>9s}  "
          f"{'INT8 2stg':>10s}  {'INT8 fused':>11s}  "
          f"{'cos(ref)':>9s}  {'cos(fused)':>11s}")
    print(f"  {'-'*5} {'-'*5} {'-'*5}  {'-'*9}  {'-'*10}  {'-'*11}  "
          f"{'-'*9}  {'-'*11}")

    for M, K, N in [(1024, 2048, 2048), (2048, 4096, 4096),
                    (4096, 4096, 4096), (4096, 8192, 4096)]:
        A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        B_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device=device)

        # ---- BF16 baseline ----
        t_bf16 = _cuda_timer(
            lambda: torch.mm(A_bf16, B_bf16), warmup, repeat)
        ref_out = torch.mm(A_bf16.float(), B_bf16.float())

        # ---- INT8 two-stage (Triton quant + Triton matmul) ----
        def run_int8_ref():
            a_i8, a_sc, a_mask = quantize_int8_int4_two_stage_groupwise(
                A_bf16, group_size, topk_k)
            b_i8, b_sc = quantize_int8_groupwise_along_k(B_bf16, group_size)
            num_g = K // group_size
            mask_2d = a_mask.reshape(M, -1)[:, :K]
            return scaled_int8_mm_two_stage(a_i8, b_i8, a_sc, b_sc, mask_2d, group_size)

        t_ref_int8 = _cuda_timer(run_int8_ref, warmup, repeat)
        out_ref = run_int8_ref()
        cos_ref = _cosine_sim(ref_out, out_ref)

        # ---- INT8 two-stage (CUDA fused quant + Triton matmul) ----
        def run_int8_fused():
            topk, others, a_sc = quantize_int8_int4_cuda_fused(
                A_bf16, group_size, topk_k)
            b_i8, b_sc = quantize_int8_groupwise_along_k(B_bf16, group_size)
            a_combined = topk + others  # combine for matmul mask path
            # build mask from topk (non-zero positions)
            mask_2d = (topk != 0)
            return scaled_int8_mm_two_stage(
                a_combined.to(torch.int8), b_i8, a_sc, b_sc, mask_2d, group_size)

        t_fused_int8 = _cuda_timer(run_int8_fused, warmup, repeat)
        out_fused = run_int8_fused()
        cos_fused = _cosine_sim(ref_out, out_fused)

        print(f"  {M:5d} {K:5d} {N:5d}  {t_bf16:8.0f}µ  "
              f"{t_ref_int8:9.0f}µ  {t_fused_int8:10.0f}µ  "
              f"{cos_ref:9.6f}  {cos_fused:11.6f}")


# ---------------------------------------------------------------------------
# 4. Quantization quality (how much error the quantization introduces)
# ---------------------------------------------------------------------------
def bench_accuracy(topk_k=16, group_size=64, device="cuda"):
    print(f"\n{'='*70}")
    print(f"  Quantization error analysis   topk_k={topk_k}")
    print(f"{'='*70}")
    print(f"  {'M':>6s} {'K':>6s}  {'cos(deq,orig)':>14s}  {'rmse':>10s}  "
          f"{'max_err':>10s}")
    print(f"  {'-'*6} {'-'*6}  {'-'*14}  {'-'*10}  {'-'*10}")

    for M, K in [(1024, 2048), (2048, 4096), (4096, 8192)]:
        A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        A_f32 = A.float()

        combined, scales, mask = quantize_int8_int4_cuda_compat(
            A, group_size, topk_k)
        mask_2d = mask.reshape(M, -1)[:, :K]
        deq = _dequantize(combined, scales, mask_2d, group_size)

        cos  = _cosine_sim(A_f32, deq)
        rmse = _rmse(A_f32, deq)
        maxe = (A_f32 - deq).abs().max().item()
        print(f"  {M:6d} {K:6d}  {cos:14.6f}  {rmse:10.4e}  {maxe:10.4e}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test with small sizes")
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cuda"

    print("=" * 70)
    print("  INT8+INT4 Two-Stage Quantization Benchmark")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  PyTorch: {torch.__version__}")
    print("=" * 70)

    if args.quick:
        warmup, repeat = 3, 10
    else:
        warmup, repeat = 10, 50

    for topk_k in [16, 8]:
        test_correctness(topk_k=topk_k)

        if not args.correctness_only:
            bench_quantize(topk_k=topk_k, warmup=warmup, repeat=repeat)
            bench_e2e(topk_k=topk_k, warmup=warmup, repeat=repeat)
            bench_accuracy(topk_k=topk_k)


if __name__ == "__main__":
    main()
