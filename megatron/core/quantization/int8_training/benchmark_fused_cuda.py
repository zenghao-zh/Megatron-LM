#!/usr/bin/env python3
"""
Benchmark: two-stage INT8 matmul — CUDA quant + dequant + cuBLAS vs baselines.

Strategies compared:
  1. cuBLAS BF16         — baseline, no quantization
  2. sep(cu+tri)         — CUDA quantise → Triton per-group INT8 matmul
  3. dequant+cuBLAS      — CUDA quantise → CUDA dequant → cuBLAS BF16 mm  ← best

Usage:
    python benchmark_fused_cuda.py              # full
    python benchmark_fused_cuda.py --quick      # smoke test
"""

import argparse
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from megatron.core.quantization.int8_training.int8_mm import (
    quantize_int8_int4_two_stage_groupwise,
    quantize_int8_groupwise_along_k,
    scaled_int8_mm_two_stage,
)
from megatron.core.quantization.int8_training.int8_int4_quant_cuda import (
    quantize_int8_int4_cuda_fused,
    dequant_a_cuda,
    dequant_b_cuda,
)


def _timer(fn, warmup=10, repeat=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        t0.record(); fn(); t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1) * 1e3)
    times.sort()
    return times[len(times) // 2]


def _cos(a, b):
    return F.cosine_similarity(
        a.flatten().float().unsqueeze(0),
        b.flatten().float().unsqueeze(0)).item()


@torch.no_grad()
def dequant_cublas_mm(A_bf16, B_dq, gs=64, topk_k=16):
    """Full pipeline: CUDA quant → CUDA dequant A → cuBLAS BF16 mm."""
    A_topk, A_others, A_scales = quantize_int8_int4_cuda_fused(A_bf16, gs, topk_k)
    A_dq = dequant_a_cuda(A_topk, A_others, A_scales, gs)
    return torch.mm(A_dq, B_dq)


# ── correctness ──────────────────────────────────────────────────────────────
def test_correctness(topk_k=16, gs=64, dev="cuda"):
    print(f"\n{'='*76}")
    print(f"  Correctness   topk_k={topk_k}")
    print(f"{'='*76}")

    for M, K, N in [(128, 256, 128), (256, 512, 256),
                     (1024, 2048, 1024), (2048, 4096, 2048)]:
        A = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        B_bf = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
        ref = A.float() @ B_bf.float()
        B_i8, B_sc = quantize_int8_groupwise_along_k(B_bf, gs)

        # reference: Triton quant + Triton matmul
        A_i8, A_sc, A_mask = quantize_int8_int4_two_stage_groupwise(A, gs, topk_k)
        mask_2d = A_mask.reshape(M, -1)[:, :K]
        out_ref = scaled_int8_mm_two_stage(A_i8, B_i8, A_sc, B_sc, mask_2d, gs)

        # dequant + cuBLAS
        B_dq = dequant_b_cuda(B_i8, B_sc, gs)
        out_dq = dequant_cublas_mm(A, B_dq, gs, topk_k)

        c_ref = _cos(ref, out_ref)
        c_dq  = _cos(ref, out_dq)
        c_dr  = _cos(out_ref, out_dq)

        ok = c_dq > 0.995 and c_dr > 0.998
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {M:5d}×{K:5d}×{N:5d}  "
              f"cos(fp,ref)={c_ref:.5f}  "
              f"cos(fp,dq)={c_dq:.5f}  "
              f"cos(ref,dq)={c_dr:.5f}")


# ── latency ──────────────────────────────────────────────────────────────────
def bench_e2e(topk_k=16, gs=64, dev="cuda", warmup=10, repeat=50):
    print(f"\n{'='*76}")
    print(f"  E2E latency (µs)   topk_k={topk_k}   median of {repeat}")
    print(f"{'='*76}")
    hdr = (f"  {'M':>5s} {'K':>5s} {'N':>5s}  "
           f"{'cuBLAS':>7s}  {'sep(cu+tri)':>12s}  {'dq+cuBLAS':>10s}  "
           f"{'dq/blas':>7s}  {'dq/sep':>7s}")
    print(hdr)
    print(f"  {'-'*5} {'-'*5} {'-'*5}  "
          f"{'-'*7}  {'-'*12}  {'-'*10}  {'-'*7}  {'-'*7}")

    sizes = [
        (512,  2048, 2048),
        (1024, 4096, 4096),
        (2048, 4096, 4096),
        (4096, 4096, 4096),
        (4096, 8192, 4096),
        (8192, 8192, 4096),
    ]

    for M, K, N in sizes:
        A = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        B_bf = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
        B_i8, B_sc = quantize_int8_groupwise_along_k(B_bf, gs)
        B_dq = dequant_b_cuda(B_i8, B_sc, gs)

        t_blas = _timer(lambda: torch.mm(A, B_bf), warmup, repeat)

        def _sep():
            tk, ot, sc = quantize_int8_int4_cuda_fused(A, gs, topk_k)
            comb = (tk.int() + ot.int()).clamp(-128, 127).to(torch.int8)
            mask = (tk != 0)
            return scaled_int8_mm_two_stage(comb, B_i8, sc, B_sc, mask, gs)
        t_sep = _timer(_sep, warmup, repeat)

        t_dq = _timer(
            lambda: dequant_cublas_mm(A, B_dq, gs, topk_k),
            warmup, repeat)

        r_bl = t_dq / max(t_blas, 0.1)
        r_sp = t_dq / max(t_sep, 0.1)
        print(f"  {M:5d} {K:5d} {N:5d}  "
              f"{t_blas:6.0f}µ  {t_sep:11.0f}µ  {t_dq:9.0f}µ  "
              f"{r_bl:6.2f}×  {r_sp:6.2f}×")


# ── component breakdown ──────────────────────────────────────────────────────
def bench_components(topk_k=16, gs=64, dev="cuda", warmup=10, repeat=50):
    M = K = N = 4096
    print(f"\n{'='*76}")
    print(f"  Component breakdown  {M}×{K}×{N}  topk_k={topk_k}")
    print(f"{'='*76}")

    A = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    B_bf = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
    B_i8, B_sc = quantize_int8_groupwise_along_k(B_bf, gs)

    t_blas = _timer(lambda: torch.mm(A, B_bf), warmup, repeat)
    print(f"  cuBLAS BF16 mm:          {t_blas:7.0f} µs")

    t_quant = _timer(
        lambda: quantize_int8_int4_cuda_fused(A, gs, topk_k), warmup, repeat)
    print(f"  CUDA quantise:           {t_quant:7.0f} µs")

    A_tk, A_ot, A_sc = quantize_int8_int4_cuda_fused(A, gs, topk_k)
    t_dq_a = _timer(
        lambda: dequant_a_cuda(A_tk, A_ot, A_sc, gs), warmup, repeat)
    print(f"  CUDA dequant A:          {t_dq_a:7.0f} µs")

    t_dq_b = _timer(
        lambda: dequant_b_cuda(B_i8, B_sc, gs), warmup, repeat)
    print(f"  CUDA dequant B:          {t_dq_b:7.0f} µs  (one-time)")

    A_i8, A_sc2, A_mask = quantize_int8_int4_two_stage_groupwise(A, gs, topk_k)
    mask_2d = A_mask.reshape(M, -1)[:, :K]
    t_tri = _timer(
        lambda: scaled_int8_mm_two_stage(A_i8, B_i8, A_sc2, B_sc, mask_2d, gs),
        warmup, repeat)
    print(f"  Triton two-stage mm:     {t_tri:7.0f} µs")

    print(f"\n  ─── Totals ───")
    total_sep = t_quant + t_tri
    print(f"  Separate (quant+tri):    {total_sep:7.0f} µs  ({total_sep/t_blas:.1f}× cuBLAS)")
    total_dq = t_quant + t_dq_a + t_blas
    print(f"  Dequant+cuBLAS (CUDA):   {total_dq:7.0f} µs  ({total_dq/t_blas:.1f}× cuBLAS)  ← best")
    print(f"  (B dequant is one-time: {t_dq_b:.0f} µs)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    w, r = (5, 20) if args.quick else (10, 50)

    print("=" * 76)
    print("  Two-stage INT8 matmul — CUDA quant + dequant + cuBLAS")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  PyTorch: {torch.__version__}")
    print("=" * 76)

    for topk in [16]:
        test_correctness(topk_k=topk)
        bench_e2e(topk_k=topk, warmup=w, repeat=r)

    bench_components(topk_k=16, warmup=w, repeat=r)


if __name__ == "__main__":
    main()
