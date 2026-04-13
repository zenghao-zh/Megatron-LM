#!/usr/bin/env python3
"""
Benchmark: two-stage INT8 matmul — CUDA quant + dequant + cuBLAS vs baselines.

Strategies compared:
  1. cuBLAS BF16         — baseline, no quantization
  2. sep(cu+tri)         — CUDA quantise → Triton per-group INT8 matmul
  3. dequant+cuBLAS      — CUDA quantise → CUDA dequant → cuBLAS BF16 mm
  4. fused B + cuBLAS    — same as 3, but B uses fused quant+dequant kernel
  5. Hadamard + fused    — block-diagonal Hadamard rotation before quantization

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
    quant_dequant_a_cuda,
    dequant_a_cuda,
    dequant_b_cuda,
    quant_dequant_b_cuda,
)
from megatron.core.quantization.hadamard import random_hadamard_matrix, hadamard_rotate


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

        # dequant + cuBLAS (separate B path)
        B_dq_sep = dequant_b_cuda(B_i8, B_sc, gs)
        out_sep = dequant_cublas_mm(A, B_dq_sep, gs, topk_k)

        # dequant + cuBLAS (fused B path)
        B_dq_fused = quant_dequant_b_cuda(B_bf, gs)
        out_fused = dequant_cublas_mm(A, B_dq_fused, gs, topk_k)

        c_ref = _cos(ref, out_ref)
        c_sep = _cos(ref, out_sep)
        c_fsd = _cos(ref, out_fused)
        c_b   = _cos(B_dq_sep, B_dq_fused)

        ok = c_sep > 0.995 and c_fsd > 0.995 and c_b > 0.9999
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {M:5d}×{K:5d}×{N:5d}  "
              f"cos(fp,ref)={c_ref:.5f}  "
              f"cos(fp,sep)={c_sep:.5f}  "
              f"cos(fp,fused)={c_fsd:.5f}  "
              f"B_dq cos={c_b:.6f}")


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


# ── full pipeline including B quantization ────────────────────────────────────
def bench_full_pipeline(topk_k=16, gs=64, dev="cuda", warmup=10, repeat=50):
    print(f"\n{'='*76}")
    print(f"  Full pipeline (incl. B quant) (µs)  topk_k={topk_k}  median of {repeat}")
    print(f"{'='*76}")
    hdr = (f"  {'M':>5s} {'K':>5s} {'N':>5s}  "
           f"{'cuBLAS':>7s}  {'old(tri+cu)':>12s}  {'new(fused)':>11s}  "
           f"{'speedup':>8s}  {'vs blas':>8s}")
    print(hdr)
    print(f"  {'-'*5} {'-'*5} {'-'*5}  "
          f"{'-'*7}  {'-'*12}  {'-'*11}  {'-'*8}  {'-'*8}")

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

        t_blas = _timer(lambda: torch.mm(A, B_bf), warmup, repeat)

        def _old():
            A_dq = quant_dequant_a_cuda(A, gs, topk_k)
            B_i8, B_sc = quantize_int8_groupwise_along_k(B_bf, gs)
            B_dq = dequant_b_cuda(B_i8, B_sc, gs)
            return torch.mm(A_dq, B_dq)

        def _new():
            A_dq = quant_dequant_a_cuda(A, gs, topk_k)
            B_dq = quant_dequant_b_cuda(B_bf, gs)
            return torch.mm(A_dq, B_dq)

        t_old = _timer(_old, warmup, repeat)
        t_new = _timer(_new, warmup, repeat)

        speedup = t_old / max(t_new, 0.1)
        vs_blas = t_new / max(t_blas, 0.1)
        print(f"  {M:5d} {K:5d} {N:5d}  "
              f"{t_blas:6.0f}µ  {t_old:11.0f}µ  {t_new:10.0f}µ  "
              f"{speedup:7.2f}×  {vs_blas:7.2f}×")


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
    print(f"  CUDA quantise A:         {t_quant:7.0f} µs")

    A_tk, A_ot, A_sc = quantize_int8_int4_cuda_fused(A, gs, topk_k)
    t_dq_a = _timer(
        lambda: dequant_a_cuda(A_tk, A_ot, A_sc, gs), warmup, repeat)
    print(f"  CUDA dequant A:          {t_dq_a:7.0f} µs")

    t_dq_b_old = _timer(
        lambda: dequant_b_cuda(B_i8, B_sc, gs), warmup, repeat)
    print(f"  CUDA dequant B (old):    {t_dq_b_old:7.0f} µs")

    t_b_quant = _timer(
        lambda: quantize_int8_groupwise_along_k(B_bf, gs), warmup, repeat)
    print(f"  Triton quant B:          {t_b_quant:7.0f} µs")

    t_dq_b_fused = _timer(
        lambda: quant_dequant_b_cuda(B_bf, gs), warmup, repeat)
    print(f"  CUDA fused B (new):      {t_dq_b_fused:7.0f} µs")

    A_i8, A_sc2, A_mask = quantize_int8_int4_two_stage_groupwise(A, gs, topk_k)
    mask_2d = A_mask.reshape(M, -1)[:, :K]
    t_tri = _timer(
        lambda: scaled_int8_mm_two_stage(A_i8, B_i8, A_sc2, B_sc, mask_2d, gs),
        warmup, repeat)
    print(f"  Triton two-stage mm:     {t_tri:7.0f} µs")

    print(f"\n  ─── Totals ───")
    total_sep = t_quant + t_tri
    print(f"  Separate (quant+tri):    {total_sep:7.0f} µs  ({total_sep/t_blas:.1f}× cuBLAS)")
    total_dq_old = t_quant + t_dq_a + t_b_quant + t_dq_b_old + t_blas
    print(f"  Old B (tri+cu+cuBLAS):   {total_dq_old:7.0f} µs  ({total_dq_old/t_blas:.1f}× cuBLAS)")
    total_dq_new = t_quant + t_dq_a + t_dq_b_fused + t_blas
    print(f"  New B (fused+cuBLAS):    {total_dq_new:7.0f} µs  ({total_dq_new/t_blas:.1f}× cuBLAS)  ← best")
    saved = total_dq_old - total_dq_new
    print(f"  B saved:                 {saved:7.0f} µs  "
          f"(old B: {t_b_quant+t_dq_b_old:.0f} → new B: {t_dq_b_fused:.0f} µs)")


# ── Hadamard helpers ──────────────────────────────────────────────────────────

_H_CACHE = {}

def _get_H(gs, dev, dtype):
    key = (gs, str(dev))
    if key not in _H_CACHE:
        rng = torch.random.get_rng_state()
        torch.manual_seed(42)
        _H_CACHE[key] = random_hadamard_matrix(gs, dev)
        torch.random.set_rng_state(rng)
    return _H_CACHE[key].to(dtype=dtype, device=dev)


def _hadamard_rotate_AB(A, B, gs, dev):
    """Apply block-diagonal Hadamard: A_rot=A@H^T per group, B_rot=H^T@B per group."""
    H = _get_H(gs, dev, A.dtype)
    A_rot = hadamard_rotate(A, H)           # per-group right-multiply by H
    B_rot = hadamard_rotate(B, H, dim=0)    # per-group: effectively H^T @ B_g
    return A_rot, B_rot


# ── Hadamard quality & latency ───────────────────────────────────────────────

def bench_hadamard(topk_k=16, gs=64, dev="cuda", warmup=10, repeat=50):
    """Compare quantization quality and latency with/without Hadamard rotation."""
    print(f"\n{'='*76}")
    print(f"  Hadamard rotation effect   topk_k={topk_k}   gs={gs}")
    print(f"{'='*76}")
    print(f"  {'M':>5s} {'K':>5s} {'N':>5s}  "
          f"{'cos(no_had)':>11s}  {'cos(had)':>9s}  {'Δcos':>7s}  "
          f"{'t_noH':>6s}  {'t_H':>6s}  {'t_rot':>6s}")
    print(f"  {'-'*5} {'-'*5} {'-'*5}  "
          f"{'-'*11}  {'-'*9}  {'-'*7}  "
          f"{'-'*6}  {'-'*6}  {'-'*6}")

    sizes = [
        (512,  2048, 2048),
        (1024, 4096, 4096),
        (2048, 4096, 4096),
        (4096, 4096, 4096),
        (4096, 8192, 4096),
    ]

    for M, K, N in sizes:
        A = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        B = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
        ref = A.float() @ B.float()

        # ---- without Hadamard ----
        def _no_had():
            A_dq = quant_dequant_a_cuda(A, gs, topk_k)
            B_dq = quant_dequant_b_cuda(B, gs)
            return torch.mm(A_dq, B_dq)

        t_no = _timer(_no_had, warmup, repeat)
        out_no = _no_had()
        cos_no = _cos(ref, out_no)

        # ---- with Hadamard ----
        def _had():
            Ar, Br = _hadamard_rotate_AB(A, B, gs, dev)
            A_dq = quant_dequant_a_cuda(Ar, gs, topk_k)
            B_dq = quant_dequant_b_cuda(Br, gs)
            return torch.mm(A_dq, B_dq)

        t_had = _timer(_had, warmup, repeat)
        out_had = _had()
        cos_had = _cos(ref, out_had)

        # ---- rotation-only overhead ----
        t_rot = _timer(lambda: _hadamard_rotate_AB(A, B, gs, dev), warmup, repeat)

        delta = cos_had - cos_no
        print(f"  {M:5d} {K:5d} {N:5d}  "
              f"{cos_no:11.6f}  {cos_had:9.6f}  {delta:+7.4f}  "
              f"{t_no:5.0f}µ  {t_had:5.0f}µ  {t_rot:5.0f}µ")


def bench_hadamard_correctness(gs=64, dev="cuda"):
    """Verify that Hadamard rotation preserves matmul result (before quantization)."""
    print(f"\n{'='*76}")
    print(f"  Hadamard rotation correctness (no quantization)")
    print(f"{'='*76}")

    for M, K, N in [(128, 256, 128), (1024, 2048, 1024), (4096, 4096, 4096)]:
        A = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        B = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
        ref = A.float() @ B.float()

        Ar, Br = _hadamard_rotate_AB(A, B, gs, dev)
        out_rot = Ar.float() @ Br.float()
        c = _cos(ref, out_rot)
        ok = c > 0.99999
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {M:5d}×{K:5d}×{N:5d}  cos(ref, rotated)={c:.8f}")


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
        bench_full_pipeline(topk_k=topk, warmup=w, repeat=r)

    bench_components(topk_k=16, warmup=w, repeat=r)

    bench_hadamard_correctness()
    bench_hadamard(topk_k=16, warmup=w, repeat=r)


if __name__ == "__main__":
    main()
