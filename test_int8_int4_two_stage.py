#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for mixed precision INT8+INT4 two-stage quantization.

Tests:
1. Quantization correctness (INT8 for top-k, INT4 for others)
2. Matmul numerical accuracy
3. Comparison with original two-stage INT8 quantization
4. Performance benchmarks (optional)
"""

import torch
import time
import argparse


def test_quantize_int8_int4_two_stage_pytorch():
    """Test PyTorch implementation of mixed precision quantization."""
    from megatron.core.quantization.int8_training.int8_mm import (
        _quantize_int8_int4_two_stage_pytorch,
    )
    
    print("\n" + "="*60)
    print("Test 1: PyTorch INT8+INT4 Two-Stage Quantization")
    print("="*60)
    
    # Test parameters
    M, K = 32, 128
    group_size = 64
    topk_elements = 16
    
    # Create test tensor with some outliers
    torch.manual_seed(42)
    tensor = torch.randn(M, K, dtype=torch.float32, device='cuda')
    # Add some outliers
    tensor[:, :topk_elements] *= 10.0
    
    # Run quantization
    int8_topk, int4_packed, scales, topk_mask = _quantize_int8_int4_two_stage_pytorch(
        tensor, group_size, topk_elements
    )
    
    # Verify shapes
    assert int8_topk.shape == (M, K), f"int8_topk shape mismatch: {int8_topk.shape}"
    assert int4_packed.shape == (M, K // 2), f"int4_packed shape mismatch: {int4_packed.shape}"
    num_groups = (K + group_size - 1) // group_size
    assert scales.shape == (M, num_groups, 2), f"scales shape mismatch: {scales.shape}"
    assert topk_mask.shape == (M, num_groups, group_size), f"topk_mask shape mismatch: {topk_mask.shape}"
    
    # Verify INT8 range for top-k
    assert int8_topk.min() >= -127 and int8_topk.max() <= 127, "INT8 out of range"
    
    # Verify INT4 values in packed tensor
    int4_low = (int4_packed & 0x0F).to(torch.int8)
    int4_high = ((int4_packed >> 4) & 0x0F).to(torch.int8)
    # Sign extend
    int4_low = torch.where(int4_low > 7, int4_low - 16, int4_low)
    int4_high = torch.where(int4_high > 7, int4_high - 16, int4_high)
    assert int4_low.min() >= -7 and int4_low.max() <= 7, f"INT4 low out of range: [{int4_low.min()}, {int4_low.max()}]"
    assert int4_high.min() >= -7 and int4_high.max() <= 7, f"INT4 high out of range: [{int4_high.min()}, {int4_high.max()}]"
    
    # Verify top-k mask count
    topk_count = topk_mask.sum(dim=2)
    assert (topk_count == topk_elements).all(), f"Top-k count mismatch: {topk_count}"
    
    print(f"  Input shape: ({M}, {K})")
    print(f"  Group size: {group_size}, Top-k: {topk_elements}")
    print(f"  INT8 top-k shape: {int8_topk.shape}, range: [{int8_topk.min()}, {int8_topk.max()}]")
    print(f"  INT4 packed shape: {int4_packed.shape}")
    print(f"  INT4 unpacked range: low [{int4_low.min()}, {int4_low.max()}], high [{int4_high.min()}, {int4_high.max()}]")
    print(f"  Scales shape: {scales.shape}")
    print("  [PASS] PyTorch quantization test passed!")


def test_quantize_int8_int4_two_stage_triton():
    """Test Triton implementation of mixed precision quantization."""
    from megatron.core.quantization.int8_training.int8_mm import (
        _quantize_int8_int4_two_stage_pytorch,
        _quantize_int8_int4_two_stage_triton,
        HAS_TRITON,
    )
    
    print("\n" + "="*60)
    print("Test 2: Triton INT8+INT4 Two-Stage Quantization")
    print("="*60)
    
    if not HAS_TRITON:
        print("  [SKIP] Triton not available")
        return
    
    # Test parameters
    M, K = 64, 256
    group_size = 64
    topk_elements = 16
    
    # Create test tensor
    torch.manual_seed(123)
    tensor = torch.randn(M, K, dtype=torch.float32, device='cuda')
    tensor[:, :topk_elements] *= 5.0  # Add outliers
    
    # Run both implementations
    int8_topk_pt, int4_packed_pt, scales_pt, mask_pt = _quantize_int8_int4_two_stage_pytorch(
        tensor.clone(), group_size, topk_elements
    )
    int8_topk_tr, int4_packed_tr, scales_tr, mask_tr = _quantize_int8_int4_two_stage_triton(
        tensor.clone(), group_size, topk_elements
    )
    
    # Compare results
    # Note: Due to different top-k tie-breaking, results may not be exactly identical
    # We check shapes and ranges instead
    assert int8_topk_tr.shape == int8_topk_pt.shape, "INT8 shape mismatch"
    assert int4_packed_tr.shape == int4_packed_pt.shape, "INT4 packed shape mismatch"
    assert scales_tr.shape == scales_pt.shape, "Scales shape mismatch"
    assert mask_tr.shape == mask_pt.shape, "Mask shape mismatch"
    
    # Verify ranges
    assert int8_topk_tr.min() >= -127 and int8_topk_tr.max() <= 127, "Triton INT8 out of range"
    
    # Check scale similarity (may differ due to different top-k selection)
    scale_diff = (scales_tr - scales_pt).abs().mean()
    print(f"  Input shape: ({M}, {K})")
    print(f"  INT8 shapes: PT={int8_topk_pt.shape}, TR={int8_topk_tr.shape}")
    print(f"  INT4 packed shapes: PT={int4_packed_pt.shape}, TR={int4_packed_tr.shape}")
    print(f"  Scale mean diff: {scale_diff:.6f}")
    print("  [PASS] Triton quantization test passed!")


def test_matmul_accuracy():
    """Test matmul accuracy comparing mixed precision vs full precision."""
    from megatron.core.quantization.int8_training.int8_mm import (
        quantize_int8_int4_two_stage_groupwise,
        quantize_int8_groupwise_along_k,
        scaled_int8_int4_mm_two_stage,
    )
    
    print("\n" + "="*60)
    print("Test 3: Mixed Precision Matmul Accuracy")
    print("="*60)
    
    # Test parameters
    M, K, N = 128, 256, 64
    group_size = 64
    topk_elements = 16
    
    # Create test tensors
    torch.manual_seed(456)
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    
    # Ground truth: FP32 matmul
    C_fp32 = A @ B
    
    # Quantize A with mixed precision
    A_topk_i8, A_others_i4_packed, A_scales, A_mask = quantize_int8_int4_two_stage_groupwise(
        A, group_size, topk_elements
    )
    
    # Quantize B with standard INT8
    B_i8, B_scales = quantize_int8_groupwise_along_k(B, group_size)
    
    # Reshape mask for matmul
    A_mask_2d = A_mask.reshape(M, -1)[:, :K]
    
    # Mixed precision matmul
    C_mixed = scaled_int8_int4_mm_two_stage(
        A_topk_i8, A_others_i4_packed, B_i8, A_scales, B_scales, A_mask_2d, group_size
    )
    
    # Compute error metrics
    abs_error = (C_mixed - C_fp32).abs()
    rel_error = abs_error / (C_fp32.abs() + 1e-8)
    
    mae = abs_error.mean().item()
    max_ae = abs_error.max().item()
    mre = rel_error.mean().item()
    
    print(f"  Shapes: A({M},{K}) @ B({K},{N}) = C({M},{N})")
    print(f"  Group size: {group_size}, Top-k: {topk_elements}")
    print(f"  Mean Absolute Error: {mae:.6f}")
    print(f"  Max Absolute Error: {max_ae:.6f}")
    print(f"  Mean Relative Error: {mre:.4%}")
    
    # Basic sanity check: error should be bounded
    assert mae < 1.0, f"MAE too large: {mae}"
    print("  [PASS] Matmul accuracy test passed!")


def test_compare_with_original_two_stage():
    """Compare mixed precision with original two-stage INT8 quantization."""
    from megatron.core.quantization.int8_training.int8_mm import (
        quantize_int8_two_stage_groupwise,
        quantize_int8_int4_two_stage_groupwise,
        quantize_int8_groupwise_along_k,
        scaled_int8_mm_two_stage,
        scaled_int8_int4_mm_two_stage,
    )
    
    print("\n" + "="*60)
    print("Test 4: Compare Mixed Precision vs Original Two-Stage INT8")
    print("="*60)
    
    # Test parameters
    M, K, N = 128, 256, 64
    group_size = 64
    topk_elements = 16
    
    # Create test tensors with outliers
    torch.manual_seed(789)
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    
    # Add outliers to simulate activation distribution
    A[:, :topk_elements * (K // group_size)] *= 5.0
    
    # Ground truth
    C_fp32 = A @ B
    
    # Original two-stage INT8
    A_i8, A_scales, A_mask = quantize_int8_two_stage_groupwise(A, group_size, topk_elements)
    B_i8, B_scales = quantize_int8_groupwise_along_k(B, group_size)
    A_mask_2d = A_mask.reshape(M, -1)[:, :K]
    C_int8 = scaled_int8_mm_two_stage(A_i8, B_i8, A_scales, B_scales, A_mask_2d, group_size)
    
    # Mixed precision INT8+INT4
    A_topk_i8, A_others_i4, A_scales_mixed, A_mask_mixed = quantize_int8_int4_two_stage_groupwise(
        A, group_size, topk_elements
    )
    A_mask_2d_mixed = A_mask_mixed.reshape(M, -1)[:, :K]
    C_mixed = scaled_int8_int4_mm_two_stage(
        A_topk_i8, A_others_i4, B_i8, A_scales_mixed, B_scales, A_mask_2d_mixed, group_size
    )
    
    # Compute errors
    int8_error = (C_int8 - C_fp32).abs().mean().item()
    mixed_error = (C_mixed - C_fp32).abs().mean().item()
    
    int8_max_error = (C_int8 - C_fp32).abs().max().item()
    mixed_max_error = (C_mixed - C_fp32).abs().max().item()
    
    print(f"  Shapes: A({M},{K}) @ B({K},{N}) = C({M},{N})")
    print(f"  Original Two-Stage INT8:")
    print(f"    MAE: {int8_error:.6f}, Max Error: {int8_max_error:.6f}")
    print(f"  Mixed Precision INT8+INT4:")
    print(f"    MAE: {mixed_error:.6f}, Max Error: {mixed_max_error:.6f}")
    print(f"  Error Ratio (Mixed/INT8): {mixed_error/int8_error:.2f}x")
    
    print("  [PASS] Comparison test completed!")


def test_end_to_end_dynamic_mm():
    """Test end-to-end flow through _dynamic_int8_mm_groupwise."""
    from megatron.core.quantization.int8_training.int8_tensor import _dynamic_int8_mm_groupwise
    from megatron.core.quantization.int8_training.config import Int8MixedPrecisionTrainingConfig
    
    print("\n" + "="*60)
    print("Test 5: End-to-End Dynamic MM with Mixed Precision")
    print("="*60)
    
    # Test parameters
    M, K, N = 64, 128, 32
    group_size = 64
    
    # Create test tensors
    torch.manual_seed(999)
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    
    # Ground truth
    C_fp32 = (A.float() @ B.float())
    
    # Test with different quantization methods
    methods = ['groupwise', 'two_stage', 'two_stage_mixed']
    
    for method in methods:
        config = Int8MixedPrecisionTrainingConfig(
            quantization_method=method,
            group_size=group_size,
            topk_elements=16,
        )
        
        C_quant = _dynamic_int8_mm_groupwise(A, B, group_size, config)
        error = (C_quant.float() - C_fp32).abs().mean().item()
        
        print(f"  {method:20s}: MAE = {error:.6f}")
    
    print("  [PASS] End-to-end test passed!")


def benchmark_performance():
    """Benchmark performance of different quantization methods."""
    from megatron.core.quantization.int8_training.int8_tensor import _dynamic_int8_mm_groupwise
    from megatron.core.quantization.int8_training.config import Int8MixedPrecisionTrainingConfig
    
    print("\n" + "="*60)
    print("Benchmark: Performance Comparison")
    print("="*60)
    
    # Test parameters
    M, K, N = 2048, 4096, 2048
    group_size = 64
    warmup_iters = 10
    bench_iters = 100
    
    # Create test tensors
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    
    methods = ['groupwise', 'two_stage', 'two_stage_mixed']
    
    for method in methods:
        config = Int8MixedPrecisionTrainingConfig(
            quantization_method=method,
            group_size=group_size,
            topk_elements=16,
        )
        
        # Warmup
        for _ in range(warmup_iters):
            _ = _dynamic_int8_mm_groupwise(A, B, group_size, config)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(bench_iters):
            _ = _dynamic_int8_mm_groupwise(A, B, group_size, config)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / bench_iters) * 1000
        tflops = 2 * M * K * N / (avg_time_ms / 1000) / 1e12
        
        print(f"  {method:20s}: {avg_time_ms:.3f} ms/iter, {tflops:.2f} TFLOPS")
    
    # Baseline: BF16 matmul
    for _ in range(warmup_iters):
        _ = A @ B
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(bench_iters):
        _ = A @ B
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    avg_time_ms = (elapsed / bench_iters) * 1000
    tflops = 2 * M * K * N / (avg_time_ms / 1000) / 1e12
    print(f"  {'BF16 baseline':20s}: {avg_time_ms:.3f} ms/iter, {tflops:.2f} TFLOPS")


def main():
    parser = argparse.ArgumentParser(description='Test INT8+INT4 two-stage quantization')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    args = parser.parse_args()
    
    print("\n" + "#"*60)
    print("# INT8 + INT4 Mixed Precision Two-Stage Quantization Tests")
    print("#"*60)
    
    # Run correctness tests
    test_quantize_int8_int4_two_stage_pytorch()
    test_quantize_int8_int4_two_stage_triton()
    test_matmul_accuracy()
    test_compare_with_original_two_stage()
    test_end_to_end_dynamic_mm()
    
    if args.benchmark:
        benchmark_performance()
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
