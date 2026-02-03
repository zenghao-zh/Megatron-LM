#!/usr/bin/env python3
"""Test script for Triton-accelerated two-stage INT8 quantization.

This script verifies:
1. Correctness: Triton kernel output matches PyTorch implementation
2. Performance: Benchmark Triton vs PyTorch implementation
"""

import torch
import time
import argparse


def test_correctness(M, K, group_size=64, topk_elements=16, eps=1e-20, verbose=True):
    """Test that Triton and PyTorch implementations produce the same results."""
    from megatron.core.quantization.int8_training.int8_mm import (
        _quantize_int8_two_stage_groupwise_triton,
        _quantize_int8_two_stage_groupwise_pytorch,
        HAS_TRITON,
    )
    
    if not HAS_TRITON:
        print("Triton not available, skipping test")
        return True
    
    # Create random input tensor
    torch.manual_seed(42)
    tensor = torch.randn(M, K, device='cuda', dtype=torch.float32)
    
    # Run both implementations
    int8_triton, scales_triton, mask_triton = _quantize_int8_two_stage_groupwise_triton(
        tensor.clone(), group_size, topk_elements, eps
    )
    int8_pytorch, scales_pytorch, mask_pytorch = _quantize_int8_two_stage_groupwise_pytorch(
        tensor.clone(), group_size, topk_elements, False, eps
    )
    
    # Check shapes
    assert int8_triton.shape == int8_pytorch.shape, f"int8 shape mismatch: {int8_triton.shape} vs {int8_pytorch.shape}"
    assert scales_triton.shape == scales_pytorch.shape, f"scales shape mismatch: {scales_triton.shape} vs {scales_pytorch.shape}"
    assert mask_triton.shape == mask_pytorch.shape, f"mask shape mismatch: {mask_triton.shape} vs {mask_pytorch.shape}"
    
    # Check values
    # Note: Due to tie-breaking differences in top-k selection, exact match may not always work
    # We check that the quantization error is within acceptable bounds
    
    # Dequantize both results
    num_groups = scales_triton.shape[1]
    padded_K = num_groups * group_size
    
    # Reshape for dequantization
    int8_triton_grouped = int8_triton.reshape(M, -1)
    if int8_triton_grouped.shape[1] < padded_K:
        int8_triton_grouped = torch.nn.functional.pad(int8_triton_grouped, (0, padded_K - int8_triton_grouped.shape[1]))
    int8_triton_grouped = int8_triton_grouped[:, :padded_K].reshape(M, num_groups, group_size)
    
    int8_pytorch_grouped = int8_pytorch.reshape(M, -1)
    if int8_pytorch_grouped.shape[1] < padded_K:
        int8_pytorch_grouped = torch.nn.functional.pad(int8_pytorch_grouped, (0, padded_K - int8_pytorch_grouped.shape[1]))
    int8_pytorch_grouped = int8_pytorch_grouped[:, :padded_K].reshape(M, num_groups, group_size)
    
    # Dequantize triton result
    scale_topk_t = scales_triton[:, :, 0:1]  # [M, num_groups, 1]
    scale_others_t = scales_triton[:, :, 1:2]  # [M, num_groups, 1]
    scales_t = torch.where(mask_triton, scale_topk_t, scale_others_t)
    dequant_triton = (int8_triton_grouped.float() * scales_t).reshape(M, -1)[:, :K]
    
    # Dequantize pytorch result
    scale_topk_p = scales_pytorch[:, :, 0:1]  # [M, num_groups, 1]
    scale_others_p = scales_pytorch[:, :, 1:2]  # [M, num_groups, 1]
    scales_p = torch.where(mask_pytorch, scale_topk_p, scale_others_p)
    dequant_pytorch = (int8_pytorch_grouped.float() * scales_p).reshape(M, -1)[:, :K]
    
    # Compare dequantized results (should be very close)
    max_diff = (dequant_triton - dequant_pytorch).abs().max().item()
    rel_error = max_diff / (tensor.abs().max().item() + eps)
    
    # Also compare to original
    error_triton = (dequant_triton - tensor).abs().mean().item()
    error_pytorch = (dequant_pytorch - tensor).abs().mean().item()
    
    if verbose:
        print(f"  Shape: M={M}, K={K}, group_size={group_size}, topk={topk_elements}")
        print(f"  Max diff between Triton and PyTorch dequantized: {max_diff:.6f} (rel: {rel_error:.6f})")
        print(f"  Mean quantization error - Triton: {error_triton:.6f}, PyTorch: {error_pytorch:.6f}")
        
        # Check mask agreement (should be very similar, may differ on ties)
        mask_agreement = (mask_triton == mask_pytorch).float().mean().item()
        print(f"  Mask agreement: {mask_agreement*100:.2f}%")
    
    # Success criteria: relative error < 5% (due to potential tie-breaking differences)
    success = rel_error < 0.05
    if not success and verbose:
        print(f"  FAILED: Relative error too high ({rel_error:.4f} > 0.05)")
    
    return success


def benchmark(M, K, group_size=64, topk_elements=16, num_warmup=10, num_iterations=100):
    """Benchmark Triton vs PyTorch implementations."""
    from megatron.core.quantization.int8_training.int8_mm import (
        _quantize_int8_two_stage_groupwise_triton,
        _quantize_int8_two_stage_groupwise_pytorch,
        HAS_TRITON,
    )
    
    if not HAS_TRITON:
        print("Triton not available, skipping benchmark")
        return
    
    # Create random input tensor
    tensor = torch.randn(M, K, device='cuda', dtype=torch.float32)
    
    # Warmup
    for _ in range(num_warmup):
        _quantize_int8_two_stage_groupwise_triton(tensor.clone(), group_size, topk_elements)
        _quantize_int8_two_stage_groupwise_pytorch(tensor.clone(), group_size, topk_elements)
    
    torch.cuda.synchronize()
    
    # Benchmark Triton
    start = time.time()
    for _ in range(num_iterations):
        _quantize_int8_two_stage_groupwise_triton(tensor.clone(), group_size, topk_elements)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_iterations * 1000  # ms
    
    # Benchmark PyTorch
    start = time.time()
    for _ in range(num_iterations):
        _quantize_int8_two_stage_groupwise_pytorch(tensor.clone(), group_size, topk_elements)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_iterations * 1000  # ms
    
    speedup = pytorch_time / triton_time
    
    print(f"  Shape: M={M}, K={K}, group_size={group_size}, topk={topk_elements}")
    print(f"  Triton:  {triton_time:.4f} ms")
    print(f"  PyTorch: {pytorch_time:.4f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    return triton_time, pytorch_time, speedup


def main():
    parser = argparse.ArgumentParser(description='Test two-stage INT8 quantization')
    parser.add_argument('--test', action='store_true', help='Run correctness tests')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--all', action='store_true', help='Run all tests and benchmarks')
    args = parser.parse_args()
    
    if not args.test and not args.benchmark and not args.all:
        args.all = True  # Default to running all
    
    if args.test or args.all:
        print("=" * 60)
        print("Correctness Tests")
        print("=" * 60)
        
        test_cases = [
            # (M, K, group_size, topk_elements)
            (128, 256, 64, 16),
            (256, 512, 64, 16),
            (512, 1024, 64, 16),
            (1024, 2048, 64, 16),
            (4096, 4096, 64, 16),
            # Different group sizes
            (256, 512, 32, 8),
            (256, 512, 128, 32),
            # Edge cases
            (128, 100, 64, 16),  # K not divisible by group_size
            (64, 64, 64, 16),   # Small tensor
        ]
        
        all_passed = True
        for M, K, group_size, topk_elements in test_cases:
            print(f"\nTest case: M={M}, K={K}, group_size={group_size}, topk={topk_elements}")
            passed = test_correctness(M, K, group_size, topk_elements)
            if passed:
                print("  PASSED")
            else:
                print("  FAILED")
                all_passed = False
        
        print("\n" + "=" * 60)
        if all_passed:
            print("All correctness tests PASSED!")
        else:
            print("Some correctness tests FAILED!")
        print("=" * 60)
    
    if args.benchmark or args.all:
        print("\n" + "=" * 60)
        print("Performance Benchmarks")
        print("=" * 60)
        
        benchmark_cases = [
            # Typical LLM shapes
            (1024, 4096, 64, 16),   # Medium batch
            (4096, 4096, 64, 16),   # Large square
            (8192, 4096, 64, 16),   # Large batch
            (16384, 4096, 64, 16),  # Very large batch
            (4096, 8192, 64, 16),   # Wide tensor
            (4096, 16384, 64, 16),  # Very wide tensor
        ]
        
        results = []
        for M, K, group_size, topk_elements in benchmark_cases:
            print(f"\nBenchmark: M={M}, K={K}")
            result = benchmark(M, K, group_size, topk_elements)
            if result:
                results.append((M, K, *result))
        
        if results:
            print("\n" + "=" * 60)
            print("Summary")
            print("=" * 60)
            print(f"{'M':>6} {'K':>6} {'Triton (ms)':>12} {'PyTorch (ms)':>12} {'Speedup':>8}")
            print("-" * 50)
            for M, K, triton_time, pytorch_time, speedup in results:
                print(f"{M:>6} {K:>6} {triton_time:>12.4f} {pytorch_time:>12.4f} {speedup:>8.2f}x")
            
            avg_speedup = sum(r[4] for r in results) / len(results)
            print("-" * 50)
            print(f"Average speedup: {avg_speedup:.2f}x")


if __name__ == '__main__':
    main()
