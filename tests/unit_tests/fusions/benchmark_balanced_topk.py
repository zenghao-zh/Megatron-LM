#!/usr/bin/env python3
"""
Benchmark script to compare FusedBalancedTopkFunction and BalancedTopkFunction
Tests both performance (runtime) and accuracy (numerical precision).
Includes large-scale test cases based on real MOE training configurations.
"""

import torch
import time
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import argparse

# Import the functions to compare
try:
    from megatron.core.fusions.fused_balanced_topk import FusedBalancedTopkFunction
    from megatron.core.transformer.moe.experts import BalancedTopkFunction
    HAVE_MEGATRON = True
except ImportError:
    HAVE_MEGATRON = False
    print("Warning: Megatron-LM not found. Using local implementations for testing.")
    
    # Provide local implementations for testing if Megatron is not available
    class BalancedTopkFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, tokens_per_expert, k, bank_size, bias, num_assigned_tokens, training=True):
            expert_indices = torch.repeat_interleave(
                torch.arange(len(tokens_per_expert), device=input.device),
                torch.tensor(tokens_per_expert, device=input.device)
            )
            H = input.shape[-1]
            
            # Reshape input to bank format
            x = input.view(-1, H//bank_size, bank_size)
            
            # Get corresponding bias and reshape
            expert_bias_expanded = bias[expert_indices]
            bias_reshaped = expert_bias_expanded.view(-1, H//bank_size, bank_size)
            
            # Batch compute topk
            _, topk_indices = (x.abs() + bias_reshaped).topk(k, dim=-1)
            
            # Create mask
            mask = torch.zeros_like(x, dtype=x.dtype)
            mask = mask.scatter(-1, topk_indices, 1).view_as(input)
            
            output = input * mask
            
            ctx.save_for_backward(mask)

            if training:
                with torch.no_grad():
                    mask_bool = (mask != 0).int()
                    expert_indices_expanded = expert_indices.unsqueeze(1).expand_as(mask_bool)
                    num_assigned_tokens.scatter_add_(0, expert_indices_expanded, mask_bool)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            mask, = ctx.saved_tensors
            grad_input = grad_output * mask
            return grad_input, None, None, None, None, None, None

    class FusedBalancedTopkFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, tokens_per_expert, k, bank_size, bias, num_assigned_tokens, training=True):
            # Simplified fused implementation (no actual jit optimization for demo)
            expert_indices = torch.repeat_interleave(
                torch.arange(len(tokens_per_expert), device=input.device),
                torch.tensor(tokens_per_expert, device=input.device)
            )
            H = input.shape[-1]
            
            x = input.view(-1, H//bank_size, bank_size)
            expert_bias_expanded = bias[expert_indices]
            bias_reshaped = expert_bias_expanded.view(-1, H//bank_size, bank_size)
            
            _, topk_indices = (x.abs() + bias_reshaped).topk(k, dim=-1)
            mask = torch.zeros_like(x, dtype=x.dtype)
            mask = mask.scatter(-1, topk_indices, 1).view_as(input)
            output = input * mask

            ctx.save_for_backward(mask)

            if training:
                mask_bool = (mask != 0).int()
                expert_indices_expanded = expert_indices.unsqueeze(1).expand_as(mask_bool)
                num_assigned_tokens.scatter_add_(0, expert_indices_expanded, mask_bool)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            mask, = ctx.saved_tensors
            return grad_output * mask, None, None, None, None, None, None


class BenchmarkConfig:
    """Configuration for benchmark tests"""
    def __init__(self, large_scale=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16
        self.num_warmup = 10
        self.num_trials = 100
        
        if large_scale:
            # Large-scale configurations based on real MOE training
            self.test_configs = [
                # Basic configurations (smaller for comparison)
                (1024, 4096, 8, 16, 32),
                (2048, 4096, 8, 16, 32),
                
                # Medium-scale configurations  
                (4096, 1024, 16, 16, 64),
                (8192, 1024, 16, 16, 64),
                
                # Large-scale configurations based on your MOE config
                # (total_tokens, hidden_size, num_experts, k, bank_size)
                
                # Based on moe-ffn-hidden-size=576, act-sparse-topk=16, act-sparse-bank-size=64
                (2048, 576, 32, 16, 64),    # Micro-batch scenario
                (4096, 576, 32, 16, 64),    # Medium batch
                (8192, 576, 32, 16, 64),    # Large batch
                (16384, 576, 32, 16, 64),   # Very large batch
                
                # Based on main hidden-size=1024 with sparse activation
                (2048, 1024, 32, 16, 64),   
                (4096, 1024, 32, 16, 64),   
                (8192, 1024, 32, 16, 64),   
                
                # Based on shared expert intermediate size=1152
                (2048, 1152, 32, 16, 64),   
                (4096, 1152, 32, 16, 64),   
                
                # Extreme large-scale (simulating full sequence length)
                (8192, 576, 32, 16, 64),    # Full sequence length scenario
                (8192, 1024, 32, 16, 64),   # Full sequence with main hidden size
                
                # Different bank sizes for comparison
                (4096, 576, 32, 32, 64),    # Larger k
                (4096, 576, 32, 8, 64),     # Smaller k
                (4096, 576, 32, 16, 128),   # Larger bank size
            ]
        else:
            # Standard test configurations
            self.test_configs = [
                # (total_tokens, hidden_size, num_experts, k, bank_size)
                (1024, 4096, 8, 16, 32),
                (2048, 4096, 8, 16, 32),
                (4096, 4096, 8, 16, 32),
                (1024, 8192, 16, 32, 64),
                (2048, 8192, 16, 32, 64),
            ]


def generate_test_data(total_tokens: int, hidden_size: int, num_experts: int, 
                      k: int, bank_size: int, device: torch.device, dtype: torch.dtype):
    """Generate test data for benchmarking"""
    
    # Generate input tensor
    input_tensor = torch.randn(total_tokens, hidden_size, device=device, dtype=dtype)
    
    # Generate tokens per expert (roughly balanced with some variation)
    base_tokens = total_tokens // num_experts
    remainder = total_tokens % num_experts
    tokens_per_expert = [base_tokens] * num_experts
    for i in range(remainder):
        tokens_per_expert[i] += 1
    
    # Add some realistic imbalance (as in real MOE scenarios)
    if num_experts >= 8:
        # Make some experts get more tokens (realistic load imbalance)
        for i in range(min(3, num_experts//4)):
            if tokens_per_expert[i] > 1:
                tokens_per_expert[i] -= 1
                tokens_per_expert[(i + num_experts//2) % num_experts] += 1
    
    # Generate bias
    bias = torch.randn(num_experts, hidden_size, device=device, dtype=torch.float32) * 0.1
    
    # Initialize statistics tensor
    num_assigned_tokens = torch.zeros(num_experts, hidden_size, device=device, dtype=torch.int)
    
    return input_tensor, tokens_per_expert, bias, num_assigned_tokens


def benchmark_function(func, input_tensor: torch.Tensor, tokens_per_expert: List[int], 
                      k: int, bank_size: int, bias: torch.Tensor, 
                      num_assigned_tokens: torch.Tensor, num_warmup: int, num_trials: int):
    """Benchmark a function's performance"""
    
    device = input_tensor.device
    
    # Warmup
    for _ in range(num_warmup):
        num_assigned_tokens.zero_()
        _ = func.apply(input_tensor, tokens_per_expert, k, bank_size, bias, num_assigned_tokens, True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    times = []
    memory_usage = []
    
    for _ in range(num_trials):
        num_assigned_tokens.zero_()
        
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_time = time.perf_counter()
        else:
            start_time = time.perf_counter()
            
        output = func.apply(input_tensor, tokens_per_expert, k, bank_size, bias, num_assigned_tokens, True)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
        
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    avg_memory = np.mean(memory_usage) if memory_usage else 0
    return output, np.array(times), avg_memory


def compute_accuracy_metrics(output1: torch.Tensor, output2: torch.Tensor):
    """Compute accuracy metrics between two outputs"""
    
    # Convert to float32 for accurate comparison
    out1_f32 = output1.float()
    out2_f32 = output2.float()
    
    # Absolute difference
    abs_diff = torch.abs(out1_f32 - out2_f32)
    
    # Relative difference
    rel_diff = abs_diff / (torch.abs(out1_f32) + 1e-8)
    
    metrics = {
        'max_abs_diff': abs_diff.max().item(),
        'mean_abs_diff': abs_diff.mean().item(),
        'max_rel_diff': rel_diff.max().item(),
        'mean_rel_diff': rel_diff.mean().item(),
        'mse': torch.nn.functional.mse_loss(out1_f32, out2_f32).item(),
        'cosine_similarity': torch.nn.functional.cosine_similarity(
            out1_f32.flatten(), out2_f32.flatten(), dim=0).item(),
        'l2_norm_diff': torch.norm(out1_f32 - out2_f32).item(),
        'l2_norm_ratio': (torch.norm(out1_f32 - out2_f32) / torch.norm(out1_f32)).item()
    }
    
    return metrics


def run_benchmark(large_scale=False):
    """Run the complete benchmark"""
    
    config = BenchmarkConfig(large_scale=large_scale)
    print(f"Running {'LARGE-SCALE' if large_scale else 'standard'} benchmark on {config.device} with dtype {config.dtype}")
    print(f"Warmup iterations: {config.num_warmup}, Trial iterations: {config.num_trials}")
    print(f"Number of test configurations: {len(config.test_configs)}")
    print("-" * 80)
    
    results = []
    
    for i, (total_tokens, hidden_size, num_experts, k, bank_size) in enumerate(config.test_configs):
        print(f"\nTest {i+1}/{len(config.test_configs)}: tokens={total_tokens}, hidden={hidden_size}, experts={num_experts}, k={k}, bank_size={bank_size}")
        
        # Check memory requirements
        estimated_memory = (total_tokens * hidden_size * 4) / 1024**2  # MB for float32
        print(f"  Estimated memory: {estimated_memory:.1f} MB")
        
        try:
            # Generate test data
            input_tensor, tokens_per_expert, bias, num_assigned_tokens1 = generate_test_data(
                total_tokens, hidden_size, num_experts, k, bank_size, config.device, config.dtype
            )
            
            # Create separate copies for each function
            num_assigned_tokens2 = num_assigned_tokens1.clone()
            
            # Benchmark BalancedTopkFunction
            print("  Benchmarking BalancedTopkFunction...")
            output1, times1, memory1 = benchmark_function(
                BalancedTopkFunction, input_tensor, tokens_per_expert, k, bank_size, 
                bias, num_assigned_tokens1, config.num_warmup, config.num_trials
            )
            
            # Benchmark FusedBalancedTopkFunction
            print("  Benchmarking FusedBalancedTopkFunction...")
            output2, times2, memory2 = benchmark_function(
                FusedBalancedTopkFunction, input_tensor, tokens_per_expert, k, bank_size, 
                bias, num_assigned_tokens2, config.num_warmup, config.num_trials
            )
            
            # Compute accuracy metrics
            accuracy_metrics = compute_accuracy_metrics(output1, output2)
            
            # Compute performance metrics
            perf_metrics = {
                'balanced_mean_ms': times1.mean(),
                'balanced_std_ms': times1.std(),
                'balanced_min_ms': times1.min(),
                'balanced_max_ms': times1.max(),
                'balanced_memory_mb': memory1,
                'fused_mean_ms': times2.mean(),
                'fused_std_ms': times2.std(),
                'fused_min_ms': times2.min(),
                'fused_max_ms': times2.max(),
                'fused_memory_mb': memory2,
                'speedup': times1.mean() / times2.mean(),
                'memory_ratio': memory1 / memory2 if memory2 > 0 else 1.0,
            }
            
            # Store results
            test_result = {
                'config': (total_tokens, hidden_size, num_experts, k, bank_size),
                'performance': perf_metrics,
                'accuracy': accuracy_metrics,
            }
            results.append(test_result)
            
            # Print results for this test
            print(f"  Performance:")
            print(f"    BalancedTopkFunction:      {perf_metrics['balanced_mean_ms']:.3f} ± {perf_metrics['balanced_std_ms']:.3f} ms ({perf_metrics['balanced_memory_mb']:.1f} MB)")
            print(f"    FusedBalancedTopkFunction: {perf_metrics['fused_mean_ms']:.3f} ± {perf_metrics['fused_std_ms']:.3f} ms ({perf_metrics['fused_memory_mb']:.1f} MB)")
            print(f"    Speedup: {perf_metrics['speedup']:.2f}x, Memory ratio: {perf_metrics['memory_ratio']:.2f}x")
            print(f"  Accuracy:")
            print(f"    Max absolute diff: {accuracy_metrics['max_abs_diff']:.2e}")
            print(f"    Mean absolute diff: {accuracy_metrics['mean_abs_diff']:.2e}")
            print(f"    Cosine similarity: {accuracy_metrics['cosine_similarity']:.6f}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ⚠️ Skipped due to insufficient memory: {e}")
                continue
            else:
                print(f"  ✗ Error: {e}")
                continue
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            continue
    
    return results


def plot_results(results, large_scale=False):
    """Create visualization plots for the benchmark results"""
    
    if not results:
        print("No results to plot.")
        return
    
    # Extract data for plotting
    test_names = []
    for i, r in enumerate(results):
        config = r['config']
        test_names.append(f"T{i+1}\n{config[0]}×{config[1]}\nE{config[2]}")
    
    balanced_times = [r['performance']['balanced_mean_ms'] for r in results]
    fused_times = [r['performance']['fused_mean_ms'] for r in results]
    speedups = [r['performance']['speedup'] for r in results]
    max_abs_diffs = [r['accuracy']['max_abs_diff'] for r in results]
    cosine_sims = [r['accuracy']['cosine_similarity'] for r in results]
    memory_ratios = [r['performance']['memory_ratio'] for r in results]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Runtime comparison
    x = np.arange(len(test_names))
    width = 0.35
    
    ax1.bar(x - width/2, balanced_times, width, label='BalancedTopkFunction', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, fused_times, width, label='FusedBalancedTopkFunction', alpha=0.8, color='orange')
    ax1.set_xlabel('Test Cases')
    ax1.set_ylabel('Runtime (ms)')
    ax1.set_title('Runtime Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup and Memory Ratio
    ax2_twin = ax2.twinx()
    bars1 = ax2.bar(x - width/2, speedups, width, alpha=0.8, color='green', label='Speedup')
    bars2 = ax2_twin.bar(x + width/2, memory_ratios, width, alpha=0.8, color='purple', label='Memory Ratio')
    ax2.set_xlabel('Test Cases')
    ax2.set_ylabel('Speedup (x)', color='green')
    ax2_twin.set_ylabel('Memory Ratio (x)', color='purple')
    ax2.set_title('Performance Improvements')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 3: Maximum absolute difference (handle zeros)
    valid_diffs = [d if d > 0 else 1e-10 for d in max_abs_diffs]
    ax3.semilogy(test_names, valid_diffs, 'o-', color='red', markersize=6)
    ax3.set_xlabel('Test Cases')
    ax3.set_ylabel('Max Absolute Difference (log scale)')
    ax3.set_title('Numerical Accuracy: Maximum Absolute Difference')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cosine similarity
    ax4.plot(test_names, cosine_sims, 'o-', color='blue', markersize=6, linewidth=2)
    ax4.set_xlabel('Test Cases')
    ax4.set_ylabel('Cosine Similarity')
    ax4.set_title('Numerical Accuracy: Cosine Similarity')
    ax4.set_ylim([min(cosine_sims) - 0.0001, 1.0001])
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'balanced_topk_benchmark_large.png' if large_scale else 'balanced_topk_benchmark.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved as '{filename}'")


def main():
    parser = argparse.ArgumentParser(description='Benchmark BalancedTopkFunction vs FusedBalancedTopkFunction')
    parser.add_argument('--plot', action='store_true', help='Generate plots (requires matplotlib)')
    parser.add_argument('--trials', type=int, default=100, help='Number of trial runs (default: 100)')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup runs (default: 10)')
    parser.add_argument('--large-scale', action='store_true', help='Run large-scale tests based on MOE training config')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Running on CPU may give different performance characteristics.")
    
    # Run benchmark
    results = run_benchmark(large_scale=args.large_scale)
    
    if not results:
        print("No benchmark results available.")
        return
    
    # Print summary
    print("\n" + "="*80)
    print(f"BENCHMARK SUMMARY ({'LARGE-SCALE' if args.large_scale else 'STANDARD'})")
    print("="*80)
    
    speedups = [r['performance']['speedup'] for r in results]
    memory_ratios = [r['performance']['memory_ratio'] for r in results]
    cosine_sims = [r['accuracy']['cosine_similarity'] for r in results]
    max_abs_diffs = [r['accuracy']['max_abs_diff'] for r in results]
    
    avg_speedup = np.mean(speedups)
    min_speedup = np.min(speedups)
    max_speedup = np.max(speedups)
    
    avg_memory_ratio = np.mean(memory_ratios)
    avg_cosine_sim = np.mean(cosine_sims)
    min_cosine_sim = np.min(cosine_sims)
    max_abs_diff = np.max(max_abs_diffs)
    
    print(f"Performance ({len(results)} test cases):")
    print(f"  Average speedup: {avg_speedup:.2f}x")
    print(f"  Speedup range: {min_speedup:.2f}x - {max_speedup:.2f}x")
    print(f"  Average memory ratio: {avg_memory_ratio:.2f}x")
    print(f"\nAccuracy:")
    print(f"  Average cosine similarity: {avg_cosine_sim:.6f}")
    print(f"  Minimum cosine similarity: {min_cosine_sim:.6f}")
    print(f"  Maximum absolute difference: {max_abs_diff:.2e}")
    
    # Find best performing configurations
    best_speedup_idx = np.argmax(speedups)
    best_config = results[best_speedup_idx]['config']
    print(f"\nBest speedup ({speedups[best_speedup_idx]:.2f}x) achieved with:")
    print(f"  Tokens: {best_config[0]}, Hidden: {best_config[1]}, Experts: {best_config[2]}, k: {best_config[3]}, Bank: {best_config[4]}")
    
    # Generate plots if requested
    if args.plot:
        try:
            plot_results(results, large_scale=args.large_scale)
        except ImportError:
            print("Warning: matplotlib not available. Skipping plot generation.")
    
    print(f"\nBenchmark completed successfully!")


if __name__ == "__main__":
    main() 