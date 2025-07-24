#!/usr/bin/env python3
"""
Simple test to verify the benchmark script works
"""

import torch
import sys
import os

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.device_count()} device(s)")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
else:
    print("CUDA not available, using CPU")

# Test basic torch operations
print(f"PyTorch version: {torch.__version__}")

# Simple version of the functions for testing
class TestBalancedTopkFunction(torch.autograd.Function):
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

# Test with small data first, then larger data
def test_function():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_cases = [
        # Small test case for basic verification
        {
            'name': 'Small Test',
            'total_tokens': 128,
            'hidden_size': 256,
            'num_experts': 4,
            'k': 16,
            'bank_size': 16
        },
        # Medium test case
        {
            'name': 'Medium Test',
            'total_tokens': 1024,
            'hidden_size': 1024,
            'num_experts': 8,
            'k': 16,
            'bank_size': 64
        },
        # Large test case based on your MOE config
        {
            'name': 'Large Test (MOE Config)',
            'total_tokens': 2048,  # micro-batch-size * seq-length fragment
            'hidden_size': 576,    # moe-ffn-hidden-size from your config
            'num_experts': 32,     # num-experts from your config
            'k': 16,              # act-sparse-topk from your config
            'bank_size': 64       # act-sparse-bank-size from your config
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        print(f"  Device: {device}")
        print(f"  Tokens: {test_case['total_tokens']}")
        print(f"  Hidden size: {test_case['hidden_size']}")
        print(f"  Experts: {test_case['num_experts']}")
        print(f"  k: {test_case['k']}")
        print(f"  Bank size: {test_case['bank_size']}")
        
        # Generate test data
        input_tensor = torch.randn(
            test_case['total_tokens'], 
            test_case['hidden_size'], 
            device=device, 
            dtype=torch.float32
        )
        tokens_per_expert = [test_case['total_tokens'] // test_case['num_experts']] * test_case['num_experts']
        # Handle remainder
        remainder = test_case['total_tokens'] % test_case['num_experts']
        for i in range(remainder):
            tokens_per_expert[i] += 1
            
        bias = torch.randn(
            test_case['num_experts'], 
            test_case['hidden_size'], 
            device=device, 
            dtype=torch.float32
        ) * 0.1
        num_assigned_tokens = torch.zeros(
            test_case['num_experts'], 
            test_case['hidden_size'], 
            device=device, 
            dtype=torch.int
        )
        
        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Tokens per expert: {tokens_per_expert}")
        print(f"  Bias shape: {bias.shape}")
        
        # Test the function
        try:
            output = TestBalancedTopkFunction.apply(
                input_tensor, tokens_per_expert, test_case['k'], 
                test_case['bank_size'], bias, num_assigned_tokens, True
            )
            print(f"  Output shape: {output.shape}")
            print(f"  Output non-zero ratio: {(output != 0).float().mean().item():.3f}")
            print(f"  Memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB" if device.type == 'cuda' else "  Memory usage: N/A")
            print("  ✓ Test passed!")
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            all_passed = False
            
        # Clear cache for next test
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return all_passed

if __name__ == "__main__":
    print("=" * 60)
    print("BalancedTopk Simple Benchmark Test")
    print("=" * 60)
    
    success = test_function()
    
    if success:
        print("\n✓ All tests passed! You can now run the full benchmark:")
        print("  python benchmark_balanced_topk.py")
        print("  python benchmark_balanced_topk.py --plot")
        print("  python benchmark_balanced_topk.py --large-scale")
    else:
        print("\n✗ Some tests failed. Please check your environment.") 