# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from TorchAO: https://github.com/pytorch/ao
# Modified for Megatron-LM integration.

"""
INT8 Mixed-Precision Training for Megatron-LM.

This module provides INT8 mixed-precision training functionality adapted from TorchAO.
It enables using INT8 Tensor Cores for faster training while keeping model weights
in original precision (BF16/FP32).

Usage:
    from megatron.core.quantization.int8_training import (
        apply_int8_training,
        Int8MixedPrecisionTrainingConfig,
    )
    
    # Apply to model
    config = Int8MixedPrecisionTrainingConfig(
        output=True,      # INT8 for forward
        grad_input=True,  # INT8 for backward grad_input
        grad_weight=False # Keep original precision for grad_weight (recommended)
    )
    apply_int8_training(model, config)
"""

import torch
import types
import torch.nn as nn

from .config import Int8MixedPrecisionTrainingConfig
from .int8_tensor import (
    Int8MixedPrecisionTrainingLinearWeight,
    Int8MixedPrecisionTrainingLinear,
    _Int8MixedPrecisionTrainingLinearFunction,
)


def _is_te_linear_layer(module):
    """Check if a module is a TransformerEngine linear layer."""
    module_path = module.__class__.__module__ or ""
    class_name = module.__class__.__name__
    
    if 'transformer_engine' in module_path:
        # TE Linear layers: Linear, LayerNormLinear, LayerNormMLP, etc.
        if 'Linear' in class_name or 'MLP' in class_name:
            return True
    return False


def _is_linear_layer(module):
    """Check if a module is a linear layer (including Megatron parallel layers)."""
    # Get module info
    class_name = module.__class__.__name__
    module_path = module.__class__.__module__ or ""
    
    # Skip TransformerEngine layers - handled separately
    if 'transformer_engine' in module_path:
        return False
    
    # Standard nn.Linear
    if isinstance(module, nn.Linear):
        return True
    
    # Check for Megatron parallel linear layers by class name
    # This covers ColumnParallelLinear, RowParallelLinear, and their variants
    if 'Linear' in class_name and hasattr(module, 'weight'):
        # Verify it has a 2D weight (linear layer characteristic)
        if module.weight is not None and module.weight.dim() == 2:
            return True
    
    return False


def _create_int8_te_linear_forward(original_forward, module, config):
    """Create a wrapped forward function for TE Linear that uses INT8 matmul.
    
    This replaces TE's forward with INT8 matmul for both forward and backward passes.
    The function matches Megatron's expected interface: returns (output, output_bias).
    
    Megatron's TELinear always returns (output, bias) tuple:
    - If te_return_bias=True: returns (output, bias) for external fusion
    - If te_return_bias=False: returns (output+bias, None) or (output, None)
    """
    from .int8_tensor import _dynamic_int8_mm
    
    def int8_te_forward(input, *args, **kwargs):
        # Get weight and bias from module
        weight = module.weight
        bias = getattr(module, 'bias', None)
        
        # TE returns zero-length tensor when bias=False, treat as None
        if bias is not None and bias.numel() == 0:
            bias = None
        
        # Megatron's TELinear uses te_return_bias to decide output format
        # te_return_bias = skip_bias_add and bias
        te_return_bias = getattr(module, 'te_return_bias', False)
        
        # Use INT8 matmul for forward
        if config.output and input.requires_grad:
            return _Int8TELinearFunction.apply(
                input, weight, bias, te_return_bias, config
            )
        else:
            # Fall back to original TE forward when not training or INT8 disabled
            return original_forward(input, *args, **kwargs)
    
    return int8_te_forward


class _Int8TELinearFunction(torch.autograd.Function):
    """Autograd function for INT8 matmul in TransformerEngine layers.
    
    Implements INT8 quantized matmul for both forward and backward passes,
    replacing TE's native forward with our INT8 implementation.
    
    Note: This function always returns (output, bias) tuple to match
    Megatron's expected interface for linear layers (TELinear).
    
    Args:
        input: Input tensor
        weight: Weight tensor [out_features, in_features]
        bias: Optional bias tensor
        te_return_bias: If True, return (output, bias); if False, return (output+bias, None)
        config: INT8 config
    """
    
    @staticmethod
    def forward(ctx, input, weight, bias, te_return_bias, config):
        from .int8_tensor import _dynamic_int8_mm
        
        ctx.save_for_backward(input, weight)
        ctx.config = config
        ctx.has_bias = bias is not None
        ctx.te_return_bias = te_return_bias
        
        group_size = config.group_size
        
        # Forward: output = input @ weight.T + bias
        # TE Linear stores weight as [out_features, in_features]
        if config.output:
            out = _dynamic_int8_mm(input, weight.T, group_size)
        else:
            out = input @ weight.T
        
        # Megatron interface: (output, output_bias)
        # If te_return_bias=True: return (output, bias) for external fusion
        # If te_return_bias=False: return (output+bias, None) or (output, None)
        if bias is not None:
            if te_return_bias:
                return out, bias
            else:
                return out + bias, None
        return out, None
    
    @staticmethod
    def backward(ctx, grad_output, grad_bias_output):
        from .int8_tensor import _dynamic_int8_mm
        
        input, weight = ctx.saved_tensors
        config = ctx.config
        group_size = config.group_size
        
        grad_input = grad_weight = grad_bias = None
        
        # grad_input = grad_output @ weight
        if ctx.needs_input_grad[0]:
            if config.grad_input and grad_output is not None:
                grad_input = _dynamic_int8_mm(grad_output, weight, group_size)
            elif grad_output is not None:
                grad_input = grad_output @ weight
        
        # grad_weight = grad_output.T @ input
        # Note: weight shape is [out_features, in_features]
        if ctx.needs_input_grad[1]:
            # Reshape for matmul
            grad_output_2d = grad_output.reshape(-1, weight.shape[0])
            input_2d = input.reshape(-1, weight.shape[1])
            if config.grad_weight:
                grad_weight = _dynamic_int8_mm(grad_output_2d.T, input_2d, group_size)
            else:
                grad_weight = grad_output_2d.T @ input_2d
        
        # grad_bias = sum of grad_output along batch dimensions
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.reshape(-1, weight.shape[0]).sum(0)
        
        # Return gradients for: input, weight, bias, te_return_bias, config
        return grad_input, grad_weight, grad_bias, None, None


def _wrap_te_layer(module, config):
    """Wrap a TransformerEngine layer to use INT8 matmul.
    
    This replaces the TE layer's forward with INT8 quantized matmul for both
    forward and backward passes, providing full INT8 acceleration.
    """
    # import types
    
    # Check if it's a TE Linear-like layer with weight
    if not hasattr(module, 'weight') or module.weight is None:
        return False
    
    # Skip if already wrapped
    if getattr(module, '_int8_enabled', False):
        return False
    
    # Store original forward
    original_forward = module.forward
    
    # Create wrapped forward
    wrapped_forward = _create_int8_te_linear_forward(original_forward, module, config)
    
    # Replace forward method - use a closure to capture wrapped_forward
    def make_forward(wf):
        def forward_method(self, x, *args, **kwargs):
            return wf(x, *args, **kwargs)
        return forward_method
    
    module.forward = types.MethodType(make_forward(wrapped_forward), module)
    module._int8_enabled = True
    module._int8_config = config
    module._original_forward = original_forward
    
    return True


def apply_int8_training(
    model,
    config: Int8MixedPrecisionTrainingConfig = None,
    filter_fn=None,
    verbose: bool = False,
):
    """Apply INT8 mixed-precision training to a model's Linear layers.
    
    This function applies INT8 quantization to Linear layers in two ways:
    
    1. For standard PyTorch Linear layers (nn.Linear, Megatron's ColumnParallelLinear,
       RowParallelLinear): Wraps weights with a tensor subclass that performs
       dynamic INT8 quantization during forward and backward passes.
       
    2. For TransformerEngine layers (TELinear, TEColumnParallelLinear, etc.):
       Replaces the forward method with INT8 quantized matmul implementation.
    
    Args:
        model: The model to apply INT8 training to
        config: Configuration for INT8 training. If None, uses default config
            with output=True, grad_input=True, grad_weight=False
        filter_fn: Optional function to filter which modules to apply INT8 to.
            Takes (module_name, module) and returns True to apply INT8.
            Default: apply to all Linear layers.
        verbose: If True, print detailed information about each layer.
            
    Returns:
        The modified model (in-place modification)
        
    Example:
        # Apply to all Linear layers
        apply_int8_training(model)
        
        # Apply only to transformer layers, excluding lm_head
        apply_int8_training(
            model,
            filter_fn=lambda name, mod: 'transformer' in name and 'lm_head' not in name
        )
    """
    if config is None:
        config = Int8MixedPrecisionTrainingConfig()
    
    applied_count = 0
    te_layer_count = 0
    applied_layers = []
    excluded_layers = []
    
    for name, module in model.named_modules():
        # Check if it's a linear-like layer
        is_te = _is_te_linear_layer(module)
        is_linear = _is_linear_layer(module)
        
        if not is_te and not is_linear:
            continue
            
        # Apply filter if provided
        if filter_fn is not None and not filter_fn(name, module):
            excluded_layers.append((name, module.__class__.__name__, "filtered"))
            continue
        
        # Handle TransformerEngine layers specially
        if is_te:
            if _wrap_te_layer(module, config):
                te_layer_count += 1
                applied_layers.append((name, module.__class__.__name__, "TE"))
            continue
            
        # Check if it's a linear layer (nn.Linear or Megatron parallel layers)
        if is_linear:
            # Skip if weight is None or already wrapped
            if module.weight is None:
                excluded_layers.append((name, module.__class__.__name__, "no weight"))
                continue
            if isinstance(module.weight, Int8MixedPrecisionTrainingLinearWeight):
                excluded_layers.append((name, module.__class__.__name__, "already wrapped"))
                continue
                
            # Wrap the weight with INT8 training tensor subclass
            new_weight = Int8MixedPrecisionTrainingLinearWeight(
                module.weight.data, config
            )
            module.weight = nn.Parameter(new_weight, requires_grad=True)
            applied_count += 1
            applied_layers.append((name, module.__class__.__name__, "PyTorch"))
    
    # Print summary
    print(f"  INT8 applied to {applied_count} PyTorch layers")
    if te_layer_count > 0:
        print(f"  INT8 applied to {te_layer_count} TransformerEngine layers")
    
    if verbose:
        print(f"\n  === INT8 Enabled Layers ({len(applied_layers)}) ===")
        for name, cls, layer_type in applied_layers:
            print(f"    [INT8] {name} ({cls}) - {layer_type}")
        
        if excluded_layers:
            print(f"\n  === Excluded Layers ({len(excluded_layers)}) ===")
            for name, cls, reason in excluded_layers:
                print(f"    [SKIP] {name} ({cls}) - {reason}")
    
    return model


def _default_int8_filter(name, module):
    """Default filter: exclude lm_head, output layers, and attention layers.
    
    These layers are more sensitive to INT8 precision loss:
    - lm_head/output_layer: directly affect output logits
    - attention layers (Q/K/V/O projections): critical for model quality
    
    By excluding them, we apply INT8 only to FFN layers which are more robust
    to quantization while still providing significant speedup.
    """
    # Exclude output/lm_head layers - they're sensitive to quantization
    exclude_patterns = [
        'lm_head',
        'output_layer', 
        'final_linear',
        'head',
        # Attention projection layers (Q, K, V, O)
        'query',           # query projection
        'key',             # key projection  
        'value',           # value projection
        'q_proj',          # alternative naming
        'k_proj',
        'v_proj',
        'qkv',             # fused QKV projection
        'linear_qkv',      # Megatron naming
        'dense',           # attention output (common in many architectures)
        'o_proj',          # output projection (alternative naming)
        'out_proj',        # output projection
        'linear_proj',     # Megatron attention output
        'attention.linear_proj',  # More specific Megatron pattern
    ]
    name_lower = name.lower()
    for pattern in exclude_patterns:
        if pattern in name_lower:
            return False
    return True


def apply_int8_training_from_args(model, args):
    """Apply INT8 training based on command-line arguments.
    
    Args:
        model: The model to apply INT8 training to
        args: Parsed arguments containing INT8 training config
        
    Returns:
        The modified model (in-place modification)
    """
    if not getattr(args, 'int8_mixed_precision_training', False):
        return model
    
    config = Int8MixedPrecisionTrainingConfig(
        output=getattr(args, 'int8_mp_output', True),
        grad_input=getattr(args, 'int8_mp_grad_input', True),
        grad_weight=getattr(args, 'int8_mp_grad_weight', False),
        group_size=getattr(args, 'int8_mp_group_size', 64),
    )
    
    # Use default filter to exclude lm_head unless user explicitly wants all layers
    filter_fn = None
    if not getattr(args, 'int8_mp_all_layers', False):
        filter_fn = _default_int8_filter
    
    # Check if verbose output is requested
    verbose = getattr(args, 'int8_mp_verbose', False)
    
    return apply_int8_training(model, config, filter_fn=filter_fn, verbose=verbose)


__all__ = [
    'Int8MixedPrecisionTrainingConfig',
    'Int8MixedPrecisionTrainingLinearWeight',
    'Int8MixedPrecisionTrainingLinear',
    'apply_int8_training',
    'apply_int8_training_from_args',
]

