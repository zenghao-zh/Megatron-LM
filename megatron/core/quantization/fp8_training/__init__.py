# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from TorchAO: https://github.com/pytorch/ao
# Modified for Megatron-LM integration with FP8 support.

"""
FP8 Mixed-Precision Training for Megatron-LM.

This module provides FP8 mixed-precision training functionality adapted from TorchAO.
It enables using FP8 Tensor Cores for faster training while keeping model weights
in original precision (BF16/FP32).

FP8 Formats:
    - E4M3 (float8_e4m3fn): Higher precision, range [-448, 448]
        Recommended for forward pass activations and weights
    - E5M2 (float8_e5m2): Larger dynamic range [-57344, 57344]  
        Recommended for backward pass gradients

Usage:
    from megatron.core.quantization.fp8_training import (
        apply_fp8_training,
        FP8MixedPrecisionTrainingConfig,
    )
    
    # Apply to model
    config = FP8MixedPrecisionTrainingConfig(
        output=True,       # FP8 for forward
        grad_input=True,   # FP8 for backward grad_input
        grad_weight=False, # Keep original precision for grad_weight (recommended)
        group_size=64,     # Group-wise quantization
        forward_dtype='e4m3',   # E4M3 for forward (higher precision)
        backward_dtype='e5m2',  # E5M2 for backward (larger range)
    )
    apply_fp8_training(model, config)
"""

import torch
import types
import torch.nn as nn

from megatron.training.utils import print_rank_0

from megatron.core.quantization.fp8_training.config import FP8MixedPrecisionTrainingConfig
from megatron.core.quantization.fp8_training.fp8_tensor import (
    FP8MixedPrecisionTrainingLinearWeight,
    FP8MixedPrecisionTrainingLinear,
    _FP8MixedPrecisionTrainingLinearFunction,
    _dynamic_fp8_mm,
    _dynamic_fp8_mm_groupwise,
)
from megatron.core.quantization.fp8_training.fp8_mm import (
    quantize_fp8_groupwise,
    scaled_fp8_mm_groupwise,
)


def _is_te_linear_layer(module):
    """Check if a module is a TransformerEngine linear layer."""
    module_path = module.__class__.__module__ or ""
    class_name = module.__class__.__name__
    
    if 'transformer_engine' in module_path:
        if 'Linear' in class_name or 'MLP' in class_name:
            return True
    return False


def _is_linear_layer(module):
    """Check if a module is a linear layer (including Megatron parallel layers)."""
    class_name = module.__class__.__name__
    module_path = module.__class__.__module__ or ""
    
    # Skip TransformerEngine layers - handled separately
    if 'transformer_engine' in module_path:
        return False
    
    # Standard nn.Linear
    if isinstance(module, nn.Linear):
        return True
    
    # Check for Megatron parallel linear layers by class name
    if 'Linear' in class_name and hasattr(module, 'weight'):
        if module.weight is not None and module.weight.dim() == 2:
            return True
    
    return False


def _create_fp8_te_linear_forward(original_forward, module, config, layer_name="unknown"):
    """Create a wrapped forward function for TE Linear that uses FP8 matmul.
    
    This replaces TE's forward with FP8 matmul for both forward and backward passes.
    The function matches Megatron's expected interface: returns (output, output_bias).
    """
    
    def fp8_te_forward(input, *args, **kwargs):
        # Get weight and bias from module
        weight = module.weight
        bias = getattr(module, 'bias', None)
        
        # TE returns zero-length tensor when bias=False, treat as None
        if bias is not None and bias.numel() == 0:
            bias = None
        
        # Megatron's TELinear uses te_return_bias to decide output format
        te_return_bias = getattr(module, 'te_return_bias', False)
        
        # Use FP8 matmul for forward
        if config.output and input.requires_grad:
            return _FP8TELinearFunction.apply(
                input, weight, bias, te_return_bias, config, layer_name
            )
        else:
            # Fall back to original TE forward when not training or FP8 disabled
            return original_forward(input, *args, **kwargs)
    
    return fp8_te_forward


class _FP8TELinearFunction(torch.autograd.Function):
    """Autograd function for FP8 matmul in TransformerEngine layers.
    
    Implements FP8 quantized matmul for both forward and backward passes,
    replacing TE's native forward with our FP8 implementation.
    """
    
    @staticmethod
    def forward(ctx, input, weight, bias, te_return_bias, config, layer_name="unknown"):
        from .fp8_tensor import _dynamic_fp8_mm
        
        ctx.save_for_backward(input, weight)
        ctx.config = config
        ctx.has_bias = bias is not None
        ctx.te_return_bias = te_return_bias
        ctx.layer_name = layer_name
        
        group_size = config.group_size
        
        # Forward: output = input @ weight.T + bias
        # TE Linear stores weight as [out_features, in_features]
        if config.output:
            out = _dynamic_fp8_mm(input, weight.T, group_size, config, is_backward=False)
        else:
            out = input @ weight.T
        
        # Megatron interface: (output, output_bias)
        if bias is not None:
            if te_return_bias:
                return out, bias
            else:
                return out + bias, None
        return out, None
    
    @staticmethod
    def backward(ctx, grad_output, grad_bias_output):
        from .fp8_tensor import _dynamic_fp8_mm
        
        layer_name = ctx.layer_name if hasattr(ctx, 'layer_name') else 'unknown'
        
        input, weight = ctx.saved_tensors
        config = ctx.config
        group_size = config.group_size
        
        grad_input = grad_weight = grad_bias = None
        
        # grad_input = grad_output @ weight
        if ctx.needs_input_grad[0] and grad_output is not None:
            if config.grad_input:
                grad_input = _dynamic_fp8_mm(grad_output, weight, group_size, config, is_backward=True)
            else:
                grad_input = grad_output @ weight
        
        # grad_weight = grad_output.T @ input
        if ctx.needs_input_grad[1]:
            grad_output_2d = grad_output.reshape(-1, weight.shape[0])
            input_2d = input.reshape(-1, weight.shape[1])
            if config.grad_weight:
                grad_weight = _dynamic_fp8_mm(grad_output_2d.T, input_2d, group_size, config, is_backward=True)
            else:
                grad_weight = grad_output_2d.T @ input_2d
        
        # grad_bias = sum of grad_output along batch dimensions
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.reshape(-1, weight.shape[0]).sum(0)
        
        # Return gradients for: input, weight, bias, te_return_bias, config, layer_name
        return grad_input, grad_weight, grad_bias, None, None, None


def _wrap_te_layer(module, config, layer_name="unknown"):
    """Wrap a TransformerEngine layer to use FP8 matmul.
    
    This replaces the TE layer's forward with FP8 quantized matmul for both
    forward and backward passes, providing full FP8 acceleration.
    """
    # Check if it's a TE Linear-like layer with weight
    if not hasattr(module, 'weight') or module.weight is None:
        return False
    
    # Skip if already wrapped
    if getattr(module, '_fp8_enabled', False):
        return False
    
    # Store original forward
    original_forward = module.forward
    
    # Create wrapped forward with layer name
    wrapped_forward = _create_fp8_te_linear_forward(original_forward, module, config, layer_name)
    
    # Replace forward method
    def make_forward(wf):
        def forward_method(self, x, *args, **kwargs):
            return wf(x, *args, **kwargs)
        return forward_method
    
    module.forward = types.MethodType(make_forward(wrapped_forward), module)
    module._fp8_enabled = True
    module._fp8_config = config
    module._original_forward = original_forward
    
    return True


def apply_fp8_training(
    model,
    config: FP8MixedPrecisionTrainingConfig = None,
    filter_fn=None,
    verbose: bool = False,
):
    """Apply FP8 mixed-precision training to a model's Linear layers.
    
    This function applies FP8 quantization to Linear layers in two ways:
    
    1. For standard PyTorch Linear layers (nn.Linear, Megatron's ColumnParallelLinear,
       RowParallelLinear): Wraps weights with a tensor subclass that performs
       dynamic FP8 quantization during forward and backward passes.
       
    2. For TransformerEngine layers (TELinear, TEColumnParallelLinear, etc.):
       Replaces the forward method with FP8 quantized matmul implementation.
    
    Args:
        model: The model to apply FP8 training to
        config: Configuration for FP8 training. If None, uses default config
            with output=True, grad_input=True, grad_weight=False
        filter_fn: Optional function to filter which modules to apply FP8 to.
            Takes (module_name, module) and returns True to apply FP8.
            Default: apply to all Linear layers.
        verbose: If True, print detailed information about each layer.
            
    Returns:
        The modified model (in-place modification)
        
    Example:
        # Apply to all Linear layers
        apply_fp8_training(model)
        
        # Apply only to transformer layers, excluding lm_head
        apply_fp8_training(
            model,
            filter_fn=lambda name, mod: 'transformer' in name and 'lm_head' not in name
        )
    """
    if config is None:
        config = FP8MixedPrecisionTrainingConfig()
    
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
            if _wrap_te_layer(module, config, layer_name=name):
                te_layer_count += 1
                applied_layers.append((name, module.__class__.__name__, "TE"))
            continue
            
        # Check if it's a linear layer
        if is_linear:
            # Skip if weight is None or already wrapped
            if module.weight is None:
                excluded_layers.append((name, module.__class__.__name__, "no weight"))
                continue
            if isinstance(module.weight, FP8MixedPrecisionTrainingLinearWeight):
                excluded_layers.append((name, module.__class__.__name__, "already wrapped"))
                continue
                
            # Wrap the weight with FP8 training tensor subclass
            new_weight = FP8MixedPrecisionTrainingLinearWeight(
                module.weight.data, config
            )
            module.weight = nn.Parameter(new_weight, requires_grad=True)
            applied_count += 1
            applied_layers.append((name, module.__class__.__name__, "PyTorch"))
    
    # Print summary
    print(f"  FP8 applied to {applied_count} PyTorch layers")
    if te_layer_count > 0:
        print(f"  FP8 applied to {te_layer_count} TransformerEngine layers")
    
    if verbose:
        print(f"\n  === FP8 Enabled Layers ({len(applied_layers)}) ===")
        for name, cls, layer_type in applied_layers:
            print(f"    [FP8] {name} ({cls}) - {layer_type}")
        
        if excluded_layers:
            print(f"\n  === Excluded Layers ({len(excluded_layers)}) ===")
            for name, cls, reason in excluded_layers:
                print(f"    [SKIP] {name} ({cls}) - {reason}")
    
    return model


def _default_fp8_filter(name, module):
    """Default filter: exclude lm_head, output layers, and attention layers.
    
    These layers are more sensitive to FP8 precision loss.
    """
    exclude_patterns = [
        'lm_head',
        'output_layer', 
        'final_linear',
        'head',
        # Attention projection layers (Q, K, V, O)
        'query',
        'key',
        'value',
        'q_proj',
        'k_proj',
        'v_proj',
        'qkv',
        'linear_qkv',
        'dense',
        'o_proj',
        'out_proj',
        'linear_proj',
        'attention.linear_proj',
    ]
    name_lower = name.lower()
    for pattern in exclude_patterns:
        if pattern in name_lower:
            return False
    return True


def apply_fp8_training_from_args(model, args):
    """Apply FP8 training based on command-line arguments.
    
    Args:
        model: The model to apply FP8 training to
        args: Parsed arguments containing FP8 training config
        
    Returns:
        The modified model (in-place modification)
    """
    if not getattr(args, 'fp8_mixed_precision_training', False):
        return model
    
    # Check if we should enable backward gradually
    enable_backward_at_iter = getattr(args, 'fp8_mp_enable_backward_at_iter', None)
    
    # Get current iteration
    current_iteration = getattr(args, 'iteration', 0)
    
    # Get FP8 dtypes
    forward_dtype = getattr(args, 'fp8_mp_forward_dtype', 'e4m3')
    backward_dtype = getattr(args, 'fp8_mp_backward_dtype', 'e5m2')
    
    if enable_backward_at_iter is not None and current_iteration < enable_backward_at_iter:
        # Start with only forward FP8, backward will be enabled later
        config = FP8MixedPrecisionTrainingConfig(
            output=getattr(args, 'fp8_mp_output', True),
            grad_input=False,
            grad_weight=False,
            group_size=getattr(args, 'fp8_mp_group_size', 64),
            forward_dtype=forward_dtype,
            backward_dtype=backward_dtype,
        )
        print_rank_0(f'  FP8 backward will be enabled at iteration {enable_backward_at_iter}')
    elif enable_backward_at_iter is not None and current_iteration >= enable_backward_at_iter:
        # Resume training: already past the threshold
        config = FP8MixedPrecisionTrainingConfig(
            output=getattr(args, 'fp8_mp_output', True),
            grad_input=getattr(args, 'fp8_mp_grad_input', True),
            grad_weight=getattr(args, 'fp8_mp_grad_weight', False),
            group_size=getattr(args, 'fp8_mp_group_size', 64),
            forward_dtype=forward_dtype,
            backward_dtype=backward_dtype,
        )
        print_rank_0(f'  FP8 backward already enabled (resumed at iteration {current_iteration} >= {enable_backward_at_iter})')
    else:
        # Use normal config from command line
        config = FP8MixedPrecisionTrainingConfig(
            output=getattr(args, 'fp8_mp_output', True),
            grad_input=getattr(args, 'fp8_mp_grad_input', True),
            grad_weight=getattr(args, 'fp8_mp_grad_weight', False),
            group_size=getattr(args, 'fp8_mp_group_size', 64),
            forward_dtype=forward_dtype,
            backward_dtype=backward_dtype,
        )
    
    # Print FP8 dtype info
    print_rank_0(f'  FP8 forward dtype: {forward_dtype}, backward dtype: {backward_dtype}')
    print_rank_0(f'  Using group-wise quantization with group_size={config.group_size}')
    
    # Use default filter to exclude lm_head unless user explicitly wants all layers
    filter_fn = None
    if not getattr(args, 'fp8_mp_all_layers', False):
        filter_fn = _default_fp8_filter
    
    # Check if verbose output is requested
    verbose = getattr(args, 'fp8_mp_verbose', False)
    
    return apply_fp8_training(model, config, filter_fn=filter_fn, verbose=verbose)


def update_fp8_backward_config(model, grad_input=True, grad_weight=False, verbose=False):
    """Update FP8 backward configuration for all layers with FP8 enabled.
    
    This function updates the config of all layers that already have FP8 training applied,
    allowing dynamic enabling of backward FP8 during training.
    
    Args:
        model: The model (or list of models) to update
        grad_input: Whether to enable FP8 for grad_input computation
        grad_weight: Whether to enable FP8 for grad_weight computation
        verbose: Whether to print detailed information
        
    Returns:
        Number of layers updated
    """
    # Handle model list (e.g., from pipeline parallelism)
    if isinstance(model, list):
        total_updated = 0
        for model_module in model:
            total_updated += update_fp8_backward_config(model_module, grad_input, grad_weight, verbose)
        return total_updated
    
    updated_count = 0
    te_updated_count = 0
    
    for name, module in model.named_modules():
        # Check for PyTorch Linear layers with wrapped weights
        if hasattr(module, 'weight') and isinstance(module.weight, FP8MixedPrecisionTrainingLinearWeight):
            old_config = module.weight.config
            # Update config in-place
            module.weight.config.grad_input = grad_input
            module.weight.config.grad_weight = grad_weight
            updated_count += 1
            
            if verbose:
                print(f"  Updated {name}: grad_input {old_config.grad_input}->{grad_input}, "
                      f"grad_weight {old_config.grad_weight}->{grad_weight}")
        
        # Check for TransformerEngine layers with FP8 enabled
        elif getattr(module, '_fp8_enabled', False):
            if hasattr(module, '_fp8_config'):
                old_config = module._fp8_config
                old_config.grad_input = grad_input
                old_config.grad_weight = grad_weight
                te_updated_count += 1
                
                if verbose:
                    print(f"  Updated TE layer {name}: grad_input {old_config.grad_input}->{grad_input}, "
                          f"grad_weight {old_config.grad_weight}->{grad_weight}")
    
    if verbose or updated_count > 0 or te_updated_count > 0:
        print(f"  Updated FP8 backward config: {updated_count} PyTorch layers")
        if te_updated_count > 0:
            print(f"  Updated FP8 backward config: {te_updated_count} TE layers")
    
    return updated_count + te_updated_count


__all__ = [
    'FP8MixedPrecisionTrainingConfig',
    'FP8MixedPrecisionTrainingLinearWeight',
    'FP8MixedPrecisionTrainingLinear',
    'apply_fp8_training',
    'apply_fp8_training_from_args',
    'update_fp8_backward_config',
]
