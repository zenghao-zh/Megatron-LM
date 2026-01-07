# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from TorchAO: https://github.com/pytorch/ao
# Modified for Megatron-LM integration.

"""
INT8 mixed-precision training tensor subclass and autograd function.

This module provides the core tensor wrapper and autograd function that enables
INT8 computation in forward and backward passes while keeping weights in original precision.
"""

import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.utils._pytree as pytree
from torch import Tensor, nn

from .config import Int8MixedPrecisionTrainingConfig
from .int8_mm import (
    quantize_int8_rowwise, 
    quantize_int8_groupwise,
    scaled_int8_mm,
    scaled_int8_mm_groupwise,
)


aten = torch.ops.aten


def _implements(cls, aten_ops_or_torch_fns):
    """Decorator to register implementations for tensor subclass dispatch."""
    if not isinstance(aten_ops_or_torch_fns, (list, tuple)):
        aten_ops_or_torch_fns = [aten_ops_or_torch_fns]

    def decorator(func):
        for op in aten_ops_or_torch_fns:
            @functools.wraps(func)
            def wrapper(f, types, args, kwargs, _func=func):
                return _func(f, types, args, kwargs)
            cls._OP_TABLE[op] = wrapper
        return func

    return decorator


class Int8MixedPrecisionTrainingLinearWeight(Tensor):
    """Linear weight wrapper for INT8 mixed-precision training.
    
    The weight is stored in original precision (e.g. FP32 or BF16).
    During training, weight and activation are dynamically quantized to INT8 
    to utilize INT8 Tensor Cores, then scaled back to original precision.
    This is applied to both forward and backward passes based on config.
    """
    
    # Class-level dispatch table for operators
    _OP_TABLE: Dict[Callable, Callable] = {}

    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, data: Tensor, config: Int8MixedPrecisionTrainingConfig):
        return Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            data.stride(),
            data.storage_offset(),
            dtype=data.dtype,
            device=data.device,
        )

    @torch._dynamo.disable
    def __init__(self, data: Tensor, config: Int8MixedPrecisionTrainingConfig):
        self._data = data
        self.config = config

    def __tensor_flatten__(self):
        return ["_data"], [self.config]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None
    ):
        return cls(tensor_data_dict["_data"], *tensor_attributes)

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self._data}, config={self.config})"

    def to_original(self):
        """Return a copy of the underlying data tensor."""
        return self._data.clone()

    @classmethod
    def implements(cls, aten_ops_or_torch_fns):
        """Class method decorator to register op implementations."""
        return _implements(cls, aten_ops_or_torch_fns)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # Check if we have a custom implementation for this op
        if func in cls._OP_TABLE:
            return cls._OP_TABLE[func](func, types, args, kwargs)
        
        # Default handling
        config = None

        def unwrap(x):
            nonlocal config
            if config is None:
                config = x.config
            else:
                assert x.config == config
            return x._data

        out = func(
            *pytree.tree_map_only(cls, unwrap, args),
            **pytree.tree_map_only(cls, unwrap, kwargs),
        )

        if func is aten.copy_.default:
            return args[0]
        elif func in {
            aten.t.default,
            aten.detach.default,
            aten.empty_like.default,
            aten.new_zeros.default,
            aten.slice.Tensor,
            aten.view.default,
            aten.as_strided.default,
            aten._to_copy.default,
            aten._pin_memory.default,
            aten.split.Tensor,
            aten.clone.default,
        }:
            return pytree.tree_map_only(Tensor, lambda x: cls(x, config), out)
        else:
            return out

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
            
        # Check if we have a custom implementation
        if func in cls._OP_TABLE:
            return cls._OP_TABLE[func](func, types, args, kwargs)
        
        # Check if this is an external library op (e.g., TransformerEngine, triton)
        # by looking at the function's module path
        func_module = getattr(func, '__module__', '') or ''
        is_external_op = (
            'transformer_engine' in func_module or 
            'triton' in func_module or
            'tex.' in str(func) or  # TransformerEngine ops
            (hasattr(func, '__self__') and 'transformer_engine' in str(type(func.__self__)))
        )
        
        if is_external_op:
            # For external library ops, unwrap tensors to avoid compatibility issues
            def unwrap(x):
                if isinstance(x, cls):
                    return x._data
                return x
            
            unwrapped_args = pytree.tree_map(unwrap, args)
            unwrapped_kwargs = pytree.tree_map(unwrap, kwargs)
            
            with torch._C.DisableTorchFunctionSubclass():
                return func(*unwrapped_args, **unwrapped_kwargs)
        
        # Let torch dispatch handle standard tensor ops properly
        # This includes detach, clone, etc. which need to preserve the subclass
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    # FSDP all-gather extension
    def fsdp_pre_all_gather(
        self,
        mesh,
        outer_size=None,
        outer_stride=None,
        module=None,
        mp_policy=None,
    ):
        data = self._data
        if mp_policy is not None:
            data = data.to(mp_policy.param_dtype)
        return (data,), (self.config,)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[Tensor] = None,
    ):
        (data,) = all_gather_outputs
        (config,) = metadata
        if out is not None:
            assert isinstance(out, Int8MixedPrecisionTrainingLinearWeight)
            assert out.config == config
            return
        return Int8MixedPrecisionTrainingLinearWeight(data, config), all_gather_outputs


_INT8_MM_CALL_COUNT = 0
_INT8_MM_DEBUG = False  # Set to True to verify INT8 is being called


def _dynamic_int8_mm(A: Tensor, B: Tensor, group_size: int = 0) -> Tensor:
    """Dynamically quantize A and B to INT8 for matmul, then scale back.
    
    Supports both row-wise (group_size=0) and group-wise (group_size>0) quantization.
    
    Args:
        A: Activation tensor, may have more than 2 dims
        B: Weight tensor, must be exactly 2-dim
        group_size: Quantization granularity. 0=row-wise, >0=group-wise
        
    Returns:
        Result tensor in original precision
    """
    global _INT8_MM_CALL_COUNT
    _INT8_MM_CALL_COUNT += 1
    if _INT8_MM_DEBUG and _INT8_MM_CALL_COUNT <= 10:
        print(f"[INT8] _dynamic_int8_mm called #{_INT8_MM_CALL_COUNT}: "
              f"A.shape={tuple(A.shape)}, B.shape={tuple(B.shape)}, group_size={group_size}")
    
    # A may have more than 2 dims, while B must be exactly 2-dim
    A_2d = A.reshape(-1, A.shape[-1])
    
    if group_size > 0:
        # Group-wise quantization
        return _dynamic_int8_mm_groupwise(A, B, group_size)
    
    # Row-wise quantization (original implementation)
    # Quantize A row-wise
    A_i8, A_scale_rowwise = quantize_int8_rowwise(A_2d.contiguous())
    
    # Quantize B column-wise (via B.T row-wise)
    # B.T.contiguous() ensures the transposed view is made contiguous BEFORE quantization
    B_t_contig = B.T.contiguous()
    B_t_i8, B_scale_colwise = quantize_int8_rowwise(B_t_contig)
    
    # B_t_i8 is [N, K], we need [K, N] for matmul A @ B
    # B_t_i8.T would be non-contiguous, so we use .T.contiguous()
    B_i8 = B_t_i8.T.contiguous()
    
    out = scaled_int8_mm(
        A_i8.contiguous(),
        B_i8,
        A_scale_rowwise.contiguous(),
        B_scale_colwise.contiguous(),
    )
    return out.view(*A.shape[:-1], out.shape[-1])


def _dynamic_int8_mm_groupwise(A: Tensor, B: Tensor, group_size: int = 64) -> Tensor:
    """Dynamically quantize A and B to INT8 with group-wise scaling.
    
    Each group of `group_size` elements along K dimension has its own scale.
    
    Args:
        A: Activation tensor [..., K]
        B: Weight tensor [K, N]
        group_size: Number of elements per quantization group
        
    Returns:
        Result tensor in original precision
    """
    # A may have more than 2 dims
    orig_shape = A.shape
    A_2d = A.reshape(-1, A.shape[-1])  # [M, K]
    K = A_2d.shape[-1]
    
    # Quantize A with group-wise scaling along K
    A_i8, A_scales = quantize_int8_groupwise(A_2d.contiguous(), group_size)
    # A_i8: [M, K], A_scales: [M, num_groups]
    
    # Quantize B with group-wise scaling along K (B is [K, N])
    # We need to quantize along K dimension (rows of B)
    # Transpose to [N, K], quantize, then transpose back
    B_t = B.T.contiguous()  # [N, K]
    B_t_i8, B_scales = quantize_int8_groupwise(B_t, group_size)
    # B_t_i8: [N, K], B_scales: [N, num_groups]
    
    # For matmul A @ B, we need B in [K, N] format
    B_i8 = B_t_i8.T.contiguous()  # [K, N]
    
    # Group-wise scaled matmul
    out = scaled_int8_mm_groupwise(
        A_i8.contiguous(),
        B_i8,
        A_scales.contiguous(),
        B_scales.contiguous(),
        group_size,
    )
    
    return out.view(*orig_shape[:-1], out.shape[-1])


class _Int8MixedPrecisionTrainingLinearFunction(torch.autograd.Function):
    """Autograd function for INT8 mixed-precision linear layer."""
    
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Union[Int8MixedPrecisionTrainingLinearWeight, Tensor],
        bias: Optional[Tensor],
        config: Optional[Int8MixedPrecisionTrainingConfig] = None,
    ):
        # Unpack tensor subclass if necessary
        if isinstance(weight, Int8MixedPrecisionTrainingLinearWeight):
            config = weight.config
            weight = weight._data

        ctx.config = config
        ctx.save_for_backward(input, weight)
        ctx.bias = bias is not None

        # Cast weight to input dtype if needed
        weight = weight.to(input.dtype)
        group_size = config.group_size

        if config.output:
            out = _dynamic_int8_mm(input, weight.T, group_size)
        else:
            out = input @ weight.T
        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        weight = weight.to(input.dtype)
        config = ctx.config
        group_size = config.group_size

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            if config.grad_input:
                grad_input = _dynamic_int8_mm(grad_output, weight, group_size)
            else:
                grad_input = grad_output @ weight

        if ctx.needs_input_grad[1]:
            grad_output_2d = grad_output.view(-1, weight.shape[0])
            input_2d = input.view(-1, weight.shape[1])
            if config.grad_weight:
                grad_weight = _dynamic_int8_mm(input_2d.T, grad_output_2d, group_size).T
            else:
                grad_weight = grad_output_2d.T @ input_2d

        if ctx.needs_input_grad[2] and ctx.bias:
            grad_bias = grad_output.view(-1, weight.shape[0]).sum(0)

        return grad_input, grad_weight, grad_bias, None


# Register the custom F.linear implementation for the tensor subclass
@Int8MixedPrecisionTrainingLinearWeight.implements(torch.nn.functional.linear)
def _int8_linear_impl(func, types, args, kwargs):
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_gpu_dtype()
        args = tuple(x.to(dtype) if x is not None else x for x in args)
    return _Int8MixedPrecisionTrainingLinearFunction.apply(*args, **kwargs)


class _Int8MatmulFunction(torch.autograd.Function):
    """Autograd function for INT8 matmul: A @ B where B is Int8 weight (possibly transposed)."""
    
    @staticmethod
    def forward(ctx, input: Tensor, weight_data: Tensor, config: Int8MixedPrecisionTrainingConfig):
        """
        Compute input @ weight_data using INT8 quantization.
        
        Args:
            input: Activation tensor [..., K]
            weight_data: Weight tensor [K, N] (already transposed if needed)
            config: INT8 training config
        """
        ctx.config = config
        ctx.save_for_backward(input, weight_data)
        group_size = config.group_size
        
        if config.output:
            out = _dynamic_int8_mm(input, weight_data, group_size)
        else:
            out = input @ weight_data
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight_data = ctx.saved_tensors
        config = ctx.config
        group_size = config.group_size
        
        grad_input = grad_weight = None
        
        # grad_input = grad_output @ weight_data.T
        if ctx.needs_input_grad[0]:
            if config.grad_input:
                grad_input = _dynamic_int8_mm(grad_output, weight_data.T, group_size)
            else:
                grad_input = grad_output @ weight_data.T
        
        # grad_weight = input.T @ grad_output -> need grad for weight_data which is [K, N]
        # So grad_weight_data = input.T @ grad_output where input is [..., K], grad_output is [..., N]
        if ctx.needs_input_grad[1]:
            input_2d = input.reshape(-1, input.shape[-1])  # [batch, K]
            grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])  # [batch, N]
            if config.grad_weight:
                grad_weight = _dynamic_int8_mm(input_2d.T, grad_output_2d, group_size)  # [K, N]
            else:
                grad_weight = input_2d.T @ grad_output_2d  # [K, N]
        
        return grad_input, grad_weight, None


def _get_int8_weight_and_config(tensor):
    """Extract weight data and config from Int8 tensor or its transpose."""
    if isinstance(tensor, Int8MixedPrecisionTrainingLinearWeight):
        return tensor._data, tensor.config, False
    return None, None, False


# Register aten.mm for Int8 weight support (used by Megatron's parallel linear layers)
@Int8MixedPrecisionTrainingLinearWeight.implements(aten.mm.default)
def _int8_mm_dispatch(func, types, args, kwargs):
    """Handle aten.mm when one operand is Int8MixedPrecisionTrainingLinearWeight."""
    A, B = args[0], args[1]
    
    # Case 1: A @ B where B is Int8 weight (common: input @ weight.T)
    if isinstance(B, Int8MixedPrecisionTrainingLinearWeight):
        return _Int8MatmulFunction.apply(A, B._data, B.config)
    
    # Case 2: A @ B where A is Int8 weight (rare, but handle for completeness)
    if isinstance(A, Int8MixedPrecisionTrainingLinearWeight):
        # For A @ B where A is weight, we need special handling
        # This typically happens in grad_weight computation
        config = A.config
        if config.grad_weight:
            return _dynamic_int8_mm(A._data, B, config.group_size)
        else:
            return A._data @ B
    
    # Fallback (should not reach here)
    raise RuntimeError("_int8_mm_dispatch called but no Int8 weight found")


# Also register aten.matmul for broader coverage (torch.matmul may dispatch here for 2D tensors)
@Int8MixedPrecisionTrainingLinearWeight.implements(aten.matmul.default)
def _int8_matmul_dispatch(func, types, args, kwargs):
    """Handle aten.matmul when one operand is Int8MixedPrecisionTrainingLinearWeight."""
    A, B = args[0], args[1]
    
    # For 2D @ 2D, behavior is same as aten.mm
    if A.dim() == 2 and B.dim() == 2:
        if isinstance(B, Int8MixedPrecisionTrainingLinearWeight):
            return _Int8MatmulFunction.apply(A, B._data, B.config)
        if isinstance(A, Int8MixedPrecisionTrainingLinearWeight):
            config = A.config
            if config.grad_weight:
                return _dynamic_int8_mm(A._data, B, config.group_size)
            else:
                return A._data @ B
    
    # For higher-dim matmul, handle B being Int8 weight
    if isinstance(B, Int8MixedPrecisionTrainingLinearWeight):
        return _Int8MatmulFunction.apply(A, B._data, B.config)
    
    if isinstance(A, Int8MixedPrecisionTrainingLinearWeight):
        config = A.config
        if config.grad_weight:
            return _dynamic_int8_mm(A._data, B, config.group_size)
        else:
            return A._data @ B
    
    raise RuntimeError("_int8_matmul_dispatch called but no Int8 weight found")


class Int8MixedPrecisionTrainingLinear(nn.Linear):
    """Drop-in replacement for nn.Linear with INT8 mixed-precision training."""
    
    def __init__(
        self, *args, config: Int8MixedPrecisionTrainingConfig, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

    def forward(self, input: Tensor) -> Tensor:
        return _Int8MixedPrecisionTrainingLinearFunction.apply(
            input, self.weight, self.bias, self.config
        )
