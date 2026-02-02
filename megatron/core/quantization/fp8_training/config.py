# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from TorchAO: https://github.com/pytorch/ao
# Modified for Megatron-LM integration with FP8 support.

"""
Configuration for FP8 mixed-precision training.
"""

from dataclasses import dataclass


@dataclass
class FP8MixedPrecisionTrainingConfig:
    """Configuration for FP8 mixed-precision training.
    
    Controls which matmuls in the Linear layer use FP8:
    - Forward: output = input @ weight.T
    - Backward: grad_input = grad_output @ weight
    - Backward: grad_weight = grad_output.T @ input
    
    FP8 Formats:
    - E4M3 (float8_e4m3fn): 4 exponent bits, 3 mantissa bits
        Range: [-448, 448], higher precision, good for activations/weights
    - E5M2 (float8_e5m2): 5 exponent bits, 2 mantissa bits  
        Range: [-57344, 57344], larger range, good for gradients
    
    Attributes:
        output: Apply FP8 to forward matmul (default: True)
        grad_input: Apply FP8 to backward grad_input matmul (default: True)
        grad_weight: Apply FP8 to backward grad_weight matmul (default: False)
            Note: Disabling grad_weight FP8 is recommended for better convergence
        group_size: Quantization granularity along K dimension (default: 64)
            - 0: Row-wise quantization (one scale per row)
            - 64: Group-wise quantization (one scale per 64 elements)
            Group-wise is recommended for Tensor Parallelism compatibility.
        forward_dtype: FP8 dtype for forward pass (default: 'e4m3')
            - 'e4m3': torch.float8_e4m3fn (higher precision)
            - 'e5m2': torch.float8_e5m2 (larger range)
        backward_dtype: FP8 dtype for backward pass (default: 'e5m2')
            - 'e4m3': torch.float8_e4m3fn
            - 'e5m2': torch.float8_e5m2 (recommended for gradients due to larger range)
    """
    output: bool = True
    grad_input: bool = True
    grad_weight: bool = False  # Default False for better convergence
    group_size: int = 64  # Default 64 for TP compatibility
    forward_dtype: str = 'e4m3'  # 'e4m3' or 'e5m2'
    backward_dtype: str = 'e5m2'  # 'e4m3' or 'e5m2' (e5m2 recommended for gradients)
    
    def __repr__(self):
        return (
            f"FP8MixedPrecisionTrainingConfig("
            f"output={self.output}, "
            f"grad_input={self.grad_input}, "
            f"grad_weight={self.grad_weight}, "
            f"group_size={self.group_size}, "
            f"forward_dtype={self.forward_dtype}, "
            f"backward_dtype={self.backward_dtype})"
        )
    
    def get_forward_dtype(self):
        """Get the torch dtype for forward pass."""
        import torch
        if self.forward_dtype == 'e4m3':
            return torch.float8_e4m3fn
        elif self.forward_dtype == 'e5m2':
            return torch.float8_e5m2
        else:
            raise ValueError(f"Unknown forward_dtype: {self.forward_dtype}")
    
    def get_backward_dtype(self):
        """Get the torch dtype for backward pass."""
        import torch
        if self.backward_dtype == 'e4m3':
            return torch.float8_e4m3fn
        elif self.backward_dtype == 'e5m2':
            return torch.float8_e5m2
        else:
            raise ValueError(f"Unknown backward_dtype: {self.backward_dtype}")
