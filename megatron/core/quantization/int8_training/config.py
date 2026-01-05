# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from TorchAO: https://github.com/pytorch/ao
# Modified for Megatron-LM integration.

"""
Configuration for INT8 mixed-precision training.
"""

from dataclasses import dataclass


@dataclass
class Int8MixedPrecisionTrainingConfig:
    """Configuration for INT8 mixed-precision training.
    
    Controls which matmuls in the Linear layer use INT8:
    - Forward: output = input @ weight.T
    - Backward: grad_input = grad_output @ weight
    - Backward: grad_weight = grad_output.T @ input
    
    Attributes:
        output: Apply INT8 to forward matmul (default: True)
        grad_input: Apply INT8 to backward grad_input matmul (default: True)  
        grad_weight: Apply INT8 to backward grad_weight matmul (default: False)
            Note: Disabling grad_weight INT8 is recommended for better convergence
    """
    output: bool = True
    grad_input: bool = True
    grad_weight: bool = False  # Default False for better convergence
    
    def __repr__(self):
        return (
            f"Int8MixedPrecisionTrainingConfig("
            f"output={self.output}, "
            f"grad_input={self.grad_input}, "
            f"grad_weight={self.grad_weight})"
        )

