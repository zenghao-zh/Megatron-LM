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
        grad_input: Apply INT8 to backward grad_input matmul (default: False)
            Note: Disabling grad_input INT8 is recommended - gradient errors accumulate
            across layers and hurt convergence more than forward errors.
        grad_weight: Apply INT8 to backward grad_weight matmul (default: False)
            Note: Disabling grad_weight INT8 is recommended for better convergence
        group_size: Quantization granularity along K dimension (default: 64)
            - 0: Row-wise quantization (one scale per row)
            - 64: Group-wise quantization (one scale per 64 elements)
            Group-wise is recommended for Tensor Parallelism compatibility.
        quantization_method: Quantization method (default: 'groupwise')
            - 'groupwise': Single-stage group-wise quantization (one scale per group)
            - 'two_stage': Two-stage quantization (separate scales for top-k outliers and others)
            - 'two_stage_mixed': Mixed precision two-stage (INT8 for top-k, INT4 for others)
        topk_elements: Number of top-k elements per group for two-stage quantization (default: 16)
            Only used when quantization_method='two_stage' or 'two_stage_mixed'.
            For group_size=64, topk_elements=16 means top 25% outliers get their own scale.
    """
    output: bool = True
    grad_input: bool = True
    grad_weight: bool = False  # Default False for better convergence
    group_size: int = 64  # Default 64 for TP compatibility
    quantization_method: str = 'groupwise'  # 'groupwise', 'two_stage', or 'two_stage_mixed'
    topk_elements: int = 16  # Number of top-k elements for two-stage quantization
    
    def __repr__(self):
        return (
            f"Int8MixedPrecisionTrainingConfig("
            f"output={self.output}, "
            f"grad_input={self.grad_input}, "
            f"grad_weight={self.grad_weight}, "
            f"group_size={self.group_size}, "
            f"quantization_method={self.quantization_method}, "
            f"topk_elements={self.topk_elements})"
        )

