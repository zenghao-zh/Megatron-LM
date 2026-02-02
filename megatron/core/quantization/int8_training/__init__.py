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

from megatron.training.utils import print_rank_0

from megatron.core.quantization.int8_training.config import Int8MixedPrecisionTrainingConfig
from megatron.core.quantization.int8_training.int8_tensor import (
    Int8MixedPrecisionTrainingLinearWeight,
    Int8MixedPrecisionTrainingLinear,
    _Int8MixedPrecisionTrainingLinearFunction,
    _dynamic_int8_mm,
    _dynamic_int8_mm_groupwise,
)
from megatron.core.quantization.int8_training.int8_mm import (
    quantize_int8_two_stage_groupwise,
    scaled_int8_mm_two_stage,
)



# from .int8_tensor import _dynamic_int8_mm
# import torch.distributed as dist
# import matplotlib
# matplotlib.use('Agg')  # 无界面后端
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import re


# def _should_plot_analysis(layer_name, config):
#     """判断是否需要进行可视化分析，返回 (should_plot, layer_id, iteration)
    
#     注意：画图功能独立于 config.grad_input 设置
#     - 画图时总是计算 INT8 和 FP32 的对比，用于分析量化效果
#     - 但是否真正使用 INT8 结果取决于 config.grad_input
#     """
#     import re
#     import torch.distributed as dist
    
#     # 只在 rank 0 画图
#     try:
#         if dist.is_initialized():
#             rank = dist.get_rank()
#             if rank != 0:
#                 return False, -1, 0
#     except:
#         pass  # 非分布式环境
    
#     # 提取层号
#     match = re.search(r'layers\.(\d+)\.', layer_name)
#     if not match:
#         return False, -1, 0
    
#     layer_id = int(match.group(1))
#     if layer_id not in [0, 10, 20, 31]:
#         return False, layer_id, 0
    
#     # 获取当前迭代次数
#     try:
#         from megatron.training import get_args
#         args = get_args()
#         iteration = args.iteration if hasattr(args, 'iteration') else 0
#     except:
#         iteration = 0
    
#     # 配置：指定需要画图的 iterations（可以是列表或每N步）
#     # 选项1：只在特定 iterations 画图
#     target_iterations = [8000, 8010]  # 指定需要画图的迭代次数
#     should_plot = iteration in target_iterations
    
#     # 选项2：每 N 步画一次（注释掉选项1，启用这一行）
#     # should_plot = (iteration % 100 == 0)
    
#     # 选项3：调试模式 - 每次都画图（训练时记得关闭！性能影响很大）
#     # should_plot = True
    
#     return should_plot, layer_id, iteration


# def _quantize_fp8_e4m3(tensor):
#     """FP8 E4M3 量化，返回量化值和反量化值"""
#     import torch
#     max_fp8 = 448.0
#     amax = tensor.abs().max()
#     scale = amax / max_fp8
#     scale = scale.clamp(min=1e-12)
    
#     quantized = tensor / scale
#     quantized = quantized.clamp(-max_fp8, max_fp8)
#     quantized_fp8 = torch.round(quantized * 8) / 8
#     dequantized = quantized_fp8 * scale
    
#     return quantized_fp8, dequantized, scale


# def _compute_quantization_data(grad_output, weight, group_size):
#     """计算所有量化数据，返回字典"""
#     import torch
#     from .int8_tensor import quantize_int8_rowwise, quantize_int8_groupwise, _dynamic_int8_mm
    
#     data = {}
    
#     # FP32 结果（真实值）
#     data['grad_input_fp32'] = grad_output @ weight
#     data['grad_input_int8'] = _dynamic_int8_mm(grad_output, weight, group_size)
    
#     # FP8 量化
#     data['grad_output_fp8_quant'], data['grad_output_fp8_dequant'], data['fp8_gout_scale'] = _quantize_fp8_e4m3(grad_output)
#     data['weight_fp8_quant'], data['weight_fp8_dequant'], data['fp8_weight_scale'] = _quantize_fp8_e4m3(weight)
#     data['grad_input_fp8'] = data['grad_output_fp8_dequant'] @ data['weight_fp8_dequant']
    
#     # INT8 量化
#     grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
#     weight_2d = weight.contiguous()
    
#     if group_size > 0:
#         grad_output_i8, grad_output_scale = quantize_int8_groupwise(grad_output_2d, group_size)
#         weight_2d_t = weight_2d.T.contiguous()
#         weight_i8, weight_scale = quantize_int8_groupwise(weight_2d_t, group_size)
        
#         M, K = grad_output_2d.shape
#         grad_output_scale_expanded = grad_output_scale.unsqueeze(-1).expand(-1, -1, group_size).reshape(M, -1)[:, :K]
#         data['grad_output_dequant'] = (grad_output_i8.float() * grad_output_scale_expanded).reshape(grad_output.shape)
        
#         N, K_w = weight_2d_t.shape
#         weight_scale_expanded = weight_scale.unsqueeze(-1).expand(-1, -1, group_size).reshape(N, -1)[:, :K_w]
#         data['weight_dequant'] = (weight_i8.float() * weight_scale_expanded).T
#     else:
#         grad_output_i8, grad_output_scale = quantize_int8_rowwise(grad_output_2d)
#         weight_t_contig = weight_2d.T.contiguous()
#         weight_t_i8, weight_scale = quantize_int8_rowwise(weight_t_contig)
        
#         data['grad_output_dequant'] = (grad_output_i8.float() * grad_output_scale.unsqueeze(1)).reshape(grad_output.shape)
#         data['weight_dequant'] = (weight_t_i8.float() * weight_scale.unsqueeze(1)).T
    
#     data['grad_output_i8'] = grad_output_i8
#     data['weight_i8'] = weight_i8
    
#     return data


# def _plot_histogram(ax, data, bins, title, xlabel='Value', ylabel='Frequency', stats_text=None, color='blue', edgecolor='darkblue', **kwargs):
#     """绘制直方图的通用函数"""
#     ax.hist(data, bins=bins, edgecolor=edgecolor, alpha=0.8, color=color, **kwargs)
#     ax.set_xlabel(xlabel, fontsize=10)
#     ax.set_ylabel(ylabel, fontsize=10)
#     ax.set_title(title, fontsize=11, fontweight='bold')
#     ax.grid(True, alpha=0.3)
    
#     if stats_text:
#         ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
#                 fontsize=8, va='top', ha='right',
#                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))


# def _plot_quantization_analysis(grad_output, weight, data, layer_id, layer_name, iteration):
#     """创建完整的量化分析图表"""
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import os
    
#     # 转换为 numpy
#     np_data = {k: v.detach().float().cpu().flatten().numpy() for k, v in data.items() if hasattr(v, 'detach')}
    
#     # 原始数据
#     grad_output_bf16 = grad_output.detach().float().cpu().flatten().numpy()
#     weight_bf16 = weight.detach().float().cpu().flatten().numpy()
    
#     # 误差计算
#     error = np_data['grad_input_int8'] - np_data['grad_input_fp32']
#     fp8_error = np_data['grad_input_fp8'] - np_data['grad_input_fp32']
    
#     # 创建 4x5 子图
#     fig, axes = plt.subplots(4, 5, figsize=(28, 20))
#     fig.suptitle(f'Layer {layer_id}: {layer_name} (Iteration {iteration})', fontsize=17, fontweight='bold')
    
#     # 第一行：grad_output
#     _plot_histogram(axes[0, 0], grad_output_bf16, 100, 'grad_output: BF16 Original', 
#                     stats_text=f'Mean: {grad_output_bf16.mean():.3e}\nStd: {grad_output_bf16.std():.3e}')
    
#     grad_output_log = np.log10(np.abs(grad_output_bf16) + 1e-10)
#     _plot_histogram(axes[0, 1], grad_output_log, 100, 'grad_output: BF16 (log scale)', 
#                     xlabel='log10(|Value|)', color='royalblue', edgecolor='navy')
    
#     _plot_histogram(axes[0, 2], np_data['grad_output_i8'].astype(np.int8), 255, 
#                     'grad_output: INT8 Quantized [-127, 127]', xlabel='INT8 Value',
#                     color='red', edgecolor='darkred', range=(-127, 128))
#     axes[0, 2].set_xlim(-128, 128)
    
#     _plot_histogram(axes[0, 3], np_data['grad_output_dequant'], 100, 'grad_output: INT8 Dequant',
#                     color='violet', edgecolor='darkviolet')
    
#     _plot_histogram(axes[0, 4], np_data['grad_output_fp8_dequant'], 100, 'grad_output: FP8 E4M3 Dequant',
#                     color='gold', edgecolor='darkgoldenrod')
    
#     # 第二行：weight
#     _plot_histogram(axes[1, 0], weight_bf16, 100, 'Weight: BF16 Original',
#                     color='green', edgecolor='darkgreen')
    
#     weight_log = np.log10(np.abs(weight_bf16) + 1e-10)
#     _plot_histogram(axes[1, 1], weight_log, 100, 'Weight: BF16 (log scale)',
#                     xlabel='log10(|Value|)', color='limegreen', edgecolor='darkgreen')
    
#     _plot_histogram(axes[1, 2], np_data['weight_i8'].astype(np.int8), 255,
#                     'Weight: INT8 Quantized [-127, 127]', xlabel='INT8 Value',
#                     color='magenta', edgecolor='darkmagenta', range=(-127, 128))
#     axes[1, 2].set_xlim(-128, 128)
    
#     _plot_histogram(axes[1, 3], np_data['weight_dequant'], 100, 'Weight: INT8 Dequant',
#                     color='cyan', edgecolor='darkcyan')
    
#     _plot_histogram(axes[1, 4], np_data['weight_fp8_dequant'], 100, 'Weight: FP8 E4M3 Dequant',
#                     color='gold', edgecolor='darkgoldenrod')
    
#     # 第三行：grad_input 结果和误差
#     _plot_histogram(axes[2, 0], np_data['grad_input_fp32'], 100, 'grad_input: BF16 Result',
#                     color='purple', edgecolor='darkviolet')
    
#     ginput_log = np.log10(np.abs(np_data['grad_input_fp32']) + 1e-10)
#     _plot_histogram(axes[2, 1], ginput_log, 100, 'grad_input: BF16 (log scale)',
#                     xlabel='log10(|Value|)', color='mediumpurple', edgecolor='indigo')
    
#     _plot_histogram(axes[2, 2], np_data['grad_input_int8'], 100, 'grad_input: INT8 Result',
#                     color='orange', edgecolor='darkorange')
    
#     # INT8 误差
#     int8_mae = np.abs(error).mean()
#     _plot_histogram(axes[2, 3], error, 100, 'grad_input: INT8 Error',
#                     xlabel='Error (INT8-BF16)', color='crimson', edgecolor='black',
#                     stats_text=f'MAE: {int8_mae:.2e}\nMax: {np.abs(error).max():.2e}')
#     axes[2, 3].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
#     # FP8 误差
#     fp8_mae = np.abs(fp8_error).mean()
#     _plot_histogram(axes[2, 4], fp8_error, 100, 'grad_input: FP8 Error',
#                     xlabel='Error (FP8-BF16)', color='olive', edgecolor='black',
#                     stats_text=f'MAE: {fp8_mae:.2e}\nMax: {np.abs(fp8_error).max():.2e}')
#     axes[2, 4].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
#     # 第四行：FP8 详细分析和对比表
#     _plot_histogram(axes[3, 0], np_data['grad_output_fp8_quant'], 100, 'FP8 E4M3: grad_output Quantized',
#                     xlabel='Quantized Value', color='gold', edgecolor='darkgoldenrod')
    
#     fp8_quant_log = np.log10(np.abs(np_data['grad_output_fp8_quant']) + 1e-10)
#     _plot_histogram(axes[3, 1], fp8_quant_log, 100, 'FP8 E4M3: Quantized (log)',
#                     xlabel='log10(|Quant Value|)', color='khaki', edgecolor='darkgoldenrod')
    
#     _plot_histogram(axes[3, 2], np_data['grad_input_fp8'], 100, 'FP8 E4M3: grad_input Result',
#                     color='yellowgreen', edgecolor='darkolivegreen')
    
#     _plot_histogram(axes[3, 3], np_data['weight_fp8_quant'], 100, 'FP8 E4M3: Weight Quantized',
#                     xlabel='Quantized Value', color='limegreen', edgecolor='darkgreen')
    
#     # 对比表
#     axes[3, 4].axis('off')
#     gout_int8_mae = np.abs(grad_output_bf16 - np_data['grad_output_dequant']).mean()
#     gout_fp8_mae = np.abs(grad_output_bf16 - np_data['grad_output_fp8_dequant']).mean()
#     weight_int8_mae = np.abs(weight_bf16 - np_data['weight_dequant']).mean()
#     weight_fp8_mae = np.abs(weight_bf16 - np_data['weight_fp8_dequant']).mean()
    
#     comparison = f'''Quantization Comparison
    
# grad_input Error:
#   INT8 MAE: {int8_mae:.2e}
#   FP8  MAE: {fp8_mae:.2e}
#   Ratio: {(fp8_mae/int8_mae):.3f}x

# grad_output Dequant MAE:
#   INT8: {gout_int8_mae:.2e}
#   FP8:  {gout_fp8_mae:.2e}

# Weight Dequant MAE:
#   INT8: {weight_int8_mae:.2e}
#   FP8:  {weight_fp8_mae:.2e}'''
    
#     axes[3, 4].text(0.05, 0.5, comparison, transform=axes[3, 4].transAxes,
#                     fontsize=8, va='center', ha='left', family='monospace',
#                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
#     # 保存
#     save_dir = '/root/workspace/Megatron-LM/grad_input_int8_analysis'
#     os.makedirs(save_dir, exist_ok=True)
#     safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
#     save_path = os.path.join(save_dir, f'layer{layer_id}_iter{iteration}_{safe_layer_name}.png')
#     plt.tight_layout(pad=2.0)
#     plt.savefig(save_path, dpi=120, bbox_inches='tight')
#     plt.close()
    
#     print(f"\n{'='*80}")
#     print(f"[Quantization Analysis] Layer {layer_id} | Iteration {iteration} | {layer_name}")
#     print(f"INT8 MAE: {int8_mae:.6e} | FP8 MAE: {fp8_mae:.6e} | Ratio: {(fp8_mae/int8_mae):.4f}")
#     print(f"Saved to: {save_path}")
#     print(f"{'='*80}\n")


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


def _create_int8_te_linear_forward(original_forward, module, config, layer_name="unknown"):
    """Create a wrapped forward function for TE Linear that uses INT8 matmul.
    
    This replaces TE's forward with INT8 matmul for both forward and backward passes.
    The function matches Megatron's expected interface: returns (output, output_bias).
    
    Megatron's TELinear always returns (output, bias) tuple:
    - If te_return_bias=True: returns (output, bias) for external fusion
    - If te_return_bias=False: returns (output+bias, None) or (output, None)
    """
    
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
                input, weight, bias, te_return_bias, config, layer_name
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
    def forward(ctx, input, weight, bias, te_return_bias, config, layer_name="unknown"):
        from .int8_tensor import _dynamic_int8_mm
        
        ctx.save_for_backward(input, weight)
        ctx.config = config
        ctx.has_bias = bias is not None
        ctx.te_return_bias = te_return_bias
        ctx.layer_name = layer_name  # Save layer name for backward
        
        group_size = config.group_size
        
        # Forward: output = input @ weight.T + bias
        # TE Linear stores weight as [out_features, in_features]
        if config.output:
            out = _dynamic_int8_mm(input, weight.T, group_size, config)
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
        
        layer_name = ctx.layer_name if hasattr(ctx, 'layer_name') else 'unknown'
        
        input, weight = ctx.saved_tensors
        config = ctx.config
        group_size = config.group_size
        
        grad_input = grad_weight = grad_bias = None

        # print_rank_0(f"config: {config.grad_input}")
        
        # grad_input = grad_output @ weight
        if ctx.needs_input_grad[0] and grad_output is not None:
            # # 检查是否需要进行可视化分析
            # should_plot, layer_id, iteration = _should_plot_analysis(layer_name, config)
            
            # # 如果需要画图，计算并可视化量化前后的结果
            # if should_plot:
            #     # 计算所有量化数据
            #     quant_data = _compute_quantization_data(grad_output, weight, group_size)
                
            #     # 绘制分析图
            #     _plot_quantization_analysis(grad_output, weight, quant_data, layer_id, layer_name, iteration)
                
            #     # 根据 config.grad_input 决定是否使用 INT8 结果
            #     if config.grad_input:
            #         grad_input = quant_data['grad_input_int8']
            #     else:
            #         grad_input = grad_output @ weight
            # elif config.grad_input
            if config.grad_input:
                # 不需要画图，直接计算 INT8 结果
                grad_input = _dynamic_int8_mm(grad_output, weight, group_size, config)
            else:
                # 不使用 INT8
                grad_input = grad_output @ weight
        
        # grad_weight = grad_output.T @ input
        # Note: weight shape is [out_features, in_features]
        if ctx.needs_input_grad[1]:
            # Reshape for matmul
            grad_output_2d = grad_output.reshape(-1, weight.shape[0])
            input_2d = input.reshape(-1, weight.shape[1])
            if config.grad_weight:
                grad_weight = _dynamic_int8_mm(grad_output_2d.T, input_2d, group_size, config)
            else:
                grad_weight = grad_output_2d.T @ input_2d
        
        # grad_bias = sum of grad_output along batch dimensions
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.reshape(-1, weight.shape[0]).sum(0)
        
        # Return gradients for: input, weight, bias, te_return_bias, config, layer_name
        return grad_input, grad_weight, grad_bias, None, None, None

def _wrap_te_layer(module, config, layer_name="unknown"):
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
    
    # Create wrapped forward with layer name
    wrapped_forward = _create_int8_te_linear_forward(original_forward, module, config, layer_name)
    
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
            if _wrap_te_layer(module, config, layer_name=name):
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
    
    # Check if we should enable backward gradually
    enable_backward_at_iter = getattr(args, 'int8_mp_enable_backward_at_iter', None)
    
    # Get current iteration (important for resume training)
    current_iteration = getattr(args, 'iteration', 0)
    
    # Determine quantization method based on --int8-mp-two-stage flag
    quantization_method = 'two_stage' if getattr(args, 'int8_mp_two_stage', False) else 'groupwise'
    topk_elements = getattr(args, 'int8_mp_topk', 16)
    
    if enable_backward_at_iter is not None and current_iteration < enable_backward_at_iter:
        # Start with only forward INT8, backward will be enabled later
        config = Int8MixedPrecisionTrainingConfig(
            output=getattr(args, 'int8_mp_output', True),
            grad_input=False,  # Disable initially
            grad_weight=False,  # Disable initially
            group_size=getattr(args, 'int8_mp_group_size', 64),
            quantization_method=quantization_method,
            topk_elements=topk_elements,
        )
        print_rank_0(f'  INT8 backward will be enabled at iteration {enable_backward_at_iter}')
    elif enable_backward_at_iter is not None and current_iteration >= enable_backward_at_iter:
        # Resume training: already past the threshold, enable backward immediately
        config = Int8MixedPrecisionTrainingConfig(
            output=getattr(args, 'int8_mp_output', True),
            grad_input=getattr(args, 'int8_mp_grad_input', True),
            grad_weight=getattr(args, 'int8_mp_grad_weight', False),
            group_size=getattr(args, 'int8_mp_group_size', 64),
            quantization_method=quantization_method,
            topk_elements=topk_elements,
        )
        print_rank_0(f'  INT8 backward already enabled (resumed at iteration {current_iteration} >= {enable_backward_at_iter})')
    else:
        # Use normal config from command line
        config = Int8MixedPrecisionTrainingConfig(
            output=getattr(args, 'int8_mp_output', True),
            grad_input=getattr(args, 'int8_mp_grad_input', True),
            grad_weight=getattr(args, 'int8_mp_grad_weight', False),
            group_size=getattr(args, 'int8_mp_group_size', 64),
            quantization_method=quantization_method,
            topk_elements=topk_elements,
        )
    
    # Print quantization method info
    if quantization_method == 'two_stage':
        print_rank_0(f'  Using two-stage quantization: top-{topk_elements} outliers per group')
    else:
        print_rank_0(f'  Using standard group-wise quantization')
    
    # Use default filter to exclude lm_head unless user explicitly wants all layers
    filter_fn = None
    if not getattr(args, 'int8_mp_all_layers', False):
        filter_fn = _default_int8_filter
    
    # Check if verbose output is requested
    verbose = getattr(args, 'int8_mp_verbose', False)
    
    return apply_int8_training(model, config, filter_fn=filter_fn, verbose=verbose)


def update_int8_backward_config(model, grad_input=True, grad_weight=False, verbose=False):
    """Update INT8 backward configuration for all layers with INT8 enabled.
    
    This function updates the config of all layers that already have INT8 training applied,
    allowing dynamic enabling of backward INT8 during training.
    
    Args:
        model: The model (or list of models) to update
        grad_input: Whether to enable INT8 for grad_input computation
        grad_weight: Whether to enable INT8 for grad_weight computation
        verbose: Whether to print detailed information
        
    Returns:
        Number of layers updated
    """
    # Handle model list (e.g., from pipeline parallelism)
    if isinstance(model, list):
        total_updated = 0
        for model_module in model:
            total_updated += update_int8_backward_config(model_module, grad_input, grad_weight, verbose)
        return total_updated
    
    updated_count = 0
    te_updated_count = 0
    
    for name, module in model.named_modules():
        # Check for PyTorch Linear layers with wrapped weights
        if hasattr(module, 'weight') and isinstance(module.weight, Int8MixedPrecisionTrainingLinearWeight):
            old_config = module.weight.config
            # Update config in-place
            module.weight.config.grad_input = grad_input
            module.weight.config.grad_weight = grad_weight
            updated_count += 1
            
            if verbose:
                print(f"  Updated {name}: grad_input {old_config.grad_input}->{grad_input}, "
                      f"grad_weight {old_config.grad_weight}->{grad_weight}")
        
        # Check for TransformerEngine layers with INT8 enabled
        elif getattr(module, '_int8_enabled', False):
            # TE layers store config in the wrapped forward closure
            # We need to update the config object that was passed to _create_int8_te_linear_forward
            # The config is accessible through the module's forward method
            if hasattr(module, '_int8_config'):
                old_config = module._int8_config
                old_config.grad_input = grad_input
                old_config.grad_weight = grad_weight
                te_updated_count += 1
                
                if verbose:
                    print(f"  Updated TE layer {name}: grad_input {old_config.grad_input}->{grad_input}, "
                          f"grad_weight {old_config.grad_weight}->{grad_weight}")
    
    if verbose or updated_count > 0 or te_updated_count > 0:
        print(f"  Updated INT8 backward config: {updated_count} PyTorch layers")
        if te_updated_count > 0:
            print(f"  Updated INT8 backward config: {te_updated_count} TE layers")
    
    return updated_count + te_updated_count


__all__ = [
    'Int8MixedPrecisionTrainingConfig',
    'Int8MixedPrecisionTrainingLinearWeight',
    'Int8MixedPrecisionTrainingLinear',
    'apply_int8_training',
    'apply_int8_training_from_args',
    'update_int8_backward_config',
]

