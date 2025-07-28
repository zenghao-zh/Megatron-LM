import torch
import torch.nn.functional as F

from megatron.core.jit import jit_fuser
from megatron.core.utils import nvtx_decorator
from megatron.core.fusions.fused_balanced_topk_triton import fused_balanced_topk_triton_sort_optimized


@jit_fuser
def fused_balanced_topk(input, tokens_per_expert_tensor, k, bank_size, bias, num_assigned_tokens, training=True):
    # 创建专家索引
    expert_indices = torch.repeat_interleave(
        torch.arange(tokens_per_expert_tensor.shape[0], device=input.device),
        tokens_per_expert_tensor
    )

    H = input.shape[-1]
    
    # 重塑input为bank格式
    x = input.view(-1, H // bank_size, bank_size)
    
    # 获取对应的bias并重塑
    expert_bias_expanded = bias[expert_indices]
    bias_reshaped = expert_bias_expanded.view(-1, H // bank_size, bank_size)
    
    # 批量计算topk
    _, topk_indices = (x.abs() + bias_reshaped).topk(k, dim=-1)
    
    # 创建mask
    mask = torch.zeros_like(x, dtype=x.dtype)
    mask = mask.scatter(-1, topk_indices, 1).view_as(input)
    
    output = input * mask

    # 原地更新统计信息（torch.compile 支持）
    # 使用与原始 BalancedTopkFunction 完全相同的统计方式
    if training:
        with torch.no_grad():
            mask_bool = (mask != 0).int()  # [total_tokens, hidden_size_per_partition]
            
            # 创建每个token对应的专家索引
            expert_indices_expanded = expert_indices.unsqueeze(1).expand_as(mask_bool)
            
            # 按专家分组求和 - 与原始实现完全一致
            num_assigned_tokens.scatter_add_(
                0,  # 在专家维度上求和
                expert_indices_expanded, 
                mask_bool
            )

    return output, mask

@jit_fuser
def fused_balanced_topk_back(g, mask):
    topk_grad = g * mask

    return topk_grad

class FusedBalancedTopkFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, tokens_per_expert, k, bank_size, bias, num_assigned_tokens, training=True):

        output, mask = fused_balanced_topk(input, torch.tensor(tokens_per_expert, device=input.device), k, bank_size, bias, num_assigned_tokens, training)

        ctx.save_for_backward(mask)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return fused_balanced_topk_back(grad_output, mask), None, None, None, None, None, None

