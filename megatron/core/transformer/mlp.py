# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.tensor_parallel.utils import divide
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.transformer.moe.moe_utils import save_to_aux_losses_tracker
from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl, bias_swiglu_without_silu_impl
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    get_tensor_model_parallel_group_if_none,
    nvtx_range_pop,
    nvtx_range_push,
)
from megatron.core.transformer.identity_op import identity

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


# pylint: disable=missing-class-docstring
@dataclass
class MLPSubmodules:
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None

@dataclass
class BalancedTopkMLPSubmodules:
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None
    predictor: Union[ModuleSpec, type] = None

class BalancedTopkFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k, bank_size, bias, num_assigned_tokens, training = True):
        
        # ctx.save_for_backward(mask)
        H = input.shape[-1]
        
        # 重塑input为bank格式
        x = input.view(-1, H//bank_size, bank_size)  # [total_tokens, H//bank_size, bank_size]
        
        # 获取对应的bias并重塑
        bias_reshaped = bias.view(-1, H//bank_size, bank_size)
        
        # 批量计算topk
        _, topk_indices = (x + bias_reshaped).topk(k, dim=-1)
        
        # 创建mask
        mask = torch.zeros_like(x, dtype=x.dtype)
        mask = mask.scatter(-1, topk_indices, 1).view_as(input)
        
        output = input * mask
        
        ctx.save_for_backward(mask)

        if training:  
            with torch.no_grad():
                # 并行化版本：使用 scatter_add_ 进行分组求和
                mask_bool = (mask != 0).int()  # [total_tokens, hidden_size_per_partition]
                
                # 创建每个token对应的专家索引
                num_assigned_tokens += mask_bool.sum(dim = tuple(range(input.ndim - 1)))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors  # 恢复前向传播保存的掩码

        grad_input = grad_output*mask
        return grad_input, None, None, None, None, None

class BalancedTopkModule(MegatronModule):
    def __init__(self, config, hidden_size, topk, bank_size, tp_group=None):
        super().__init__(config)
        
        # 获取专家张量并行组信息
        if tp_group is not None:
            self.tp_group = tp_group
            tp_size = self.tp_group.size()
            tp_rank = self.tp_group.rank()
        else:
            # 向后兼容：使用全局专家张量并行组
            self.tp_group = parallel_state.get_expert_tensor_parallel_group()
            tp_size = parallel_state.get_expert_tensor_parallel_world_size()
            tp_rank = parallel_state.get_expert_tensor_parallel_rank()
        
        # 计算 TP 分片后的 hidden_size
        self.hidden_size_per_partition = divide(hidden_size, tp_size)

        assert self.hidden_size_per_partition % bank_size == 0, "hidden_size_per_partition must be divisible by bank_size"
        
        # 注册 buffer，考虑 TP 分片
        self.register_buffer('balanced_bias', 
                           torch.zeros(self.hidden_size_per_partition, 
                                     dtype=torch.float32))
        self.register_buffer('num_assigned_tokens', 
                           torch.zeros(self.hidden_size_per_partition, 
                                     dtype=torch.int))

        self.topk = topk
        self.bank_size = bank_size
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.act_sparse_enable_fused_balanced_topk = getattr(config, 'act_sparse_enable_fused_balanced_topk', False)

    
    def forward(self, x, k = None):

        # 确保输入张量的最后一个维度与分片后的 hidden_size 匹配
        if x.shape[-1] != self.hidden_size_per_partition:
            raise ValueError(f"Expected input hidden size {self.hidden_size_per_partition} "
                           f"but got {x.shape[-1]} for TP rank {self.tp_rank}")
        
        mask = BalancedTopkFunction.apply(x, self.topk if k is None else k, 
                    self.bank_size, self.balanced_bias, self.num_assigned_tokens, self.training)
        
            

        return mask, self.num_assigned_tokens

    def reset_num_assigned_tokens(self):
        self.num_assigned_tokens.zero_()

class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss, 
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        ctx.shape = loss.numel()
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(ctx.shape, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss

class TopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, k: int = 16, bank_size: int = 64):
        x = input.view(-1, input.shape[-1]//bank_size, bank_size)
        _, topk_indices = x.abs().topk(k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, topk_indices, 1)
        mask = mask.view_as(input)

        ctx.save_for_backward(mask)

        return input*mask

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        grad_output = grad_output*mask
       
        return grad_output, None, None

class BalancedTopkMLP(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        is_expert: bool = False,
        input_size: Optional[int] = None,
        ffn_hidden_size: int = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.layer_number = None

        self.input_size = input_size if input_size != None else self.config.hidden_size

        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        if ffn_hidden_size is None:
            if is_expert:
                raise ValueError("MoE MLP requires `ffn_hidden_size`, but it was not provided.")
            warnings.warn(
                "MLP requires ffn_hidden_size, but it was not provided. Using \
                    config.ffn_hidden_size by default.",
                DeprecationWarning,
                stacklevel=2,
            )
            ffn_hidden_size = self.config.ffn_hidden_size

        # If this is a gated linear unit we double the output width
        # see https://arxiv.org/pdf/2002.05202.pdf
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc1',
            tp_group=tp_group,
        )

        self.activation_func = self.config.activation_func

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc2',
            tp_group=tp_group,
        )
        assert self.config.act_sparse_training, "predictor is only supported in activation sparse training"
        self.predictor = build_module(
            submodules.predictor,
            self.config,
            input_size = self.config.hidden_size,
            ffn_hidden_size = self.config.act_sparse_predictor_hidden_size,
            # submodules.predictor.submodules,
            act_sparse_training=self.config.act_sparse_training,
        )

        self.topk_module = BalancedTopkModule(
            config,
            self.config.ffn_hidden_size, 
            self.config.act_sparse_topk, 
            self.config.act_sparse_bank_size,
            tp_group=tp_group,
        )

        self.train_predictor_independently = False

    def set_train_predictor_independently(self, train_predictor_independently: bool):
        self.train_predictor_independently = train_predictor_independently

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MLP."""
        self.layer_number = layer_number

    def forward(self, hidden_states, per_token_scale=None):
        """Perform the forward pass through the MLP block."""
        # [s, b, 4 * h/p]
        nvtx_range_push(suffix="linear_fc1")
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        nvtx_range_pop(suffix="linear_fc1")
        if self.train_predictor_independently:
            y_1, _ = torch.chunk(intermediate_parallel, 2, -1)
        nvtx_range_push(suffix="activation")
        if self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                else:
                    raise ValueError("Only support fusion of swiglu with per_token_scale in MLP.")
            else:
                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(
                            intermediate_parallel, bias_parallel
                        )
                    else:
                        assert self.config.add_bias_linear is True
                        intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
                elif self.activation_func == F.silu and self.config.gated_linear_unit:
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                        self.config.cpu_offloading
                        and self.config.cpu_offloading_activations
                        and HAVE_TE,
                    )
                elif self.activation_func == identity and self.config.gated_linear_unit:
                    intermediate_parallel = bias_swiglu_without_silu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                        self.config.cpu_offloading
                        and self.config.cpu_offloading_activations
                        and HAVE_TE,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return self.config.activation_func(x[0]) * x[1]

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        nvtx_range_pop(suffix="activation")

        nvtx_range_push(suffix="predictor")
        if self.train_predictor_independently:
            pred_mask = torch.sigmoid(self.predictor(hidden_states.detach())[0])
            target = torch.sigmoid(y_1)
            predictor_loss = F.mse_loss(pred_mask, target.detach(), reduction='mean')
            save_to_aux_losses_tracker(
                "predictor_loss",
                predictor_loss,
                self.layer_number,
                self.config.num_layers,
            )
            intermediate_parallel = AddAuxiliaryLoss.apply(intermediate_parallel, predictor_loss)

            intermediate_parallel = intermediate_parallel * TopKFunction.apply(target, self.topk_module.topk, self.topk_module.bank_size)
        else: 
            pred_mask = torch.sigmoid(self.predictor(hidden_states)[0])
            topk_mask, *_ = self.topk_module(pred_mask)
            intermediate_parallel = intermediate_parallel * topk_mask

            # gate_proj, up_proj = torch.chunk(intermediate_parallel, 2, -1)
            # gate_proj_topk = self.topk_module(self.activation_func(gate_proj))[0]
            # intermediate_parallel = gate_proj_topk*up_proj
        nvtx_range_pop(suffix="predictor")

        # [s, b, h]
        nvtx_range_push(suffix="linear_fc2")
        output, output_bias = self.linear_fc2(intermediate_parallel)
        nvtx_range_pop(suffix="linear_fc2")

        if per_token_scale is not None:
            assert output_bias is None, "Bias is not supported with per_token_scale"

        return output, output_bias

    # pylint: disable=missing-function-docstring
    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        sharded_state_dict = {}
        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(f'{prefix}{name}.', sharded_offsets, metadata)
            if self.config.gated_linear_unit and name == 'linear_fc1':
                for k, v in sub_sd.items():
                    if k in (f'{prefix}{name}.weight', f'{prefix}{name}.bias'):
                        sub_sd[k] = apply_swiglu_sharded_factory(v, sharded_offsets)
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict

class MLP(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        is_expert: bool = False,
        input_size: Optional[int] = None,
        ffn_hidden_size: int = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        act_sparse_training: Optional[bool] = False,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.act_sparse_training = act_sparse_training

        self.input_size = input_size if input_size != None else self.config.hidden_size

        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        if ffn_hidden_size is None:
            if is_expert:
                raise ValueError("MoE MLP requires `ffn_hidden_size`, but it was not provided.")
            warnings.warn(
                "MLP requires ffn_hidden_size, but it was not provided. Using \
                    config.ffn_hidden_size by default.",
                DeprecationWarning,
                stacklevel=2,
            )
            ffn_hidden_size = self.config.ffn_hidden_size

        # If this is a gated linear unit we double the output width
        # see https://arxiv.org/pdf/2002.05202.pdf
        if self.config.gated_linear_unit and not self.act_sparse_training:
            ffn_hidden_size *= 2

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc1',
            tp_group=tp_group,
            **({'gather_output': False} if not self.act_sparse_training else {}),
        )

        self.activation_func = self.config.activation_func if not self.act_sparse_training else None

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.config.ffn_hidden_size if not self.act_sparse_training else ffn_hidden_size,
            self.config.hidden_size if not self.act_sparse_training else self.config.ffn_hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc2',
            tp_group=tp_group,
            **({'input_is_parallel': True} if not self.act_sparse_training else {'gather_output': False}),
        )

    def forward(self, hidden_states, per_token_scale=None):
        """Perform the forward pass through the MLP block."""
        # [s, b, 4 * h/p]
        nvtx_range_push(suffix="linear_fc1")
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        nvtx_range_pop(suffix="linear_fc1")

        nvtx_range_push(suffix="activation")
        if self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                else:
                    raise ValueError("Only support fusion of swiglu with per_token_scale in MLP.")
            else:
                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(
                            intermediate_parallel, bias_parallel
                        )
                    else:
                        assert self.config.add_bias_linear is True
                        intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
                elif self.activation_func == F.silu and self.config.gated_linear_unit:
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                        self.config.cpu_offloading
                        and self.config.cpu_offloading_activations
                        and HAVE_TE,
                    )
                elif self.activation_func is None:
                    intermediate_parallel = intermediate_parallel
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit and not self.act_sparse_training:

                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return self.config.activation_func(x[0]) * x[1]

                intermediate_parallel = glu(intermediate_parallel)
            elif self.act_sparse_training is None:
                intermediate_parallel = intermediate_parallel
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        nvtx_range_pop(suffix="activation")

        # [s, b, h]
        nvtx_range_push(suffix="linear_fc2")
        output, output_bias = self.linear_fc2(intermediate_parallel)
        nvtx_range_pop(suffix="linear_fc2")

        if per_token_scale is not None:
            assert output_bias is None, "Bias is not supported with per_token_scale"

        return output, output_bias

    # pylint: disable=missing-function-docstring
    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        sharded_state_dict = {}
        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(f'{prefix}{name}.', sharded_offsets, metadata)
            if self.config.gated_linear_unit and name == 'linear_fc1' and 'predictor' not in prefix:
                for k, v in sub_sd.items():
                    if k in (f'{prefix}{name}.weight', f'{prefix}{name}.bias'):
                        sub_sd[k] = apply_swiglu_sharded_factory(v, sharded_offsets)
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict


# pylint: disable=missing-function-docstring
def apply_swiglu_sharded_factory(original_sh_ten, sharded_offsets):
    # We must split the tensor into 2 parts, each sharded separately.
    # This requires a ShardedTensorFactory which `chunk`s during saving
    # and `cat`s during loading

    swiglu_shard_axis = 0
    prepend_axis_num = len(sharded_offsets)
    original_shape = original_sh_ten.local_shape
    original_numel = int(np.prod(original_shape))
    local_axis_size = original_shape[swiglu_shard_axis]
    assert (
        original_sh_ten.global_offset[swiglu_shard_axis + prepend_axis_num] % local_axis_size == 0
    )
    rank_offset = (
        original_sh_ten.global_offset[swiglu_shard_axis + prepend_axis_num] // local_axis_size
    )
    axis_frag = original_sh_ten.axis_fragmentations[swiglu_shard_axis + prepend_axis_num]

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        offset_w = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag * 2)
        offset_v = (swiglu_shard_axis + prepend_axis_num, rank_offset + axis_frag, axis_frag * 2)
        if flattened_range is None:
            tensor_w, tensor_v = torch.chunk(t, 2, dim=swiglu_shard_axis)
            return [
                ShardedTensor.from_rank_offsets(
                    key,
                    tensor_w,
                    *sharded_offsets,
                    offset_w,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                ),
                ShardedTensor.from_rank_offsets(
                    key,
                    tensor_v,
                    *sharded_offsets,
                    offset_v,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                ),
            ]
        else:
            # Here we need to map a slice `t` (`flattened_range` specifies slice start and stop)
            # of the *original* flattened tensor into slices `w` and `v` of chunked
            # and flattened tensor.
            # Example:
            # If original tensor has (16, 5) shape and flattened_range is `slice(8, 64)`,
            # then `t` has shape `(56,)` and we need to create 2 tensors:
            # w: first 32 elements of `t` with flattened_range slice(8, 40)
            # v: last 24 elements of `t` with flattened_range slice(0, 24)
            # Global offsets are the same as in the non-flattened case
            assert t.ndim == 1, (key, t.shape)
            non_flat_local_shape = (original_shape[0] // 2, *original_shape[1:])
            chunk_numel = original_numel // 2
            result = []
            if flattened_range.start < chunk_numel:
                # Non-empty `w` chunk
                tensor_w = t[: chunk_numel - flattened_range.start]
                flattened_range_w = slice(
                    flattened_range.start, min(chunk_numel, flattened_range.stop)
                )
                assert len(tensor_w) == flattened_range_w.stop - flattened_range_w.start
                result.append(
                    ShardedTensor.from_rank_offsets_flat(
                        key,
                        tensor_w,
                        non_flat_local_shape,
                        *sharded_offsets,
                        offset_w,
                        replica_id=replica_id,
                        prepend_axis_num=prepend_axis_num,
                        flattened_range=flattened_range_w,
                    )
                )
            if flattened_range.stop > chunk_numel:
                # Non-empty `v` chunk
                tensor_v = t[-(flattened_range.stop - chunk_numel) :]
                flattened_range_v = slice(
                    max(chunk_numel, flattened_range.start) - chunk_numel,
                    flattened_range.stop - chunk_numel,
                )
                assert len(tensor_v) == flattened_range_v.stop - flattened_range_v.start, (
                    len(tensor_v),
                    flattened_range_v,
                )

                result.append(
                    ShardedTensor.from_rank_offsets_flat(
                        key,
                        tensor_v,
                        non_flat_local_shape,
                        *sharded_offsets,
                        offset_v,
                        replica_id=replica_id,
                        prepend_axis_num=prepend_axis_num,
                        flattened_range=flattened_range_v,
                    )
                )
            assert sum(sh_ten.data.numel() for sh_ten in result) == t.numel(), (result, t.shape)
            return result

    def sh_ten_merge_fn(sub_state_dict):
        with torch.no_grad():
            return torch.cat(sub_state_dict)

    return ShardedTensorFactory(
        original_sh_ten.key,
        original_sh_ten.data,
        sh_ten_build_fn,
        sh_ten_merge_fn,
        original_sh_ten.replica_id,
        flattened_range=original_sh_ten.flattened_range,
    )
