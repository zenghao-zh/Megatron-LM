# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import importlib.util
import json
import os
import re
import sys
import types

# Add the tools directory to sys.path to import hf_moe
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from packaging import version

from transformers import AutoTokenizer, GPT2Config, LlamaForCausalLM, LlamaConfig
from hf_moe import MoEConfig, MoEForCausalLM
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils import check_torch_load_is_safe
from safetensors.torch import save_file as safe_save_file


def add_checkpointing_args(parser):
    parser.add_argument("--megatron-path", type=str, default=None, help="Base directory of Megatron repository")
    parser.add_argument(
        "--convert_checkpoint_from_megatron_to_transformers",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a Transformers checkpoint. "
            "If False, convert a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the converted checkpoint.",
    )
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    return parser


def add_megatron_checkpoint_args(parser):
    parser.add_argument(
        "--target_tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The tensor model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_pipeline_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The pipeline model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_data_parallel_size",
        type=int,
        default=1,
        help=(
            "The data parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_params_dtype",
        type=str,
        default="fp32",
        help=(
            "The dtype of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--make_vocab_size_divisible_by",
        type=int,
        default=128,
        help=(
            "Pad the vocab size to be divisible by this value. "
            "This is added for computational efficiency reasons. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--use_distributed_optimizer",
        action="store_true",
        help=(
            "If True, use the distributed optimizer. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    return parser


def add_transformers_checkpoint_args(parser):
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help=(
            "The name of the pre-trained tokenizer to save. "
            "If not None, the tokenizer will be saved. "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="10GB",
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )

    return parser


# The simple map of names for "automated" rules.
megatron_to_transformers = {
    # "attention.dense": ".attn.c_proj.",
    # "self_attention.dense": ".attn.c_proj.",
    # "mlp.dense_h_to_4h": ".mlp.c_fc.",
    # "mlp.dense_4h_to_h": ".mlp.c_proj.",
}
transformers_to_megatron = {v[1:-1]: k for k, v in megatron_to_transformers.items()}

tensor_parallel_params = [
    # # megatron-lm layers to merge across tp ranks
    # "self_attention.query_key_value.weight",
    # "self_attention.query_key_value.bias",
    # "self_attention.dense.weight",
    # "mlp.dense_h_to_4h.weight",
    # "mlp.dense_h_to_4h.bias",
    # "mlp.dense_4h_to_h.weight",
    # # deprecated
    # "attention.query_key_value.weight",
    # "attention.query_key_value.bias",
    # "attention.dense.weight",
    # # transformers layers to split across tp ranks
    # "attn.c_attn.weight",
    # "attn.c_attn.bias",
    # "attn.c_proj.weight",
    # "mlp.c_fc.weight",
    # "mlp.c_fc.bias",
    # "mlp.c_proj.weight",
]


def recursive_print(name, val, spaces=0):
    """
    Recursively print the structure of a checkpoint. This function is taken from `convert_megatron_gpt2_checkpoint.py`

    Args:
        name (str): the name of the current tensor parameter
        val (Tuple(int)): the shape of the current tensor parameter
        spaces (int): the number of spaces to print before the output for a nested structure
    """
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def megatron_to_transformers_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :] for compatibility with later versions
    of NVIDIA Megatron-LM. The inverse operation is performed inside Megatron-LM to read checkpoints:
    https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209 If param is the weight tensor of the
    self-attention block, the returned tensor will have to be transposed one more time to be read by HuggingFace GPT2.
    This function is taken from `convert_megatron_gpt2_checkpoint.py`

    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def transformers_to_megatron_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to the one compatible with respective NVIDIA Megatron-LM checkpoint versions. Input
    is [num_splits * num_heads * hidden_size, :] and output is [num_heads * hidden_size * num_splits, :] for version
    1.0 and [num_heads * num_splits * hidden_size, :] for version 2.0 and later. If param is the weight tensor of the
    self-attention block, the param needs to be already transposed before calling this function.

    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    # Input is [num_splits * num_heads * hidden_size, :]
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def merge_transformers_sharded_states(path, num_checkpoints):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.

    Args:
        path (str): the path to the sharded checkpoints
        num_checkpoints (int): the number of checkpoints to merge
    """
    state_dict = {}
    for i in range(1, num_checkpoints + 1):
        checkpoint_path = os.path.join(path, f"pytorch_model-{i:05d}-of-{num_checkpoints:05d}.bin")
        check_torch_load_is_safe()
        current_chunk = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict.update(current_chunk)
    return state_dict


def get_megatron_sharded_states(args, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.

    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        sub_dir_name = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}"
        for checkpoint_name in ["model_optim_rng.pt", "model_rng.pt"]:
            checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
            if os.path.isfile(checkpoint_path):
                break
        check_torch_load_is_safe()
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        tp_state_dicts.append(state_dict)
    return tp_state_dicts

def get_element_from_dict_by_layer_re(d, layer_re):
    """
    Get element from dictionary by layer regex.
    """
    result = {}
    for key, val in d.items():
        if layer_re.match(key):
            result[key] = val
    return result

def get_element_from_dict_by_path(d, path):
    """
    Get element from dictionary by path. If element is not present, recursively add empty dictionaries.

    Args:
        d (dict): the dictionary to get the element from
        path (list): the path to the element which is delimited by "."
    """
    path = [path]
    for k in path:
        if k not in d:
            d[k] = {}
        d = d[k]
    return d

def is_weight_with_number(weight_or_bias):
    """判断是否是 weight + 数字"""
    return weight_or_bias.startswith("weight") and len(weight_or_bias) > 6 and weight_or_bias[6:].isdigit()

def extract_weight_number(weight_or_bias):
    """提取 weight 后面的数字"""
    if is_weight_with_number(weight_or_bias):
        return int(weight_or_bias[6:])  # "weight" 长度是6，所以从索引6开始
    return None


def convert_checkpoint_from_megatron_to_transformers(args):
    """
    Convert NVIDIA Megatron-LM checkpoint to HuggingFace Transformers checkpoint. This handles Megatron checkpoints
    with different tensor parallelism and pipeline parallelism sizes. It saves the converted checkpoint into shards
    using HuggingFace Transformers checkpoint sharding functionality. This greatly extends the functionality of
    `convert_megatron_gpt2_checkpoint.py`

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    # Load Megatron-LM checkpoint arguments from the state dict
    sub_dirs = os.listdir(args.load_path)
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000"]
    rank0_checkpoint_path = None
    for sub_dir in possible_sub_dirs:
        if sub_dir in sub_dirs:
            rank0_checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir))[0]
            rank0_checkpoint_path = os.path.join(args.load_path, sub_dir, rank0_checkpoint_name)
            break
    if rank0_checkpoint_path is None:
        rank0_checkpoint_path = os.path.join(args.load_path, "common.pt")
    print(f"Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}")
    check_torch_load_is_safe()
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu", weights_only=False)
    megatron_args = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint instead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    # Create Transformers GPT2 config from Megatron-LM arguments
    if megatron_args is not None:
        if megatron_args.swiglu:
            activation_function = "silu"
        elif megatron_args.bias_gelu_fusion:
            activation_function = "gelu_fast"
        elif megatron_args.openai_gelu:
            activation_function = "gelu_new"
        else:
            activation_function = "gelu"
    else:
        # in the very early days this used to be "gelu_new"
        activation_function = "gelu_new"
    vocab_size = (
        megatron_args.padded_vocab_size
        if getattr(megatron_args, "orig_vocab_size", None) is None
        else megatron_args.orig_vocab_size
    )
    print(vocab_size)

    config = MoEConfig(
        vocab_size=vocab_size,
        hidden_size=megatron_args.hidden_size,
        intermediate_size=megatron_args.ffn_hidden_size,
        num_hidden_layers=megatron_args.num_layers,
        num_attention_heads=megatron_args.num_attention_heads,
        num_key_value_heads=megatron_args.num_query_groups,
        hidden_act=activation_function,
        max_position_embeddings=megatron_args.max_position_embeddings,
        initializer_range=megatron_args.init_method_std,
        rms_norm_eps=megatron_args.norm_epsilon,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=megatron_args.rotary_base,
        rope_scaling=None,
        attention_bias=megatron_args.add_qkv_bias,
        attention_dropout=megatron_args.attention_dropout,
        mlp_bias=megatron_args.add_bias_linear,
        head_dim=None,
        # moe related
        moe_layer_class="MoELayer",
        moe_expert_class="MLP" if not megatron_args.act_sparse_training else "BalancedTopkMLP",
        moe_intermediate_size=megatron_args.moe_ffn_hidden_size,
        n_shared_experts=megatron_args.moe_shared_expert_intermediate_size//megatron_args.moe_ffn_hidden_size,
        n_routed_experts=megatron_args.num_experts,
        moe_layer_freq=megatron_args.moe_layer_freq,
        aux_loss_alpha=megatron_args.moe_aux_loss_coeff,
        num_experts_per_tok=megatron_args.moe_router_topk,
        scoring_func=megatron_args.moe_router_score_function,
        seq_aux=True,
        norm_topk_prob=False,
        first_k_dense_replace=0,
        # balanced topk mlp related
        predictor_hidden_size=megatron_args.act_sparse_predictor_hidden_size,
        predictor_bank_size=megatron_args.act_sparse_bank_size,
        predictor_topk=megatron_args.act_sparse_topk,
        predictor_btopk_coeff=megatron_args.act_sparse_btopk_coeff,
    )

    output_state_dict = {}

    checkpoint_version = state_dict.get("checkpoint_version", 0.0)
    tp_size = megatron_args.tensor_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size
    dtype = torch.bfloat16
    # The regex to extract layer names.
    # layer_re = re.compile(r"decoder\.layers\.(\d+)\.(.+)\.(weight)(\d*)$")
    layer_re = re.compile(r"decoder\.layers\.(\d+)\.(.+)\.(weight\d*|layer_norm_weight\d*|bias\d*|layer_norm_bias\d*|balanced_bias|num_assigned_tokens)$")

    # Convert.
    print("Converting")

    # Embeddings
    print("Converting embeddings")
    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, 0)

    # Convert and store the position embeddings.
    # position_embeddings = get_element_from_dict_by_path(
    #     tp_state_dicts[0], "model.language_model.embedding.position_embeddings.weight"
    # )
    # output_state_dict["model.embed_tokens.weight"] = position_embeddings.to(dtype)

    # Convert and store the word embeddings.
    word_embeddings = torch.cat(
        [
            get_element_from_dict_by_path(
                tp_state_dicts[tp_rank]["model"], "embedding.word_embeddings.weight"
            )
            for tp_rank in range(tp_size)
        ],
        dim=0,
    )
    word_embeddings = word_embeddings[:vocab_size].to(dtype)
    output_state_dict["model.embed_tokens.weight"] = word_embeddings

    # Transformer Layers
    print("Converting transformer layers")
    num_layers = config.num_hidden_layers // pp_size

    for pp_rank in range(pp_size):
        if pp_size > 0:
            print(f"Converting pipeline parallel rank {pp_rank}")
            tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, pp_rank)

        # The transformer.
        path = (
            "model.language_model.transformer"
            if "transformer" in get_element_from_dict_by_path(tp_state_dicts[0], "model.language_model").keys()
            else "model.language_model.encoder"
        )

        for key, val in get_element_from_dict_by_layer_re(tp_state_dicts[0]["model"], layer_re).items():
            print("processing key: ", key)
            m = layer_re.match(key)
            # Stop if that's not a layer
            assert m is not None, "key: " + key + " is not a correct layer name"

            # The index of the layer.
            layer_idx = int(m.group(1)) + pp_rank * num_layers
            # The name of the operation.
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)

            layer_name = f"model.layers.{layer_idx}"

            if op_name + "." + weight_or_bias not in tensor_parallel_params:
                params = val.to(dtype)
            else:
                ## TODO: implement tensor parallel for moe
                dim = 1 if op_name in ["self_attention.dense", "mlp.dense_4h_to_h", "attention.dense"] else 0
                params = torch.cat(
                    [val]
                    + [
                        get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                        for tp_rank in range(1, tp_size)
                    ],
                    dim=dim,
                ).to(dtype)
            
            # process layernorms
            if "layernorm" in op_name or "layer_norm" in weight_or_bias:
                if "linear_qkv" in op_name:
                    output_state_dict[layer_name + "." + "input_layernorm.weight"] = params
                elif "pre_mlp_layernorm" in op_name:
                    output_state_dict[layer_name + "." + "post_attention_layernorm.weight"] = params
                else:
                    raise ValueError(f"Unknown layernorm in {op_name}")
            
            # process attention
            elif "self_attention" in op_name and "linear_qkv" in op_name and weight_or_bias == "weight":
                q_shape = config.head_dim * config.num_attention_heads
                k_shape = config.head_dim * config.num_key_value_heads
                v_shape = config.head_dim * config.num_key_value_heads
                # val shape: [qkv_out, in] where qkv_out = q_shape + k_shape + v_shape
                # Split val into q, k, v along the output dimension (dim=0)
                q_weight, k_weight, v_weight = torch.split(params, [q_shape, k_shape, v_shape], dim=0)
                output_state_dict[layer_name + ".self_attn.q_proj.weight"] = q_weight
                output_state_dict[layer_name + ".self_attn.k_proj.weight"] = k_weight
                output_state_dict[layer_name + ".self_attn.v_proj.weight"] = v_weight

            elif "self_attention" in op_name and "linear_qkv" in op_name and weight_or_bias == "bias":
                # TODO: implement for bias
                pass

            elif "self_attention" in op_name and "linear_proj" in op_name and weight_or_bias == "weight":
                output_state_dict[layer_name + ".self_attn.o_proj.weight"] = params
            
            elif "self_attention" in op_name and "linear_proj" in op_name and weight_or_bias == "bias":
                # TODO: implement for bias
                pass
            # process route
            elif "router" in op_name and weight_or_bias == "weight":
                output_state_dict[layer_name + ".mlp.gate.weight"] = params
            # process mlp/moe
            elif "mlp.experts" in op_name and "linear_fc1" in op_name and is_weight_with_number(weight_or_bias):
                expert_idx = extract_weight_number(weight_or_bias)
                if "predictors" in op_name:
                    output_state_dict[layer_name + f".mlp.experts.{expert_idx}.predictor.fc_1.weight"] = params
                else:
                    # linear_fc1 weight: [2 * hidden_size, expert_in]
                    # split into two: first half for gate, second half for up
                    fc1_weight = params
                    hidden_size = fc1_weight.shape[0] // 2
                    gate_weight = fc1_weight[:hidden_size, :]
                    up_weight = fc1_weight[hidden_size:, :]
                    output_state_dict[layer_name + f".mlp.experts.{expert_idx}.gate_proj.weight"] = gate_weight
                    output_state_dict[layer_name + f".mlp.experts.{expert_idx}.up_proj.weight"] = up_weight
            elif "mlp.experts" in op_name and "linear_fc1" in op_name and weight_or_bias == "bias":
                pass
            elif "mlp.experts" in op_name and "linear_fc2" in op_name and is_weight_with_number(weight_or_bias):
                expert_idx = extract_weight_number(weight_or_bias)
                if "predictors" in op_name:
                    output_state_dict[layer_name + f".mlp.experts.{expert_idx}.predictor.fc_2.weight"] = params
                else:
                    output_state_dict[layer_name + f".mlp.experts.{expert_idx}.down_proj.weight"] = params
            elif "mlp.experts" in op_name and "linear_fc2" in op_name and weight_or_bias == "bias":
                pass
            elif "mlp.experts" in op_name and ("balanced_bias" in weight_or_bias or "num_assigned_tokens" in weight_or_bias):
                for expert_idx in range(config.n_routed_experts):
                    output_state_dict[layer_name + f".mlp.experts.{expert_idx}.topk_model.{weight_or_bias}"] = params[expert_idx]
            elif "mlp.shared_experts" in op_name  and "linear_fc1" in op_name and weight_or_bias=='weight':
                fc1_weight = params
                hidden_size = fc1_weight.shape[0] // 2
                gate_weight = fc1_weight[:hidden_size, :]
                up_weight = fc1_weight[hidden_size:, :]
                output_state_dict[layer_name + ".mlp.shared_experts.gate_proj.weight"] = gate_weight
                output_state_dict[layer_name + ".mlp.shared_experts.up_proj.weight"] = up_weight
            elif "mlp.shared_experts" in op_name  and "linear_fc1" in op_name and weight_or_bias=='bias':
                pass
            elif "mlp.shared_experts" in op_name  and "linear_fc2" in op_name and weight_or_bias=='weight':
                output_state_dict[layer_name + ".mlp.shared_experts.down_proj.weight"] = params
            elif "mlp.shared_experts" in op_name  and "linear_fc2" in op_name and weight_or_bias=='bias':
                pass
            else:
                raise ValueError(f"Unknown param/layer in {op_name}")

    # process final layernorm
    if config.num_hidden_layers != (layer_idx + 1):
        raise ValueError(f"Expected {config.num_hidden_layers} layers but found {layer_idx + 1}")

    # The final layernorm.
    print("Converting final layernorm")
    output_state_dict["model.norm.weight"] = tp_state_dicts[0]["model"]["decoder.final_layernorm.weight"].to(dtype)

    # For LM head, transformers' wants the matrix to weight embeddings.
    print("Converting LM head")
    lm_head_weight = torch.cat(
        [
            tp_state_dicts[tp_rank]["model"]["output_layer.weight"]
            for tp_rank in range(tp_size)
        ],
        dim=0,
    )
    output_state_dict["lm_head.weight"] = lm_head_weight.to(dtype)

    # It should be done!
    print("Conversion from Megatron-LM to Transformers is done!")

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # Add tokenizer class info to config
    # see https://github.com/huggingface/transformers/issues/13906)

    if args.tokenizer_name is None:
        tokenizer_name = "openai-community/gpt2"
    else:
        tokenizer_name = args.tokenizer_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer_class = type(tokenizer).__name__
    config.tokenizer_class = tokenizer_class

    # Store the config to file.
    print("Saving config")
    config.architectures = ["MoEForCausalLM"]
    config.save_pretrained(args.save_path)

    # Save tokenizer based on args
    if args.tokenizer_name is not None:
        print(f"Adding {tokenizer_class} tokenizer files")
        tokenizer.save_pretrained(args.save_path)

    # Store the state_dict to file.
    max_shard_size = int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    state_dict_split = split_torch_state_dict_into_shards(output_state_dict, max_shard_size=max_shard_size)
    shard = index = None
    for filename, tensors in state_dict_split.filename_to_tensors.items():
        shard = {tensor: output_state_dict[tensor] for tensor in tensors}
        safe_save_file(
                 shard,
                 os.path.join(args.save_path, filename),
                 metadata={"format": "pt"},
             )
        print(f"Model weights saved in {os.path.join(args.save_path, filename)}")

    # Save the model
    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        with open(os.path.join(args.save_path, "model.safetensors.index.json"), "w") as f:
            f.write(json.dumps(index, indent=2))
        print(
            f"The model is bigger than the maximum size per checkpoint ({args.max_shard_size}) and is going to be "
            f"split in {len(shard)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {os.path.join(args.save_path, 'model.safetensors.index.json')}."
        )

    # if index is None:
    #     print(f"Model weights saved in {os.path.join(args.save_path, WEIGHTS_NAME)}")
    # else:
    #     save_index_file = os.path.join(args.save_path, WEIGHTS_INDEX_NAME)
    #     # Save the index as well
    #     with open(save_index_file, "w", encoding="utf-8") as f:
    #         content = json.dumps(index, indent=2, sort_keys=True) + "\n"
    #         f.write(content)
    #     print(
    #         f"The model is bigger than the maximum size per checkpoint ({args.max_shard_size}) and is going to be "
    #         f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
    #         f"index located at {save_index_file}."
    #     )


def convert_checkpoint_from_transformers_to_megatron(args):
    """
    Convert a checkpoint from HuggingFace Transformers to Megatron-LM. This allows converted checkpoints with variable
    tensor parallelism and pipeline parallelism sizes. It takes as input a checkpoint from HuggingFace Transformers
    which can have multiple shards.

    Args:
        args (argparse.Namespace): the arguments to the script

    """
    os.makedirs(args.save_path, exist_ok=True)
    # Search in directory above this
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    megatron_exists = importlib.util.find_spec("megatron") is not None
    if megatron_exists:
        from megatron.core import package_info

        if version.parse(package_info.__version__) >= version.parse("0.6.0"):
            from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
        else:
            from megatron.tokenizer.tokenizer import _vocab_size_with_padding

    else:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        exit(1)

    # load the transformers model state dict and config
    sub_dirs = [x for x in os.listdir(args.load_path) if x.startswith("pytorch_model")]
    if len(sub_dirs) == 1:
        checkpoint_name = "pytorch_model.bin"
        check_torch_load_is_safe()
        state_dict = torch.load(os.path.join(args.load_path, checkpoint_name), map_location="cpu", weights_only=False)
    else:
        num_checkpoints = len(sub_dirs) - 1
        state_dict = merge_transformers_sharded_states(args.load_path, num_checkpoints)

    config = GPT2Config.from_pretrained(args.load_path)

    # Saving the tracker file
    tracker_filepath = os.path.join(args.save_path, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as f:
        f.write("release")

    # create `release` dir in args.load_path
    release_dir = os.path.join(args.save_path, "release")
    os.makedirs(release_dir, exist_ok=True)

    # megatron args
    megatron_args = {
        "orig_vocab_size": config.vocab_size,
        "max_position_embeddings": config.n_positions,
        "hidden_size": config.n_embd,
        "num_layers": config.n_layer,
        "num_attention_heads": config.n_head,
        "ffn_hidden_size": config.n_inner,
        "tensor_model_parallel_size": args.target_tensor_model_parallel_size,
        "pipeline_model_parallel_size": args.target_pipeline_model_parallel_size,
        "data_parallel_size": args.target_data_parallel_size,
        "make_vocab_size_divisible_by": args.make_vocab_size_divisible_by,
        "rank": 0,
        "tokenizer_type": "GPT2BPETokenizer",
    }

    if config.activation_function == "gelu":
        megatron_args["bias_gelu_fusion"] = False
        megatron_args["openai_gelu"] = False
    elif config.activation_function == "gelu_fast":
        megatron_args["bias_gelu_fusion"] = True
        megatron_args["openai_gelu"] = False
    elif config.activation_function == "gelu_new":
        megatron_args["bias_gelu_fusion"] = False
        megatron_args["openai_gelu"] = True

    margs = types.SimpleNamespace()
    for k, v in megatron_args.items():
        setattr(margs, k, v)

    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    setattr(margs, "params_dtype", dtype)

    # save dummy optim state dict
    dummy_optim_state_dict = {}
    dummy_optim_state_dict["optimizer"] = {
        "step": 0,
        "param_groups": [
            {
                "lr": 0.0,
                "beta1": 0.0,
                "beta2": 0.0,
                "eps": 0.0,
                "weight_decay": 0.0,
                "correct_bias": False,
                "params": [],
            }
        ],
    }
    if args.use_distributed_optimizer:
        for i in range(args.target_pipeline_model_parallel_size):
            for j in range(args.target_tensor_model_parallel_size):
                for k in range(args.target_data_parallel_size):
                    if args.target_pipeline_model_parallel_size == 1:
                        checkpoint_dir = f"mp_rank_{j:02d}_{k:03d}"
                    else:
                        checkpoint_dir = f"mp_rank_{j:02d}_{i:03d}_{k:03d}"
                    checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(
                        dummy_optim_state_dict,
                        os.path.join(checkpoint_dir, "optim.pt"),
                    )

    # Convert.
    print("Converting")
    output_state_dict = []
    for i in range(args.target_tensor_model_parallel_size):
        output_state_dict.append({})

    # Embedding layer
    print("converting embedding layer")
    pos_embedding = state_dict["transformer.wpe.weight"].to(dtype)
    word_embedding = state_dict["transformer.wte.weight"].to(dtype)
    orig_vocab_size = config.vocab_size
    padded_vocab_size = _vocab_size_with_padding(orig_vocab_size, margs)
    setattr(margs, "padded_vocab_size", padded_vocab_size)
    # Cut out extra padding we don't need
    if orig_vocab_size > padded_vocab_size:
        full_word_embed = word_embedding[0:padded_vocab_size, :]
    # Expanding embedding to larger size by replicating final entry
    elif orig_vocab_size < padded_vocab_size:
        padding_size = padded_vocab_size - orig_vocab_size
        full_word_embed = torch.cat((word_embedding, word_embedding[-1].unsqueeze(0).expand(padding_size, -1)))
    # Same size!
    else:
        full_word_embed = word_embedding

    # Split into new tensor model parallel sizes
    out_word_embed = torch.chunk(full_word_embed, args.target_tensor_model_parallel_size, dim=0)
    for i in range(args.target_tensor_model_parallel_size):
        pos_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.language_model.embedding.position_embeddings"
        )
        pos_emb_dict["weight"] = pos_embedding

        word_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.language_model.embedding.word_embeddings"
        )
        word_emb_dict["weight"] = out_word_embed[i].clone()

    # Transformer layers
    print("converting transformer layers")
    if config.num_attention_heads % args.target_tensor_model_parallel_size != 0:
        raise ValueError(
            f"Number of attention heads ({config.num_attention_heads}) must be divisible by number of tensor parallelism"
            f" ({args.target_tensor_model_parallel_size})"
        )

    if config.num_hidden_layers % args.target_pipeline_model_parallel_size != 0:
        raise ValueError(
            f"Number of layers ({config.num_hidden_layers}) must be divisible by number of pipeline parallelism"
            f" ({args.target_pipeline_model_parallel_size})"
        )

    num_layers = config.num_hidden_layers // args.target_pipeline_model_parallel_size

    layer_re = re.compile(r"transformer.h\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    # The number of heads.
    heads = config.n_head
    # The hidden_size per head.
    hidden_size_per_head = config.n_embd // config.n_head
    for pp_rank in range(args.target_pipeline_model_parallel_size):
        layer_offset = pp_rank * num_layers
        if pp_rank > 0:
            output_state_dict = []
            for i in range(args.target_tensor_model_parallel_size):
                output_state_dict.append({})

        for layer in range(num_layers):
            pp_layer_id = layer + layer_offset
            layers_to_copy = [
                layer_name
                for layer_name in state_dict.keys()
                if layer_name.startswith(f"transformer.h.{pp_layer_id}.")
            ]

            for layer_name in layers_to_copy:
                m = layer_re.match(layer_name)
                # Stop if that's not a layer
                if m is None:
                    break

                # The index of the layer.
                _ = int(m.group(1))
                # The name of the operation.
                op_name = m.group(2)
                # Is it a weight or a bias?
                weight_or_bias = m.group(3)

                params = state_dict[layer_name].to(dtype)
                # handle layernorm
                if op_name.startswith("ln"):
                    out_name = "input_layernorm" if op_name.endswith("1") else "post_attention_layernorm"
                    layer_name = f"layers.{layer}.{out_name}.{weight_or_bias}"

                # handle attention K, V, Q weights
                elif op_name.startswith("attn.c_attn") and weight_or_bias == "weight":
                    # transformers stores D X (3*D) but Megatron-LM expects (3*D) X D.
                    params = params.transpose(0, 1).contiguous()

                    params = transformers_to_megatron_fix_query_key_value_ordering(
                        params,
                        3.0,
                        3,
                        heads,
                        hidden_size_per_head,
                    )
                    layer_name = f"layers.{layer}.self_attention.query_key_value.{weight_or_bias}"

                # handle attention K, V, Q bias
                elif op_name.startswith("attn.c_attn") and weight_or_bias == "bias":
                    params = transformers_to_megatron_fix_query_key_value_ordering(
                        params,
                        3.0,
                        3,
                        heads,
                        hidden_size_per_head,
                    )
                    layer_name = f"layers.{layer}.self_attention.query_key_value.{weight_or_bias}"

                # handle attention and mlp weights
                elif weight_or_bias == "weight":
                    out_name = transformers_to_megatron.get(op_name, None)
                    if out_name is None:
                        continue
                    params = params.transpose(0, 1)
                    layer_name = f"layers.{layer}.{out_name}.{weight_or_bias}"

                # handle attention and mlp bias
                elif weight_or_bias == "bias":
                    out_name = transformers_to_megatron.get(op_name, None)
                    if out_name is None:
                        continue
                    layer_name = f"layers.{layer}.{out_name}.{weight_or_bias}"

                # skip
                else:
                    continue

                if op_name + "." + weight_or_bias in tensor_parallel_params:
                    dim = 1 if op_name in ["attn.c_proj", "mlp.c_proj"] else 0
                    params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=dim)

                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.encoder")
                    params_dict[layer_name] = (
                        params[i].clone() if (op_name + "." + weight_or_bias in tensor_parallel_params) else params
                    )

        if pp_rank == args.target_pipeline_model_parallel_size - 1:
            # handle final layernorm
            for weight_or_bias in ["weight", "bias"]:
                params = state_dict[f"transformer.ln_f.{weight_or_bias}"].to(dtype)
                layer_name = f"final_layernorm.{weight_or_bias}"
                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.encoder")
                    params_dict[layer_name] = params

            # add the LM head
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.word_embeddings_for_head")
                params_dict["weight"] = out_word_embed[i].clone()

        # saving the state dict as per the tp_rank and pp_rank
        for tp_rank in range(args.target_tensor_model_parallel_size):
            output_state_dict[tp_rank]["checkpoint_version"] = 3.0
            output_state_dict[tp_rank]["args"] = margs
            checkpoint_dir = (
                f"mp_rank_{tp_rank:02d}"
                if args.target_pipeline_model_parallel_size == 1
                else f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
            )
            if args.use_distributed_optimizer:
                checkpoint_name = "model_rng.pt"
            else:
                checkpoint_name = "model_optim_rng.pt"
                output_state_dict[tp_rank]["optimizer"] = dummy_optim_state_dict["optimizer"]
            checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            if args.print_checkpoint_structure:
                print(
                    f"Checkpoint structure of model state dict shard belonging to TP rank {tp_rank} and PP rank"
                    f" {pp_rank}:"
                )
                recursive_print(None, output_state_dict[tp_rank])
            torch.save(output_state_dict[tp_rank], checkpoint_path)


def main():
    parser = argparse.ArgumentParser()
    parser = add_checkpointing_args(parser)
    parser = add_megatron_checkpoint_args(parser)
    parser = add_transformers_checkpoint_args(parser)
    args = parser.parse_args()
    if args.convert_checkpoint_from_megatron_to_transformers:
        convert_checkpoint_from_megatron_to_transformers(args)
    else:
        convert_checkpoint_from_transformers_to_megatron(args)


if __name__ == "__main__":
    main()