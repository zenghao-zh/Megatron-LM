#!/bin/bash

# 设置脚本在遇到错误时退出
set -e

# 显示执行的命令
set -x

# 定义参数
# EXPERIMENT_NAME=smollm-360m-int8-fw-mlp-4gpu
# LOAD_PATH="/root/data/megatron-models/checkpoints/${EXPERIMENT_NAME}/iter_0097656"
# SAVE_PATH="/root/data/megatron-models/checkpoints/${EXPERIMENT_NAME}/hf_smollm_iter_0097656_int8_oproj"
EXPERIMENT_NAME=smollm-360m-int8-fw-mlp-4gpu
LOAD_PATH="/root/data/megatron-models/checkpoints/${EXPERIMENT_NAME}/iter_0097656"
SAVE_PATH="/root/data/megatron-models/checkpoints/${EXPERIMENT_NAME}/hf_smollm_iter_0097656_int8_oproj"
TOKENIZER_NAME="/root/data/cosmo2-tokenizer"
MAX_SHARD_SIZE="10GB"

# 获取脚本所在目录作为工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 执行checkpoint转换命令 (torch_dist 分布式格式 -> HuggingFace SmolLM/LLaMA)
python tools/checkpoint/checkpoint_reshaping_and_interoperability.py \
    --convert_checkpoint_from_megatron_to_transformers \
    --load_path "$LOAD_PATH" \
    --save_path "$SAVE_PATH" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --max_shard_size "$MAX_SHARD_SIZE" \
    --print-checkpoint-structure

echo "Checkpoint conversion completed successfully!"
