#!/bin/bash

# 设置脚本在遇到错误时退出
set -e

# 显示执行的命令
set -x

# 定义参数
LOAD_PATH="/root/workspace/Megatron-LM/checkpoints/moe-0.6B-input2-topk-4x-torch/torch/iter_0018750"
SAVE_PATH="/root/workspace/Megatron-LM/checkpoints/moe-0.6B-input2-topk-4x-torch/torch/trfs_checkpoint_iter_iter_0018750"
TOKENIZER_NAME="/root/data/llama"
MAX_SHARD_SIZE="10GB"

# 获取脚本所在目录作为工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 执行checkpoint转换命令
python tools/checkpoint/checkpoint_reshaping_and_interoperability.py \
    --convert_checkpoint_from_megatron_to_transformers \
    --load_path "$LOAD_PATH" \
    --save_path "$SAVE_PATH" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --max_shard_size "$MAX_SHARD_SIZE" \
    --print-checkpoint-structure

echo "Checkpoint conversion completed successfully!" 