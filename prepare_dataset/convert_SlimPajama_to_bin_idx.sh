#!/bin/bash

# 设置基础路径
INPUT_DIR="/ssd_1234/haozeng/data/json_data"
OUTPUT_DIR="/ssd_1234/haozeng/data/slimpajama"
TOKENIZER_MODEL="/ssd_1234/haozeng/data/llama/tokenizer.model"

# 定义要处理的 chunk 数量（根据实际情况修改）
NUM_CHUNKS=10  # 假设有 chunk1 到 chunk8

for i in $(seq 7 10)
do
    INPUT_FILE="${INPUT_DIR}/SlimPajama_train_chunk${i}.json"
    OUTPUT_PREFIX="${OUTPUT_DIR}/slimpajama_train_chunk${i}"
    echo "Start to process chunk $i"
    python tools/preprocess_data.py \
        --input "$INPUT_FILE" \
        --output-prefix "$OUTPUT_PREFIX" \
        --tokenizer-type Llama2Tokenizer \
        --tokenizer-model "$TOKENIZER_MODEL" \
        --json-keys text \
        --workers 32 \
        --append-eod
    
    echo "Completed processing chunk $i"
done

echo "All chunks processed successfully!"

python tools/merge_datasets.py \
    --input "/ssd_1234/haozeng/data/slimpajama" \
    --output-prefix "/ssd_1234/haozeng/data/slimpajama/merged_slimpajama"