#!/bin/bash

# 获取所有符合条件的 JSON 文件路径
files=(/ssd_1234/haozeng/data/json_data/starcode_train_*.json)

# 获取总文件数
total=${#files[@]}

# 初始化计数器
i=0

# 遍历每个文件进行处理
for file in "${files[@]}"; do
    # 更新计数器
    i=$((i + 1))
    remaining=$((total - i))

    # 提取文件名（不含路径和后缀）
    filename=$(basename "$file" .json)

    # 输出当前处理信息
    echo "Processing: $filename (Remaining: $remaining) ..."

    # 执行预处理命令
    python tools/preprocess_data.py \
        --input $file \
        --output-prefix /ssd_1234/haozeng/data/starcode/$filename \
        --tokenizer-type Llama2Tokenizer \
        --tokenizer-model /ssd_1234/haozeng/data/llama/tokenizer.model \
        --json-keys content \
        --workers 32 \
        --append-eod
done

echo "All files have been processed. Starting to merge datasets"


python tools/merge_datasets.py \
    --input "/ssd_1234/haozeng/data/starcode" \
    --output-prefix "/ssd_1234/haozeng/data/starcode/merged_starcode"