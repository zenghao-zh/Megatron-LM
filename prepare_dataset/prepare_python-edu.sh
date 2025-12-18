# 重新处理这个 JSON 文件
python tools/preprocess_data.py \
    --input /root/data/json_data/smollm_corpus_train_python-edu.json \
    --output-prefix /root/data/smollm_corpus/smollm_corpus_train_python-edu \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /root/data/cosmo2-tokenizer \
    --json-keys text \
    --workers 32 \
    --append-eod