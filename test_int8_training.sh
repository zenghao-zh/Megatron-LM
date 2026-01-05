#!/bin/bash
# Test script for INT8 mixed-precision training

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6099
NNODES=1

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# Tiny model for quick testing
MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 64
    --max-position-embeddings 64
    --num-layers 2
    --hidden-size 64
    --num-attention-heads 2
    --ffn-hidden-size 128
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --vocab-size 256
)

TRAINING_ARGS=(
    --seed 42
    --micro-batch-size 1
    --global-batch-size 1
    --lr 1e-4
    --train-iters 3
    --lr-warmup-iters 1
    --lr-decay-style constant
    --min-lr 1e-6
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
    --no-gradient-accumulation-fusion
    # INT8 mixed-precision training
    --int8-mixed-precision-training
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
)

DATA_ARGS=(
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size 256
)

LOGGING_ARGS=(
    --log-interval 1
)

echo "=========================================="
echo "Testing INT8 Mixed-Precision Training"
echo "=========================================="

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}

echo "=========================================="
echo "Test completed!"
echo "=========================================="

