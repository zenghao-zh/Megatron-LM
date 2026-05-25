export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
EXPERIMENT_NAME=smollm-130m-btopk
# DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
CHECKPOINT_PATH=/root/data/megatron-models/checkpoints/$EXPERIMENT_NAME
# VOCAB_FILE=vocab.json
# MERGE_FILE=merges.txt
DATA_PATH="/root/data/smollm_corpus/merged_smollm_corpus"
TOKENIZER_MODEL=/root/data/cosmo2-tokenizer
WANDB_PATH=/root/workspace/Megatron-LM/wandb
TENSORBOARD_PATH=/root/workspace/Megatron-LM/tensorboard

MAX_TRAIN_SAMPLES=19200000
LR_WARMUP_SAMPLES=$(( 5000 * 128 ))


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 8192
    --max-position-embeddings 8192
    --num-layers 30
    --hidden-size 576
    --num-attention-heads 9
    --ffn-hidden-size 1536
    --init-method-std 0.006
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    # --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 3
    --no-masked-softmax-fusion
    --position-embedding-type rope
    --rotary-base 10000
    --attention-softmax-in-fp32
    --vocab-size 49152
    # --use-cpu-initialization
    # INT8 训练现在支持 TransformerEngine 层（默认）和 local 实现
    # 使用 TransformerEngine（默认）: INT8 通过 forward 包装实现
    # 使用 local: INT8 通过 tensor subclass 实现（可能更稳定）
    # --transformer-impl local
    # --no-persist-layer-norm
)

TRAINING_ARGS=(
    --seed 3407
    --micro-batch-size 16
    --global-batch-size 128
    --lr 3e-3
    --train-samples $MAX_TRAIN_SAMPLES
    --lr-warmup-samples $LR_WARMUP_SAMPLES
    --lr-decay-style constant
    # --lr-decay-multi-step 0.6 0.3 0.1
    --min-lr 1e-8
    --lr-warmup-init 1e-8
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --norm-epsilon 1e-5
    --clip-grad 1.0
    --bf16
    ## 激活稀疏训练参数
    --act-sparse-training
    --act-sparse-predictor-hidden-size 64
    --act-sparse-bank-size 64
    --act-sparse-topk 16
    --act-sparse-btopk-coeff 0.001
    --act-sparse-swiglu-without-silu
    # --act-sparse-affinity-bias 0.01
    # --act-sparse-affinity-bias-max 0.1
    # --act-sparse-affinity-start-step 0
    # --act-sparse-affinity-cluster-interval 200
    # --act-sparse-bias-threshold 0
    # --int8-mixed-precision-training
    # --int8-mp-verbose  # 打印每个层的INT8状态
    # --no-int8-mp-grad-input
)

MOE_ARGS=(
    # --num-experts 64
    # --moe-grouped-gemm
    # --moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, none. Default is aux_loss.
    # --moe-router-topk 8
    # --moe-router-dtype fp32
    # --moe-aux-loss-coeff 5e-3
    # --moe-token-dispatcher-type alltoall
    # --moe-ffn-hidden-size 40
    # --moe-shared-expert-intermediate-size 0 # shared-experts 2
    #--moe-expert-capacity-factor 1.2
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
   # --expert-model-parallel-size 1
    # --pipeline-model-parallel-size 2
    --use-distributed-optimizer
    --sequence-parallel
    # --use-torch-fsdp2
    # --torch-fsdp2-no-reshard-after-forward
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_MODEL
    --data-path $DATA_PATH
    --split 1,0,0
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000
    --eval-interval 10000
    --eval-iters 1
    --save $CHECKPOINT_PATH
    # --load $CHECKPOINT_PATH
    --wandb-project megatron-training-smollm-135M
    --wandb-exp-name $EXPERIMENT_NAME
    --wandb-save-dir $WANDB_PATH
    --log-timers-to-tensorboard
    --tensorboard-dir $TENSORBOARD_PATH
    # --wandb-project benchmark_training
    # --wandb-exp-name moe8x2-7B
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH
)


# TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} 