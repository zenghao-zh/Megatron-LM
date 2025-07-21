export CUDA_DEVICE_MAX_CONNECTIONS=1


GPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
# DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
CHECKPOINT_PATH=/ssd_1234/haozeng/workspace/Megatron-LM/checkpoints/moe-0.6b-baseline-test
CHECKPOINT_PATH_TORCH=/ssd_1234/haozeng/workspace/Megatron-LM/checkpoints/moe-0.6b-baseline-test-torch
# VOCAB_FILE=vocab.json
# MERGE_FILE=merges.txt
DATA_PATH="0.693584 /ssd_1234/haozeng/data/slimpajama/merged_slimpajama 0.306416 /ssd_1234/haozeng/data/starcode/merged_starcode"
TOKENIZER_MODEL=/ssd_1234/haozeng/data/llama/tokenizer.model
WANDB_PATH=/ssd_1234/haozeng/workspace/Megatron-LM/wandb
TENSORBOARD_PATH=/ssd_1234/haozeng/workspace/Megatron-LM/tensorboard

MAX_TRAIN_SAMPLES=38400000
LR_WARMUP_SAMPLES=$(( 1000 * 2048 ))


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
    --num-layers 8
    --hidden-size 1024
    --num-attention-heads 8
    --init-method-std 0.006
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 2
    --no-masked-softmax-fusion
    --position-embedding-type rope
    --rotary-base 10000
    --attention-softmax-in-fp32
    # --use-cpu-initialization
)

MOE_ARGS=(
    --num-experts 32
    --moe-grouped-gemm
    --moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, none. Default is aux_loss.
    --moe-router-topk 2
    --moe-router-dtype fp32
    --moe-aux-loss-coeff 1e-2
    --moe-token-dispatcher-type alltoall
    --moe-ffn-hidden-size 512
    --moe-shared-expert-intermediate-size 1152 # shared-experts 2
    #--moe-expert-capacity-factor 1.2
)

TRAINING_ARGS=(
    --seed 3407
    --micro-batch-size 8
    --global-batch-size 2048
    --lr 3e-4
    --train-samples $MAX_TRAIN_SAMPLES
    --lr-warmup-samples $LR_WARMUP_SAMPLES
    --lr-decay-style constant
    # --lr-decay-multi-step 0.6 0.3 0.1
    --min-lr 1e-8
    --lr-warmup-init 1e-8
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --norm-epsilon 1e-6
    --clip-grad 1.0
    --bf16

    ## 激活稀疏训练参数
    --act-sparse-training
    --act-sparse-predictor-hidden-size 64
    --act-sparse-bank-size 64
    --act-sparse-topk 16
    --act-sparse-btopk-coeff 0.001
    # --act-sparse-swiglu-without-silu
)

MODEL_PARALLEL_ARGS=(
    # --tensor-model-parallel-size 1
    --expert-model-parallel-size 2
    --use-distributed-optimizer
    --sequence-parallel
    # --use-torch-fsdp2
    # --torch-fsdp2-no-reshard-after-forward
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model $TOKENIZER_MODEL
    --data-path $DATA_PATH
    --split 1,0,0
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 3000
    --eval-interval 3000
    --eval-iters 1
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    # --wandb-project megatron-training
    # --wandb-exp-name MOE-0.6B-btopk-4x
    # --wandb-save-dir $WANDB_PATH
    --log-timers-to-tensorboard
    --tensorboard-dir $TENSORBOARD_PATH
    --ckpt-convert-format torch
    --ckpt-convert-save $CHECKPOINT_PATH_TORCH
    # --ckpt-step 15000
    # --wandb-project benchmark_training
    # --wandb-exp-name moe8x2-7B
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH
)


# TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

