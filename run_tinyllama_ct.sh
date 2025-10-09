export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
EXPERIMENT_NAME=tinyllama-120m
# DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
CHECKPOINT_PATH=/root/data/megatron-models/checkpoints/$EXPERIMENT_NAME
# VOCAB_FILE=vocab.json
# MERGE_FILE=merges.txt
DATA_PATH="0.693584 /root/data/slimpajama/merged_slimpajama 0.306416 /root/data/starcode/merged_starcode"
TOKENIZER_MODEL=/root/data/llama/tokenizer.model
WANDB_PATH=/root/workspace/Megatron-LM/wandb
TENSORBOARD_PATH=/root/workspace/Megatron-LM/tensorboard

MAX_TRAIN_SAMPLES=100000000
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
    --seq-length 2048
    --max-position-embeddings 2048
    --num-layers 12
    --hidden-size 768
    --num-attention-heads 12
    --ffn-hidden-size 2048
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
    --vocab-size 32000
    # --use-cpu-initialization
)

TRAINING_ARGS=(
    --seed 3407
    --micro-batch-size 32
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
    --norm-epsilon 1e-5
    --clip-grad 1.0
    --bf16
    --override-opt_param-scheduler
    ## 激活稀疏训练参数
    # --act-sparse-training
    # --act-sparse-predictor-hidden-size 64
    # --act-sparse-bank-size 64
    # --act-sparse-topk 16
    # --act-sparse-btopk-coeff 0.001
    # --act-sparse-swiglu-without-silu

    ## Chain of Expert训练
    # --use-coe-layer               
    # --coe-communication-steps 2
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
   # --expert-model-parallel-size 1
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
    --save-interval 5000
    --eval-interval 5000
    --eval-iters 1
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --wandb-project megatron-training-tinyllama
    --wandb-exp-name $EXPERIMENT_NAME
    --wandb-save-dir $WANDB_PATH
    --log-timers-to-tensorboard
    --tensorboard-dir $TENSORBOARD_PATH
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
