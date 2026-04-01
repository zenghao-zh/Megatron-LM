#!/bin/bash
# =============================================================================
# Lesson 01: 最简 Nsight Systems 采集脚本
#
# 与 run_smollm.sh 的区别（用 diff 对比即可看到）:
#   1. TRAINING_ARGS 新增 --profile / --profile-step-start / --profile-step-end
#   2. torchrun 前面用 nsys profile ... 包裹
#   3. 关闭 wandb / tensorboard / checkpoint，减少干扰
#
# 用法:
#   cd <Megatron-LM 根目录>
#   bash tutorials/nsight_profiling/01_quick_start/run_nsys_basic.sh
#
# 产出:
#   nsys_profiles/ 目录下会生成 .nsys-rep 文件（每张卡一个）
#   以及自动生成的 .sqlite 统计摘要
# =============================================================================

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---------- 集群 / 路径配置（与 run_smollm.sh 保持一致） ----------
GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6003
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
EXPERIMENT_NAME=smollm-360m-nsys-demo

DATA_PATH="/root/data/smollm_corpus/merged_smollm_corpus"
TOKENIZER_MODEL=/root/data/cosmo2-tokenizer

# ---------- [新增] Nsight Systems 输出目录 ----------
NSYS_OUTPUT_DIR="./nsys_profiles"
mkdir -p "$NSYS_OUTPUT_DIR"

# ---------- [新增] nsys profile 命令参数 ----------
#
# 关键参数说明:
#   --capture-range=cudaProfilerApi
#       只在 cudaProfilerStart() ~ cudaProfilerStop() 之间采集。
#       Megatron 的 --profile 会在 profile-step-start 时调用 cudaProfilerStart()，
#       在 profile-step-end 时调用 cudaProfilerStop()。两者配合使用。
#
#   --trace cuda,nvtx,osrt,cudnn,cublas
#       采集 CUDA runtime API、NVTX 标注、OS runtime、cuDNN、cuBLAS 信息。
#
#   --stats true
#       采集结束后自动输出统计摘要到终端。
#
#   --output ... %q{RANK}
#       每个进程生成独立文件，%q{RANK} 会被替换为进程的 RANK 环境变量。
#
NSYS_CMD=(
    nsys profile
    --output "${NSYS_OUTPUT_DIR}/${EXPERIMENT_NAME}_%q{RANK}"
    --force-overwrite true
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --trace cuda,nvtx,osrt,cudnn,cublas
    --stats true
)

# ---------- 分布式参数 ----------
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# ---------- 模型参数（与 run_smollm.sh 完全一致） ----------
MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 2048
    --max-position-embeddings 2048
    --num-layers 32
    --hidden-size 960
    --num-attention-heads 15
    --ffn-hidden-size 2560
    --init-method-std 0.006
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --group-query-attention
    --num-query-groups 5
    --no-masked-softmax-fusion
    --rotary-base 10000
    --attention-softmax-in-fp32
    --vocab-size 49152
)

# ---------- 训练参数 ----------
# [新增] --profile 系列参数: 让 Megatron 在 step 5~8 之间激活 CUDA Profiler
TRAINING_ARGS=(
    --seed 3407
    --micro-batch-size 16
    --global-batch-size 512
    --lr 3e-3
    --train-samples 50000000
    --lr-warmup-samples $(( 1000 * 2048 ))
    --lr-decay-style constant
    --min-lr 1e-8
    --lr-warmup-init 1e-8
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --norm-epsilon 1e-5
    --clip-grad 1.0
    --bf16
    --int8-mixed-precision-training
    --int8-mp-group-size 64
    --int8-mp-two-stage-mixed
    --int8-mp-topk 16
    --int8-mp-grad-weight
    # ---- 以下三行是 profiling 专用 ----
    --profile
    --profile-step-start 5
    --profile-step-end 8
)

# ---------- 并行参数 ----------
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
)

# ---------- 数据参数 ----------
DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_MODEL
    --data-path $DATA_PATH
    --split 1,0,0
)

# ---------- 日志参数（精简版，关闭 wandb / checkpoint） ----------
EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 100000
    --eval-interval 100000
    --eval-iters 0
)

echo "============================================"
echo " Nsight Systems Profiling"
echo "   Profiling steps : 5 ~ 8"
echo "   Output directory: ${NSYS_OUTPUT_DIR}"
echo "============================================"

# ---------- 启动! ----------
# nsys 包裹 torchrun，对整个进程树进行采集
CUDA_VISIBLE_DEVICES=0,1,2,3 ${NSYS_CMD[@]} \
    torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

echo ""
echo "============================================"
echo " Profiling 完成!"
echo " 报告文件:"
ls -lh ${NSYS_OUTPUT_DIR}/*.nsys-rep 2>/dev/null
echo "============================================"
