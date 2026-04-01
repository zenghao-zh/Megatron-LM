#!/bin/bash
# =============================================================================
# Lesson 04: Nsight Compute 单 Kernel 深度分析脚本
#
# 重要: ncu 开销极大（100x+），只用于分析 1~2 次 kernel 调用。
#       务必限制 --launch-count 和 --launch-skip。
#
# 用法:
#   cd <Megatron-LM 根目录>
#
#   # 分析 GEMM kernel（默认）:
#   bash tutorials/nsight_profiling/04_ncu_kernel_deep_dive/run_ncu_single_kernel.sh
#
#   # 分析指定 kernel（用正则匹配名称）:
#   NCU_KERNEL="flash_fwd" \
#   bash tutorials/nsight_profiling/04_ncu_kernel_deep_dive/run_ncu_single_kernel.sh
#
# 产出:
#   ncu_profiles/ 目录下生成 .ncu-rep 文件
# =============================================================================

set -e

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---------- ncu 分析配置 ----------
NCU_KERNEL_REGEX="${NCU_KERNEL:-gemm}"   # 要分析的 kernel 名称（正则）
NCU_LAUNCH_SKIP="${NCU_SKIP:-500}"       # 跳过前 N 次调用（跳过 warmup）
NCU_LAUNCH_COUNT="${NCU_COUNT:-1}"       # 只分析几次调用
NCU_OUTPUT_DIR="./ncu_profiles"
mkdir -p "$NCU_OUTPUT_DIR"

NCU_OUTPUT_FILE="${NCU_OUTPUT_DIR}/kernel_${NCU_KERNEL_REGEX}_skip${NCU_LAUNCH_SKIP}"

# ---------- ncu 命令 ----------
NCU_CMD=(
    ncu
    --target-processes all
    --kernel-name-base demangled              # 使用 demangled 名称匹配
    --kernel-name "$NCU_KERNEL_REGEX"         # 按名称正则过滤 kernel
    --launch-skip "$NCU_LAUNCH_SKIP"          # 跳过前 N 次（避开 warmup）
    --launch-count "$NCU_LAUNCH_COUNT"        # 只采集 N 次
    --set full                                # 完整指标集
    --output "$NCU_OUTPUT_FILE"               # 输出文件
    --force-overwrite                         # 覆盖已有文件
)

# ---------- 训练参数（单 GPU，小 batch，快速启动） ----------
DATA_PATH="/root/data/smollm_corpus/merged_smollm_corpus"
TOKENIZER_MODEL=/root/data/cosmo2-tokenizer

echo "============================================"
echo " Nsight Compute Kernel Profiling"
echo "   Target kernel : ${NCU_KERNEL_REGEX}"
echo "   Launch skip   : ${NCU_LAUNCH_SKIP}"
echo "   Launch count  : ${NCU_LAUNCH_COUNT}"
echo "   Output        : ${NCU_OUTPUT_FILE}.ncu-rep"
echo ""
echo " *** 注意: ncu 开销极大，运行会非常慢 ***"
echo "============================================"

# 单 GPU，不需要 torchrun
CUDA_VISIBLE_DEVICES=0 ${NCU_CMD[@]} \
    python pretrain_gpt.py \
    --use-mcore-models \
    --disable-bias-linear \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --num-layers 4 \
    --hidden-size 960 \
    --num-attention-heads 15 \
    --ffn-hidden-size 2560 \
    --init-method-std 0.006 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --swiglu \
    --group-query-attention \
    --num-query-groups 5 \
    --no-masked-softmax-fusion \
    --rotary-base 10000 \
    --attention-softmax-in-fp32 \
    --vocab-size 49152 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --seed 3407 \
    --micro-batch-size 4 \
    --global-batch-size 4 \
    --lr 1e-4 \
    --train-samples 100 \
    --lr-warmup-samples 0 \
    --lr-decay-style constant \
    --bf16 \
    --int8-mixed-precision-training \
    --int8-mp-group-size 64 \
    --int8-mp-two-stage-mixed \
    --int8-mp-topk 16 \
    --int8-mp-grad-weight \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --data-path $DATA_PATH \
    --split 1,0,0 \
    --log-interval 1 \
    --save-interval 100000 \
    --eval-interval 100000 \
    --eval-iters 0

echo ""
echo "============================================"
echo " 分析完成!"
echo " 报告文件: ${NCU_OUTPUT_FILE}.ncu-rep"
echo ""
echo " 查看 CLI 摘要:"
echo "   ncu --import ${NCU_OUTPUT_FILE}.ncu-rep --page raw"
echo ""
echo " 或下载到本地用 Nsight Compute GUI 打开"
echo "============================================"
