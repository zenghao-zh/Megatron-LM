# Lesson 06: 常见性能瓶颈模式与优化方法

本课汇总在 Nsight profiling 中常见的性能问题，以及对应的优化策略。

---

## 瓶颈诊断决策树

```
Profile 结果出来了，看哪里？
│
├── GPU 利用率低（kernel 之间有空隙）?
│   ├── 空隙在 step 之间 ──→ [Pattern A: CPU Bottleneck]
│   ├── 空隙在通信操作前后 ──→ [Pattern C: Communication Bottleneck]
│   └── 空隙在 kernel 之间均匀分布 ──→ [Pattern B: Kernel Launch Overhead]
│
├── GPU 利用率高，但 throughput 不达预期?
│   ├── Top kernel 是 memory bound ──→ [Pattern D: Memory Bound Kernels]
│   └── Top kernel 是 compute bound ──→ [Pattern E: Compute Bound]
│
└── 某些 step 异常慢?
    ├── 周期性变慢 ──→ [Pattern F: GC / Checkpoint]
    └── Pipeline parallel 气泡 ──→ [Pattern G: Pipeline Bubbles]
```

---

## Pattern A: CPU Bottleneck

### 症状
- GPU kernel 行有大段空白
- CUDA API 行的 `cudaLaunchKernel` 调用间隔不均匀
- CPU 采样显示 Python 代码占用大量时间

### nsys 中的表现

```
CPU: ████ Python ████████ Python ████████
GPU: ████░░░░░░░░████░░░░░░░░░████░░░░░░
         ↑ GPU 在等 CPU 发射下一个 kernel
```

### 常见原因与修复

| 原因 | 检查方法 | 修复 |
|------|---------|------|
| Data loading 慢 | NVTX 标注 data loading 区域 | 增加 `--num-workers`，使用内存映射数据集 |
| Python GIL | CPU 采样显示 Python 代码热点 | 减少 Python 层计算，用 `torch.compile` |
| 复杂的 Python 控制流 | 每个 step 的 CPU 时间 >> GPU 时间 | 简化逻辑，移到 CUDA 端 |
| 过多的小 tensor 操作 | `cudaLaunchKernel` 数量极多 | Kernel fusion（见 Pattern B）|

### 关键指标
```bash
# 看 cudaLaunchKernel 的调用频率
nsys stats --report cuda_api_sum <file.nsys-rep>
# 如果 cudaLaunchKernel 调用次数 > 10000/step，可能太碎了
```

---

## Pattern B: Kernel Launch Overhead

### 症状
- GPU kernel 行有很多非常短的 kernel，中间有规律的小空隙
- `cudaLaunchKernel` 的总 CPU 时间占比很高

### nsys 中的表现

```
GPU: █░█░█░█░█░██░█░█░█░█░██░
     ↑ 大量微小 kernel，每个之间有 launch 间隙
```

### 修复方法

| 方法 | 说明 | Megatron 中的参数 |
|------|------|------------------|
| `torch.compile` | 自动 fuse 多个小 kernel | 需要手动集成 |
| cuDNN / cuBLAS fusion | 让库自动合并操作 | 默认启用 |
| 手写 fused kernel | 把多个 elementwise 操作合成一个 kernel | Megatron 已有 `fused_bias_swiglu` 等 |
| `CUDA_DEVICE_MAX_CONNECTIONS=1` | 限制 CUDA stream 数量，减少调度开销 | `run_smollm.sh` 第一行已设置 |

### 检查: 量化你的 launch overhead

```sql
-- 在 SQLite 中计算平均 kernel 间隙
sqlite3 report.sqlite <<'SQL'
WITH k AS (
    SELECT start, end, LEAD(start) OVER (ORDER BY start) AS next_start
    FROM CUPTI_ACTIVITY_KIND_KERNEL
)
SELECT
    ROUND(AVG(next_start - end) / 1e3, 1) AS avg_gap_us,
    COUNT(*) AS total_kernels
FROM k WHERE next_start > end;
SQL
-- avg_gap_us > 5 说明 launch overhead 显著
```

---

## Pattern C: Communication Bottleneck

### 症状
- NCCL kernel 占总 GPU 时间 > 15%
- 通信和计算没有重叠（串行执行）

### nsys 中的表现

```
不好（串行）:
GPU: ████ GEMM ████ | ncclAllReduce | ████ GEMM ████ | ncclAllReduce |

好（重叠）:
Stream 0: ████ GEMM ████████ GEMM ████████ GEMM ████
Stream 1:     █ AllReduce █     █ AllReduce █
```

### 修复方法

| 方法 | 说明 | Megatron 参数 |
|------|------|--------------|
| Overlap comm & compute | 通信和计算在不同 stream 上并行 | `--overlap-grad-reduce` |
| 调整 TP/PP 大小 | 减少跨节点通信 | `--tensor-model-parallel-size` |
| Sequence Parallel | 用 ReduceScatter 替代 AllReduce | `--sequence-parallel`（已启用） |
| Gradient accumulation | 减少通信频率 | 增大 `--global-batch-size` |

### 检查: 量化通信占比

```bash
# 用 Lesson 03 的 SQLite 查询
sqlite3 report.sqlite <<'SQL'
SELECT ROUND(100.0 *
    SUM(CASE WHEN shortName LIKE '%nccl%' THEN end - start ELSE 0 END) /
    SUM(end - start), 1) AS nccl_pct
FROM CUPTI_ACTIVITY_KIND_KERNEL;
SQL
```

---

## Pattern D: Memory Bound Kernels

### 症状
- `ncu` 显示 SOL Memory > 60%, SOL SM < 30%
- 大量 elementwise kernel（activation, normalization, 量化/反量化）

### nsys 中的表现

```
cuda_gpu_kern_sum 中:
  Time%  Name
  15%    void at::native::vectorized_elementwise_kernel<...>   ← memory bound
   8%    void layernorm_forward_kernel<...>                     ← memory bound
   6%    void quantize_int8_kernel<...>                         ← memory bound
```

### 修复方法

| 方法 | 说明 | 预期收益 |
|------|------|---------|
| Kernel Fusion | 把连续的 elementwise 操作合成一个 kernel | 减少 HBM 读写次数 |
| `torch.compile` | 自动发现并 fuse elementwise 操作 | 10-30% |
| Fused Operators | 用库提供的 fused 版本 | Megatron 有 `fused_bias_swiglu`, `fused_layer_norm` |
| 减少数据类型转换 | 避免 bf16 → fp32 → bf16 的往返 | 减少 2x 内存读写 |

### INT8 混合精度特有的优化点

在 SmolLM 的 INT8 训练中，量化/反量化 kernel 是典型的 memory bound 操作：

```
数据流: bf16 input → [quantize] → int8 → [gemm] → int32 → [dequantize] → bf16
                      ↑ memory bound              ↑ memory bound
```

优化方向：
- 将 quantize 与前一个操作 fuse（如 LayerNorm + Quantize）
- 将 dequantize 与后一个操作 fuse（如 Dequantize + Activation）
- 使用 groupwise quantization 时，确保 group_size 对齐 GPU cache line

---

## Pattern E: Compute Bound Kernels

### 症状
- `ncu` 显示 SOL SM > 70%, SOL Memory < 40%
- 主要是 GEMM kernel

### 这通常是好事!

GEMM kernel compute bound 说明 GPU 算力被充分利用。进一步优化需要从算法层面入手：

| 方法 | 说明 |
|------|------|
| 使用更低精度 | BF16 → INT8/FP8 GEMM（本项目已在做） |
| 减少计算量 | 稀疏注意力、MoE routing、激活稀疏 |
| Tensor Core 对齐 | 确保矩阵维度是 8/16/32 的倍数 |
| cuBLAS 调优 | `CUBLAS_WORKSPACE_CONFIG` 环境变量 |

### 检查矩阵维度对齐

SmolLM 360M 的关键维度：
- hidden_size=960 (960 / 16 = 60, 对齐 Tensor Core)
- ffn_hidden_size=2560 (2560 / 16 = 160, 对齐)
- num_attention_heads=15, head_dim=64 (64 / 16 = 4, 对齐)
- vocab_size=49152 (49152 / 16 = 3072, 对齐)

都是 16 的倍数，Tensor Core 利用率应该很好。

---

## Pattern F: GC / Checkpoint 导致的间歇性卡顿

### 症状
- 大多数 step 耗时一致，但周期性出现一个特别慢的 step
- 慢的 step 中 CPU 有长时间空闲，或有磁盘 I/O

### 修复方法

```bash
# Megatron 支持手动控制 GC
--manual-gc                    # 手动管理垃圾回收
--manual-gc-interval 100       # 每 100 步做一次 GC

# 减少 checkpoint 频率
--save-interval 5000           # profiling 时设大一点

# 异步 checkpoint
--async-save                   # 异步保存，不阻塞训练
```

---

## Pattern G: Pipeline Parallel 气泡

### 症状 (仅 PP > 1 时)
- 时间线上看到周期性的大段 GPU 空闲
- 空闲出现在 pipeline flush 阶段

### 修复方法

| 方法 | 说明 |
|------|------|
| 增加 micro-batch 数量 | 气泡比例 = (pp_size - 1) / num_microbatches |
| Virtual Pipeline Parallel | `--num-layers-per-virtual-pipeline-stage` |
| Interleaved schedule | 减少 bubble ratio |

> 当前 `run_smollm.sh` 使用 PP=1，没有 pipeline bubble 问题。

---

## 性能分析 Checklist 总表

拿到 profiling 数据后，按这个顺序检查：

| # | 检查项 | 工具 | 期望值 |
|---|--------|------|--------|
| 1 | 每个 step 的耗时 | nsys 时间线 | 各 step 一致 |
| 2 | GPU 利用率 | nsys kernel 行 | 无明显空隙 |
| 3 | Top-5 kernel 及占比 | `nsys stats --report cuda_gpu_kern_sum` | GEMM > 40% |
| 4 | NCCL 通信占比 | SQLite 查询 | < 15% (TP=1) |
| 5 | 通信-计算重叠 | nsys 时间线 | 通信隐藏在计算后面 |
| 6 | cudaLaunchKernel 次数 | `nsys stats --report cuda_api_sum` | 越少越好 |
| 7 | 同步操作次数 | `cuda_api_sum` 找 Synchronize | 应该极少 |
| 8 | Top kernel 的 SOL | `ncu` Speed of Light | SM% 或 Mem% > 60% |
| 9 | Forward / Backward 比例 | NVTX ranges | Backward ~2x Forward |
| 10 | 内存使用 | `nvidia-smi` 或 `--cuda-memory-usage` | 无 OOM，合理利用 |

---

## 推荐的优化顺序

```
1. 先确保没有 GPU 空闲（Pattern A/B）
   ↓  解决 CPU bottleneck / launch overhead
2. 优化通信（Pattern C）
   ↓  overlap / 调整并行策略
3. 优化 memory bound kernel（Pattern D）
   ↓  kernel fusion / torch.compile
4. 最后才考虑 compute bound（Pattern E）
   ↓  数据类型 / 算法优化
```

原因：前面的优化容易获得大幅度提升（10-50%），后面的优化收益递减但难度增大。

---

## 回到起点

完整的 profiling 工作流：

```
[Lesson 01] 采集 nsys profile
      ↓
[Lesson 02] 看时间线，识别瓶颈区域
      ↓
[Lesson 03] CLI 提取关键指标
      ↓
[Lesson 04] 对特定 kernel 用 ncu 深入分析
      ↓
[Lesson 05] 对自定义代码添加 NVTX 标注
      ↓
[Lesson 06] 根据本课的 Pattern 匹配优化策略
      ↓
   重新 profile，验证优化效果，循环迭代
```
