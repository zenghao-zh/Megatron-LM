# Lesson 04: Nsight Compute -- 深入分析单个 Kernel

当 `nsys` 告诉你「某个 kernel 最耗时」之后，你需要搞清楚**为什么它慢**。这就是 `ncu` (Nsight Compute) 的用武之地。

---

## nsys vs ncu 的关系

```
nsys (Lesson 01~03)                    ncu (本课)
┌─────────────────────────┐      ┌─────────────────────────┐
│ 看「全局」               │      │ 看「单个 kernel」        │
│ 所有 kernel 的排列时序    │──→  │ 一个 kernel 的内部行为    │
│ CPU/GPU 流水线           │      │ Memory/Compute 利用率    │
│ 通信 vs 计算比例         │      │ Occupancy / Warp Stalls  │
│ 开销极小 (~5%)           │      │ 开销极大 (~100x)         │
└─────────────────────────┘      └─────────────────────────┘
          全景 X 光          →→→           显微镜
```

---

## 关键原则：ncu 只看 1~2 个 kernel

ncu 会**逐条指令**地重放 kernel，开销极大。一个完整训练 step 可能有上千个 kernel，全部 profile 会跑几小时。

正确做法：
1. 先用 `nsys` 确定目标 kernel 名称（如 `sm80_xmma_gemm_bf16...`）
2. 用 `--kernel-name` 只匹配该 kernel
3. 用 `--launch-skip` 跳过 warmup 阶段的调用
4. 用 `--launch-count 1` 只采集一次

---

## 运行

### 默认：分析 GEMM kernel

```bash
cd /root/workspace/Megatron-LM
bash tutorials/nsight_profiling/04_ncu_kernel_deep_dive/run_ncu_single_kernel.sh
```

### 自定义：分析 FlashAttention kernel

```bash
NCU_KERNEL="flash_fwd" NCU_SKIP=100 NCU_COUNT=2 \
    bash tutorials/nsight_profiling/04_ncu_kernel_deep_dive/run_ncu_single_kernel.sh
```

### 自定义：分析 INT8 量化 kernel

```bash
NCU_KERNEL="quantize" NCU_SKIP=200 \
    bash tutorials/nsight_profiling/04_ncu_kernel_deep_dive/run_ncu_single_kernel.sh
```

> 脚本用 4 层（而非 32 层）、batch=4 的小模型来加速。因为 ncu 分析的是单个 kernel 的微架构行为，与模型大小无关。

---

## 脚本中的关键 ncu 参数

```bash
ncu \
    --target-processes all \          # 追踪所有子进程
    --kernel-name-base demangled \    # 用 demangled C++ 名称匹配
    --kernel-name "gemm" \            # 只匹配名称含 "gemm" 的 kernel
    --launch-skip 500 \               # 跳过前 500 次调用
    --launch-count 1 \                # 只采集 1 次
    --set full \                      # 收集完整指标集
    --output output.ncu-rep \         # 输出文件
    python pretrain_gpt.py ...
```

| 参数 | 说明 |
|------|------|
| `--kernel-name` | 正则匹配 kernel 名称。从 `nsys stats` 输出中复制 |
| `--launch-skip N` | 跳过前 N 次调用。设大一点跳过 warmup |
| `--launch-count N` | 采集 N 次调用。通常 1~3 次即可 |
| `--set full` | 完整指标集（包括所有 Section）。也可用 `--set basic` 快速预览 |
| `--section` | 只采集某个 Section（如 `SpeedOfLight`），更快 |

---

## 解读 ncu 报告

### CLI 查看

```bash
ncu --import ncu_profiles/kernel_gemm_skip500.ncu-rep --page raw
```

### GUI 查看（推荐）

下载到本地，用 Nsight Compute GUI 打开。

### 核心指标解读

#### Speed of Light (SOL) -- 最重要的指标

```
Section: GPU Speed Of Light Throughput
    SM [%]        : 72.5%     ← Compute 利用率
    Memory [%]    : 45.3%     ← Memory 带宽利用率
```

| SM% vs Memory% | 瓶颈类型 | 含义 | 优化方向 |
|-----------------|---------|------|---------|
| SM 高, Memory 低 | **Compute Bound** | 算力是瓶颈 | 减少计算量、用更快的数据类型 |
| SM 低, Memory 高 | **Memory Bound** | 带宽是瓶颈 | kernel fusion、减少数据搬运 |
| 两者都低 | **Latency Bound** | 都没充分利用 | 提高 occupancy、减少 warp stall |
| 两者都高 | 接近最优 | 硬件被充分利用 | 考虑算法层面优化 |

#### Occupancy -- GPU 并行度

```
Section: Occupancy
    Achieved Occupancy: 0.75  (75%)
    Theoretical Occupancy: 1.0
```

- 75%+ 通常足够好
- 低 occupancy 常见原因：register 用太多、shared memory 用太多

#### Warp Stall Reasons -- 为什么 SM 在等待？

```
Section: Warp State Statistics
    Stall Long Scoreboard  : 35%   ← 等待全局内存读取
    Stall Math Pipe Throttle: 25%  ← 数学管线满了（好事）
    Stall Wait             : 15%   ← 等待依赖指令完成
    Stall Short Scoreboard : 10%   ← 等待 shared memory
```

| Stall 原因 | 含义 | 优化方向 |
|-----------|------|---------|
| Long Scoreboard | 等全局内存 | 提高 cache 命中率、减少内存访问 |
| Math Pipe Throttle | 数学管线饱和 | 正常，说明是 compute bound |
| Short Scoreboard | 等 shared memory | 优化 shared memory bank conflict |
| Barrier | 等 __syncthreads() | 减少同步、平衡 warp 负载 |

#### Memory Workload -- 内存访问效率

```
Section: Memory Workload Analysis
    L1 Hit Rate: 85%
    L2 Hit Rate: 62%
    DRAM Throughput: 1200 GB/s (of 2039 GB/s peak)
```

- A800 的 HBM 峰值带宽约 2039 GB/s
- DRAM Throughput / Peak > 80% 说明带宽已经充分利用

---

## 实际案例：你可能会看到什么

### GEMM Kernel (bf16)

```
SOL SM: 80%+   SOL Memory: 30-40%
→ Compute Bound (正常，cuBLAS GEMM 优化得很好)
```

### INT8 量化 Kernel

```
SOL SM: 20-30%   SOL Memory: 70%+
→ Memory Bound (量化操作是逐元素的，计算密度低)
→ 优化方向: 与前后 kernel 融合
```

### FlashAttention Kernel

```
SOL SM: 60-70%   SOL Memory: 50-60%
→ 接近平衡 (FlashAttention 就是通过 fusion 优化内存访问的)
```

---

## ncu 常用的 Section 单独采集

如果 `--set full` 太慢，可以只采集你关心的 Section：

```bash
# 只看 Speed of Light
ncu --section SpeedOfLight --kernel-name "gemm" ...

# 只看 Occupancy
ncu --section Occupancy --kernel-name "gemm" ...

# 只看 Memory
ncu --section MemoryWorkloadAnalysis --kernel-name "gemm" ...

# 列出所有可用 Section
ncu --list-sections
```

---

## 常见问题

### Q: ncu 跑了很久还没结束？
正常。ncu 会将 kernel 重放多次来收集不同的硬件计数器。一个 `--set full` 可能需要重放 10+ 次。用 `--set basic` 会快很多。

### Q: 报错 "ERR_NVGPUCTRPERM"？
需要 root 权限或设置 `cat /proc/driver/nvidia/params/ModuleParametersPerformanceMonitor=1`：
```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/ncu.conf
# 或直接用 sudo 运行
```

### Q: 想对比两次运行的结果？
```bash
ncu --import baseline.ncu-rep --import optimized.ncu-rep --page diff
```

---

## 下一步

想在你自己的代码里添加 NVTX 标记来精确定位区域？去 [Lesson 05](../05_custom_nvtx/)。
