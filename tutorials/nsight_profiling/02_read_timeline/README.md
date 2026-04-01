# Lesson 02: 解读 Nsight Systems 时间线

拿到 `.nsys-rep` 文件后，如何从中读出性能信息？本课教你「看什么」和「怎么看」。

---

## 打开报告的两种方式

### 方式 A: GUI（推荐，适合有桌面/远程桌面的场景）

1. 在本地机器安装 [Nsight Systems](https://developer.nvidia.com/nsight-systems)（免费）
2. 把 `.nsys-rep` 文件 scp 到本地
3. 用 Nsight Systems 打开

```bash
# 在本地执行
scp user@server:/path/to/nsys_profiles/*.nsys-rep ./
# 然后双击 .nsys-rep 文件打开
```

### 方式 B: CLI（适合纯终端场景）

不需要 GUI 也能分析，详见 [Lesson 03](../03_nsys_cli_analysis/)。

---

## 时间线的结构（从上到下）

在 GUI 中打开后，你会看到类似这样的层次结构：

```
Timeline (横轴 = 时间)
│
├── CPU rows (每个线程一行)
│   ├── Python Main Thread
│   │   ├── NVTX ranges (Megatron 标注的逻辑区域)
│   │   └── CUDA API calls (cudaLaunchKernel, cudaMemcpy...)
│   ├── NCCL Thread
│   └── DataLoader Workers
│
├── GPU rows (每张卡一行)
│   ├── CUDA Kernels (实际在 GPU 上执行的 kernel)
│   │   ├── GEMM kernels (矩阵乘法)
│   │   ├── Elementwise kernels (激活函数、LayerNorm...)
│   │   └── NCCL kernels (AllReduce, AllGather...)
│   └── Memory Operations (cudaMemcpy)
│
└── NVTX Ranges (嵌套的逻辑标签)
    ├── "forward"
    │   ├── "self_attention"    (每层一个)
    │   ├── "self_attn_bda"     (attention 后的 bias-dropout-add)
    │   ├── "mlp"               (每层一个)
    │   └── "mlp_bda"           (MLP 后的 bias-dropout-add)
    └── "backward"
```

---

## Megatron 内置的 NVTX 标注

Megatron 在 `megatron/core/transformer/transformer_layer.py` 中为每个 Transformer 层的关键模块添加了 NVTX 标注：

| NVTX Range 名称 | 对应模块 | 说明 |
|-----------------|----------|------|
| `self_attention` | `TransformerLayer.self_attention` | QKV 投影 + Attention 计算 + Output 投影 |
| `self_attn_bda` | bias-dropout-add after attention | 残差连接 + Dropout |
| `mlp` | `TransformerLayer.mlp` | FFN (SwiGLU) 的前向计算 |
| `mlp_bda` | bias-dropout-add after MLP | 残差连接 + Dropout |

在时间线上，一个完整的 training step 看起来像：

```
|<------ Forward Pass ------>|<------ Backward Pass ------>|<-- Optimizer -->|
|                            |                              |                 |
| [attn][bda][mlp][bda] x32  | [mlp_bw][attn_bw] x32      | [Adam update]   |
|  (32 layers)               |  (32 layers, 反序)           |                 |
```

---

## 五个关键观察点

### 1. GPU 利用率：有没有空隙？

**怎么看：** 观察 GPU Kernel 行。如果 kernel 之间有明显空白（gap），说明 GPU 在等待。

```
好的情况（kernel 紧密排列）：
GPU: ████████████████████████████████████████

坏的情况（有 gap）：
GPU: ████░░░████░░░████░░████░░░░████████░░░
         ↑       ↑
      CPU 在干活  同步等待
```

**常见原因：**
- CPU-bound: Python 端计算太慢，GPU 等不到新 kernel
- 同步点: `torch.cuda.synchronize()` 或隐式同步
- 数据加载: DataLoader 跟不上

### 2. 通信 vs 计算：通信比例多大？

**怎么看：** 在 GPU kernel 行中，NCCL kernel（名称含 `ncclKernel` 或 `nccl`）代表集合通信。

```
理想情况（通信与计算重叠）：
GPU Stream 0: ████ GEMM ████████ GEMM ████████
GPU Stream 1:   ████ AllReduce ████   ████ AllReduce ████
                 ↑ 计算和通信并行

不理想（通信串行）：
GPU: ████ GEMM ████ | AllReduce | ████ GEMM ████ | AllReduce |
                     ↑ GPU 空等通信完成
```

**健康指标：** 通信时间 / 总 step 时间 < 15% 为佳。

### 3. 最耗时的 Kernel 是哪些？

**怎么看：** 右键 GPU kernel 行 -> Show in Events View，按 Duration 排序。

典型的 Top-5 kernel 类型：
- `sm80_xmma_gemm_*` / `cutlass_*`: GEMM（矩阵乘法），通常是最大头
- `void at::native::*`: PyTorch elementwise 操作
- `ncclKernel_AllReduce_*`: NCCL 通信
- `void flash_*`: FlashAttention kernel
- `void layernorm_*`: LayerNorm / RMSNorm

### 4. Forward vs Backward 的时间比

**怎么看：** 利用 NVTX 标注，测量 Forward 和 Backward 的总时长。

```
典型比例（BF16 训练）：
Forward  : ████████████         (~30-35%)
Backward : ██████████████████████  (~55-60%)
Optimizer: ████                 (~5-10%)
```

如果 Backward 远超 2x Forward，检查是否有不必要的 recomputation 或激活重计算配置。

### 5. 每个 Step 的耗时是否一致？

**怎么看：** 对比多个 step 的总时长。

```
正常：
Step 5: |████████████████| 1.2s
Step 6: |████████████████| 1.2s
Step 7: |████████████████| 1.2s

异常（某些 step 明显更长）：
Step 5: |████████████████| 1.2s
Step 6: |██████████████████████████| 2.0s   <-- 为什么？
Step 7: |████████████████| 1.2s
```

常见原因：GC (垃圾回收)、checkpoint saving、eval step、动态 batch size。

---

## 实用操作技巧（GUI）

| 操作 | 快捷键/方法 |
|------|------------|
| 放大时间线 | 鼠标滚轮 / Ctrl+滚轮 |
| 选中一个时间区间 | 鼠标拖选 |
| 查看选中区间的统计 | 选中后看底部 Summary 面板 |
| 搜索特定 kernel | Ctrl+F，输入 kernel 名称 |
| 测量两点间时间 | 左键点第一个点，右键点第二个点 |
| 只看某一行 | 双击行名展开/折叠 |

---

## 实际例子：SmolLM 360M 你会看到什么

基于 `run_smollm.sh` 的配置（32 层, hidden=960, heads=15, bf16+int8）:

- **Forward 中 MLP 占比大**: SwiGLU MLP 的 FFN hidden=2560，有 gate + up + down 三个矩阵乘，加上 INT8 量化/反量化的额外 kernel
- **Self-Attention 中 FlashAttention 是主体**: seq_len=2048，GQA (15 heads, 5 groups)
- **Sequence Parallel AllGather/ReduceScatter**: 因为启用了 `--sequence-parallel`，每层前后有 AllGather 和 ReduceScatter
- **INT8 量化 kernel**: 你会看到额外的量化/反量化 kernel 穿插在 GEMM 之间

---

## Checklist: 你应该能回答这些问题

看完时间线后，试着回答：

- [ ] 一个完整的 training step 耗时多少？
- [ ] Forward / Backward / Optimizer 各占多少比例？
- [ ] 最耗时的 3 个 kernel 是什么？
- [ ] GPU 利用率大约多少？（有没有明显空隙？）
- [ ] 通信占总时间的百分比？
- [ ] 通信和计算有没有重叠？

---

## 下一步

想要不用 GUI，直接在终端里提取这些指标？去 [Lesson 03](../03_nsys_cli_analysis/)。
