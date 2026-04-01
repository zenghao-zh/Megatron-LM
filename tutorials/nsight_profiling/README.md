# Nsight Performance Profiling Tutorial

使用 NVIDIA Nsight Systems & Nsight Compute 分析 Megatron-LM 训练性能的实战教程。

以 `run_smollm.sh`（SmolLM 360M, INT8 混合精度, 4-GPU 训练）为贯穿全程的示例。

---

## 环境要求

| 组件 | 版本 |
|------|------|
| GPU | NVIDIA A800-SXM4-80GB x 8 |
| CUDA | 12.4 |
| Nsight Systems (`nsys`) | 2024.2+ |
| Nsight Compute (`ncu`) | 2024.1+ |
| Megatron-LM | 本仓库 (含 `--profile` 支持) |

快速检查：
```bash
nsys --version && ncu --version && nvidia-smi --query-gpu=name --format=csv,noheader
```

---

## 课程目录

| # | 课程 | 难度 | 内容 |
|---|------|------|------|
| **01** | [Quick Start](01_quick_start/) | Easy | 5 分钟跑通第一次 nsys profiling |
| **02** | [解读时间线](02_read_timeline/) | Medium | NVTX 标注、kernel 排布、如何识别瓶颈 |
| **03** | [CLI 分析](03_nsys_cli_analysis/) | Medium | 不用 GUI，纯命令行提取关键指标 |
| **04** | [Nsight Compute 深入](04_ncu_kernel_deep_dive/) | Hard | 单 kernel 微架构分析 (SOL, Occupancy, Stalls) |
| **05** | [自定义 NVTX 标注](05_custom_nvtx/) | Hard | 在你的代码中添加精确的 NVTX 标记 |
| **06** | [优化模式总结](06_optimization_patterns/) | Expert | 7 种常见瓶颈模式与对应修复方法 |

---

## 推荐学习路径

```
新手:     01 → 02 → 03          (掌握基本 profiling 工作流)
进阶:     01 → 02 → 03 → 04     (能深入分析单个 kernel)
完整:     01 → 02 → 03 → 04 → 05 → 06  (完整性能优化能力)
```

---

## 核心概念速览

### 两个工具的分工

```
nsys (Nsight Systems)              ncu (Nsight Compute)
┌─────────────────────┐      ┌─────────────────────────┐
│  系统级时间线分析     │      │  单 Kernel 微架构分析    │
│                     │      │                         │
│  - GPU 利用率       │      │  - Compute 利用率 (SOL)  │
│  - Kernel 排布      │ ──→  │  - Memory 带宽利用率     │
│  - 通信 vs 计算     │      │  - Occupancy            │
│  - Forward/Backward │      │  - Warp Stall 原因       │
│                     │      │                         │
│  开销: ~5%          │      │  开销: ~100x            │
└─────────────────────┘      └─────────────────────────┘
       先用这个                    再用这个
```

### Megatron 的 profiling 内置支持

Megatron 已经内置了 `--profile` 参数，会在指定 step 范围内自动调用 `cudaProfilerStart/Stop` 并发射 NVTX 标注。你只需要：

1. 在训练参数中加 `--profile --profile-step-start 5 --profile-step-end 8`
2. 用 `nsys profile --capture-range=cudaProfilerApi ...` 包裹 `torchrun`

详见 [Lesson 01](01_quick_start/)。

---

## 文件结构

```
tutorials/nsight_profiling/
├── README.md                           ← 你在这里
├── 01_quick_start/
│   ├── README.md                       # 教程文档
│   └── run_nsys_basic.sh              # 可直接运行的采集脚本
├── 02_read_timeline/
│   └── README.md                       # 时间线解读图文教程
├── 03_nsys_cli_analysis/
│   ├── README.md                       # CLI 分析教程 + SQLite 查询示例
│   └── analyze_nsys_report.sh          # 一键分析脚本
├── 04_ncu_kernel_deep_dive/
│   ├── README.md                       # Nsight Compute 教程
│   └── run_ncu_single_kernel.sh        # 单 kernel 采集脚本
├── 05_custom_nvtx/
│   ├── README.md                       # NVTX 标注教程
│   └── nvtx_example.py                # 独立可运行示例
└── 06_optimization_patterns/
    └── README.md                       # 瓶颈模式与优化方法汇总
```

---

## 快速开始

```bash
# 1. 采集
cd /root/workspace/Megatron-LM
bash tutorials/nsight_profiling/01_quick_start/run_nsys_basic.sh

# 2. 分析
bash tutorials/nsight_profiling/03_nsys_cli_analysis/analyze_nsys_report.sh \
     nsys_profiles/smollm-360m-nsys-demo_0.nsys-rep

# 3. 针对特定 kernel 深入
NCU_KERNEL="gemm" bash tutorials/nsight_profiling/04_ncu_kernel_deep_dive/run_ncu_single_kernel.sh
```
