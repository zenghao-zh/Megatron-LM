# Lesson 01: Quick Start -- 5 分钟跑通第一次 Nsight Profiling

## 两个工具，先认清分工

| 工具 | 全称 | 定位 | 类比 |
|------|------|------|------|
| `nsys` | Nsight Systems | **系统级**时间线分析 | 「全景X光」—— 看整体骨架 |
| `ncu`  | Nsight Compute | **单 kernel 级**微架构分析 | 「显微镜」—— 看单个细胞 |

**黄金法则：先 `nsys` 定位瓶颈区域，再 `ncu` 深入分析具体 kernel。**

本课只用 `nsys`，`ncu` 留到 [Lesson 04](../04_ncu_kernel_deep_dive/)。

---

## Megatron-LM 内置的 Profiling 机制

你不需要自己在代码里插 `cudaProfilerStart()`——Megatron 已经内置了。只需要加三个命令行参数：

```bash
--profile                 # 启用 profiling
--profile-step-start 5    # 从第 5 个 global step 开始采集
--profile-step-end   8    # 到第 8 个 global step 停止采集
```

**它的工作原理：**

```
step 0  1  2  3  4 | 5  6  7 | 8  9  10 ...
                    |         |
        warmup      |  采集区  |  正常训练
                    |         |
          cudaProfilerStart()  cudaProfilerStop()
          emit_nvtx(True)
```

在 `megatron/training/training.py` 中，当 `iteration == profile_step_start` 时：
```python
torch.cuda.cudart().cudaProfilerStart()
torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
```

当 `iteration == profile_step_end` 时：
```python
torch.cuda.cudart().cudaProfilerStop()
```

这就是为什么 nsys 必须配合 `--capture-range=cudaProfilerApi` 使用——它只在这对 Start/Stop 之间真正记录数据。

---

## 运行 Profiling

### 第一步：确认环境

```bash
nsys --version    # 应该输出 2024.x 或更高
nvidia-smi        # 确认 GPU 可用
```

### 第二步：运行采集脚本

```bash
cd /root/workspace/Megatron-LM
bash tutorials/nsight_profiling/01_quick_start/run_nsys_basic.sh
```

脚本会：
1. 正常启动 4-GPU 分布式训练
2. 前 5 个 step 正常运行（warmup，不采集）
3. Step 5~8 期间 nsys 开始录制
4. Step 8 之后停止录制，训练可以继续或手动 Ctrl+C 终止
5. 在 `nsys_profiles/` 目录下生成 `.nsys-rep` 文件

### 第三步：确认产出

```bash
ls -lh nsys_profiles/
# 预期输出:
#   smollm-360m-nsys-demo_0.nsys-rep   (rank 0)
#   smollm-360m-nsys-demo_1.nsys-rep   (rank 1)
#   ...
```

---

## 脚本对比：与 run_smollm.sh 有什么不同？

只有 **3 处改动**：

### 改动 1: TRAINING_ARGS 新增 profile 参数

```bash
# 原版 run_smollm.sh: 无 profile 参数
# 新增:
    --profile
    --profile-step-start 5
    --profile-step-end 8
```

### 改动 2: torchrun 前面包裹 nsys profile

```bash
# 原版:
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun ... pretrain_gpt.py ...

# 改为:
CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile \
    --output ./nsys_profiles/output_%q{RANK} \
    --force-overwrite true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --trace cuda,nvtx,osrt,cudnn,cublas \
    --stats true \
    torchrun ... pretrain_gpt.py ...
```

### 改动 3: 关闭 wandb / tensorboard / checkpoint

Profiling 时不需要这些，减少 I/O 干扰。

---

## nsys profile 参数速查

| 参数 | 作用 | 建议值 |
|------|------|--------|
| `--output` | 输出文件路径，`%q{RANK}` 按 rank 分文件 | `./nsys_profiles/name_%q{RANK}` |
| `--force-overwrite true` | 覆盖已有文件 | `true` |
| `--capture-range=cudaProfilerApi` | 只在 cudaProfilerStart/Stop 之间采集 | 必须配合 Megatron `--profile` |
| `--capture-range-end=stop` | 遇到 cudaProfilerStop 就结束 | 搭配使用 |
| `--trace` | 采集哪些事件 | `cuda,nvtx,osrt,cudnn,cublas` |
| `--stats true` | 采集后自动打印统计摘要 | `true` |
| `--cuda-memory-usage true` | 追踪 GPU 内存分配 | 可选，会增加开销 |
| `--sample cpu` | CPU 采样 | 可选，看 Python 热点 |

---

## 常见问题

### Q: 为什么要跳过前几个 step？
前几个 step 包含模型初始化、JIT 编译、CUDA context 创建等一次性开销，不能代表稳态性能。建议 `--profile-step-start` >= 3。

### Q: profile 多少个 step 合适？
3~5 个 step 足以看清模式。太多会导致 `.nsys-rep` 文件过大（几十 GB），打开和分析都很慢。

### Q: 只想 profile rank 0 怎么办？
加 `--profile-ranks 0`（Megatron 参数，默认就是 `[0]`）。

### Q: 采集结束后训练还在跑？
Ctrl+C 终止即可。profiling 数据在 `cudaProfilerStop()` 时已经写入文件。

---

## 下一步

拿到了 `.nsys-rep` 文件？去 [Lesson 02](../02_read_timeline/) 学习如何解读时间线。
