# BalancedTopk 函数性能对比基准测试

这个基准测试脚本用于对比 `FusedBalancedTopkFunction` 和 `BalancedTopkFunction` 的性能和精度。

## 文件说明

- `benchmark_balanced_topk.py` - 主要的基准测试脚本
- `balanced_topk_simple_test.py` - 简单的功能验证测试
- `balanced_topk_benchmark.png` - 标准测试的可视化结果图表
- `balanced_topk_benchmark_large.png` - 大规模测试的可视化结果图表

## 功能特点

### 性能测试
- 测量运行时间（毫秒）
- 计算加速比
- 支持 CUDA 和 CPU 设备
- 自动预热和多次试验取平均值

### 精度测试
- 最大绝对误差
- 平均绝对误差
- 最大相对误差
- 余弦相似度
- L2 范数差异

## 使用方法

### 基本测试
```bash
# 简单功能验证
python balanced_topk_simple_test.py

# 基本基准测试
python benchmark_balanced_topk.py

# 自定义参数
python benchmark_balanced_topk.py --trials 100 --warmup 20

# 生成可视化图表
python benchmark_balanced_topk.py --plot

# 运行大规模测试（基于真实 MOE 训练配置）
python benchmark_balanced_topk.py --large-scale

# 大规模测试 + 可视化
python benchmark_balanced_topk.py --large-scale --plot
```

### 命令行参数
- `--plot`: 生成可视化图表（需要 matplotlib）
- `--trials N`: 设置试验次数（默认 100）
- `--warmup N`: 设置预热次数（默认 10）
- `--large-scale`: 运行大规模测试配置（18个配置 vs 5个标准配置）

## 测试配置

### 标准测试配置
| 配置 | Tokens | Hidden Size | Experts | TopK | Bank Size |
|------|--------|-------------|---------|------|-----------|
| 1    | 1024   | 4096        | 8       | 16   | 32        |
| 2    | 2048   | 4096        | 8       | 16   | 32        |
| 3    | 4096   | 4096        | 8       | 16   | 32        |
| 4    | 1024   | 8192        | 16      | 32   | 64        |
| 5    | 2048   | 8192        | 16      | 32   | 64        |

### 大规模测试配置（基于真实 MOE 训练配置）
包含 18 个大规模测试配置，涵盖：

- **基于 `moe-ffn-hidden-size=576`** 的配置（2048-16384 tokens）
- **基于 `hidden-size=1024`** 的配置（2048-8192 tokens）  
- **基于 `moe-shared-expert-intermediate-size=1152`** 的配置
- **32个专家**的大规模专家配置
- **不同 k 值**（8, 16, 32）和 **bank size**（64, 128）的对比

## 基准测试结果

### 标准测试结果
基于标准 5 个测试配置：

#### 性能表现
- **平均加速比**: 1.07x
- **加速比范围**: 0.97x - 1.30x
- **最佳性能提升**: 在大规模配置下（2048 tokens, 8192 hidden size）达到 1.30x 加速

### 大规模测试结果
基于真实 MOE 训练配置的 18 个大规模测试：

#### 性能表现
- **平均加速比**: 0.97x（基本持平）
- **加速比范围**: 0.62x - 1.18x
- **最佳性能提升**: 1.18x（在 4096 tokens, 576 hidden, 32 experts, k=32 配置下）
- **平均内存比例**: 1.60x（FusedBalancedTopkFunction 内存使用更高效）

#### 关键发现
- **内存优势明显**: FusedBalancedTopkFunction 平均节省 37% 内存
- **大规模稳定性**: 在 16384 tokens 的极大规模下仍然稳定运行
- **参数敏感性**: k=32 时性能提升最明显，k=8 时性能有所下降

### 精度表现（所有测试）
- **余弦相似度**: 1.000000（完美匹配）
- **最大绝对误差**: 0.00e+00（无误差）
- **数值稳定性**: 两个实现在数值上完全一致

## 结果分析

### 性能优势
1. **FusedBalancedTopkFunction** 在特定配置下表现更好（如 k=32）
2. **内存效率显著提升**：平均节省 37% 内存使用
3. **大规模稳定性好**：在极大规模（16384 tokens）下仍然稳定
4. JIT 编译优化在特定参数组合下更有效

### 参数影响分析
1. **k 值影响**：k=32 时性能提升最明显（1.18x），k=8 时反而下降
2. **专家数量**：32个专家的大规模配置下表现稳定
3. **隐藏层大小**：576-1152 范围内表现良好

### 精度保证
1. 两个实现在数值上完全等价
2. 没有精度损失
3. 可以安全地替换使用

### 实际 MOE 训练建议
1. **推荐使用 FusedBalancedTopkFunction**：内存效率明显提升
2. **最佳参数组合**：k=32, bank_size=64 在大多数情况下表现最好
3. **大规模部署**：在高内存压力下，内存节省效果尤为重要

## 环境要求

- PyTorch >= 1.8
- CUDA 支持（推荐）
- matplotlib（用于生成图表，可选）
- numpy

## 实现细节

### BalancedTopkFunction
- 标准的 PyTorch autograd.Function 实现
- 逐步计算每个操作
- 简单明了的代码结构

### FusedBalancedTopkFunction
- 使用 `@jit_fuser` 装饰器优化
- 融合多个操作减少内存访问
- 针对 torch.compile 优化

## 扩展使用

### 自定义测试配置
```python
# 修改 benchmark_balanced_topk.py 中的 test_configs
self.test_configs = [
    # (total_tokens, hidden_size, num_experts, k, bank_size)
    (your_tokens, your_hidden_size, your_experts, your_k, your_bank_size),
]
```

### 添加新的精度指标
```python
def compute_accuracy_metrics(output1, output2):
    # 添加你的自定义精度指标
    metrics['your_metric'] = your_calculation(output1, output2)
    return metrics
```

## 注意事项

1. **参数约束**: `k` 必须小于等于 `bank_size`
2. **内存使用**: 大规模测试可能需要大量 GPU 内存
3. **设备一致性**: 确保所有张量在同一设备上
4. **数据类型**: 默认使用 float16，可能影响精度

## 故障排除

### 常见错误
1. **"selected index k out of range"**: 检查 k <= bank_size
2. **CUDA out of memory**: 减少测试规模或使用 CPU
3. **导入错误**: 确保 Megatron-LM 正确安装

### 性能建议
1. 使用 CUDA 设备获得更准确的性能对比
2. 增加试验次数以获得更稳定的结果
3. 关闭其他 GPU 程序以减少干扰

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个基准测试工具。 