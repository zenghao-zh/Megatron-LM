# Lesson 05: 自定义 NVTX 标注

Megatron 内置的 NVTX 标注覆盖了 `self_attention`、`mlp` 等高层模块。但如果你新增了自定义代码（比如 INT8 量化 kernel），需要自己添加 NVTX 标注才能在时间线上精确定位。

---

## NVTX 是什么？

NVTX (NVIDIA Tools Extension) 是一套轻量级的标注 API。它不影响程序行为，只在 profiler 运行时才有意义：

```
没有 NVTX 的时间线（只看到一堆 kernel，不知道谁属于谁）:
GPU: ████ ██ ███████ ██ ████ ██████ ███ ████ ██ ███████

有 NVTX 的时间线（一目了然）:
NVTX: |-- self_attention --|-- mlp --|-- self_attention --|-- mlp --|
GPU:  ████ ██ ███████      ██ ████   ██████ ███           ████ ██ ██
```

---

## 四种添加方式

### 方式 1: 手动 push / pop

最基础，完全手动控制：

```python
torch.cuda.nvtx.range_push("my_operation")
# ... your code ...
torch.cuda.nvtx.range_pop()
```

优点：灵活。缺点：忘记 pop 会导致后续标注全部错位。

### 方式 2: Context Manager (推荐)

用 `with` 语句自动配对，不会忘记 pop：

```python
with torch.cuda.nvtx.range("my_operation"):
    # ... your code ...
```

### 方式 3: 装饰器

标注整个函数：

```python
def nvtx_annotate(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            torch.cuda.nvtx.range_push(name)
            result = func(*args, **kwargs)
            torch.cuda.nvtx.range_pop()
            return result
        return wrapper
    return decorator

@nvtx_annotate("my_function")
def my_function(x):
    return x * 2
```

### 方式 4: Megatron 内置工具

Megatron 在 `megatron/core/utils.py` 中提供了更完善的 NVTX 工具：

```python
from megatron.core.utils import nvtx_range_push, nvtx_range_pop, nvtx_decorator

# push/pop 方式
nvtx_range_push(suffix="my_operation")
# ... your code ...
nvtx_range_pop(suffix="my_operation")

# 装饰器方式
@nvtx_decorator(message="MyFunction", color="blue")
def my_function(x):
    return x * 2
```

Megatron 版本的优势：
- 有 `_nvtx_enabled` 全局开关，不 profiling 时零开销
- 自动记录调用者的模块路径作为 range 名称
- push/pop 配对检查，错位时抛异常

---

## 运行示例

```bash
cd /root/workspace/Megatron-LM

# 用 nsys 采集 NVTX 标注
nsys profile \
    --trace cuda,nvtx \
    --output nsys_profiles/nvtx_demo \
    --force-overwrite true \
    --stats true \
    python tutorials/nsight_profiling/05_custom_nvtx/nvtx_example.py
```

然后查看 NVTX 统计：

```bash
nsys stats --report nvtx_sum nsys_profiles/nvtx_demo.nsys-rep
```

你会看到类似：

```
 Time (%)  Total Time (ns)  Instances  Avg (ns)  Range Name
 --------  ---------------  ---------  --------  ----------
    35.2        234567890          3   78189296  training_step_*
    18.1        120345678          3   40115226  forward
    15.3        101234567          3   33744855  backward
     8.2         54567890          3   18189296  self_attention
     7.1         47234567          3   15744855  feed_forward
     ...
```

---

## 嵌套 Range 的层级结构

NVTX range 可以嵌套，形成树状结构：

```python
with torch.cuda.nvtx.range("training_step"):        # Level 0
    with torch.cuda.nvtx.range("forward"):           # Level 1
        with torch.cuda.nvtx.range("self_attention"): # Level 2
            with torch.cuda.nvtx.range("qkv_proj"):   # Level 3
                ...
```

在时间线中显示为：

```
Level 0: |-------------- training_step_0 --------------|
Level 1: |---- forward ----|---- backward ----|
Level 2: |-- attn --|-- ffn --|
Level 3: |qkv|score|out|  |up|down|
```

---

## 在 Megatron 中添加你自己的标注

### 例子：标注 INT8 量化操作

假设你想在 INT8 mixed-precision 的量化/反量化步骤上加标注：

```python
# 在你的量化代码中
import torch

class Int8LinearWithNVTX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, scale):
        with torch.cuda.nvtx.range("int8_quantize"):
            input_int8 = quantize_to_int8(input, scale)

        with torch.cuda.nvtx.range("int8_gemm"):
            output = torch.matmul(input_int8.float(), weight.float())

        with torch.cuda.nvtx.range("int8_dequantize"):
            output = dequantize_from_int8(output, scale)

        return output
```

### 例子：标注数据加载

```python
with torch.cuda.nvtx.range("data_loading"):
    batch = next(data_iterator)

with torch.cuda.nvtx.range("data_to_gpu"):
    batch = {k: v.cuda() for k, v in batch.items()}
```

---

## 性能注意事项

- NVTX 标注本身开销极小（纳秒级），不影响训练性能
- 但只在 `nsys profile` 运行时才会被记录
- Megatron 的 `configure_nvtx_profiling(enabled=False)` 可以彻底禁用（连纳秒级开销都没有）
- 不要在超内层循环里加标注（如逐 token 级别），太多标注反而让时间线难以阅读

---

## 下一步

现在你能精确地标注和定位任何代码区域了。去 [Lesson 06](../06_optimization_patterns/) 学习看到瓶颈后该如何优化。
