# Two-Stage Mixed-Precision INT8 GEMM 算子需求

## 目标

实现 CUDA kernel 完成带 **per-group dual-scale** 的 INT8 矩阵乘法。

当前因 cuBLAS 不支持 per-group scaling，被迫 dequant 回 BF16 再调 cuBLAS BF16 mm。4096³ 耗时 808 µs，目标 ≤ 500 µs。

## 量化方案

对 A 的每组 64 个元素（沿 K 维），按绝对值排序后：
- **top-k**（k=16）：量化到 INT8 [-127, 127]，scale = max|topk| / 127
- **others**（48 个）：量化到 INT4 范围 [-7, 7]（存储仍为 INT8），scale = max|others| / 7

量化后 A 拆分为两个互补的稀疏 INT8 矩阵，每组两套 scale。B 为标准 group-wise INT8 量化（每组一个 scale）。

## 计算公式

```
C[m, n] = Σ_g  s_b[n,g] * (
    s_topk[m,g] * Σ_{k ∈ topk(g)}   A_topk[m,k]   * B[k,n]
  + s_others[m,g] * Σ_{k ∈ others(g)} A_others[m,k] * B[k,n]
)
```

其中 `g = 0..K/64-1`，topk 和 others 互补不重叠，合并覆盖完整 group。

每个 group 需要 **两次 INT8 MMA**（共享同一个 B tile），分别乘以不同的 scale 后累加。

## 接口

```python
def scaled_int8_mm_two_stage(
    A_topk:   Tensor,  # [M, K] int8   topk 位置有值, 其余为 0
    A_others: Tensor,  # [M, K] int8   others 位置有值, 其余为 0
    B:        Tensor,  # [K, N] int8
    A_scales: Tensor,  # [M, K//64, 2] float32   [:,g,0]=s_topk  [:,g,1]=s_others
    B_scales: Tensor,  # [N, K//64]    float32
) -> Tensor:           # [M, N] bfloat16
```

**约束**: K 是 64 的倍数，group_size 固定 64，topk=16 或 8，硬件 SM80+。

## 性能分析

A800 INT8 Tensor Core 吞吐是 BF16 的 2×（624 TOPS vs 312 TFLOPS），但本算子每组需要 2 次 MMA（topk + others），所以计算量与 BF16 单次 GEMM 持平。收益来自：
- INT8 数据量是 BF16 的一半 → global memory / shared memory 加载带宽减半
- B tile 两次 MMA 共享 → 实际带宽需求约为 BF16 的 0.75×
- 软件流水线可以更好地隐藏 rescale 延迟

## 性能目标（A800）

| M×K×N | cuBLAS BF16 | 当前方案 (dequant+BF16) | 目标 |
|---|---|---|---|
| 4096×4096×4096 | 516 µs | 808 µs | **≤ 500 µs** |

## 正确性对照

`int8_mm.py` 中的 Triton 函数 `scaled_int8_mm_two_stage()` 为参考实现，cosine similarity > 0.9999。

## 集成位置

`int8_tensor.py` → `_dynamic_int8_mm_groupwise()` → `use_two_stage_mixed` 分支，替换当前的 dequant + `torch.mm`。

量化部分（A_topk / A_others / A_scales 的生成）已有高性能 CUDA kernel（`csrc/quant_int8_int4.cu`），无需重复实现。
