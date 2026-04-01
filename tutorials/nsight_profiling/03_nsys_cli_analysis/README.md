# Lesson 03: 纯 CLI 分析 -- 不用 GUI 也能读懂 profiling 结果

远程服务器通常没有桌面环境。`nsys stats` 命令能直接在终端里提取所有关键指标。

---

## 快速开始

运行 Lesson 01 的脚本拿到 `.nsys-rep` 文件后：

```bash
# 方式 1: 用本课提供的一键分析脚本
bash tutorials/nsight_profiling/03_nsys_cli_analysis/analyze_nsys_report.sh \
     nsys_profiles/smollm-360m-nsys-demo_0.nsys-rep

# 方式 2: 手动运行单条命令
nsys stats --report cuda_gpu_kern_sum nsys_profiles/smollm-360m-nsys-demo_0.nsys-rep
```

---

## nsys stats 核心报告类型

### 报告 1: `cuda_gpu_kern_sum` -- GPU Kernel 排行榜

```bash
nsys stats --report cuda_gpu_kern_sum <file.nsys-rep>
```

输出示例：

```
 Time (%)  Total Time (ns)  Instances  Avg (ns)    Med (ns)    Min (ns)   Max (ns)   Name
 --------  ---------------  ---------  ----------  ----------  ---------  ---------  ----
    45.2      892345678          384    2323816     2301234     2100000    2600000    sm80_xmma_gemm_bf16...
    12.1      238765432           96    2487140     2456789     2300000    2700000    flash_fwd_kernel...
     8.3      163876543           96    1707047     1698765     1600000    1800000    ncclKernel_AllReduce...
     6.5      128345678          192     668467      654321      600000     750000    void quantize_int8...
     ...
```

**怎么读：**
- `Time (%)`: 该 kernel 类型占总 GPU 时间的百分比 -- **最重要的列**
- `Instances`: 该 kernel 被调用了多少次
- `Avg (ns)`: 平均每次执行时间
- 找到占比最高的 kernel，就是优化的首要目标

### 报告 2: `cuda_api_sum` -- CPU 端 CUDA API 调用

```bash
nsys stats --report cuda_api_sum <file.nsys-rep>
```

**怎么读：**
- `cudaLaunchKernel`: kernel 启动次数和耗时。如果总时间很长，说明 kernel 太多太碎
- `cudaMemcpyAsync`: 数据传输。正常情况下占比应该很小
- `cudaStreamSynchronize` / `cudaDeviceSynchronize`: 同步等待。如果占比大，说明有不必要的同步

### 报告 3: `nvtx_sum` -- NVTX 逻辑区域汇总

```bash
nsys stats --report nvtx_sum <file.nsys-rep>
```

**怎么读：**
- 能看到 `self_attention`、`mlp`、`self_attn_bda`、`mlp_bda` 各自的总耗时
- 对比 Forward 各模块的时间占比
- 如果 `mlp` 远大于 `self_attention`，说明 FFN 是计算主体（对于 SwiGLU 这很正常）

### 报告 4: `gpu_mem_size_sum` -- 内存操作

```bash
nsys stats --report gpu_mem_size_sum <file.nsys-rep>
```

> 注：需要采集时加 `--cuda-memory-usage true` 才有数据。

---

## 进阶：导出 SQLite 做自定义分析

`nsys-rep` 文件本质上是一个数据库。导出为 SQLite 后可以用 SQL 随意查询：

```bash
nsys export --type sqlite --output report.sqlite <file.nsys-rep>
```

### 常用 SQL 查询

#### 查询 Top-10 最耗时 kernel

```sql
sqlite3 report.sqlite <<'SQL'
SELECT
    SUBSTR(shortName, 1, 80) AS kernel_name,
    COUNT(*) AS call_count,
    SUM(end - start) / 1e6 AS total_ms,
    AVG(end - start) / 1e6 AS avg_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL
GROUP BY shortName
ORDER BY total_ms DESC
LIMIT 10;
SQL
```

#### 计算 NCCL 通信占比

```sql
sqlite3 report.sqlite <<'SQL'
SELECT
    ROUND(
        100.0 * SUM(CASE WHEN shortName LIKE '%nccl%' THEN end - start ELSE 0 END)
        / SUM(end - start),
        2
    ) AS nccl_percent,
    ROUND(
        SUM(CASE WHEN shortName LIKE '%nccl%' THEN end - start ELSE 0 END) / 1e6,
        2
    ) AS nccl_total_ms,
    ROUND(SUM(end - start) / 1e6, 2) AS all_kernel_total_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL;
SQL
```

#### 查询 NVTX Range 耗时

```sql
sqlite3 report.sqlite <<'SQL'
SELECT
    SUBSTR(text, 1, 60) AS range_name,
    COUNT(*) AS count,
    ROUND(SUM(end - start) / 1e6, 2) AS total_ms,
    ROUND(AVG(end - start) / 1e6, 2) AS avg_ms
FROM NVTX_EVENTS
WHERE eventType = 59  -- NVTX push/pop range
GROUP BY text
ORDER BY total_ms DESC
LIMIT 20;
SQL
```

#### 检测 GPU 空闲间隙

```sql
sqlite3 report.sqlite <<'SQL'
-- 找出连续两个 kernel 之间的空隙（大于 10us 的）
WITH ordered_kernels AS (
    SELECT
        start, end,
        LEAD(start) OVER (ORDER BY start) AS next_start
    FROM CUPTI_ACTIVITY_KIND_KERNEL
)
SELECT
    COUNT(*) AS gap_count,
    ROUND(AVG(next_start - end) / 1e3, 2) AS avg_gap_us,
    ROUND(MAX(next_start - end) / 1e3, 2) AS max_gap_us,
    ROUND(SUM(next_start - end) / 1e6, 2) AS total_gap_ms
FROM ordered_kernels
WHERE next_start - end > 10000;  -- 大于 10 微秒
SQL
```

---

## 一键分析脚本

本课提供的 `analyze_nsys_report.sh` 会依次执行上述所有报告，并自动导出 SQLite：

```bash
bash tutorials/nsight_profiling/03_nsys_cli_analysis/analyze_nsys_report.sh \
     nsys_profiles/smollm-360m-nsys-demo_0.nsys-rep
```

---

## 关键指标 Checklist

跑完分析后，记录以下数字：

| 指标 | 命令/查询 | 健康范围 |
|------|----------|---------|
| Top-1 kernel 占比 | `cuda_gpu_kern_sum` 第一行 | GEMM 通常 40-60% |
| NCCL 通信占比 | SQLite 查询 | < 15% 为佳 (TP=1 时) |
| cudaLaunchKernel 次数 | `cuda_api_sum` | 越少越好，太多说明 kernel 太碎 |
| 最大 GPU 空闲间隙 | SQLite 查询 | < 100us 为佳 |
| cudaDeviceSynchronize 次数 | `cuda_api_sum` | 应该为 0（训练中不应有显式同步） |

---

## 下一步

发现某个特定 kernel 耗时异常？去 [Lesson 04](../04_ncu_kernel_deep_dive/) 用 Nsight Compute 深入分析它的微架构行为。
