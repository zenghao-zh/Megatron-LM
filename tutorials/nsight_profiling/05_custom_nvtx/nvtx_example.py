"""
Lesson 05: 自定义 NVTX 标注示例

本脚本演示三种在 PyTorch 代码中添加 NVTX 标注的方式，
使得在 nsys 时间线中可以精确看到每个代码区域的耗时。

运行方式:
    nsys profile --trace cuda,nvtx -o nvtx_demo python \
        tutorials/nsight_profiling/05_custom_nvtx/nvtx_example.py
"""

import torch
import torch.nn as nn

# ===================================================================
# 方式 1: torch.cuda.nvtx.range_push / range_pop
#         最基础的方式，手动 push/pop
# ===================================================================

def matmul_with_manual_nvtx(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    torch.cuda.nvtx.range_push("quantize_input")
    a_int8 = a.to(torch.int8)
    b_int8 = b.to(torch.int8)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("int8_matmul")
    result = torch.matmul(a_int8.float(), b_int8.float())
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("dequantize_output")
    output = result.to(torch.bfloat16)
    torch.cuda.nvtx.range_pop()

    return output


# ===================================================================
# 方式 2: torch.cuda.nvtx.range (context manager)
#         用 with 语句，自动配对 push/pop，不会忘记 pop
# ===================================================================

def matmul_with_context_manager(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    with torch.cuda.nvtx.range("quantize_input"):
        a_int8 = a.to(torch.int8)
        b_int8 = b.to(torch.int8)

    with torch.cuda.nvtx.range("int8_matmul"):
        result = torch.matmul(a_int8.float(), b_int8.float())

    with torch.cuda.nvtx.range("dequantize_output"):
        output = result.to(torch.bfloat16)

    return output


# ===================================================================
# 方式 3: 装饰器方式（适合标注整个函数）
#         用 Megatron 的 nvtx_decorator 或自己写一个
# ===================================================================

def nvtx_annotate(name: str):
    """简单的 NVTX 装饰器。Megatron 中有更完善的版本: megatron.core.utils.nvtx_decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            torch.cuda.nvtx.range_push(name)
            result = func(*args, **kwargs)
            torch.cuda.nvtx.range_pop()
            return result
        return wrapper
    return decorator


@nvtx_annotate("my_custom_layer_forward")
def custom_forward(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight)


# ===================================================================
# 方式 4: 嵌套 NVTX Range（在时间线上形成层级结构）
# ===================================================================

class AnnotatedTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ffn_up = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.ffn_down = nn.Linear(4 * hidden_size, hidden_size, bias=False)
        self.norm1 = nn.RMSNorm(hidden_size)
        self.norm2 = nn.RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 外层 range: 整个 block
        torch.cuda.nvtx.range_push("transformer_block")

        residual = x

        # 嵌套 range: attention 子模块
        with torch.cuda.nvtx.range("self_attention"):
            x = self.norm1(x)

            with torch.cuda.nvtx.range("qkv_projection"):
                qkv = self.attn_qkv(x)

            with torch.cuda.nvtx.range("attention_score"):
                q, k, v = qkv.chunk(3, dim=-1)
                scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
                attn = torch.softmax(scores, dim=-1)
                x = torch.matmul(attn, v)

            with torch.cuda.nvtx.range("output_projection"):
                x = self.attn_out(x)

        x = x + residual
        residual = x

        # 嵌套 range: FFN 子模块
        with torch.cuda.nvtx.range("feed_forward"):
            x = self.norm2(x)

            with torch.cuda.nvtx.range("ffn_up"):
                x = self.ffn_up(x)
                x = torch.nn.functional.gelu(x)

            with torch.cuda.nvtx.range("ffn_down"):
                x = self.ffn_down(x)

        x = x + residual

        torch.cuda.nvtx.range_pop()  # transformer_block
        return x


# ===================================================================
# 主函数：运行所有示例
# ===================================================================

def main():
    device = "cuda"
    torch.manual_seed(42)

    print("=== NVTX Annotation Demo ===\n")

    # 方式 1: Manual push/pop
    print("1. Manual push/pop...")
    a = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
    b = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
    with torch.cuda.nvtx.range("demo_manual_pushpop"):
        result = matmul_with_manual_nvtx(a, b)

    # 方式 2: Context manager
    print("2. Context manager...")
    with torch.cuda.nvtx.range("demo_context_manager"):
        result = matmul_with_context_manager(a, b)

    # 方式 3: Decorator
    print("3. Decorator...")
    weight = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
    x = torch.randn(16, 128, 1024, device=device, dtype=torch.bfloat16)
    with torch.cuda.nvtx.range("demo_decorator"):
        result = custom_forward(x, weight)

    # 方式 4: Nested ranges (模拟训练 step)
    print("4. Nested ranges (simulated training)...")
    model = AnnotatedTransformerBlock(hidden_size=1024).to(device).to(torch.bfloat16)
    x = torch.randn(4, 128, 1024, device=device, dtype=torch.bfloat16)

    for step in range(3):
        with torch.cuda.nvtx.range(f"training_step_{step}"):

            with torch.cuda.nvtx.range("forward"):
                output = model(x)
                loss = output.sum()

            with torch.cuda.nvtx.range("backward"):
                loss.backward()

    torch.cuda.synchronize()
    print("\nDone! Open the .nsys-rep file to see the NVTX ranges in the timeline.")


if __name__ == "__main__":
    main()
