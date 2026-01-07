import torch
import triton
import triton.language as tl

from megatron.core.quantization.int8_training.int8_mm import (
    quantize_int8_groupwise,
    _groupwise_configs_16,
)

@triton.jit
def test_kernel(
    A_ptr, B_ptr, C_ptr, A_scale_ptr, B_scale_ptr,
    M, N, K, num_groups,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    stride_as_m, stride_as_g, stride_bs_n, stride_bs_g,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
):
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    width = GROUP_M * grid_n
    group_id = pid // width
    group_sz = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_sz)
    pid_n = (pid % width) // group_sz

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_block in range(0, K, BLOCK_K):
        rk = tl.arange(0, BLOCK_K)
        
        A = A_ptr + (ram[:, None] * stride_am + (k_block + rk[None, :]) * stride_ak)
        B = B_ptr + ((k_block + rk[:, None]) * stride_bk + rbn[None, :] * stride_bn)
        
        k_mask = (k_block + rk) < K
        a = tl.load(A, mask=k_mask[None, :], other=0)
        b = tl.load(B, mask=k_mask[:, None], other=0)
        
        partial = tl.dot(a, b).to(tl.float32)
        group_idx = k_block // BLOCK_K
        
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        
        a_scale = tl.load(
            A_scale_ptr + idx_m * stride_as_m + group_idx * stride_as_g,
            mask=idx_m < M
        ).to(tl.float32)
        b_scale = tl.load(
            B_scale_ptr + idx_n * stride_bs_n + group_idx * stride_bs_g,
            mask=idx_n < N
        ).to(tl.float32)
        
        acc += partial * a_scale * b_scale

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    
    xindex = idx_m * stride_cm + idx_n * stride_cn
    tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)


def test_specific_size(M, K, N, group_size=16, BLOCK_M=32, BLOCK_N=32, BLOCK_K=16):
    device = 'cuda'
    dtype = torch.bfloat16
    num_groups = K // group_size
    
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    ref = (A.float() @ B.float())  # Keep as float32 for comparison

    A_i8, A_scales = quantize_int8_groupwise(A, group_size)
    B_t = B.T.contiguous()
    B_t_i8, B_scales = quantize_int8_groupwise(B_t, group_size)
    B_i8 = B_t_i8.T.contiguous()

    # Use float32 for output (kernel accumulates in float32)
    C = torch.zeros(M, N, device=device, dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    test_kernel[grid](
        A_i8, B_i8, C, A_scales, B_scales,
        M, N, K, num_groups,
        *A_i8.stride(), *B_i8.stride(), *C.stride(),
        *A_scales.stride(), *B_scales.stride(),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    torch.cuda.synchronize()
    err = (C - ref).abs()
    return err.max().item(), err.mean().item()


if __name__ == "__main__":
    # Test 1: The case that was working in test_groupwise_debug.py
    print("Test 1: M=256, K=256, N=256 (same as test_groupwise_debug.py)")
    torch.manual_seed(42)
    max_err, mean_err = test_specific_size(256, 256, 256)
    print(f"Max error: {max_err:.4f}, Mean error: {mean_err:.4f}")
    
    # Test 2: The failing case
    print("\nTest 2: M=512, K=1024, N=4096 (the failing case)")
    torch.manual_seed(123)
    max_err, mean_err = test_specific_size(512, 1024, 4096)
    print(f"Max error: {max_err:.4f}, Mean error: {mean_err:.4f}")
    
    # Test 3: Different seed for the failing case
    print("\nTest 3: M=512, K=1024, N=4096 with different seed")
    torch.manual_seed(42)
    max_err, mean_err = test_specific_size(512, 1024, 4096)
    print(f"Max error: {max_err:.4f}, Mean error: {mean_err:.4f}")
    
    # Test 4: Check if num_groups matters
    print("\nTest 4: num_groups analysis")
    for K in [256, 512, 1024, 2048]:
        torch.manual_seed(42)
        max_err, mean_err = test_specific_size(512, K, 4096)
        print(f"K={K}, num_groups={K//16}: max_err={max_err:.4f}")
    
    # Test all configs
    print("\n" + "="*60)
    print("Testing all configs for M=512, K=1024, N=4096:")
    print(f"{'BLOCK_M':<10} {'BLOCK_N':<10} {'BLOCK_K':<10} {'Max Err':<12} {'Status':<10}")
    print("-" * 52)
    
    device = 'cuda'
    dtype = torch.bfloat16
    torch.manual_seed(123)
    M, K, N = 512, 1024, 4096
    group_size = 16
    num_groups = K // group_size

    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    ref = A.float() @ B.float()  # Keep as float32

    A_i8, A_scales = quantize_int8_groupwise(A, group_size)
    B_t = B.T.contiguous()
    B_t_i8, B_scales = quantize_int8_groupwise(B_t, group_size)
    B_i8 = B_t_i8.T.contiguous()

    for config in _groupwise_configs_16:
        BLOCK_M = config.kwargs['BLOCK_M']
        BLOCK_N = config.kwargs['BLOCK_N']
        BLOCK_K = config.kwargs['BLOCK_K']
        
        C = torch.zeros(M, N, device=device, dtype=torch.float32)
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        
        try:
            test_kernel[grid](
                A_i8, B_i8, C, A_scales, B_scales,
                M, N, K, num_groups,
                *A_i8.stride(), *B_i8.stride(), *C.stride(),
                *A_scales.stride(), *B_scales.stride(),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )
            torch.cuda.synchronize()
            err = (C - ref).abs()
            max_err = err.max().item()
            status = "OK" if max_err < 10 else "BAD"
            print(f"{BLOCK_M:<10} {BLOCK_N:<10} {BLOCK_K:<10} {max_err:<12.4f} {status:<10}")
        except Exception as e:
            print(f"{BLOCK_M:<10} {BLOCK_N:<10} {BLOCK_K:<10} {'ERROR':<12} {str(e)[:30]}")

