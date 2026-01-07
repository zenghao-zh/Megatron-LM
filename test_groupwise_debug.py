import torch
import triton
import triton.language as tl

@triton.jit
def debug_groupwise_kernel(
    A_ptr, B_ptr, C_ptr, A_scale_ptr, B_scale_ptr,
    M, N, K, num_groups,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    stride_as_m, stride_as_g, stride_bs_n, stride_bs_g,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
):
    """Debug version of groupwise kernel."""
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # Re-order program ID
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
        
        # BLOCK_K == GROUP_SIZE, so group_idx = k_block // BLOCK_K
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


def test_kernel():
    device = 'cuda'
    dtype = torch.bfloat16
    
    M, K, N = 256, 256, 256
    group_size = 16
    num_groups = K // group_size
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, group_size  # Use small blocks
    
    # Create test data
    torch.manual_seed(42)
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    
    # Quantize
    from megatron.core.quantization.int8_training.int8_mm import quantize_int8_groupwise
    
    A_i8, A_scales = quantize_int8_groupwise(A, group_size)
    B_t = B.T.contiguous()
    B_t_i8, B_scales = quantize_int8_groupwise(B_t, group_size)
    B_i8 = B_t_i8.T.contiguous()
    
    print(f"M={M}, K={K}, N={N}, group_size={group_size}, num_groups={num_groups}")
    print(f"A_i8: {A_i8.shape}, stride={A_i8.stride()}")
    print(f"B_i8: {B_i8.shape}, stride={B_i8.stride()}")
    print(f"A_scales: {A_scales.shape}, stride={A_scales.stride()}")
    print(f"B_scales: {B_scales.shape}, stride={B_scales.stride()}")
    
    # Check B is contiguous with expected stride
    print(f"B_i8 is contiguous: {B_i8.is_contiguous()}")
    print(f"Expected stride for B_i8 [K,N]: ({N}, 1)")
    
    # Make sure everything is contiguous
    A_i8 = A_i8.contiguous()
    B_i8 = B_i8.contiguous()
    A_scales = A_scales.contiguous()
    B_scales = B_scales.contiguous()
    
    C = torch.zeros(M, N, device=device, dtype=torch.float32)
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    try:
        debug_groupwise_kernel[grid](
            A_i8, B_i8, C, A_scales, B_scales,
            M, N, K, num_groups,
            *A_i8.stride(), *B_i8.stride(), *C.stride(),
            *A_scales.stride(), *B_scales.stride(),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        torch.cuda.synchronize()
        print("SUCCESS!")
        
        # Compare with reference
        ref = (A.float() @ B.float()).to(dtype)
        err = (C.to(dtype) - ref).abs()
        print(f"Max error: {err.max().item():.4f}")
        print(f"Mean error: {err.mean().item():.4f}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_kernel()

