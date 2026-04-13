/*
 * CUDA kernel for INT8+INT4 two-stage groupwise quantization.
 * Optimized for group_size=64, topk_k=8 or 16.
 *
 * Key optimizations over the Triton baseline:
 *   1. Warp-level bitonic sort entirely in registers (21 compare-swap steps,
 *      zero shared memory) to find the top-k threshold.
 *   2. Warp shuffle primitives for all inter-thread communication.
 *   3. Fused pre-split output: directly writes A_topk and A_others, removing
 *      the need for a separate torch.where masking pass before matmul.
 *   4. Native BF16 / FP16 input support (reads 2 bytes instead of 4).
 *   5. Vectorised 2-element loads per thread → perfectly coalesced memory.
 *
 * Two output modes (compile-time template switch):
 *   FUSED  = true  → (topk_out, others_out, scales)   for direct matmul
 *   FUSED  = false → (combined_out, scales, mask)      for API compatibility
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define WARP_SIZE       32
#define FULL_MASK       0xFFFFFFFFu
#define GROUP_SIZE      64
#define WARPS_PER_BLOCK 8

/* ================================================================
 *  to_float: type-dispatched device load → float
 * ================================================================ */
template <typename T>
__device__ __forceinline__ float to_float(const T* ptr, int idx);

template <>
__device__ __forceinline__ float to_float<float>(const float* ptr, int idx) {
    return ptr[idx];
}
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(
        const __nv_bfloat16* ptr, int idx) {
    return __bfloat162float(ptr[idx]);
}
template <>
__device__ __forceinline__ float to_float<__half>(
        const __half* ptr, int idx) {
    return __half2float(ptr[idx]);
}

/* ================================================================
 *  Warp-level bitonic sort: 64 elements, descending, in-register
 *
 *  Each of the 32 lanes holds 2 values (v0 at position 2*lane,
 *  v1 at position 2*lane+1). The full network has 6 stages and
 *  21 compare-swap steps, all executed via warp shuffles (d>=2)
 *  or local register swaps (d==1).
 *
 *  Element distance d maps to warp shuffle delta = d/2 (for d>=2)
 *  because element p lives in lane p/2, register p%2.
 * ================================================================ */

/* Local (d=1) compare-swap: v0 vs v1 within a lane. */
#define BSORT_LOCAL(s) do {                                     \
    int _dir = ((lane << 1) >> (s)) & 1;                       \
    float _hi = fmaxf(v0, v1), _lo = fminf(v0, v1);           \
    v0 = _dir ? _lo : _hi;                                     \
    v1 = _dir ? _hi : _lo;                                     \
} while (0)

/*
 * Shuffle (d>=2) compare-swap: exchange between lanes.
 * For descending sort, a pair at element distance d:
 *   lower position takes max if block is descending (dir=0)
 *   lower position takes min if block is ascending  (dir=1)
 *   → take_max = lower XOR ascending
 */
#define BSORT_SHFL(s, d) do {                                          \
    int _delta = (d) >> 1;                                             \
    float _o0 = __shfl_xor_sync(FULL_MASK, v0, _delta);               \
    float _o1 = __shfl_xor_sync(FULL_MASK, v1, _delta);               \
    {                                                                  \
        int _p = lane << 1;                                            \
        bool _lower = (_p & (d)) == 0;                                 \
        bool _asc   = ((_p >> (s)) & 1) != 0;                         \
        float _h = fmaxf(v0, _o0), _l = fminf(v0, _o0);              \
        v0 = (_lower ^ _asc) ? _h : _l;                               \
    }                                                                  \
    {                                                                  \
        int _p = (lane << 1) | 1;                                     \
        bool _lower = (_p & (d)) == 0;                                 \
        bool _asc   = ((_p >> (s)) & 1) != 0;                         \
        float _h = fmaxf(v1, _o1), _l = fminf(v1, _o1);              \
        v1 = (_lower ^ _asc) ? _h : _l;                               \
    }                                                                  \
} while (0)

__device__ __forceinline__ void bitonic_sort_64_desc(
        float& v0, float& v1, int lane) {
    /* stage 1  (block=2,  1 step)  */  BSORT_LOCAL(1);
    /* stage 2  (block=4,  2 steps) */  BSORT_SHFL(2,2);   BSORT_LOCAL(2);
    /* stage 3  (block=8,  3 steps) */  BSORT_SHFL(3,4);   BSORT_SHFL(3,2);
                                        BSORT_LOCAL(3);
    /* stage 4  (block=16, 4 steps) */  BSORT_SHFL(4,8);   BSORT_SHFL(4,4);
                                        BSORT_SHFL(4,2);   BSORT_LOCAL(4);
    /* stage 5  (block=32, 5 steps) */  BSORT_SHFL(5,16);  BSORT_SHFL(5,8);
                                        BSORT_SHFL(5,4);   BSORT_SHFL(5,2);
                                        BSORT_LOCAL(5);
    /* stage 6  (block=64, 6 steps) */  BSORT_SHFL(6,32);  BSORT_SHFL(6,16);
                                        BSORT_SHFL(6,8);   BSORT_SHFL(6,4);
                                        BSORT_SHFL(6,2);   BSORT_LOCAL(6);
}

/* ================================================================
 *  Warp-level primitives
 * ================================================================ */

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, o));
    return v;
}

__device__ __forceinline__ int warp_reduce_sum(int v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(FULL_MASK, v, o);
    return v;
}

/* Inclusive prefix-sum within a warp, then subtract own value → exclusive. */
__device__ __forceinline__ int warp_exclusive_prefix_sum(int v, int lane) {
    int s = v;
    #pragma unroll
    for (int d = 1; d < WARP_SIZE; d <<= 1) {
        int n = __shfl_up_sync(FULL_MASK, s, d);
        if (lane >= d) s += n;
    }
    return s - v;
}

/* ================================================================
 *  Main quantization kernel
 *
 *  Grid : (ceil(M*num_groups / WARPS_PER_BLOCK))
 *  Block: (WARPS_PER_BLOCK * 32)
 *
 *  Each warp handles one (row, group) pair.
 * ================================================================ */

template <typename scalar_t, int TOPK_K, bool FUSED>
__global__ void quant_int8_int4_kernel(
        const scalar_t* __restrict__ input,
        int8_t*  __restrict__ out_a,       // topk (FUSED) or combined (!FUSED)
        int8_t*  __restrict__ out_b,       // others (FUSED) or unused (!FUSED)
        float*   __restrict__ scales_out,  // [M, num_groups, 2]
        bool*    __restrict__ mask_out,    // nullptr (FUSED) or [M, ng, 64]
        int M, int K, int num_groups, float eps)
{
    const int lane  = threadIdx.x & 31;
    const int gwarp = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (gwarp >= M * num_groups) return;

    const int row = gwarp / num_groups;
    const int grp = gwarp % num_groups;
    const int bk  = grp * GROUP_SIZE;
    const int i0  = row * K + bk + (lane << 1);

    const bool v0_ok = bk + (lane << 1)     < K;
    const bool v1_ok = bk + (lane << 1) + 1 < K;
    float val0 = v0_ok ? to_float(input, i0)     : 0.f;
    float val1 = v1_ok ? to_float(input, i0 + 1) : 0.f;
    float a0 = fabsf(val0), a1 = fabsf(val1);

    /* ---------- bitonic sort on abs copies ---------- */
    float sv0 = a0, sv1 = a1;
    bitonic_sort_64_desc(sv0, sv1, lane);

    /* Threshold = (TOPK_K)-th largest absolute value */
    const int thr_lane = (TOPK_K - 1) >> 1;
    const int thr_reg  = (TOPK_K - 1) & 1;
    float threshold = __shfl_sync(FULL_MASK, thr_reg ? sv1 : sv0, thr_lane);
    float max_topk  = __shfl_sync(FULL_MASK, sv0, 0);

    /* ---------- build top-k mask with stable tie-breaking ---------- */
    bool ab0 = a0 > threshold, ab1 = a1 > threshold;
    int  tot_above = warp_reduce_sum((int)ab0 + (int)ab1);
    int  need      = TOPK_K - tot_above;
    bool tie0 = (a0 == threshold), tie1 = (a1 == threshold);
    int  pfx  = warp_exclusive_prefix_sum((int)tie0 + (int)tie1, lane);
    bool sel0 = tie0 && (pfx                < need);
    bool sel1 = tie1 && (pfx + (int)tie0    < need);
    bool tk0  = ab0 || sel0;
    bool tk1  = ab1 || sel1;

    /* ---------- scale computation ---------- */
    float max_oth = warp_reduce_max(
            fmaxf(tk0 ? 0.f : a0, tk1 ? 0.f : a1));

    float sc_tk  = max_topk / 127.f;
    float sc_ot  = max_oth  / 7.f;
    float inv_tk = 1.f / fmaxf(sc_tk, eps);
    float inv_ot = 1.f / fmaxf(sc_ot, eps);

    /* ---------- quantize ---------- */
    int q0 = __float2int_rn(val0 * (tk0 ? inv_tk : inv_ot));
    int q1 = __float2int_rn(val1 * (tk1 ? inv_tk : inv_ot));
    int8_t c0 = (int8_t)(tk0 ? max(-127, min(127, q0)) : max(-7, min(7, q0)));
    int8_t c1 = (int8_t)(tk1 ? max(-127, min(127, q1)) : max(-7, min(7, q1)));

    /* ---------- store ---------- */
    if (FUSED) {
        if (v0_ok) { out_a[i0]   = tk0 ? c0 : (int8_t)0;
                     out_b[i0]   = tk0 ? (int8_t)0 : c0; }
        if (v1_ok) { out_a[i0+1] = tk1 ? c1 : (int8_t)0;
                     out_b[i0+1] = tk1 ? (int8_t)0 : c1; }
    } else {
        if (v0_ok) out_a[i0]   = c0;
        if (v1_ok) out_a[i0+1] = c1;
        int mi = row * num_groups * GROUP_SIZE + grp * GROUP_SIZE + (lane << 1);
        if (v0_ok) mask_out[mi]   = tk0;
        if (v1_ok) mask_out[mi+1] = tk1;
    }

    if (lane == 0) {
        int si = (row * num_groups + grp) << 1;
        scales_out[si]     = sc_tk;
        scales_out[si + 1] = sc_ot;
    }
}

/* ================================================================
 *  Kernel launcher (handles dtype + topk_k dispatch)
 * ================================================================ */
template <bool FUSED>
static void launch_kernel(
        torch::Tensor input,
        int8_t* out_a, int8_t* out_b,
        float* scales, bool* mask,
        int M, int K, int ng, int topk_k, float eps) {

    int total_warps = M * ng;
    int num_blocks  = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(num_blocks);
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto ic = input.contiguous();

#define DISPATCH_TOPK(scalar_type, ptr, TK) \
    quant_int8_int4_kernel<scalar_type, TK, FUSED> \
        <<<grid, block, 0, stream>>>( \
            ptr, out_a, out_b, scales, mask, M, K, ng, eps)

#define DISPATCH_DTYPE(TK) do {                                      \
    if (ic.scalar_type() == torch::kBFloat16) {                      \
        auto* p = reinterpret_cast<const __nv_bfloat16*>(            \
                      ic.data_ptr());                                \
        DISPATCH_TOPK(__nv_bfloat16, p, TK);                        \
    } else if (ic.scalar_type() == torch::kFloat16) {                \
        auto* p = reinterpret_cast<const __half*>(ic.data_ptr());    \
        DISPATCH_TOPK(__half, p, TK);                                \
    } else {                                                         \
        auto f32 = ic.to(torch::kFloat32);                           \
        DISPATCH_TOPK(float, f32.data_ptr<float>(), TK);            \
    }                                                                \
} while (0)

    if      (topk_k == 16) { DISPATCH_DTYPE(16); }
    else if (topk_k ==  8) { DISPATCH_DTYPE( 8); }
    else    TORCH_CHECK(false, "topk_k must be 8 or 16, got ", topk_k);

#undef DISPATCH_DTYPE
#undef DISPATCH_TOPK

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

/* ================================================================
 *  Python-visible wrappers
 * ================================================================ */

/*
 * Fused mode: returns (topk_out [M,K] i8, others_out [M,K] i8,
 *                       scales [M, ng, 2] fp32)
 * Directly usable by the two-stage matmul kernel without pre-split.
 */
std::vector<torch::Tensor> quantize_fused(
        torch::Tensor input, int64_t topk_k, double eps) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2-D [M, K]");
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(K % GROUP_SIZE == 0, "K must be divisible by ", GROUP_SIZE);
    int ng = K / GROUP_SIZE;

    auto oi8  = torch::TensorOptions().dtype(torch::kInt8).device(input.device());
    auto of32 = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto topk   = torch::empty({M, K}, oi8);
    auto others = torch::empty({M, K}, oi8);
    auto scales = torch::empty({M, ng, 2}, of32);

    launch_kernel<true>(input,
        topk.data_ptr<int8_t>(), others.data_ptr<int8_t>(),
        scales.data_ptr<float>(), nullptr,
        M, K, ng, topk_k, (float)eps);

    return {topk, others, scales};
}

/*
 * Compat mode: returns (combined [M,K] i8, scales [M, ng, 2] fp32,
 *                        mask [M, ng, 64] bool)
 * Drop-in replacement for the existing Python / Triton quantize function.
 */
std::vector<torch::Tensor> quantize_compat(
        torch::Tensor input, int64_t topk_k, double eps) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2-D [M, K]");
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(K % GROUP_SIZE == 0, "K must be divisible by ", GROUP_SIZE);
    int ng = K / GROUP_SIZE;

    auto oi8  = torch::TensorOptions().dtype(torch::kInt8).device(input.device());
    auto of32 = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto ob   = torch::TensorOptions().dtype(torch::kBool).device(input.device());
    auto combined = torch::empty({M, K}, oi8);
    auto scales   = torch::empty({M, ng, 2}, of32);
    auto mask     = torch::empty({M, ng, GROUP_SIZE}, ob);

    launch_kernel<false>(input,
        combined.data_ptr<int8_t>(), nullptr,
        scales.data_ptr<float>(), mask.data_ptr<bool>(),
        M, K, ng, topk_k, (float)eps);

    return {combined, scales, mask};
}

/* =====================================================================
 * Dequant kernels: convert quantised INT8 back to BF16 in a single pass.
 * Used by the "CUDA quant → dequant → cuBLAS BF16 mm" fast-path.
 * ===================================================================== */

__global__ void dequant_a_kernel(
        const int8_t* __restrict__ A_topk,
        const int8_t* __restrict__ A_others,
        const float*  __restrict__ A_scales,   /* [M, ng, 2] */
        __nv_bfloat16* __restrict__ A_dq,       /* [M, K] */
        int MK, int K, int ng, int gs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= MK) return;
    int k = idx % K;
    int m = idx / K;
    int g = k / gs;
    int si = (m * ng + g) * 2;
    float val = (float)A_topk[idx] * A_scales[si]
              + (float)A_others[idx] * A_scales[si + 1];
    A_dq[idx] = __float2bfloat16(val);
}

__global__ void dequant_b_kernel(
        const int8_t* __restrict__ B,
        const float*  __restrict__ B_scales,   /* [N, ng] */
        __nv_bfloat16* __restrict__ B_dq,       /* [K, N] */
        int KN, int N, int ng, int gs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= KN) return;
    int n = idx % N;
    int k = idx / N;
    int g = k / gs;
    B_dq[idx] = __float2bfloat16((float)B[idx] * B_scales[n * ng + g]);
}

torch::Tensor dequant_a_cuda(
        torch::Tensor A_topk, torch::Tensor A_others,
        torch::Tensor A_scales, int64_t group_size) {
    int M = A_topk.size(0), K = A_topk.size(1);
    int ng = K / group_size;
    auto out = torch::empty({M, K}, torch::TensorOptions()
                   .dtype(torch::kBFloat16).device(A_topk.device()));
    int n = M * K;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dequant_a_kernel<<<blocks, threads, 0, stream>>>(
        A_topk.data_ptr<int8_t>(), A_others.data_ptr<int8_t>(),
        A_scales.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        n, K, ng, (int)group_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor dequant_b_cuda(
        torch::Tensor B, torch::Tensor B_scales, int64_t group_size) {
    int K = B.size(0), N = B.size(1);
    int ng = K / group_size;
    auto out = torch::empty({K, N}, torch::TensorOptions()
                   .dtype(torch::kBFloat16).device(B.device()));
    int n = K * N;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dequant_b_kernel<<<blocks, threads, 0, stream>>>(
        B.data_ptr<int8_t>(), B_scales.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        n, N, ng, (int)group_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

/* ================================================================
 *  Fused quant+dequant for B: groupwise INT8 round-trip
 *
 *  Replaces the two-step (Triton quantize → CUDA dequant) pipeline
 *  with a single CUDA kernel.  Each thread owns one column and
 *  iterates over group_size rows twice: pass 1 finds max-abs,
 *  pass 2 quantizes+dequants.  The second read hits L2 cache.
 *
 *  Grid  : (num_groups, cdiv(N, BLOCK_N))
 *  Block : (BLOCK_N)
 * ================================================================ */

template <typename scalar_t>
__global__ void quant_dequant_b_fused_kernel(
        const scalar_t* __restrict__ B,
        __nv_bfloat16*  __restrict__ B_dq,
        int K, int N, int gs, float eps)
{
    const int g = blockIdx.x;
    const int n = blockIdx.y * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const int k_start = g * gs;
    const int k_end   = min(k_start + gs, K);

    float max_abs = 0.f;
    for (int k = k_start; k < k_end; ++k)
        max_abs = fmaxf(max_abs, fabsf(to_float(B, k * N + n)));

    float scale     = max_abs / 127.f;
    float inv_scale = 1.f / fmaxf(scale, eps);

    for (int k = k_start; k < k_end; ++k) {
        int   idx = k * N + n;
        float val = to_float(B, idx);
        int   q   = __float2int_rn(val * inv_scale);
        q = max(-127, min(127, q));
        B_dq[idx] = __float2bfloat16((float)q * scale);
    }
}

torch::Tensor quant_dequant_b_fused(
        torch::Tensor B, int64_t group_size) {
    TORCH_CHECK(B.dim() == 2, "B must be 2-D [K, N]");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA");
    int K = B.size(0), N = B.size(1);
    TORCH_CHECK(K % group_size == 0,
                "K must be divisible by group_size");
    int ng = K / group_size;

    auto out = torch::empty({K, N}, torch::TensorOptions()
                   .dtype(torch::kBFloat16).device(B.device()));

    constexpr int BLOCK_N = 256;
    dim3 grid(ng, (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(BLOCK_N);
    float eps = 1e-20f;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto Bc = B.contiguous();
    auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());

    if (Bc.scalar_type() == torch::kBFloat16) {
        quant_dequant_b_fused_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(Bc.data_ptr()),
            out_ptr, K, N, (int)group_size, eps);
    } else if (Bc.scalar_type() == torch::kFloat16) {
        quant_dequant_b_fused_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(Bc.data_ptr()),
            out_ptr, K, N, (int)group_size, eps);
    } else {
        auto f32 = Bc.to(torch::kFloat32);
        quant_dequant_b_fused_kernel<<<grid, block, 0, stream>>>(
            f32.data_ptr<float>(), out_ptr, K, N, (int)group_size, eps);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_fused",  &quantize_fused,
          "INT8+INT4 two-stage quantize → (topk, others, scales)");
    m.def("quantize_compat", &quantize_compat,
          "INT8+INT4 two-stage quantize → (combined, scales, mask)");
    m.def("dequant_a", &dequant_a_cuda,
          "Dequant two-stage INT8 A → BF16 (single-pass)");
    m.def("dequant_b", &dequant_b_cuda,
          "Dequant group-wise INT8 B → BF16 (single-pass)");
    m.def("quant_dequant_b", &quant_dequant_b_fused,
          "Fused quant+dequant groupwise INT8 B → BF16 (single kernel)");
}
