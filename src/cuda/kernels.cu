// tinyinfer - CUDA kernels for single-token (decode) Llama inference.
//
// Decode is memory-bound: every weight is read exactly once per token, so the kernels are
// designed around streaming weights at full DRAM bandwidth:
//   * gemv: one warp per output row, 128-bit coalesced weight loads, x staged in shared memory
//           with a bank-conflict-free padded layout, int->float via exponent tricks (no I2F).
//   * attention: single-pass "flash-decoding" with online softmax; each warp owns a slice of the
//           timeline and the partial (max, sum, acc) states are merged at the end.
//   * every kernel reads `token`/`pos` from device memory so one decode step can be captured
//     once into a CUDA graph and replayed for every token.
#include <cfloat>
#include <stdexcept>
#include <string>

#include "kernels.cuh"

namespace tinf {
namespace gpu {

namespace {

constexpr int kWarp = 32;
constexpr int kGemvWarps = 8;  // rows per block
constexpr int kAttnWarps = 8;

int g_max_dyn_smem = 48 * 1024;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// padded shared-memory index: one float of padding every 32 floats -> conflict-free strided reads
__device__ __forceinline__ int pidx(int i) { return i + (i >> 5); }

// exact int->float conversions using the 2^23 exponent trick (FADD instead of the slower I2F)
__device__ __forceinline__ float u8_biased_to_float(uint32_t v) {  // v in [0,255] -> v - 128
    return __uint_as_float(0x4B000000u | v) - 8388736.0f;
}
__device__ __forceinline__ float u4_to_float(uint32_t n) {  // n in [0,15] -> n - 8
    return __uint_as_float(0x4B000000u | n) - 8388616.0f;
}

// ------------------------------------------------------------------------------------------------
// GEMV
// ------------------------------------------------------------------------------------------------
template <int Q>  // 0 = f16, 1 = q8, 2 = q4
__global__ void __launch_bounds__(kGemvWarps* kWarp)
    gemv_kernel(const void* __restrict__ W, const __half* __restrict__ S, const float* __restrict__ x,
                float* __restrict__ y, int rows, int cols, int group, int accumulate) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[pidx(i)] = x[i];
    __syncthreads();

    const int warp = threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    const int row = blockIdx.x * kGemvWarps + warp;
    if (row >= rows) return;

    float acc = 0.f;
    if constexpr (Q == 0) {
        const uint4* w = reinterpret_cast<const uint4*>(static_cast<const __half*>(W) + (size_t)row * cols);
        const int nchunks = cols / 8;  // 8 halves per 16-byte chunk
#pragma unroll 4
        for (int j = lane; j < nchunks; j += kWarp) {
            uint4 p = __ldg(w + j);
            const __half2* h = reinterpret_cast<const __half2*>(&p);
            const int e = j * 8;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 f = __half22float2(h[k]);
                acc = fmaf(f.x, xs[pidx(e + 2 * k)], acc);
                acc = fmaf(f.y, xs[pidx(e + 2 * k + 1)], acc);
            }
        }
    } else if constexpr (Q == 1) {
        const uint4* w = reinterpret_cast<const uint4*>(static_cast<const int8_t*>(W) + (size_t)row * cols);
        const __half* sc = S + (size_t)row * (cols / group);
        const int nchunks = cols / 16;  // 16 int8 per chunk
#pragma unroll 4
        for (int j = lane; j < nchunks; j += kWarp) {
            uint4 p = __ldg(w + j);
            const uint32_t words[4] = {p.x, p.y, p.z, p.w};
            const int e = j * 16;
            float s = 0.f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                uint32_t u = words[k] ^ 0x80808080u;  // int8 -> biased uint8 per byte
#pragma unroll
                for (int b = 0; b < 4; ++b) s = fmaf(u8_biased_to_float((u >> (8 * b)) & 0xFFu), xs[pidx(e + 4 * k + b)], s);
            }
            acc = fmaf(s, __half2float(sc[e / group]), acc);
        }
    } else {
        const uint4* w = reinterpret_cast<const uint4*>(static_cast<const uint8_t*>(W) + (size_t)row * (cols / 2));
        const __half* sc = S + (size_t)row * (cols / group);
        const int nchunks = cols / 32;  // 32 nibbles per chunk
#pragma unroll 4
        for (int j = lane; j < nchunks; j += kWarp) {
            uint4 p = __ldg(w + j);
            const uint32_t words[4] = {p.x, p.y, p.z, p.w};
            const int e = j * 32;
            float s = 0.f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
#pragma unroll
                for (int n = 0; n < 8; ++n) s = fmaf(u4_to_float((words[k] >> (4 * n)) & 0xFu), xs[pidx(e + 8 * k + n)], s);
            }
            acc = fmaf(s, __half2float(sc[e / group]), acc);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = accumulate ? y[row] + acc : acc;
}

// ------------------------------------------------------------------------------------------------
// block-wide helpers
// ------------------------------------------------------------------------------------------------
__device__ float block_sum(float v) {
    __shared__ float sh[32];
    const int lane = threadIdx.x % kWarp, wid = threadIdx.x / kWarp;
    v = warp_sum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + kWarp - 1) / kWarp;
    v = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__global__ void rmsnorm_kernel(float* __restrict__ out, const float* __restrict__ x, const float* __restrict__ w,
                               int n, float eps) {
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss = fmaf(x[i], x[i], ss);
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i] * inv * w[i];
}

__global__ void embed_kernel(float* __restrict__ x, const __half* __restrict__ table, const int* __restrict__ token,
                             int dim) {
    const __half* row = table + (size_t)(*token) * dim;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim; i += gridDim.x * blockDim.x)
        x[i] = __half2float(row[i]);
}

// grid: n_heads + n_kv_heads blocks; block: head_dim/2 threads
__global__ void rope_kv_kernel(float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                               __half* __restrict__ kc, __half* __restrict__ vc, const int* __restrict__ pos_p,
                               int n_heads, int n_kv_heads, int head_dim, float theta) {
    const int pos = *pos_p;
    const int h = blockIdx.x;
    const int i = threadIdx.x;
    const int half = head_dim / 2;
    const float inv_freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
    float sn, cs;
    sincosf((float)pos * inv_freq, &sn, &cs);
    if (h < n_heads) {
        float* p = q + (size_t)h * head_dim;
        const float a = p[i], b = p[i + half];
        p[i] = a * cs - b * sn;
        p[i + half] = b * cs + a * sn;
    } else {
        const int kh = h - n_heads;
        const size_t row = (size_t)pos * n_kv_heads * head_dim + (size_t)kh * head_dim;
        const float* p = k + (size_t)kh * head_dim;
        const float a = p[i], b = p[i + half];
        kc[row + i] = __float2half_rn(a * cs - b * sn);
        kc[row + i + half] = __float2half_rn(b * cs + a * sn);
        const float* pv = v + (size_t)kh * head_dim;
        vc[row + i] = __float2half_rn(pv[i]);
        vc[row + i + half] = __float2half_rn(pv[i + half]);
    }
}

// ------------------------------------------------------------------------------------------------
// attention (decode): one block per query head, kAttnWarps warps split the timeline.
// lane owns HD/64 consecutive half2 pairs -> each K/V row is read with fully coalesced 128B+ loads.
// ------------------------------------------------------------------------------------------------
template <int HD>
__global__ void __launch_bounds__(kAttnWarps* kWarp)
    attention_kernel(float* __restrict__ out, const float* __restrict__ q, const __half* __restrict__ kc,
                     const __half* __restrict__ vc, const int* __restrict__ pos_p, int n_heads, int n_kv_heads) {
    constexpr int P = HD / 64;  // half2 pairs per lane
    __shared__ float sm_m[kAttnWarps], sm_l[kAttnWarps];
    __shared__ float sm_acc[kAttnWarps][HD];

    const int h = blockIdx.x;
    const int kvh = h / (n_heads / n_kv_heads);
    const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    const int pos = *pos_p;
    const size_t kv_stride = (size_t)n_kv_heads * HD;
    const float scale = rsqrtf((float)HD);

    float2 qv[P];
    const float* qh = q + (size_t)h * HD;
#pragma unroll
    for (int p = 0; p < P; ++p) {
        const int d = 2 * (lane + kWarp * p);
        qv[p] = make_float2(qh[d] * scale, qh[d + 1] * scale);
    }

    float m = -INFINITY, l = 0.f;
    float2 acc[P];
#pragma unroll
    for (int p = 0; p < P; ++p) acc[p] = make_float2(0.f, 0.f);

    const __half2* kbase = reinterpret_cast<const __half2*>(kc + (size_t)kvh * HD);
    const __half2* vbase = reinterpret_cast<const __half2*>(vc + (size_t)kvh * HD);
    for (int t = warp; t <= pos; t += kAttnWarps) {
        const __half2* kr = kbase + (size_t)t * (kv_stride / 2);
        float s = 0.f;
#pragma unroll
        for (int p = 0; p < P; ++p) {
            const float2 kf = __half22float2(kr[lane + kWarp * p]);
            s = fmaf(qv[p].x, kf.x, s);
            s = fmaf(qv[p].y, kf.y, s);
        }
        s = warp_sum(s);
        const float m_new = fmaxf(m, s);
        const float corr = expf(m - m_new);  // 0 on the first step (m = -inf)
        const float pr = expf(s - m_new);
        l = l * corr + pr;
        const __half2* vr = vbase + (size_t)t * (kv_stride / 2);
#pragma unroll
        for (int p = 0; p < P; ++p) {
            const float2 vf = __half22float2(vr[lane + kWarp * p]);
            acc[p].x = fmaf(pr, vf.x, acc[p].x * corr);
            acc[p].y = fmaf(pr, vf.y, acc[p].y * corr);
        }
        m = m_new;
    }

    if (lane == 0) {
        sm_m[warp] = m;
        sm_l[warp] = l;
    }
#pragma unroll
    for (int p = 0; p < P; ++p) {
        const int d = 2 * (lane + kWarp * p);
        sm_acc[warp][d] = acc[p].x;
        sm_acc[warp][d + 1] = acc[p].y;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < kAttnWarps; ++w) M = fmaxf(M, sm_m[w]);
        float L = 0.f, o = 0.f;
#pragma unroll
        for (int w = 0; w < kAttnWarps; ++w) {
            const float f = (sm_m[w] == -INFINITY) ? 0.f : expf(sm_m[w] - M);
            L = fmaf(sm_l[w], f, L);
            o = fmaf(sm_acc[w][d], f, o);
        }
        out[(size_t)h * HD + d] = o / L;
    }
}

__global__ void swiglu_kernel(float* __restrict__ g, const float* __restrict__ u, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float a = g[i];
        g[i] = a / (1.f + expf(-a)) * u[i];
    }
}

__global__ void argmax_kernel(int* __restrict__ out, const float* __restrict__ x, int n) {
    __shared__ float sv[1024];
    __shared__ int si[1024];
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i];
        if (v > best || (v == best && i < bi)) {
            best = v;
            bi = i;
        }
    }
    sv[threadIdx.x] = best;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float v = sv[threadIdx.x + s];
            const int i = si[threadIdx.x + s];
            if (v > sv[threadIdx.x] || (v == sv[threadIdx.x] && i < si[threadIdx.x])) {
                sv[threadIdx.x] = v;
                si[threadIdx.x] = i;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = si[0] == 0x7fffffff ? 0 : si[0];
}

void check_launch(const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA launch error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

size_t gemv_smem_bytes(int cols) { return (size_t)(cols + cols / 32) * sizeof(float); }

}  // namespace

void setup_kernels() {
    int dev = 0, optin = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    if (optin > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(gemv_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));
        CUDA_CHECK(cudaFuncSetAttribute(gemv_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));
        g_max_dyn_smem = optin;
    }
}

int gemv_max_cols() {
    // largest multiple of 32 whose padded size fits
    return (int)((g_max_dyn_smem / sizeof(float)) * 32 / 33) / 32 * 32;
}

void gemv(const QMatrix& W, const float* x, float* y, bool accumulate, cudaStream_t s) {
    const int cols = W.cols, rows = W.rows;
    const size_t smem = gemv_smem_bytes(cols);
    if (smem > (size_t)g_max_dyn_smem)
        throw std::runtime_error("gemv: cols=" + std::to_string(cols) + " too large for shared memory");
    const dim3 grid((rows + kGemvWarps - 1) / kGemvWarps), block(kGemvWarps * kWarp);
    const __half* sc = reinterpret_cast<const __half*>(W.scales);
    switch (W.quant) {
        case Quant::F16:
            gemv_kernel<0><<<grid, block, smem, s>>>(W.w, sc, x, y, rows, cols, W.group, accumulate);
            break;
        case Quant::Q8:
            gemv_kernel<1><<<grid, block, smem, s>>>(W.w, sc, x, y, rows, cols, W.group, accumulate);
            break;
        case Quant::Q4:
            gemv_kernel<2><<<grid, block, smem, s>>>(W.w, sc, x, y, rows, cols, W.group, accumulate);
            break;
    }
    check_launch("gemv");
}

void rmsnorm(float* out, const float* x, const float* w, int n, float eps, cudaStream_t s) {
    rmsnorm_kernel<<<1, 1024, 0, s>>>(out, x, w, n, eps);
    check_launch("rmsnorm");
}

void embed(float* x, const __half* table, const int* token, int dim, cudaStream_t s) {
    embed_kernel<<<(dim + 255) / 256, 256, 0, s>>>(x, table, token, dim);
    check_launch("embed");
}

void rope_kv(float* q, const float* k, const float* v, __half* kc, __half* vc, const int* pos, int n_heads,
             int n_kv_heads, int head_dim, float theta, cudaStream_t s) {
    rope_kv_kernel<<<n_heads + n_kv_heads, head_dim / 2, 0, s>>>(q, k, v, kc, vc, pos, n_heads, n_kv_heads,
                                                                 head_dim, theta);
    check_launch("rope_kv");
}

void attention(float* out, const float* q, const __half* kc, const __half* vc, const int* pos, int n_heads,
               int n_kv_heads, int head_dim, cudaStream_t s) {
    const dim3 block(kAttnWarps * kWarp);
    switch (head_dim) {
        case 64: attention_kernel<64><<<n_heads, block, 0, s>>>(out, q, kc, vc, pos, n_heads, n_kv_heads); break;
        case 128: attention_kernel<128><<<n_heads, block, 0, s>>>(out, q, kc, vc, pos, n_heads, n_kv_heads); break;
        case 256: attention_kernel<256><<<n_heads, block, 0, s>>>(out, q, kc, vc, pos, n_heads, n_kv_heads); break;
        default: throw std::runtime_error("attention: unsupported head_dim " + std::to_string(head_dim));
    }
    check_launch("attention");
}

void swiglu(float* g, const float* u, int n, cudaStream_t s) {
    swiglu_kernel<<<(n + 255) / 256, 256, 0, s>>>(g, u, n);
    check_launch("swiglu");
}

void argmax(int* out, const float* x, int n, cudaStream_t s) {
    argmax_kernel<<<1, 1024, 0, s>>>(out, x, n);
    check_launch("argmax");
}

}  // namespace gpu
}  // namespace tinf
