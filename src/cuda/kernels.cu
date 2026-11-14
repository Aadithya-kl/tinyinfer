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
#include <map>
#include <tuple>
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
int g_sm_count = 16;

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
// GEMV: each warp computes R consecutive rows, so every lane keeps R independent 16-byte loads in
// flight and the block's shared copy of x is amortized over 8*R rows. R is autotuned per shape.
// ------------------------------------------------------------------------------------------------
template <int Q, int R>  // Q: 0 = f16, 1 = q8, 2 = q4
__global__ void __launch_bounds__(kGemvWarps* kWarp)
    gemv_kernel(const void* __restrict__ W, const __half* __restrict__ S, const float* __restrict__ x,
                float* __restrict__ y, int rows, int cols, int group, int accumulate) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x * 4; i < cols; i += blockDim.x * 4) {  // cols % 32 == 0
        const float4 v = *reinterpret_cast<const float4*>(x + i);
        xs[pidx(i)] = v.x;
        xs[pidx(i + 1)] = v.y;
        xs[pidx(i + 2)] = v.z;
        xs[pidx(i + 3)] = v.w;
    }
    __syncthreads();

    const int warp = threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    const int row0 = (blockIdx.x * kGemvWarps + warp) * R;
    if (row0 >= rows) return;

    // tail rows are clamped to a valid row (computed redundantly, never stored)
    int rr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) rr[r] = min(row0 + r, rows - 1);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    const int ng = Q == 0 ? 1 : cols / group;

    if constexpr (Q == 0) {
        const int nchunks = cols / 8;  // 8 halves per 16-byte chunk
        for (int j = lane; j < nchunks; j += kWarp) {
            uint4 p[R];
#pragma unroll
            for (int r = 0; r < R; ++r)
                p[r] = __ldg(reinterpret_cast<const uint4*>(static_cast<const __half*>(W) + (size_t)rr[r] * cols) + j);
            const int e = j * 8;
            float xv[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) xv[k] = xs[pidx(e + k)];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const __half2* h = reinterpret_cast<const __half2*>(&p[r]);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float2 f = __half22float2(h[k]);
                    acc[r] = fmaf(f.x, xv[2 * k], acc[r]);
                    acc[r] = fmaf(f.y, xv[2 * k + 1], acc[r]);
                }
            }
        }
    } else if constexpr (Q == 1) {
        const int nchunks = cols / 16;  // 16 int8 per chunk
        for (int j = lane; j < nchunks; j += kWarp) {
            uint4 p[R];
#pragma unroll
            for (int r = 0; r < R; ++r)
                p[r] = __ldg(reinterpret_cast<const uint4*>(static_cast<const int8_t*>(W) + (size_t)rr[r] * cols) + j);
            const int e = j * 16;
            float xv[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) xv[k] = xs[pidx(e + k)];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t words[4] = {p[r].x, p[r].y, p[r].z, p[r].w};
                float sum = 0.f;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t u = words[k] ^ 0x80808080u;  // int8 -> biased uint8 per byte
#pragma unroll
                    for (int b = 0; b < 4; ++b) sum = fmaf(u8_biased_to_float((u >> (8 * b)) & 0xFFu), xv[4 * k + b], sum);
                }
                acc[r] = fmaf(sum, __half2float(S[(size_t)rr[r] * ng + e / group]), acc[r]);
            }
        }
    } else {
        const int nchunks = cols / 32;  // 32 nibbles per chunk
        for (int j = lane; j < nchunks; j += kWarp) {
            uint4 p[R];
#pragma unroll
            for (int r = 0; r < R; ++r)
                p[r] = __ldg(reinterpret_cast<const uint4*>(static_cast<const uint8_t*>(W) + (size_t)rr[r] * (cols / 2)) + j);
            const int e = j * 32;
            float xv[32];
#pragma unroll
            for (int k = 0; k < 32; ++k) xv[k] = xs[pidx(e + k)];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t words[4] = {p[r].x, p[r].y, p[r].z, p[r].w};
                float sum = 0.f;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
#pragma unroll
                    for (int n = 0; n < 8; ++n)
                        sum = fmaf(u4_to_float((words[k] >> (4 * n)) & 0xFu), xv[8 * k + n], sum);
                }
                acc[r] = fmaf(sum, __half2float(S[(size_t)rr[r] * ng + e / group]), acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = warp_sum(acc[r]);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row < rows) y[row] = accumulate ? y[row] + acc[r] : acc[r];
        }
    }
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
// attention, fallback path: one block per query head, kAttnWarps warps split the timeline.
// lane owns HD/64 half2 pairs -> each K/V row is read with fully coalesced 128B+ loads.
// ------------------------------------------------------------------------------------------------
template <int HD>
__global__ void __launch_bounds__(kAttnWarps* kWarp)
    attention_head_kernel(float* __restrict__ out, const float* __restrict__ q, const __half* __restrict__ kc,
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

// ------------------------------------------------------------------------------------------------
// attention, main path: GQA-aware split-K flash-decoding.
//   grid = (n_kv_heads, ceil(ctx / kAttnChunk)); block = 8 warps.
//   A block owns one KV head and a 128-step slice of the timeline and serves all G query heads of
//   that group, so every K/V row is read from DRAM exactly once (G=8 for TinyLlama / Llama-3).
//   Each warp runs an online softmax over every 8th step; warps are merged in shared memory and the
//   block writes an unnormalized partial (m, l, acc[HD]) per query head. attn_combine merges splits.
// ------------------------------------------------------------------------------------------------
constexpr int kAttnChunk = 128;

template <int HD, int GM>  // GM = compile-time upper bound on the group size G
__global__ void __launch_bounds__(kAttnWarps* kWarp)
    attn_split_kernel(float* __restrict__ part, const float* __restrict__ q, const __half* __restrict__ kc,
                      const __half* __restrict__ vc, const int* __restrict__ pos_p, int n_heads, int n_kv_heads,
                      int max_splits) {
    constexpr int P = HD / 64;
    __shared__ float sm_m[kAttnWarps][GM], sm_l[kAttnWarps][GM];
    __shared__ float sm_acc[kAttnWarps][GM][HD];

    const int kvh = blockIdx.x, split = blockIdx.y;
    const int pos = *pos_p;
    const int t0 = split * kAttnChunk;
    if (t0 > pos) return;  // uniform for the whole block
    const int t1 = min(t0 + kAttnChunk, pos + 1);
    const int G = n_heads / n_kv_heads;
    const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    const size_t stride2 = (size_t)n_kv_heads * HD / 2;  // cache row stride in half2
    const float scale = rsqrtf((float)HD);

    float2 qv[GM][P];
#pragma unroll
    for (int g = 0; g < GM; ++g) {
        const float* qh = q + (size_t)(kvh * G + min(g, G - 1)) * HD;
#pragma unroll
        for (int p = 0; p < P; ++p) {
            const int d = 2 * (lane + kWarp * p);
            qv[g][p] = make_float2(qh[d] * scale, qh[d + 1] * scale);
        }
    }
    float m[GM], l[GM];
    float2 acc[GM][P];
#pragma unroll
    for (int g = 0; g < GM; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.f;
#pragma unroll
        for (int p = 0; p < P; ++p) acc[g][p] = make_float2(0.f, 0.f);
    }

    const __half2* kb = reinterpret_cast<const __half2*>(kc) + (size_t)kvh * (HD / 2);
    const __half2* vb = reinterpret_cast<const __half2*>(vc) + (size_t)kvh * (HD / 2);
    for (int t = t0 + warp; t < t1; t += kAttnWarps) {
        float2 kf[P], vf[P];
#pragma unroll
        for (int p = 0; p < P; ++p) {
            kf[p] = __half22float2(kb[(size_t)t * stride2 + lane + kWarp * p]);
            vf[p] = __half22float2(vb[(size_t)t * stride2 + lane + kWarp * p]);
        }
        float sc[GM];
#pragma unroll
        for (int g = 0; g < GM; ++g) {
            float d = 0.f;
#pragma unroll
            for (int p = 0; p < P; ++p) d = fmaf(qv[g][p].x, kf[p].x, fmaf(qv[g][p].y, kf[p].y, d));
            sc[g] = d;
        }
#pragma unroll
        for (int g = 0; g < GM; ++g) {
            if (g < G) {  // warp-uniform
                const float sg = warp_sum(sc[g]);
                const float m_new = fmaxf(m[g], sg);
                const float corr = expf(m[g] - m_new);
                const float pr = expf(sg - m_new);
                l[g] = l[g] * corr + pr;
#pragma unroll
                for (int p = 0; p < P; ++p) {
                    acc[g][p].x = fmaf(pr, vf[p].x, acc[g][p].x * corr);
                    acc[g][p].y = fmaf(pr, vf[p].y, acc[g][p].y * corr);
                }
                m[g] = m_new;
            }
        }
    }

#pragma unroll
    for (int g = 0; g < GM; ++g) {
        if (lane == 0) {
            sm_m[warp][g] = m[g];
            sm_l[warp][g] = l[g];
        }
#pragma unroll
        for (int p = 0; p < P; ++p) {
            const int d = 2 * (lane + kWarp * p);
            sm_acc[warp][g][d] = acc[g][p].x;
            sm_acc[warp][g][d + 1] = acc[g][p].y;
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < G * HD; idx += blockDim.x) {
        const int g = idx / HD, d = idx % HD;
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < kAttnWarps; ++w) M = fmaxf(M, sm_m[w][g]);
        float L = 0.f, o = 0.f;
#pragma unroll
        for (int w = 0; w < kAttnWarps; ++w) {
            const float f = (sm_m[w][g] == -INFINITY) ? 0.f : expf(sm_m[w][g] - M);
            L = fmaf(sm_l[w][g], f, L);
            o = fmaf(sm_acc[w][g][d], f, o);
        }
        float* dst = part + ((size_t)(kvh * G + g) * max_splits + split) * (HD + 2);
        dst[2 + d] = o;
        if (d == 0) {
            dst[0] = M;
            dst[1] = L;
        }
    }
}

// grid = n_heads, block = head_dim threads: merge the valid splits of one query head
__global__ void attn_combine_kernel(float* __restrict__ out, const float* __restrict__ part,
                                    const int* __restrict__ pos_p, int head_dim, int max_splits) {
    const int h = blockIdx.x, d = threadIdx.x;
    const int ns = *pos_p / kAttnChunk + 1;
    const size_t stride = (size_t)head_dim + 2;
    const float* base = part + (size_t)h * max_splits * stride;
    float M = -INFINITY;
    for (int s = 0; s < ns; ++s) M = fmaxf(M, base[s * stride]);
    float L = 0.f, o = 0.f;
    for (int s = 0; s < ns; ++s) {
        const float f = expf(base[s * stride] - M);
        L = fmaf(base[s * stride + 1], f, L);
        o = fmaf(base[s * stride + 2 + d], f, o);
    }
    out[(size_t)h * head_dim + d] = o / L;
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

template <int Q>
void set_gemv_smem(int bytes) {
    CUDA_CHECK(cudaFuncSetAttribute(gemv_kernel<Q, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    CUDA_CHECK(cudaFuncSetAttribute(gemv_kernel<Q, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    CUDA_CHECK(cudaFuncSetAttribute(gemv_kernel<Q, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

void setup_kernels() {
    int dev = 0, optin = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&g_sm_count, cudaDevAttrMultiProcessorCount, dev));
    if (optin > 48 * 1024) {
        set_gemv_smem<0>(optin);
        set_gemv_smem<1>(optin);
        set_gemv_smem<2>(optin);
        g_max_dyn_smem = optin;
    }
}

int gemv_max_cols() {
    // largest multiple of 32 whose padded size fits
    return (int)((g_max_dyn_smem / sizeof(float)) * 32 / 33) / 32 * 32;
}

namespace {

std::map<std::tuple<int, int, int>, int>& tuned_rows_per_warp() {
    static std::map<std::tuple<int, int, int>, int> m;
    return m;
}

int pick_rows_per_warp(const QMatrix& W) {
    auto& t = tuned_rows_per_warp();
    auto it = t.find(std::make_tuple(W.rows, W.cols, (int)W.quant));
    if (it != t.end()) return it->second;
    // heuristic: as many rows per warp as possible while keeping >= 4 blocks per SM
    for (int r : {4, 2}) {
        const int blocks = (W.rows + kGemvWarps * r - 1) / (kGemvWarps * r);
        if (blocks >= 4 * g_sm_count) return r;
    }
    return 1;
}

template <int Q>
void launch_gemv(int r, dim3 grid, dim3 block, size_t smem, cudaStream_t s, const QMatrix& W, const __half* sc,
                 const float* x, float* y, int accumulate) {
    switch (r) {
        case 1: gemv_kernel<Q, 1><<<grid, block, smem, s>>>(W.w, sc, x, y, W.rows, W.cols, W.group, accumulate); break;
        case 2: gemv_kernel<Q, 2><<<grid, block, smem, s>>>(W.w, sc, x, y, W.rows, W.cols, W.group, accumulate); break;
        case 4: gemv_kernel<Q, 4><<<grid, block, smem, s>>>(W.w, sc, x, y, W.rows, W.cols, W.group, accumulate); break;
        default: throw std::runtime_error("gemv: rows_per_warp must be 1, 2 or 4");
    }
}

}  // namespace

void gemv(const QMatrix& W, const float* x, float* y, bool accumulate, cudaStream_t s, int rows_per_warp) {
    const size_t smem = gemv_smem_bytes(W.cols);
    if (smem > (size_t)g_max_dyn_smem)
        throw std::runtime_error("gemv: cols=" + std::to_string(W.cols) + " too large for shared memory");
    const int r = rows_per_warp > 0 ? rows_per_warp : pick_rows_per_warp(W);
    const int rows_per_block = kGemvWarps * r;
    const dim3 grid((W.rows + rows_per_block - 1) / rows_per_block), block(kGemvWarps * kWarp);
    const __half* sc = reinterpret_cast<const __half*>(W.scales);
    const int acc = accumulate ? 1 : 0;
    switch (W.quant) {
        case Quant::F16: launch_gemv<0>(r, grid, block, smem, s, W, sc, x, y, acc); break;
        case Quant::Q8: launch_gemv<1>(r, grid, block, smem, s, W, sc, x, y, acc); break;
        case Quant::Q4: launch_gemv<2>(r, grid, block, smem, s, W, sc, x, y, acc); break;
    }
    check_launch("gemv");
}

int autotune_gemv(const QMatrix& W, const float* x, float* y_scratch, cudaStream_t s) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    int best_r = 1;
    float best_ms = 1e30f;
    for (int r : {1, 2, 4}) {
        for (int i = 0; i < 3; ++i) gemv(W, x, y_scratch, false, s, r);  // warm up
        CUDA_CHECK(cudaEventRecord(a, s));
        const int iters = 20;
        for (int i = 0; i < iters; ++i) gemv(W, x, y_scratch, false, s, r);
        CUDA_CHECK(cudaEventRecord(b, s));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        if (ms < best_ms * 0.98f) {  // prefer smaller R unless clearly faster
            best_ms = ms;
            best_r = r;
        }
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    tuned_rows_per_warp()[std::make_tuple(W.rows, W.cols, (int)W.quant)] = best_r;
    return best_r;
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

namespace {
int group_bound(int g) { return g <= 1 ? 1 : g <= 2 ? 2 : g <= 4 ? 4 : g <= 8 ? 8 : 0; }
}  // namespace

bool attention_split_supported(int n_heads, int n_kv_heads, int head_dim) {
    if (n_kv_heads <= 0 || n_heads % n_kv_heads) return false;
    const int gm = group_bound(n_heads / n_kv_heads);
    if (gm == 0) return false;
    if (head_dim != 64 && head_dim != 128 && head_dim != 256) return false;
    return gm * head_dim <= 1024;  // static shared memory budget of the split kernel
}

size_t attention_scratch_floats(int n_heads, int head_dim, int ctx) {
    const int max_splits = (ctx + kAttnChunk - 1) / kAttnChunk;
    return (size_t)n_heads * max_splits * (head_dim + 2);
}

void attention(float* out, const float* q, const __half* kc, const __half* vc, const int* pos, int n_heads,
               int n_kv_heads, int head_dim, int ctx, float* scratch, cudaStream_t s) {
    const dim3 block(kAttnWarps * kWarp);
    if (scratch && attention_split_supported(n_heads, n_kv_heads, head_dim)) {
        const int max_splits = (ctx + kAttnChunk - 1) / kAttnChunk;
        const dim3 grid(n_kv_heads, max_splits);
        const int gm = group_bound(n_heads / n_kv_heads);
#define TINF_SPLIT(HD, GM) attn_split_kernel<HD, GM><<<grid, block, 0, s>>>(scratch, q, kc, vc, pos, n_heads, n_kv_heads, max_splits)
        switch (head_dim * 16 + gm) {
            case 64 * 16 + 1: TINF_SPLIT(64, 1); break;
            case 64 * 16 + 2: TINF_SPLIT(64, 2); break;
            case 64 * 16 + 4: TINF_SPLIT(64, 4); break;
            case 64 * 16 + 8: TINF_SPLIT(64, 8); break;
            case 128 * 16 + 1: TINF_SPLIT(128, 1); break;
            case 128 * 16 + 2: TINF_SPLIT(128, 2); break;
            case 128 * 16 + 4: TINF_SPLIT(128, 4); break;
            case 128 * 16 + 8: TINF_SPLIT(128, 8); break;
            case 256 * 16 + 1: TINF_SPLIT(256, 1); break;
            case 256 * 16 + 2: TINF_SPLIT(256, 2); break;
            case 256 * 16 + 4: TINF_SPLIT(256, 4); break;
            default: throw std::runtime_error("attention: unsupported split configuration");
        }
#undef TINF_SPLIT
        check_launch("attn_split");
        attn_combine_kernel<<<n_heads, head_dim, 0, s>>>(out, scratch, pos, head_dim, max_splits);
        check_launch("attn_combine");
        return;
    }
    switch (head_dim) {
        case 64: attention_head_kernel<64><<<n_heads, block, 0, s>>>(out, q, kc, vc, pos, n_heads, n_kv_heads); break;
        case 128: attention_head_kernel<128><<<n_heads, block, 0, s>>>(out, q, kc, vc, pos, n_heads, n_kv_heads); break;
        case 256: attention_head_kernel<256><<<n_heads, block, 0, s>>>(out, q, kc, vc, pos, n_heads, n_kv_heads); break;
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
