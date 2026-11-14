// tinyinfer - CUDA kernel launchers
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "tinf/model_file.h"

#define CUDA_CHECK(expr)                                                                         \
    do {                                                                                         \
        cudaError_t err_ = (expr);                                                               \
        if (err_ != cudaSuccess) {                                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), __FILE__, \
                         __LINE__, cudaGetErrorString(err_));                                    \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

namespace tinf {
namespace gpu {

// Must be called once per device before launching gemv on matrices wider than ~11k columns.
void setup_kernels();

// Largest `cols` supported by gemv on this device (x is staged in shared memory).
int gemv_max_cols();

// y[rows] = W[rows, cols] @ x[cols]   (y += ... when accumulate). W is a device QMatrix.
// rows_per_warp: 1, 2 or 4; 0 = autotuned value for this shape (or a heuristic if never tuned).
void gemv(const QMatrix& W, const float* x, float* y, bool accumulate, cudaStream_t s, int rows_per_warp = 0);

// Time every rows_per_warp variant on this matrix and remember the fastest for its shape.
// x must hold W.cols floats, y_scratch W.rows floats (its contents are overwritten).
// Returns the chosen rows_per_warp. Must not be called during graph capture.
int autotune_gemv(const QMatrix& W, const float* x, float* y_scratch, cudaStream_t s);

// out = rmsnorm(x) * w
void rmsnorm(float* out, const float* x, const float* w, int n, float eps, cudaStream_t s);

// x = fp32(table[*token])
void embed(float* x, const __half* table, const int* token, int dim, cudaStream_t s);

// RoPE on q (in place) and k; writes rotated k and v as fp16 into the caches at row *pos.
// kc/vc point at this layer's cache: [ctx][n_kv*hd]
void rope_kv(float* q, const float* k, const float* v, __half* kc, __half* vc, const int* pos, int n_heads,
             int n_kv_heads, int head_dim, float theta, cudaStream_t s);

// Decode attention over cache rows [0, *pos] (online softmax).
// With `scratch` (attention_scratch_floats() floats) this runs GQA-aware split-K flash-decoding:
// one block per (kv head, 128-step chunk) serving every query head of the group, then a combine
// kernel. Without scratch (or for unsupported shapes) it falls back to one block per query head.
// ctx = KV-cache capacity (sizes the split grid; must be the same for every call of a captured graph).
void attention(float* out, const float* q, const __half* kc, const __half* vc, const int* pos, int n_heads,
               int n_kv_heads, int head_dim, int ctx, float* scratch, cudaStream_t s);
size_t attention_scratch_floats(int n_heads, int head_dim, int ctx);
bool attention_split_supported(int n_heads, int n_kv_heads, int head_dim);

// g = silu(g) * u
void swiglu(float* g, const float* u, int n, cudaStream_t s);

// *out = argmax(x[0..n))  (lowest index on ties)
void argmax(int* out, const float* x, int n, cudaStream_t s);

}  // namespace gpu
}  // namespace tinf
