// tinyinfer - CPU reference operators. These define the semantics every CUDA kernel is tested against.
#pragma once

#include <cstdint>

#include "tinf/model_file.h"

namespace tinf {
namespace cpu {

// y = W x  (or y += W x when accumulate). W is [rows, cols].
void matvec(const QMatrix& W, const float* x, float* y, bool accumulate);

// out = x * rsqrt(mean(x^2) + eps) * w
void rmsnorm(float* out, const float* x, const float* w, int n, float eps);

// HF "rotate_half" RoPE applied in place to n_heads vectors of size head_dim.
void rope(float* x, int n_heads, int head_dim, int pos, float theta);

// Causal attention for one query position over cache[0..pos].
// q: [n_heads*hd]; kc/vc: fp16 bits, [ctx][n_kv*hd]; out: [n_heads*hd]
void attention(float* out, const float* q, const uint16_t* kc, const uint16_t* vc, int pos, int n_heads,
               int n_kv_heads, int head_dim);

// g = silu(g) * u
void swiglu(float* g, const float* u, int n);

}  // namespace cpu
}  // namespace tinf
