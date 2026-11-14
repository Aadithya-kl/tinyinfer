// CPU reference engine: straightforward fp32 implementation of the Llama decoder.
// It is deliberately simple; its job is to be obviously correct and to validate the CUDA path.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "tinf/cpu_ops.h"
#include "tinf/engine.h"
#include "tinf/half.h"
#include "tinf/sampler.h"

namespace tinf {
namespace cpu {

void matvec(const QMatrix& W, const float* x, float* y, bool accumulate) {
    const float* ht = half_table();
    const int rows = W.rows, cols = W.cols;
    const int ng = W.quant == Quant::F16 ? 0 : cols / W.group;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        float acc = 0.f;
        if (W.quant == Quant::F16) {
            const uint16_t* w = static_cast<const uint16_t*>(W.w) + (size_t)r * cols;
            for (int c = 0; c < cols; ++c) acc += ht[w[c]] * x[c];
        } else if (W.quant == Quant::Q8) {
            const int8_t* w = static_cast<const int8_t*>(W.w) + (size_t)r * cols;
            for (int g = 0; g < ng; ++g) {
                float s = 0.f;
                for (int c = g * W.group; c < (g + 1) * W.group; ++c) s += (float)w[c] * x[c];
                acc += s * ht[W.scales[(size_t)r * ng + g]];
            }
        } else {
            const uint8_t* w = static_cast<const uint8_t*>(W.w) + (size_t)r * (cols / 2);
            for (int g = 0; g < ng; ++g) {
                float s = 0.f;
                for (int c = g * W.group; c < (g + 1) * W.group; c += 2) {
                    uint8_t b = w[c / 2];
                    s += (float)((int)(b & 15) - 8) * x[c] + (float)((int)(b >> 4) - 8) * x[c + 1];
                }
                acc += s * ht[W.scales[(size_t)r * ng + g]];
            }
        }
        y[r] = accumulate ? y[r] + acc : acc;
    }
}

void rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
    float ss = 0.f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    float inv = 1.f / std::sqrt(ss / (float)n + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * w[i];
}

void rope(float* x, int n_heads, int head_dim, int pos, float theta) {
    const int half = head_dim / 2;
    for (int i = 0; i < half; ++i) {
        float inv_freq = 1.f / std::pow(theta, (float)(2 * i) / (float)head_dim);
        float angle = (float)pos * inv_freq;
        float c = std::cos(angle), s = std::sin(angle);
        for (int h = 0; h < n_heads; ++h) {
            float* v = x + (size_t)h * head_dim;
            float a = v[i], b = v[i + half];
            v[i] = a * c - b * s;
            v[i + half] = b * c + a * s;
        }
    }
}

void attention(float* out, const float* q, const uint16_t* kc, const uint16_t* vc, int pos, int n_heads,
               int n_kv_heads, int head_dim) {
    const float* ht = half_table();
    const int kv_dim = n_kv_heads * head_dim;
    const int group = n_heads / n_kv_heads;
    const float scale = 1.f / std::sqrt((float)head_dim);
#pragma omp parallel for schedule(static)
    for (int h = 0; h < n_heads; ++h) {
        std::vector<float> att(pos + 1);
        const float* qh = q + (size_t)h * head_dim;
        const int kvh = h / group;
        float mx = -INFINITY;
        for (int t = 0; t <= pos; ++t) {
            const uint16_t* k = kc + (size_t)t * kv_dim + (size_t)kvh * head_dim;
            float s = 0.f;
            for (int d = 0; d < head_dim; ++d) s += qh[d] * ht[k[d]];
            att[t] = s * scale;
            mx = std::max(mx, att[t]);
        }
        float sum = 0.f;
        for (int t = 0; t <= pos; ++t) {
            att[t] = std::exp(att[t] - mx);
            sum += att[t];
        }
        float* o = out + (size_t)h * head_dim;
        for (int d = 0; d < head_dim; ++d) o[d] = 0.f;
        for (int t = 0; t <= pos; ++t) {
            const uint16_t* v = vc + (size_t)t * kv_dim + (size_t)kvh * head_dim;
            float a = att[t] / sum;
            for (int d = 0; d < head_dim; ++d) o[d] += a * ht[v[d]];
        }
    }
}

void swiglu(float* g, const float* u, int n) {
    for (int i = 0; i < n; ++i) g[i] = g[i] / (1.f + std::exp(-g[i])) * u[i];
}

}  // namespace cpu

namespace {

class CpuEngine final : public Engine {
public:
    CpuEngine(const ModelFile& mf, int ctx) : cfg_(mf.config()), ctx_(ctx) {
        const Config& c = cfg_;
        tok_emb_ = load<uint16_t>(mf, "tok_embeddings");
        norm_ = load<float>(mf, "norm");
        output_ = load_matrix(mf, "output");
        layers_.resize(c.n_layers);
        for (int l = 0; l < c.n_layers; ++l) {
            std::string p = "layers." + std::to_string(l) + ".";
            Layer& L = layers_[l];
            L.attn_norm = load<float>(mf, p + "attention_norm");
            L.ffn_norm = load<float>(mf, p + "ffn_norm");
            L.wq = load_matrix(mf, p + "wq");
            L.wk = load_matrix(mf, p + "wk");
            L.wv = load_matrix(mf, p + "wv");
            L.wo = load_matrix(mf, p + "wo");
            L.w1 = load_matrix(mf, p + "w1");
            L.w2 = load_matrix(mf, p + "w2");
            L.w3 = load_matrix(mf, p + "w3");
        }
        size_t cache = (size_t)c.n_layers * ctx_ * c.kv_dim();
        kc_.assign(cache, 0);
        vc_.assign(cache, 0);
        x_.resize(c.dim);
        xb_.resize(std::max(c.dim, c.q_dim()));
        q_.resize(c.q_dim());
        k_.resize(c.kv_dim());
        v_.resize(c.kv_dim());
        att_.resize(c.q_dim());
        hb_.resize(c.hidden_dim);
        hb2_.resize(c.hidden_dim);
        logits_.resize(c.vocab_size);
    }

    void forward(int token, int pos) override {
        const Config& c = cfg_;
        if (pos < 0 || pos >= ctx_) throw std::runtime_error("position exceeds context length");
        if (token < 0 || token >= c.vocab_size) throw std::runtime_error("token id out of range");
        const float* ht = half_table();
        const uint16_t* e = tok_emb_.data() + (size_t)token * c.dim;
        for (int i = 0; i < c.dim; ++i) x_[i] = ht[e[i]];

        const int kv_dim = c.kv_dim();
        for (int l = 0; l < c.n_layers; ++l) {
            Layer& L = layers_[l];
            cpu::rmsnorm(xb_.data(), x_.data(), L.attn_norm.data(), c.dim, c.norm_eps);
            cpu::matvec(L.wq.m, xb_.data(), q_.data(), false);
            cpu::matvec(L.wk.m, xb_.data(), k_.data(), false);
            cpu::matvec(L.wv.m, xb_.data(), v_.data(), false);
            cpu::rope(q_.data(), c.n_heads, c.head_dim, pos, c.rope_theta);
            cpu::rope(k_.data(), c.n_kv_heads, c.head_dim, pos, c.rope_theta);
            uint16_t* kc = kc_.data() + (size_t)l * ctx_ * kv_dim;
            uint16_t* vc = vc_.data() + (size_t)l * ctx_ * kv_dim;
            for (int i = 0; i < kv_dim; ++i) {
                kc[(size_t)pos * kv_dim + i] = float_to_half(k_[i]);
                vc[(size_t)pos * kv_dim + i] = float_to_half(v_[i]);
            }
            cpu::attention(att_.data(), q_.data(), kc, vc, pos, c.n_heads, c.n_kv_heads, c.head_dim);
            cpu::matvec(L.wo.m, att_.data(), x_.data(), true);

            cpu::rmsnorm(xb_.data(), x_.data(), L.ffn_norm.data(), c.dim, c.norm_eps);
            cpu::matvec(L.w1.m, xb_.data(), hb_.data(), false);
            cpu::matvec(L.w3.m, xb_.data(), hb2_.data(), false);
            cpu::swiglu(hb_.data(), hb2_.data(), c.hidden_dim);
            cpu::matvec(L.w2.m, hb_.data(), x_.data(), true);
        }
        cpu::rmsnorm(xb_.data(), x_.data(), norm_.data(), c.dim, c.norm_eps);
        cpu::matvec(output_.m, xb_.data(), logits_.data(), false);
    }

    const float* logits() override { return logits_.data(); }
    int argmax() override { return tinf::argmax(logits_.data(), cfg_.vocab_size); }
    const Config& config() const override { return cfg_; }
    int context_length() const override { return ctx_; }
    std::string name() const override { return "cpu-reference"; }
    size_t weight_bytes_per_token() const override {
        size_t b = output_.m.bytes();
        for (const Layer& L : layers_)
            b += L.wq.m.bytes() + L.wk.m.bytes() + L.wv.m.bytes() + L.wo.m.bytes() + L.w1.m.bytes() +
                 L.w2.m.bytes() + L.w3.m.bytes();
        return b;
    }

private:
    struct Mat {
        std::vector<uint8_t> payload;
        std::vector<uint16_t> scales;
        QMatrix m;
    };
    struct Layer {
        std::vector<float> attn_norm, ffn_norm;
        Mat wq, wk, wv, wo, w1, w2, w3;
    };

    template <typename T>
    static std::vector<T> load(const ModelFile& mf, const std::string& name) {
        const TensorInfo& t = mf.tensor(name);
        std::vector<T> v(t.nbytes / sizeof(T));
        mf.read(t, v.data());
        return v;
    }

    Mat load_matrix(const ModelFile& mf, const std::string& name) {
        Mat M;
        const TensorInfo& t = mf.tensor(name);
        M.payload.resize(t.nbytes);
        mf.read(t, M.payload.data());
        M.m.rows = (int)t.shape[0];
        M.m.cols = (int)t.shape[1];
        M.m.quant = cfg_.quant;
        M.m.group = cfg_.group_size;
        if (cfg_.quant != Quant::F16) M.scales = load<uint16_t>(mf, name + ".scales");
        M.m.w = M.payload.data();
        M.m.scales = M.scales.empty() ? nullptr : M.scales.data();
        return M;
    }

    Config cfg_;
    int ctx_;
    std::vector<uint16_t> tok_emb_;
    std::vector<float> norm_;
    Mat output_;
    std::vector<Layer> layers_;
    std::vector<uint16_t> kc_, vc_;
    std::vector<float> x_, xb_, q_, k_, v_, att_, hb_, hb2_, logits_;
};

}  // namespace

std::unique_ptr<Engine> make_cpu_engine(const ModelFile& model, int context_length) {
    return std::make_unique<CpuEngine>(model, context_length);
}

}  // namespace tinf
