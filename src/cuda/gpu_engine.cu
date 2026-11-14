// tinyinfer - CUDA engine: weights resident in VRAM, fp16 KV cache, one decode step = one CUDA graph.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "kernels.cuh"
#include "tinf/engine.h"

namespace tinf {

GpuInfo gpu_info(int device) {
    GpuInfo info;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    info.name = prop.name;
    info.sm_count = prop.multiProcessorCount;
    info.cc_major = prop.major;
    info.cc_minor = prop.minor;
    int mem_clock_khz = 0, bus_width = 0;
    // these attributes are not available on every platform / driver; treat failure as "unknown"
    if (cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device) != cudaSuccess) mem_clock_khz = 0;
    if (cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, device) != cudaSuccess) bus_width = 0;
    cudaGetLastError();
    info.peak_bandwidth_gbs = 2.0 * mem_clock_khz * 1e3 * (bus_width / 8.0) / 1e9;  // DDR
    int prev = 0;
    CUDA_CHECK(cudaGetDevice(&prev));
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMemGetInfo(&info.free_mem, &info.total_mem));
    CUDA_CHECK(cudaSetDevice(prev));
    return info;
}

namespace {

class GpuEngine final : public Engine {
public:
    GpuEngine(const ModelFile& mf, int ctx, const GpuOptions& opts) : cfg_(mf.config()), ctx_(ctx), opts_(opts) {
        CUDA_CHECK(cudaSetDevice(opts.device));
        gpu::setup_kernels();
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        validate();

        const Config& c = cfg_;
        // staging buffer sized for the largest tensor
        size_t max_bytes = 0;
        for (const TensorInfo& t : mf.tensors()) max_bytes = std::max<size_t>(max_bytes, t.nbytes);
        staging_.resize(max_bytes);

        tok_emb_ = static_cast<__half*>(upload(mf, "tok_embeddings"));
        norm_ = static_cast<float*>(upload(mf, "norm"));
        output_ = upload_matrix(mf, "output");
        layers_.resize(c.n_layers);
        for (int l = 0; l < c.n_layers; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            Layer& L = layers_[l];
            L.attn_norm = static_cast<float*>(upload(mf, p + "attention_norm"));
            L.ffn_norm = static_cast<float*>(upload(mf, p + "ffn_norm"));
            L.wq = upload_matrix(mf, p + "wq");
            L.wk = upload_matrix(mf, p + "wk");
            L.wv = upload_matrix(mf, p + "wv");
            L.wo = upload_matrix(mf, p + "wo");
            L.w1 = upload_matrix(mf, p + "w1");
            L.w2 = upload_matrix(mf, p + "w2");
            L.w3 = upload_matrix(mf, p + "w3");
        }
        staging_.clear();
        staging_.shrink_to_fit();

        const size_t cache_elems = (size_t)c.n_layers * ctx_ * c.kv_dim();
        kcache_ = static_cast<__half*>(alloc(cache_elems * sizeof(__half)));
        vcache_ = static_cast<__half*>(alloc(cache_elems * sizeof(__half)));
        CUDA_CHECK(cudaMemset(kcache_, 0, cache_elems * sizeof(__half)));
        CUDA_CHECK(cudaMemset(vcache_, 0, cache_elems * sizeof(__half)));
        kv_bytes_ = 2 * cache_elems * sizeof(__half);

        x_ = falloc(c.dim);
        xb_ = falloc(std::max(c.dim, c.q_dim()));
        q_ = falloc(c.q_dim());
        k_ = falloc(c.kv_dim());
        v_ = falloc(c.kv_dim());
        att_ = falloc(c.q_dim());
        hb_ = falloc(c.hidden_dim);
        hb2_ = falloc(c.hidden_dim);
        logits_ = falloc(c.vocab_size);
        d_ints_ = static_cast<int*>(alloc(4 * sizeof(int)));  // [token, pos, argmax, pad]

        CUDA_CHECK(cudaMallocHost(&h_ints_, 4 * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_logits_, (size_t)c.vocab_size * sizeof(float)));
        CUDA_CHECK(cudaDeviceSynchronize());
        if (opts_.use_graph) capture();  // record the whole decode step once
    }

    ~GpuEngine() override {
        if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
        for (void* p : allocations_) cudaFree(p);
        if (h_ints_) cudaFreeHost(h_ints_);
        if (h_logits_) cudaFreeHost(h_logits_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    void forward(int token, int pos) override {
        if (pos < 0 || pos >= ctx_) throw std::runtime_error("position exceeds context length");
        if (token < 0 || token >= cfg_.vocab_size) throw std::runtime_error("token id out of range");
        h_ints_[0] = token;
        h_ints_[1] = pos;
        CUDA_CHECK(cudaMemcpyAsync(d_ints_, h_ints_, 2 * sizeof(int), cudaMemcpyHostToDevice, stream_));
        if (graph_exec_) {
            CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
        } else {
            enqueue_step();
        }
        CUDA_CHECK(cudaMemcpyAsync(h_ints_ + 2, d_ints_ + 2, sizeof(int), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        logits_valid_ = false;
    }

    const float* logits() override {
        if (!logits_valid_) {
            CUDA_CHECK(cudaMemcpyAsync(h_logits_, logits_, (size_t)cfg_.vocab_size * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            logits_valid_ = true;
        }
        return h_logits_;
    }

    int argmax() override { return h_ints_[2]; }
    const Config& config() const override { return cfg_; }
    int context_length() const override { return ctx_; }
    std::string name() const override {
        return std::string("cuda") + (opts_.use_graph ? "+graph" : "");
    }
    size_t weight_bytes_per_token() const override {
        size_t b = output_.bytes();
        for (const Layer& L : layers_)
            b += L.wq.bytes() + L.wk.bytes() + L.wv.bytes() + L.wo.bytes() + L.w1.bytes() + L.w2.bytes() +
                 L.w3.bytes();
        return b;
    }
    std::string memory_summary() const override {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "VRAM: weights %.0f MiB + KV cache %.0f MiB (ctx %d, fp16)",
                      weight_bytes_ / 1048576.0, kv_bytes_ / 1048576.0, ctx_);
        return buf;
    }

private:
    struct Layer {
        float* attn_norm = nullptr;
        float* ffn_norm = nullptr;
        QMatrix wq, wk, wv, wo, w1, w2, w3;
    };

    void validate() const {
        const Config& c = cfg_;
        if (c.head_dim != 64 && c.head_dim != 128 && c.head_dim != 256)
            throw std::runtime_error("unsupported head_dim (64/128/256)");
        const int max_cols = gpu::gemv_max_cols();
        for (int cols : {c.dim, c.q_dim(), c.hidden_dim}) {
            if (cols % 32) throw std::runtime_error("matrix widths must be multiples of 32");
            if (cols > max_cols)
                throw std::runtime_error("matrix width " + std::to_string(cols) + " exceeds gemv limit " +
                                         std::to_string(max_cols));
            if (c.quant != Quant::F16 && cols % c.group_size) throw std::runtime_error("bad group size");
        }
        if (c.quant != Quant::F16 && c.group_size % 32) throw std::runtime_error("group size must be a multiple of 32");
    }

    void* alloc(size_t bytes) {
        void* p = nullptr;
        cudaError_t e = cudaMalloc(&p, bytes);
        if (e != cudaSuccess) {
            size_t free_b = 0, total_b = 0;
            cudaGetLastError();
            cudaMemGetInfo(&free_b, &total_b);
            throw std::runtime_error("out of GPU memory allocating " + std::to_string(bytes >> 20) + " MiB (free " +
                                     std::to_string(free_b >> 20) + " MiB). Try a smaller --ctx or a q4 model.");
        }
        allocations_.push_back(p);
        return p;
    }
    float* falloc(size_t n) {
        float* p = static_cast<float*>(alloc(n * sizeof(float)));
        CUDA_CHECK(cudaMemset(p, 0, n * sizeof(float)));
        return p;
    }

    void* upload(const ModelFile& mf, const std::string& name) {
        const TensorInfo& t = mf.tensor(name);
        mf.read(t, staging_.data());
        void* d = alloc(t.nbytes);
        CUDA_CHECK(cudaMemcpy(d, staging_.data(), t.nbytes, cudaMemcpyHostToDevice));
        weight_bytes_ += t.nbytes;
        return d;
    }

    QMatrix upload_matrix(const ModelFile& mf, const std::string& name) {
        const TensorInfo& t = mf.tensor(name);
        QMatrix m;
        m.rows = (int)t.shape[0];
        m.cols = (int)t.shape[1];
        m.quant = cfg_.quant;
        m.group = cfg_.group_size;
        m.w = upload(mf, name);
        if (cfg_.quant != Quant::F16) m.scales = static_cast<const uint16_t*>(upload(mf, name + ".scales"));
        return m;
    }

    // All kernels of one decode step. Reads token/pos from d_ints_, so it is graph-capturable.
    void enqueue_step() {
        const Config& c = cfg_;
        cudaStream_t s = stream_;
        const int* d_token = d_ints_;
        const int* d_pos = d_ints_ + 1;
        const size_t layer_cache = (size_t)ctx_ * c.kv_dim();

        gpu::embed(x_, tok_emb_, d_token, c.dim, s);
        for (int l = 0; l < c.n_layers; ++l) {
            const Layer& L = layers_[l];
            __half* kc = kcache_ + l * layer_cache;
            __half* vc = vcache_ + l * layer_cache;
            gpu::rmsnorm(xb_, x_, L.attn_norm, c.dim, c.norm_eps, s);
            gpu::gemv(L.wq, xb_, q_, false, s);
            gpu::gemv(L.wk, xb_, k_, false, s);
            gpu::gemv(L.wv, xb_, v_, false, s);
            gpu::rope_kv(q_, k_, v_, kc, vc, d_pos, c.n_heads, c.n_kv_heads, c.head_dim, c.rope_theta, s);
            gpu::attention(att_, q_, kc, vc, d_pos, c.n_heads, c.n_kv_heads, c.head_dim, s);
            gpu::gemv(L.wo, att_, x_, true, s);  // residual fused into the gemv epilogue

            gpu::rmsnorm(xb_, x_, L.ffn_norm, c.dim, c.norm_eps, s);
            gpu::gemv(L.w1, xb_, hb_, false, s);
            gpu::gemv(L.w3, xb_, hb2_, false, s);
            gpu::swiglu(hb_, hb2_, c.hidden_dim, s);
            gpu::gemv(L.w2, hb_, x_, true, s);
        }
        gpu::rmsnorm(xb_, x_, norm_, c.dim, c.norm_eps, s);
        gpu::gemv(output_, xb_, logits_, false, s);
        gpu::argmax(d_ints_ + 2, logits_, c.vocab_size, s);
    }

    void capture() {
        cudaGraph_t graph = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
        enqueue_step();
        CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }

    Config cfg_;
    int ctx_;
    GpuOptions opts_;
    cudaStream_t stream_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    std::vector<void*> allocations_;
    std::vector<uint8_t> staging_;
    size_t weight_bytes_ = 0, kv_bytes_ = 0;

    __half* tok_emb_ = nullptr;
    float* norm_ = nullptr;
    QMatrix output_;
    std::vector<Layer> layers_;
    __half *kcache_ = nullptr, *vcache_ = nullptr;
    float *x_ = nullptr, *xb_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr;
    float *hb_ = nullptr, *hb2_ = nullptr, *logits_ = nullptr;
    int* d_ints_ = nullptr;
    int* h_ints_ = nullptr;
    float* h_logits_ = nullptr;
    bool logits_valid_ = false;
};

}  // namespace

std::unique_ptr<Engine> make_gpu_engine(const ModelFile& model, int context_length, const GpuOptions& opts) {
    return std::make_unique<GpuEngine>(model, context_length, opts);
}

}  // namespace tinf
