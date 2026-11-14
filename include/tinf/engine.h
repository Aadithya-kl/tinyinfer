// tinyinfer - inference engine interface (CPU reference + CUDA implementations)
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "tinf/model_file.h"

namespace tinf {

class Engine {
public:
    virtual ~Engine() = default;

    // Run one decoder step: consume `token` at position `pos` (0-based) and update the KV cache.
    virtual void forward(int token, int pos) = 0;
    // Logits of the last forward() call (host pointer, vocab_size floats).
    virtual const float* logits() = 0;
    // argmax of the last logits (computed on-device by the GPU engine; no logits copy needed).
    virtual int argmax() = 0;

    virtual const Config& config() const = 0;
    virtual int context_length() const = 0;
    virtual std::string name() const = 0;
    // Bytes of weights streamed from memory per decoded token (for bandwidth numbers).
    virtual size_t weight_bytes_per_token() const = 0;
    // Human readable memory usage (weights / KV cache), may be empty.
    virtual std::string memory_summary() const { return std::string(); }
};

std::unique_ptr<Engine> make_cpu_engine(const ModelFile& model, int context_length);

#ifdef TINF_WITH_CUDA
struct GpuOptions {
    bool use_graph = true;  // capture one decode step into a CUDA graph and replay it
    int device = 0;
};
std::unique_ptr<Engine> make_gpu_engine(const ModelFile& model, int context_length, const GpuOptions& opts);

struct GpuInfo {
    std::string name;
    int sm_count = 0;
    int cc_major = 0, cc_minor = 0;
    size_t total_mem = 0, free_mem = 0;
    double peak_bandwidth_gbs = 0.0;  // theoretical DRAM bandwidth
};
GpuInfo gpu_info(int device = 0);
#endif

}  // namespace tinf
