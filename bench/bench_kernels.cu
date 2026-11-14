// Kernel micro-benchmarks: achieved DRAM bandwidth of the gemv kernels (f16 / q8 / q4) on the
// matrix shapes of TinyLlama-1.1B and Llama-2-7B, plus decode attention at several context lengths.
#include <algorithm>
#include <cstring>
#include <vector>

#include "kernels.cuh"
#include "tinf/engine.h"
#include "tinf/quant.h"

using namespace tinf;

namespace {

template <typename F>
float time_ms(F&& launch, int iters) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    for (int i = 0; i < 5; ++i) launch();  // warm up
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / iters;
}

void* dmalloc(size_t bytes, int fill = 1) {
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    CUDA_CHECK(cudaMemset(p, fill, bytes));
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device found\n");
        return 1;
    }
    gpu::setup_kernels();
    GpuInfo gi = gpu_info(0);
    std::printf("device: %s, %d SMs, peak DRAM bandwidth ~%.0f GB/s\n\n", gi.name.c_str(), gi.sm_count,
                gi.peak_bandwidth_gbs);

    struct Shape { const char* what; int rows, cols; };
    const Shape shapes[] = {
        {"tinyllama wq/wo", 2048, 2048},  {"tinyllama w1/w3", 5632, 2048}, {"tinyllama w2", 2048, 5632},
        {"tinyllama lm_head", 32000, 2048}, {"llama2-7b wq", 4096, 4096},  {"llama2-7b w1", 11008, 4096},
        {"llama2-7b w2", 4096, 11008},
    };
    std::printf("%-20s %-12s %-5s %8s %8s %8s   %s\n", "gemv", "shape", "quant", "R=1", "R=2", "R=4", "best: time, GB/s, %peak");
    for (const Shape& s : shapes) {
        float* x = static_cast<float*>(dmalloc((size_t)s.cols * 4, 0));
        float* y = static_cast<float*>(dmalloc((size_t)s.rows * 4, 0));
        for (Quant q : {Quant::F16, Quant::Q8, Quant::Q4}) {
            QMatrix m;
            m.rows = s.rows;
            m.cols = s.cols;
            m.quant = q;
            m.group = 64;
            const size_t n_el = (size_t)s.rows * s.cols;
            const size_t payload = q == Quant::F16 ? 2 * n_el : (q == Quant::Q8 ? n_el : n_el / 2);
            m.w = dmalloc(payload);
            if (q != Quant::F16) m.scales = static_cast<const uint16_t*>(dmalloc(2 * n_el / 64, 0));
            float t[3];
            const int rs[3] = {1, 2, 4};
            for (int k = 0; k < 3; ++k) t[k] = time_ms([&] { gpu::gemv(m, x, y, false, 0, rs[k]); }, iters);
            const float ms = std::min(t[0], std::min(t[1], t[2]));
            const double gbs = (double)m.bytes() / (ms * 1e-3) / 1e9;
            char shape[32];
            std::snprintf(shape, sizeof(shape), "%dx%d", s.rows, s.cols);
            std::printf("%-20s %-12s %-5s %7.1fu %7.1fu %7.1fu   %6.1f us %6.1f GB/s %4.0f%%\n", s.what, shape,
                        quant_name(q), t[0] * 1e3, t[1] * 1e3, t[2] * 1e3, ms * 1e3, gbs,
                        gi.peak_bandwidth_gbs > 0 ? 100.0 * gbs / gi.peak_bandwidth_gbs : 0.0);
            cudaFree((void*)m.w);
            if (m.scales) cudaFree((void*)m.scales);
        }
        cudaFree(x);
        cudaFree(y);
    }

    std::printf("\n%-20s %-6s %22s %22s\n", "attention", "ctx", "per-head: us  GB/s", "split-K GQA: us  GB/s");
    struct A { const char* what; int H, KV, D; };
    for (A a : {A{"tinyllama (32/4x64)", 32, 4, 64}, A{"llama2-7b (32x128)", 32, 32, 128}}) {
        for (int ctx : {128, 512, 2048}) {
            const size_t kv = (size_t)ctx * a.KV * a.D;
            __half* kc = static_cast<__half*>(dmalloc(kv * 2, 0));
            __half* vc = static_cast<__half*>(dmalloc(kv * 2, 0));
            float* q = static_cast<float*>(dmalloc((size_t)a.H * a.D * 4, 0));
            float* o = static_cast<float*>(dmalloc((size_t)a.H * a.D * 4, 0));
            int* pos = static_cast<int*>(dmalloc(4, 0));
            int p = ctx - 1;
            CUDA_CHECK(cudaMemcpy(pos, &p, 4, cudaMemcpyHostToDevice));
            float* scratch = static_cast<float*>(dmalloc(gpu::attention_scratch_floats(a.H, a.D, ctx) * 4, 0));
            const float ms1 = time_ms([&] { gpu::attention(o, q, kc, vc, pos, a.H, a.KV, a.D, ctx, nullptr, 0); }, iters);
            const float ms2 = time_ms([&] { gpu::attention(o, q, kc, vc, pos, a.H, a.KV, a.D, ctx, scratch, 0); }, iters);
            const double bytes = 2.0 * kv * 2;  // K + V, fp16, read once
            std::printf("%-20s %-6d %10.1f %10.1f %11.1f %10.1f\n", a.what, ctx, ms1 * 1e3, bytes / (ms1 * 1e-3) / 1e9,
                        ms2 * 1e3, bytes / (ms2 * 1e-3) / 1e9);
            cudaFree(kc); cudaFree(vc); cudaFree(q); cudaFree(o); cudaFree(pos); cudaFree(scratch);
        }
    }
    return 0;
}
