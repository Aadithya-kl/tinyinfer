// GPU kernel tests: every CUDA kernel is compared against the CPU reference operator.
#include <cstring>
#include <vector>

#include "kernels.cuh"
#include "test_util.h"
#include "tinf/cpu_ops.h"
#include "tinf/half.h"
#include "tinf/quant.h"
#include "tinf/sampler.h"

using namespace tinf;

namespace {

template <typename T>
T* to_device(const std::vector<T>& v) {
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(T) + 16));
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

template <typename T>
std::vector<T> to_host(const T* d, size_t n) {
    std::vector<T> v(n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

void test_gemv(tt::Rng& rng) {
    std::printf("[gemv]\n");
    struct Shape { int rows, cols; };
    const Shape shapes[] = {{8, 64}, {1001, 256}, {2048, 2048}, {5632, 2048}, {2048, 5632}, {32000, 2048}, {333, 11008}};
    for (Quant q : {Quant::F16, Quant::Q8, Quant::Q4}) {
        for (const Shape& sh : shapes) {
            if (sh.cols > gpu::gemv_max_cols()) continue;
            const int group = 64;
            std::vector<float> w = rng.normals((size_t)sh.rows * sh.cols, 0.02f);
            std::vector<float> x = rng.normals(sh.cols), y0 = rng.normals(sh.rows);
            QuantizedMatrix qm = quantize(w.data(), sh.rows, sh.cols, q, group);
            QMatrix hm = qm.view(sh.rows, sh.cols, q, group);

            QMatrix dm = hm;
            dm.w = to_device(qm.payload);
            dm.scales = qm.scales.empty() ? nullptr : to_device(qm.scales);
            float* dx = to_device(x);
            float* dy = to_device(y0);

            for (int accumulate = 0; accumulate <= 1; ++accumulate) {
                std::vector<float> ref = y0;
                cpu::matvec(hm, x.data(), ref.data(), accumulate != 0);
                CUDA_CHECK(cudaMemcpy(dy, y0.data(), y0.size() * 4, cudaMemcpyHostToDevice));
                gpu::gemv(dm, dx, dy, accumulate != 0, 0);
                std::vector<float> got = to_host(dy, sh.rows);
                char name[96];
                std::snprintf(name, sizeof(name), "%s %5dx%-5d %s", quant_name(q), sh.rows, sh.cols,
                              accumulate ? "(y+=Wx)" : "(y=Wx)");
                tt::expect_close(name, got.data(), ref.data(), sh.rows, 1e-4);
            }
            cudaFree((void*)dm.w);
            if (dm.scales) cudaFree((void*)dm.scales);
            cudaFree(dx);
            cudaFree(dy);
        }
    }
}

void test_rmsnorm_swiglu_embed(tt::Rng& rng) {
    std::printf("[rmsnorm / swiglu / embed / argmax]\n");
    for (int n : {64, 2048, 4096, 5000}) {
        std::vector<float> x = rng.normals(n, 3.f), w = rng.normals(n), ref(n);
        cpu::rmsnorm(ref.data(), x.data(), w.data(), n, 1e-5f);
        float *dx = to_device(x), *dw = to_device(w), *dout = to_device(std::vector<float>(n));
        gpu::rmsnorm(dout, dx, dw, n, 1e-5f, 0);
        auto got = to_host(dout, n);
        tt::expect_close(("rmsnorm n=" + std::to_string(n)).c_str(), got.data(), ref.data(), n, 1e-5);
        cudaFree(dx); cudaFree(dw); cudaFree(dout);
    }
    {
        const int n = 5632;
        std::vector<float> g = rng.normals(n, 4.f), u = rng.normals(n), ref = g;
        cpu::swiglu(ref.data(), u.data(), n);
        float *dg = to_device(g), *du = to_device(u);
        gpu::swiglu(dg, du, n, 0);
        auto got = to_host(dg, n);
        tt::expect_close("swiglu n=5632", got.data(), ref.data(), n, 1e-6);
        cudaFree(dg); cudaFree(du);
    }
    {
        const int V = 1000, D = 256, token = 777;
        std::vector<uint16_t> table((size_t)V * D);
        for (auto& h : table) h = float_to_half(rng.normal());
        std::vector<float> ref(D);
        for (int i = 0; i < D; ++i) ref[i] = half_to_float(table[(size_t)token * D + i]);
        uint16_t* dt = to_device(table);
        int* dtok = to_device(std::vector<int>{token});
        float* dx = to_device(std::vector<float>(D));
        gpu::embed(dx, reinterpret_cast<__half*>(dt), dtok, D, 0);
        auto got = to_host(dx, D);
        tt::expect_close("embed", got.data(), ref.data(), D, 0.0);
        cudaFree(dt); cudaFree(dtok); cudaFree(dx);
    }
    for (int n : {32000, 50257, 7}) {
        std::vector<float> x = rng.normals(n);
        x[n / 3] = 100.f;
        x[n - 1] = 100.f;  // tie: lowest index must win
        float* dx = to_device(x);
        int* dout = to_device(std::vector<int>{-1});
        gpu::argmax(dout, dx, n, 0);
        int got = to_host(dout, 1)[0];
        bool ok = got == tinf::argmax(x.data(), n) && got == n / 3;
        std::printf("  argmax n=%-38d %s\n", n, ok ? "ok" : "FAIL");
        if (!ok) ++tt::failures();
        cudaFree(dx); cudaFree(dout);
    }
}

void test_rope_attention(tt::Rng& rng) {
    std::printf("[rope + kv cache + attention]\n");
    struct Cfg { int H, KV, D, ctx; };
    for (Cfg c : {Cfg{32, 4, 64, 600}, Cfg{32, 32, 128, 300}, Cfg{8, 2, 256, 70}}) {
        const int kv_dim = c.KV * c.D;
        std::vector<uint16_t> kc_ref((size_t)c.ctx * kv_dim, 0), vc_ref((size_t)c.ctx * kv_dim, 0);
        __half* dkc = to_device(std::vector<__half>((size_t)c.ctx * kv_dim));
        __half* dvc = to_device(std::vector<__half>((size_t)c.ctx * kv_dim));
        float* dq = to_device(std::vector<float>(c.H * c.D));
        float* dk = to_device(std::vector<float>(kv_dim));
        float* dv = to_device(std::vector<float>(kv_dim));
        float* dout = to_device(std::vector<float>(c.H * c.D));
        int* dpos = to_device(std::vector<int>{0});
        double worst_q = 0, worst_att = 0;
        bool cache_ok = true;
        for (int pos = 0; pos < c.ctx; ++pos) {
            std::vector<float> q = rng.normals(c.H * c.D), k = rng.normals(kv_dim), v = rng.normals(kv_dim);
            CUDA_CHECK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dk, k.data(), k.size() * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dpos, &pos, 4, cudaMemcpyHostToDevice));
            gpu::rope_kv(dq, dk, dv, dkc, dvc, dpos, c.H, c.KV, c.D, 10000.f, 0);
            gpu::attention(dout, dq, dkc, dvc, dpos, c.H, c.KV, c.D, 0);

            cpu::rope(q.data(), c.H, c.D, pos, 10000.f);
            cpu::rope(k.data(), c.KV, c.D, pos, 10000.f);
            for (int i = 0; i < kv_dim; ++i) {
                kc_ref[(size_t)pos * kv_dim + i] = float_to_half(k[i]);
                vc_ref[(size_t)pos * kv_dim + i] = float_to_half(v[i]);
            }
            std::vector<float> ref(c.H * c.D);
            cpu::attention(ref.data(), q.data(), kc_ref.data(), vc_ref.data(), pos, c.H, c.KV, c.D);

            if (pos % 37 == 0 || pos == c.ctx - 1) {
                auto gq = to_host(dq, q.size());
                auto gout = to_host(dout, ref.size());
                double mq = 0, mr = 0, ma = 0, mar = 0;
                for (size_t i = 0; i < q.size(); ++i) {
                    mq = std::fmax(mq, std::fabs(gq[i] - q[i]));
                    mr = std::fmax(mr, std::fabs(q[i]));
                }
                for (size_t i = 0; i < ref.size(); ++i) {
                    ma = std::fmax(ma, std::fabs(gout[i] - ref[i]));
                    mar = std::fmax(mar, std::fabs(ref[i]));
                }
                worst_q = std::fmax(worst_q, mq / mr);
                worst_att = std::fmax(worst_att, ma / mar);
                // cached k may differ by 1 fp16 ulp due to sin/cos rounding
                auto gk = to_host(reinterpret_cast<uint16_t*>(dkc) + (size_t)pos * kv_dim, kv_dim);
                for (int i = 0; i < kv_dim; ++i) {
                    float a = half_to_float(gk[i]), b = half_to_float(kc_ref[(size_t)pos * kv_dim + i]);
                    if (std::fabs(a - b) > 2e-3f * std::fmax(1.f, std::fabs(b))) cache_ok = false;
                }
            }
        }
        char name[96];
        std::snprintf(name, sizeof(name), "H=%d KV=%d D=%d ctx=%d", c.H, c.KV, c.D, c.ctx);
        bool ok = worst_q < 5e-4 && worst_att < 5e-3 && cache_ok;
        std::printf("  %-28s rope %.1e  attn %.1e  cache %s  %s\n", name, worst_q, worst_att, cache_ok ? "ok" : "BAD",
                    ok ? "ok" : "FAIL");
        if (!ok) ++tt::failures();
        cudaFree(dkc); cudaFree(dvc); cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout); cudaFree(dpos);
    }
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device found - skipping GPU tests\n");
        return 0;
    }
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    std::printf("device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
    gpu::setup_kernels();
    tt::Rng rng(1234);
    test_gemv(rng);
    test_rmsnorm_swiglu_embed(rng);
    test_rope_attention(rng);
    return tt::finish("tinf_tests");
}
