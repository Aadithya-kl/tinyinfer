// tiny test helpers shared by the CPU and GPU test binaries
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace tt {

inline int& failures() {
    static int n = 0;
    return n;
}

#define TT_CHECK(cond)                                                                \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::printf("  FAILED: %s  (%s:%d)\n", #cond, __FILE__, __LINE__);       \
            ++tt::failures();                                                         \
        }                                                                             \
    } while (0)

struct Rng {
    std::mt19937 gen;
    explicit Rng(uint32_t seed) : gen(seed) {}
    float normal(float std = 1.f) { return std::normal_distribution<float>(0.f, std)(gen); }
    float uniform(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(gen); }
    int integer(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(gen); }
    std::vector<float> normals(size_t n, float std = 1.f) {
        std::vector<float> v(n);
        for (auto& x : v) x = normal(std);
        return v;
    }
};

// max |a-b| / max(|b|) ; prints and records a failure when above tol
inline bool expect_close(const char* what, const float* a, const float* b, size_t n, double tol) {
    double md = 0.0, mref = 0.0;
    for (size_t i = 0; i < n; ++i) {
        md = std::fmax(md, std::fabs((double)a[i] - (double)b[i]));
        mref = std::fmax(mref, std::fabs((double)b[i]));
    }
    double rel = md / std::fmax(mref, 1e-12);
    bool ok = rel <= tol && std::isfinite(rel);
    std::printf("  %-44s rel err %.2e  %s\n", what, rel, ok ? "ok" : "FAIL");
    if (!ok) ++failures();
    return ok;
}

inline int finish(const char* suite) {
    if (failures() == 0) {
        std::printf("\n%s: all tests passed\n", suite);
        return 0;
    }
    std::printf("\n%s: %d failure(s)\n", suite, failures());
    return 1;
}

}  // namespace tt
