// tinyinfer - token sampling (greedy / temperature + nucleus + min-p)
#pragma once

#include <cstdint>
#include <vector>

namespace tinf {

class Sampler {
public:
    Sampler(int vocab_size, float temperature, float top_p, uint64_t seed);
    bool greedy() const { return temperature_ <= 0.f; }
    // min-p: drop tokens whose probability is below min_p * p(most likely token). 0 disables it.
    void set_min_p(float min_p) { min_p_ = min_p; }
    int sample(const float* logits);

private:
    float next_float();  // [0, 1)
    int vocab_;
    float temperature_, top_p_;
    float min_p_ = 0.f;
    uint64_t state_;
    std::vector<float> probs_;
    std::vector<int> idx_;
};

int argmax(const float* x, int n);

}  // namespace tinf
