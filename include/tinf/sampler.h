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
    // repetition penalty over the last `last_n` tokens (llama.cpp style). penalty 1 or last_n 0 disables it.
    void set_repeat_penalty(float penalty, int last_n);
    // record a token (prompt or generated) for the repetition penalty
    void accept(int token);
    void reset_history() { history_.clear(); }
    // false when the engine's on-device argmax gives the same answer as sample()
    bool needs_logits() const { return !greedy() || penalty_active(); }
    int sample(const float* logits);

private:
    float next_float();  // [0, 1)
    bool penalty_active() const { return repeat_penalty_ != 1.f && repeat_last_n_ > 0; }
    int vocab_;
    float temperature_, top_p_;
    float min_p_ = 0.f;
    float repeat_penalty_ = 1.f;
    int repeat_last_n_ = 0;
    std::vector<int> history_;
    std::vector<float> penalized_;
    std::vector<uint8_t> seen_;
    uint64_t state_;
    std::vector<float> probs_;
    std::vector<int> idx_;
};

int argmax(const float* x, int n);

}  // namespace tinf
