#include "tinf/sampler.h"

#include <algorithm>
#include <cmath>

namespace tinf {

int argmax(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (x[i] > x[best]) best = i;
    return best;
}

Sampler::Sampler(int vocab_size, float temperature, float top_p, uint64_t seed)
    : vocab_(vocab_size), temperature_(temperature), top_p_(top_p), state_(seed ? seed : 0x9E3779B97F4A7C15ull),
      probs_(vocab_size), idx_(vocab_size) {}

float Sampler::next_float() {  // xorshift64*
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    uint64_t r = state_ * 0x2545F4914F6CDD1Dull;
    return (float)(r >> 40) / 16777216.0f;
}

void Sampler::set_repeat_penalty(float penalty, int last_n) {
    repeat_penalty_ = penalty;
    repeat_last_n_ = last_n;
    if (penalty_active()) {
        penalized_.resize(vocab_);
        seen_.assign(vocab_, 0);
    }
}

void Sampler::accept(int token) {
    if (!penalty_active()) return;
    history_.push_back(token);
    if ((int)history_.size() > 2 * repeat_last_n_)  // trim in batches
        history_.erase(history_.begin(), history_.end() - repeat_last_n_);
}

int Sampler::sample(const float* logits) {
    if (penalty_active() && !history_.empty()) {
        std::copy(logits, logits + vocab_, penalized_.begin());
        const size_t n = std::min(history_.size(), (size_t)repeat_last_n_);
        for (size_t i = history_.size() - n; i < history_.size(); ++i) {
            const int t = history_[i];
            if (t < 0 || t >= vocab_ || seen_[t]) continue;
            seen_[t] = 1;  // each distinct token is penalised once
            float& l = penalized_[t];
            l = l > 0.f ? l / repeat_penalty_ : l * repeat_penalty_;
        }
        for (size_t i = history_.size() - n; i < history_.size(); ++i)
            if (history_[i] >= 0 && history_[i] < vocab_) seen_[history_[i]] = 0;
        logits = penalized_.data();
    }
    if (greedy()) return argmax(logits, vocab_);

    float mx = logits[0];
    for (int i = 1; i < vocab_; ++i) mx = std::max(mx, logits[i]);
    double sum = 0.0;
    for (int i = 0; i < vocab_; ++i) {
        probs_[i] = std::exp((logits[i] - mx) / temperature_);
        sum += probs_[i];
    }
    if (min_p_ > 0.f) {
        // the most likely token has exp(0) = 1 before normalisation
        const float thr = min_p_;
        double kept = 0.0;
        for (int i = 0; i < vocab_; ++i) {
            if (probs_[i] < thr) probs_[i] = 0.f;
            kept += probs_[i];
        }
        sum = kept;
    }
    for (int i = 0; i < vocab_; ++i) probs_[i] = (float)(probs_[i] / sum);

    float r = next_float();
    if (top_p_ <= 0.f || top_p_ >= 1.f) {
        float cdf = 0.f;
        for (int i = 0; i < vocab_; ++i) {
            cdf += probs_[i];
            if (r < cdf) return i;
        }
        return vocab_ - 1;
    }

    // nucleus: tokens below (1-p)/(n-1) can never be part of the nucleus, drop them before sorting
    const float cutoff = (1.f - top_p_) / (float)(vocab_ - 1);
    int n = 0;
    for (int i = 0; i < vocab_; ++i)
        if (probs_[i] >= cutoff) idx_[n++] = i;
    std::sort(idx_.begin(), idx_.begin() + n, [&](int a, int b) { return probs_[a] > probs_[b]; });
    float cum = 0.f;
    int last = n - 1;
    for (int i = 0; i < n; ++i) {
        cum += probs_[idx_[i]];
        if (cum > top_p_) {
            last = i;
            break;
        }
    }
    float target = r * cum, cdf = 0.f;
    for (int i = 0; i <= last; ++i) {
        cdf += probs_[idx_[i]];
        if (target < cdf) return idx_[i];
    }
    return idx_[last];
}

}  // namespace tinf
