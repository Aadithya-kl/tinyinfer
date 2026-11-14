#include "tinf/tokenizer.h"

#include <climits>

namespace tinf {

namespace {
inline uint64_t pair_key(int a, int b) { return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; }

// Length of the UTF-8 sequence starting with lead byte c (1 for invalid bytes).
inline size_t utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x6) return 2;
    if ((c >> 4) == 0xE) return 3;
    if ((c >> 3) == 0x1E) return 4;
    return 1;
}
}  // namespace

Tokenizer::Tokenizer(const TokenizerData& data) : d_(data) {
    piece_to_id_.reserve(d_.pieces.size() * 2);
    for (size_t i = 0; i < d_.pieces.size(); ++i) {
        if (d_.types[i] != TOK_NORMAL) continue;
        piece_to_id_.emplace(d_.pieces[i], (int)i);  // first id wins for duplicate texts
    }
    size_t n = d_.merges.size() / 3;
    merges_.reserve(n * 2);
    for (size_t r = 0; r < n; ++r) {
        int a = d_.merges[3 * r], b = d_.merges[3 * r + 1], c = d_.merges[3 * r + 2];
        merges_.emplace(pair_key(a, b), std::make_pair((int)r, c));  // keep lowest rank
    }
}

std::vector<int> Tokenizer::encode(const std::string& text, bool prefix) const {
    std::vector<int> ids;
    if (text.empty()) return ids;
    std::string s = (prefix && d_.add_prefix) ? " " + text : text;

    // 1) split into unicode characters, falling back to <0xXX> byte tokens
    for (size_t i = 0; i < s.size();) {
        size_t n = utf8_len((unsigned char)s[i]);
        if (i + n > s.size()) n = 1;
        auto it = piece_to_id_.find(s.substr(i, n));
        if (it != piece_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            for (size_t k = 0; k < n; ++k) {
                int b = d_.byte_ids[(unsigned char)s[i + k]];
                ids.push_back(b >= 0 ? b : d_.unk);
            }
        }
        i += n;
    }

    // 2) BPE: repeatedly apply the lowest-rank merge (leftmost on ties), exactly like HF.
    auto rank_of = [&](size_t i, int* merged) -> int {
        auto it = merges_.find(pair_key(ids[i], ids[i + 1]));
        if (it == merges_.end()) return INT_MAX;
        if (merged) *merged = it->second.second;
        return it->second.first;
    };
    std::vector<int> ranks(ids.size() > 1 ? ids.size() - 1 : 0);
    for (size_t i = 0; i + 1 < ids.size(); ++i) ranks[i] = rank_of(i, nullptr);
    while (!ranks.empty()) {
        size_t best = 0;
        for (size_t i = 1; i < ranks.size(); ++i)
            if (ranks[i] < ranks[best]) best = i;
        if (ranks[best] == INT_MAX) break;
        int merged = 0;
        rank_of(best, &merged);
        ids[best] = merged;
        ids.erase(ids.begin() + (long)best + 1);
        ranks.erase(ranks.begin() + (long)best);
        if (best < ranks.size()) ranks[best] = rank_of(best, nullptr);
        if (best > 0) ranks[best - 1] = rank_of(best - 1, nullptr);
    }
    return ids;
}

std::string Tokenizer::decode(int prev, int token) const {
    if (token < 0 || token >= (int)d_.pieces.size()) return std::string();
    if (d_.types[token] == TOK_CONTROL) return std::string();
    const std::string& p = d_.pieces[token];
    if (d_.types[token] == TOK_NORMAL && prev == d_.bos && !p.empty() && p[0] == ' ') return p.substr(1);
    return p;
}

}  // namespace tinf
