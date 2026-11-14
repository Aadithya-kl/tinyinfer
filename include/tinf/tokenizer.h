// tinyinfer - SentencePiece-style BPE tokenizer (Llama-2 family), byte-exact with HF `tokenizers`
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tinf/model_file.h"

namespace tinf {

class Tokenizer {
public:
    explicit Tokenizer(const TokenizerData& data);

    // Encode text. `prefix` controls the SentencePiece dummy-prefix space (only applied when the
    // tokenizer was trained with one). Use prefix=false for segments that continue a sequence.
    std::vector<int> encode(const std::string& text, bool prefix = true) const;

    // Text for `token` when it follows `prev` (handles byte tokens, control tokens, BOS space strip).
    std::string decode(int prev, int token) const;

    int bos() const { return d_.bos; }
    int eos() const { return d_.eos; }
    int vocab_size() const { return (int)d_.pieces.size(); }

private:
    TokenizerData d_;
    std::unordered_map<std::string, int> piece_to_id_;
    std::unordered_map<uint64_t, std::pair<int, int>> merges_;  // (left,right) -> (rank, merged)
};

}  // namespace tinf
