// tinyinfer - .tinf model file reader (format: docs/FORMAT.md)
#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace tinf {

enum class DType : uint32_t { F32 = 0, F16 = 1, I8 = 2, U4 = 3 };
enum class Quant : uint32_t { F16 = 0, Q8 = 1, Q4 = 2 };
enum class ChatTemplate : uint32_t { None = 0, Zephyr = 1, Llama2 = 2 };
enum TokenType : uint8_t { TOK_NORMAL = 0, TOK_BYTE = 1, TOK_CONTROL = 2 };

const char* quant_name(Quant q);
const char* template_name(ChatTemplate t);

struct Config {
    int dim = 0;
    int hidden_dim = 0;
    int n_layers = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    int max_seq_len = 0;
    float rope_theta = 10000.f;
    float norm_eps = 1e-5f;
    Quant quant = Quant::F16;
    int group_size = 64;
    ChatTemplate chat_template = ChatTemplate::None;

    int q_dim() const { return n_heads * head_dim; }
    int kv_dim() const { return n_kv_heads * head_dim; }
};

struct TensorInfo {
    std::string name;
    DType dtype = DType::F32;
    uint32_t ndim = 0;
    uint64_t shape[2] = {1, 1};
    uint64_t offset = 0;
    uint64_t nbytes = 0;
};

struct TokenizerData {
    std::vector<std::string> pieces;  // decoded text of each token ("▁" already replaced by ' ')
    std::vector<uint8_t> types;       // TokenType
    std::vector<int32_t> merges;      // (left, right, merged) triples in rank order
    int bos = -1, eos = -1, unk = -1;
    bool add_prefix = true;
    int byte_ids[256];
};

class ModelFile {
public:
    explicit ModelFile(const std::string& path);

    const Config& config() const { return cfg_; }
    const TokenizerData& tokenizer() const { return tok_; }
    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const std::string& path() const { return path_; }
    uint64_t file_size() const { return file_size_; }

    bool has(const std::string& name) const { return index_.count(name) != 0; }
    const TensorInfo& tensor(const std::string& name) const;
    // Reads a tensor's raw bytes into dst (dst must hold info.nbytes).
    void read(const TensorInfo& info, void* dst) const;
    std::vector<uint8_t> read(const std::string& name) const;

private:
    std::string path_;
    mutable std::ifstream f_;
    uint64_t file_size_ = 0;
    Config cfg_;
    TokenizerData tok_;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, size_t> index_;
};

// A (possibly quantized) weight matrix [rows, cols], row-major. Used by both engines.
struct QMatrix {
    const void* w = nullptr;           // f16 / int8 / packed uint4 payload
    const uint16_t* scales = nullptr;  // fp16 bits, [rows, cols/group] (q8/q4 only)
    int rows = 0, cols = 0;
    Quant quant = Quant::F16;
    int group = 64;
    size_t bytes() const;  // payload + scales
};

}  // namespace tinf
