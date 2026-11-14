#include "tinf/model_file.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "tinf/half.h"

namespace tinf {

namespace {

template <typename T>
T get(const uint8_t*& p, const uint8_t* end) {
    if (p + sizeof(T) > end) throw std::runtime_error("tinf: truncated file");
    T v;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
}

std::string get_str(const uint8_t*& p, const uint8_t* end, size_t n) {
    if (p + n > end) throw std::runtime_error("tinf: truncated file");
    std::string s(reinterpret_cast<const char*>(p), n);
    p += n;
    return s;
}

size_t dtype_bytes(DType dt, uint64_t n) {
    switch (dt) {
        case DType::F32: return (size_t)(4 * n);
        case DType::F16: return (size_t)(2 * n);
        case DType::I8: return (size_t)n;
        case DType::U4: return (size_t)(n / 2);
    }
    return 0;
}

}  // namespace

const char* quant_name(Quant q) {
    switch (q) {
        case Quant::F16: return "f16";
        case Quant::Q8: return "q8";
        case Quant::Q4: return "q4";
    }
    return "?";
}

const char* template_name(ChatTemplate t) {
    switch (t) {
        case ChatTemplate::None: return "none";
        case ChatTemplate::Zephyr: return "zephyr";
        case ChatTemplate::Llama2: return "llama2";
    }
    return "?";
}

size_t QMatrix::bytes() const {
    size_t n = (size_t)rows * cols;
    switch (quant) {
        case Quant::F16: return 2 * n;
        case Quant::Q8: return n + 2 * n / group;
        case Quant::Q4: return n / 2 + 2 * n / group;
    }
    return 0;
}

const float* half_table() {
    static std::vector<float> table = [] {
        std::vector<float> t(65536);
        for (uint32_t i = 0; i < 65536; ++i) t[i] = half_to_float((uint16_t)i);
        return t;
    }();
    return table.data();
}

ModelFile::ModelFile(const std::string& path) : path_(path), f_(path, std::ios::binary) {
    if (!f_) throw std::runtime_error("cannot open model file: " + path);
    f_.seekg(0, std::ios::end);
    file_size_ = (uint64_t)f_.tellg();
    f_.seekg(0);

    // ---- header (128 bytes)
    std::vector<uint8_t> hdr(128);
    if (file_size_ < hdr.size() || !f_.read(reinterpret_cast<char*>(hdr.data()), hdr.size()))
        throw std::runtime_error("not a .tinf file (too small): " + path);
    const uint8_t* p = hdr.data();
    const uint8_t* end = p + hdr.size();
    if (std::memcmp(p, "TINF", 4) != 0)
        throw std::runtime_error("not a .tinf file (bad magic): " + path);
    p += 4;
    uint32_t version = get<uint32_t>(p, end);
    if (version != 1) throw std::runtime_error("unsupported .tinf version " + std::to_string(version));
    cfg_.dim = (int)get<uint32_t>(p, end);
    cfg_.hidden_dim = (int)get<uint32_t>(p, end);
    cfg_.n_layers = (int)get<uint32_t>(p, end);
    cfg_.n_heads = (int)get<uint32_t>(p, end);
    cfg_.n_kv_heads = (int)get<uint32_t>(p, end);
    cfg_.head_dim = (int)get<uint32_t>(p, end);
    cfg_.vocab_size = (int)get<uint32_t>(p, end);
    cfg_.max_seq_len = (int)get<uint32_t>(p, end);
    cfg_.rope_theta = get<float>(p, end);
    cfg_.norm_eps = get<float>(p, end);
    cfg_.quant = (Quant)get<uint32_t>(p, end);
    cfg_.group_size = (int)get<uint32_t>(p, end);
    cfg_.chat_template = (ChatTemplate)get<uint32_t>(p, end);
    uint64_t tok_offset = get<uint64_t>(p, end);
    uint64_t tok_size = get<uint64_t>(p, end);
    uint64_t dir_offset = get<uint64_t>(p, end);
    uint32_t n_tensors = get<uint32_t>(p, end);
    if ((uint32_t)cfg_.quant > 2) throw std::runtime_error("bad quant type in header");
    if (dir_offset < tok_offset + tok_size || dir_offset > file_size_)
        throw std::runtime_error("corrupt header offsets");

    // ---- tokenizer + directory (contiguous)
    uint64_t meta_end = std::min<uint64_t>(file_size_, dir_offset + 64ull * 1024 * 1024);
    std::vector<uint8_t> meta(meta_end - tok_offset);
    f_.seekg((std::streamoff)tok_offset);
    if (!f_.read(reinterpret_cast<char*>(meta.data()), (std::streamsize)meta.size()))
        throw std::runtime_error("failed to read tokenizer/directory");

    p = meta.data();
    end = meta.data() + tok_size;
    uint32_t n_vocab = get<uint32_t>(p, end);
    uint32_t n_merges = get<uint32_t>(p, end);
    tok_.bos = get<int32_t>(p, end);
    tok_.eos = get<int32_t>(p, end);
    tok_.unk = get<int32_t>(p, end);
    tok_.add_prefix = get<uint32_t>(p, end) != 0;
    for (int i = 0; i < 256; ++i) tok_.byte_ids[i] = get<int32_t>(p, end);
    tok_.pieces.resize(n_vocab);
    tok_.types.resize(n_vocab);
    for (uint32_t i = 0; i < n_vocab; ++i) {
        tok_.types[i] = get<uint8_t>(p, end);
        uint32_t len = get<uint32_t>(p, end);
        tok_.pieces[i] = get_str(p, end, len);
    }
    tok_.merges.resize((size_t)n_merges * 3);
    for (size_t i = 0; i < tok_.merges.size(); ++i) tok_.merges[i] = get<int32_t>(p, end);

    p = meta.data() + (dir_offset - tok_offset);
    end = meta.data() + meta.size();
    tensors_.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; ++i) {
        TensorInfo t;
        uint16_t nlen = get<uint16_t>(p, end);
        t.name = get_str(p, end, nlen);
        t.dtype = (DType)get<uint32_t>(p, end);
        t.ndim = get<uint32_t>(p, end);
        t.shape[0] = get<uint64_t>(p, end);
        t.shape[1] = get<uint64_t>(p, end);
        t.offset = get<uint64_t>(p, end);
        t.nbytes = get<uint64_t>(p, end);
        if ((uint32_t)t.dtype > 3 || t.nbytes != dtype_bytes(t.dtype, t.shape[0] * t.shape[1]) ||
            t.offset + t.nbytes > file_size_)
            throw std::runtime_error("corrupt tensor entry: " + t.name);
        index_[t.name] = tensors_.size();
        tensors_.push_back(std::move(t));
    }

    if (cfg_.vocab_size != (int)n_vocab) throw std::runtime_error("tokenizer/model vocab size mismatch");
}

const TensorInfo& ModelFile::tensor(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end()) throw std::runtime_error("tensor not found in model file: " + name);
    return tensors_[it->second];
}

void ModelFile::read(const TensorInfo& info, void* dst) const {
    f_.clear();
    f_.seekg((std::streamoff)info.offset);
    // read in chunks: some C runtimes misbehave on single reads > 2 GiB
    char* out = static_cast<char*>(dst);
    uint64_t left = info.nbytes;
    while (left > 0) {
        std::streamsize n = (std::streamsize)std::min<uint64_t>(left, 1ull << 30);
        if (!f_.read(out, n)) throw std::runtime_error("failed to read tensor: " + info.name);
        out += n;
        left -= (uint64_t)n;
    }
}

std::vector<uint8_t> ModelFile::read(const std::string& name) const {
    const TensorInfo& t = tensor(name);
    std::vector<uint8_t> buf(t.nbytes);
    read(t, buf.data());
    return buf;
}

}  // namespace tinf
