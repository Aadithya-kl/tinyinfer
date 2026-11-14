// tinyinfer - command line interface
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tinf/engine.h"
#include "tinf/model_file.h"
#include "tinf/sampler.h"
#include "tinf/tokenizer.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

using namespace tinf;
using Clock = std::chrono::steady_clock;

namespace {

const char* kVersion = "0.1.0";

struct Args {
    std::string model;
    std::string prompt;
    bool has_prompt = false;
    bool raw = false;
    std::string system;
    std::string tmpl;
    int max_tokens = 256;
    float temperature = 0.7f;
    float top_p = 0.9f;
    uint64_t seed = 0;
    int ctx = 0;
    bool cpu = false;
    bool no_graph = false;
    int device = 0;
    int bench = 0;
    int verify = 0;
    std::string dump_logits;
    std::string tokens;
    bool tokenize = false;
    bool quiet = false;
};

void usage() {
    std::printf(
        "tinyinfer %s - a from-scratch CUDA inference engine for Llama-family LLMs\n\n"
        "usage: tinyinfer -m MODEL.tinf [options]\n\n"
        "generation:\n"
        "  -p, --prompt TEXT      one-shot prompt (chat-formatted unless --raw); omit for interactive chat\n"
        "  -f, --prompt-file F    read the prompt from a UTF-8 text file\n"
        "      --raw              plain text completion, no chat template\n"
        "  -s, --system TEXT      system prompt for chat\n"
        "      --template NAME    chat template: zephyr | llama2 | none (default: from model file)\n"
        "  -n, --max-tokens N     max new tokens per reply (default 256)\n"
        "  -t, --temperature F    sampling temperature, 0 = greedy (default 0.7)\n"
        "      --top-p F          nucleus sampling threshold (default 0.9)\n"
        "      --seed N           RNG seed (default: time)\n"
        "  -c, --ctx N            KV-cache length (default min(2048, model max))\n\n"
        "engine:\n"
        "      --cpu              use the CPU reference engine\n"
        "      --no-graph         disable CUDA-graph replay (launch kernels one by one)\n"
        "      --device N         CUDA device index (default 0)\n\n"
        "tools:\n"
        "      --bench N          decode N tokens greedily and report tokens/s and bandwidth\n"
        "      --verify N         run N tokens through GPU and CPU engines and compare logits\n"
        "      --tokenize         print the token ids of --prompt and exit\n"
        "      --tokens IDS       comma-separated token ids to use as input (testing)\n"
        "      --dump-logits FILE write float32 logits [T, vocab] for the input tokens and exit\n"
        "  -q, --quiet            no banner / stats\n"
        "  -h, --help\n",
        kVersion);
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

Args parse_args(int argc, char** argv) {
    Args a;
    auto need = [&](int& i) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "-m" || s == "--model") a.model = need(i);
        else if (s == "-p" || s == "--prompt") { a.prompt = need(i); a.has_prompt = true; }
        else if (s == "-f" || s == "--prompt-file") { a.prompt = read_file(need(i)); a.has_prompt = true; }
        else if (s == "--raw") a.raw = true;
        else if (s == "-s" || s == "--system") a.system = need(i);
        else if (s == "--template") a.tmpl = need(i);
        else if (s == "-n" || s == "--max-tokens") a.max_tokens = std::stoi(need(i));
        else if (s == "-t" || s == "--temperature") a.temperature = std::stof(need(i));
        else if (s == "--top-p") a.top_p = std::stof(need(i));
        else if (s == "--seed") a.seed = std::stoull(need(i));
        else if (s == "-c" || s == "--ctx") a.ctx = std::stoi(need(i));
        else if (s == "--cpu") a.cpu = true;
        else if (s == "--no-graph") a.no_graph = true;
        else if (s == "--device") a.device = std::stoi(need(i));
        else if (s == "--bench") a.bench = std::stoi(need(i));
        else if (s == "--verify") a.verify = std::stoi(need(i));
        else if (s == "--tokenize") a.tokenize = true;
        else if (s == "--tokens") a.tokens = need(i);
        else if (s == "--dump-logits") a.dump_logits = need(i);
        else if (s == "-q" || s == "--quiet") a.quiet = true;
        else if (s == "-h" || s == "--help") { usage(); std::exit(0); }
        else if (s == "--version") { std::printf("tinyinfer %s\n", kVersion); std::exit(0); }
        else throw std::runtime_error("unknown argument: " + s + "  (see --help)");
    }
    if (a.model.empty()) throw std::runtime_error("missing -m MODEL.tinf  (see --help)");
    return a;
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> ids;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty()) ids.push_back(std::stoi(item));
    return ids;
}

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::unique_ptr<Engine> make_engine(const ModelFile& mf, const Args& a, int ctx, bool force_cpu = false) {
#ifdef TINF_WITH_CUDA
    if (!a.cpu && !force_cpu) {
        GpuOptions o;
        o.use_graph = !a.no_graph;
        o.device = a.device;
        return make_gpu_engine(mf, ctx, o);
    }
#else
    (void)a;
    (void)force_cpu;
#endif
    return make_cpu_engine(mf, ctx);
}

// ------------------------------------------------------------------------------------------------
// generation session
// ------------------------------------------------------------------------------------------------
struct Stats {
    int prompt_tokens = 0, gen_tokens = 0;
    double prompt_s = 0, gen_s = 0;
};

class Session {
public:
    Session(Engine& e, const Tokenizer& t, Sampler& s) : eng_(e), tok_(t), sampler_(s) {}

    int pos() const { return pos_; }
    int remaining() const { return eng_.context_length() - pos_; }
    void reset() {
        pos_ = 0;
        prev_ = -1;
    }

    // Feed prompt tokens. Returns false if they do not fit in the context.
    bool feed(const std::vector<int>& toks, Stats& st) {
        if ((int)toks.size() >= remaining()) return false;
        auto t0 = Clock::now();
        for (int t : toks) {
            eng_.forward(t, pos_++);
            prev_ = t;
        }
        st.prompt_tokens += (int)toks.size();
        st.prompt_s += seconds_since(t0);
        return true;
    }

    // Generate until EOS / max_tokens / context full. Streams text to stdout.
    void generate(int max_tokens, Stats& st, bool print = true) {
        auto t0 = Clock::now();
        int n = 0;
        while (n < max_tokens && pos_ < eng_.context_length()) {
            int next = sampler_.greedy() ? eng_.argmax() : sampler_.sample(eng_.logits());
            if (next == tok_.eos()) break;
            if (print) {
                std::string piece = tok_.decode(prev_, next);
                std::fwrite(piece.data(), 1, piece.size(), stdout);
                std::fflush(stdout);
            }
            ++n;
            eng_.forward(next, pos_++);
            prev_ = next;
        }
        if (pos_ >= eng_.context_length()) std::fprintf(stderr, "\n[context full]");
        st.gen_tokens += n;
        st.gen_s += seconds_since(t0);
    }

private:
    Engine& eng_;
    const Tokenizer& tok_;
    Sampler& sampler_;
    int pos_ = 0;
    int prev_ = -1;
};

void print_stats(const Stats& st) {
    std::fprintf(stderr, "\n[prompt %d tok, %.1f tok/s | generated %d tok, %.1f tok/s]\n", st.prompt_tokens,
                 st.prompt_s > 0 ? st.prompt_tokens / st.prompt_s : 0.0, st.gen_tokens,
                 st.gen_s > 0 ? st.gen_tokens / st.gen_s : 0.0);
}

// Token sequence for one user turn. `first` = first turn of the conversation.
std::vector<int> chat_turn(const Tokenizer& tok, ChatTemplate tmpl, const std::string& system,
                           const std::string& user, bool first) {
    std::vector<int> out;
    auto append = [&](const std::vector<int>& v) { out.insert(out.end(), v.begin(), v.end()); };
    switch (tmpl) {
        case ChatTemplate::Zephyr: {
            // <|system|>\n{sys}</s>\n<|user|>\n{msg}</s>\n<|assistant|>\n
            if (first) {
                out.push_back(tok.bos());
                if (!system.empty()) {
                    append(tok.encode("<|system|>\n" + system, true));
                    out.push_back(tok.eos());
                    append(tok.encode("\n<|user|>\n" + user, false));
                } else {
                    append(tok.encode("<|user|>\n" + user, true));
                }
            } else {
                out.push_back(tok.eos());  // close the previous assistant reply
                append(tok.encode("\n<|user|>\n" + user, false));
            }
            out.push_back(tok.eos());
            append(tok.encode("\n<|assistant|>\n", false));
            break;
        }
        case ChatTemplate::Llama2: {
            // <s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{msg} [/INST]
            if (!first) out.push_back(tok.eos());
            out.push_back(tok.bos());
            std::string text = "[INST] ";
            if (first && !system.empty()) text += "<<SYS>>\n" + system + "\n<</SYS>>\n\n";
            text += user + " [/INST]";
            append(tok.encode(text, true));
            break;
        }
        case ChatTemplate::None: {
            if (first) out.push_back(tok.bos());
            append(tok.encode(user, first));
            break;
        }
    }
    return out;
}

ChatTemplate resolve_template(const Args& a, const Config& c) {
    if (a.raw) return ChatTemplate::None;
    if (a.tmpl.empty()) return c.chat_template;
    if (a.tmpl == "zephyr") return ChatTemplate::Zephyr;
    if (a.tmpl == "llama2") return ChatTemplate::Llama2;
    if (a.tmpl == "none") return ChatTemplate::None;
    throw std::runtime_error("unknown --template " + a.tmpl);
}

// ------------------------------------------------------------------------------------------------
// modes
// ------------------------------------------------------------------------------------------------
int run_bench(Engine& eng, const Tokenizer& tok, int n) {
    const int warmup = 8;
    n = std::min(n, eng.context_length() - warmup - 2);
    if (n <= 0) throw std::runtime_error("context too small for benchmark");
    int pos = 0;
    eng.forward(tok.bos(), pos++);
    for (int i = 0; i < warmup; ++i) eng.forward(eng.argmax(), pos++);

    auto t0 = Clock::now();
    for (int i = 0; i < n; ++i) eng.forward(eng.argmax(), pos++);
    const double s = seconds_since(t0);

    const double tps = n / s;
    const double gb = (double)eng.weight_bytes_per_token() / 1e9;
    std::printf("engine           : %s\n", eng.name().c_str());
    std::printf("weights / token  : %.1f MB\n", gb * 1e3);
    std::printf("decode           : %d tokens in %.3f s\n", n, s);
    std::printf("throughput       : %.1f tok/s  (%.2f ms/token)\n", tps, 1e3 / tps);
    std::printf("weight bandwidth : %.1f GB/s", gb * tps);
#ifdef TINF_WITH_CUDA
    if (eng.name().rfind("cuda", 0) == 0) {
        GpuInfo gi = gpu_info();
        if (gi.peak_bandwidth_gbs > 0)
            std::printf("  (%.0f%% of %.0f GB/s peak on %s)", 100.0 * gb * tps / gi.peak_bandwidth_gbs,
                        gi.peak_bandwidth_gbs, gi.name.c_str());
    }
#endif
    std::printf("\n");
    return 0;
}

int run_verify(const ModelFile& mf, const Args& a, const Tokenizer& tok, int n) {
#ifndef TINF_WITH_CUDA
    (void)mf; (void)a; (void)tok; (void)n;
    throw std::runtime_error("--verify needs a CUDA build");
#else
    std::vector<int> ids = a.tokens.empty() ? std::vector<int>{tok.bos()} : parse_ids(a.tokens);
    if (a.tokens.empty()) {
        std::string text = a.has_prompt ? a.prompt
                                        : "The quick brown fox jumps over the lazy dog. CUDA kernels stream weights "
                                          "from memory, so decoding is bandwidth bound.";
        auto t = tok.encode(text, true);
        ids.insert(ids.end(), t.begin(), t.end());
    }
    const size_t base = ids.size();  // repeat the prompt (after BOS) until we have n tokens
    while ((int)ids.size() < n) ids.push_back(base > 1 ? ids[1 + (ids.size() - base) % (base - 1)] : tok.bos());
    ids.resize(n);
    const int ctx = n + 1;
    std::printf("building engines (ctx %d) ...\n", ctx);
    auto gpu = make_engine(mf, a, ctx);
    auto cpu = make_cpu_engine(mf, ctx);
    const int V = mf.config().vocab_size;
    double worst = 0.0;
    int mismatches = 0;
    std::printf("%5s %8s %12s %12s %8s\n", "pos", "token", "max|diff|", "rel", "argmax");
    for (int p = 0; p < n; ++p) {
        gpu->forward(ids[p], p);
        cpu->forward(ids[p], p);
        const float* g = gpu->logits();
        const float* c = cpu->logits();
        double md = 0.0, mref = 0.0;
        for (int i = 0; i < V; ++i) {
            md = std::max(md, (double)std::fabs(g[i] - c[i]));
            mref = std::max(mref, (double)std::fabs(c[i]));
        }
        const double rel = md / std::max(mref, 1e-6);
        const bool same = tinf::argmax(g, V) == tinf::argmax(c, V);
        mismatches += same ? 0 : 1;
        worst = std::max(worst, rel);
        std::printf("%5d %8d %12.3e %12.3e %8s\n", p, ids[p], md, rel, same ? "ok" : "DIFF");
    }
    const bool pass = worst < 2e-3 && mismatches <= std::max(1, n / 20);
    std::printf("\nworst relative error %.3e, argmax mismatches %d/%d -> %s\n", worst, mismatches, n,
                pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
#endif
}

int run_dump(Engine& eng, const std::vector<int>& ids, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    const int V = eng.config().vocab_size;
    for (size_t p = 0; p < ids.size(); ++p) {
        eng.forward(ids[p], (int)p);
        f.write(reinterpret_cast<const char*>(eng.logits()), (std::streamsize)V * sizeof(float));
    }
    return 0;
}

int run(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    ModelFile mf(a.model);
    const Config& c = mf.config();
    Tokenizer tok(mf.tokenizer());

    if (a.tokenize) {
        auto ids = tok.encode(a.prompt, true);
        for (size_t i = 0; i < ids.size(); ++i) std::printf(i ? " %d" : "%d", ids[i]);
        std::printf("\n");
        return 0;
    }

    int ctx = a.ctx > 0 ? a.ctx : std::min(2048, c.max_seq_len);
    ctx = std::min(ctx, c.max_seq_len);

    if (!a.quiet) {
        std::fprintf(stderr, "tinyinfer %s | %s | dim %d, %d layers, %d/%d heads, vocab %d, %s",
                     kVersion, mf.path().c_str(), c.dim, c.n_layers, c.n_heads, c.n_kv_heads, c.vocab_size,
                     quant_name(c.quant));
        if (c.quant != Quant::F16) std::fprintf(stderr, " (g%d)", c.group_size);
        std::fprintf(stderr, "\n");
#ifdef TINF_WITH_CUDA
        if (!a.cpu) {
            GpuInfo gi = gpu_info(a.device);
            std::fprintf(stderr, "GPU: %s (sm_%d%d, %d SMs, %.1f GiB, ~%.0f GB/s)\n", gi.name.c_str(), gi.cc_major,
                         gi.cc_minor, gi.sm_count, gi.total_mem / 1073741824.0, gi.peak_bandwidth_gbs);
        }
#endif
    }

    if (a.verify > 0) return run_verify(mf, a, tok, a.verify);

    if (!a.dump_logits.empty()) {
        std::vector<int> ids = a.tokens.empty() ? std::vector<int>{} : parse_ids(a.tokens);
        if (ids.empty()) {
            ids.push_back(tok.bos());
            auto t = tok.encode(a.prompt, true);
            ids.insert(ids.end(), t.begin(), t.end());
        }
        auto eng = make_engine(mf, a, std::max<int>(ctx, (int)ids.size()));
        return run_dump(*eng, ids, a.dump_logits);
    }

    if (a.bench > 0) ctx = std::min(c.max_seq_len, std::max(ctx, a.bench + 16));
    auto t_load = Clock::now();
    auto eng = make_engine(mf, a, ctx);
    if (!a.quiet) {
        std::fprintf(stderr, "engine %s loaded in %.2f s", eng->name().c_str(), seconds_since(t_load));
        std::string ms = eng->memory_summary();
        if (!ms.empty()) std::fprintf(stderr, " | %s", ms.c_str());
        std::fprintf(stderr, "\n");
    }

    if (a.bench > 0) return run_bench(*eng, tok, a.bench);

    uint64_t seed = a.seed ? a.seed : (uint64_t)std::chrono::system_clock::now().time_since_epoch().count();
    Sampler sampler(c.vocab_size, a.temperature, a.top_p, seed);
    Session session(*eng, tok, sampler);
    const ChatTemplate tmpl = resolve_template(a, c);

    if (a.has_prompt) {
        Stats st;
        std::vector<int> ids;
        if (!a.tokens.empty()) {
            ids = parse_ids(a.tokens);
        } else if (tmpl == ChatTemplate::None) {
            ids.push_back(tok.bos());
            auto t = tok.encode(a.prompt, true);
            ids.insert(ids.end(), t.begin(), t.end());
            std::fwrite(a.prompt.data(), 1, a.prompt.size(), stdout);
        } else {
            ids = chat_turn(tok, tmpl, a.system, a.prompt, true);
        }
        if (!session.feed(ids, st)) throw std::runtime_error("prompt does not fit in the context (--ctx)");
        session.generate(a.max_tokens, st);
        std::printf("\n");
        if (!a.quiet) print_stats(st);
        return 0;
    }

    // interactive chat
    std::fprintf(stderr, "chat template: %s | commands: /reset, /exit\n", template_name(tmpl));
    bool first = true;
    std::string line;
    while (true) {
        std::fprintf(stderr, "\n> ");
        if (!std::getline(std::cin, line)) break;
        if (line == "/exit" || line == "/quit") break;
        if (line == "/reset") {
            session.reset();
            first = true;
            std::fprintf(stderr, "[conversation reset]\n");
            continue;
        }
        if (line.empty()) continue;
        std::vector<int> ids = chat_turn(tok, tmpl, a.system, line, first);
        Stats st;
        if (!session.feed(ids, st)) {
            std::fprintf(stderr, "[context full - conversation reset]\n");
            session.reset();
            first = true;
            ids = chat_turn(tok, tmpl, a.system, line, true);
            if (!session.feed(ids, st)) {
                std::fprintf(stderr, "[message too long for --ctx %d]\n", eng->context_length());
                continue;
            }
        }
        first = false;
        session.generate(a.max_tokens, st);
        std::printf("\n");
        if (!a.quiet) print_stats(st);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
