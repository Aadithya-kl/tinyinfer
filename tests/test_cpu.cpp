// CPU unit tests: fp16 conversion, quantization, tokenizer, sampler, reference ops. No GPU needed.
#include <cstring>

#include "test_util.h"
#include "tinf/cpu_ops.h"
#include "tinf/half.h"
#include "tinf/quant.h"
#include "tinf/sampler.h"
#include "tinf/tokenizer.h"

using namespace tinf;

static float bits_to_float(uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

static void test_half() {
    std::printf("[fp16]\n");
    int bad = 0;
    for (uint32_t h = 0; h < 65536; ++h) {
        uint16_t hb = (uint16_t)h;
        bool nan = ((hb >> 10) & 31) == 31 && (hb & 1023);
        if (nan) continue;
        if (float_to_half(half_to_float(hb)) != hb) ++bad;
    }
    TT_CHECK(bad == 0);
    TT_CHECK(float_to_half(1.0f) == 0x3C00);
    TT_CHECK(float_to_half(-2.0f) == 0xC000);
    TT_CHECK(float_to_half(65504.f) == 0x7BFF);
    TT_CHECK(float_to_half(65520.f) == 0x7C00);              // rounds to +inf
    TT_CHECK(float_to_half(1.0f + 1.0f / 2048) == 0x3C00);   // tie -> even
    TT_CHECK(float_to_half(1.0f + 3.0f / 2048) == 0x3C02);   // tie -> even (up)
    TT_CHECK(float_to_half(bits_to_float(0x33800000u)) == 0x0001);  // 2^-24, smallest subnormal
    TT_CHECK(float_to_half(bits_to_float(0x33000000u)) == 0x0000);  // 2^-25 tie -> 0
    TT_CHECK(float_to_half(bits_to_float(0x33400000u)) == 0x0001);  // 1.5 * 2^-25
    TT_CHECK(half_to_float(0x3555) > 0.333f && half_to_float(0x3555) < 0.3334f);
    std::printf("  exhaustive round trip + rounding cases     %s\n", bad == 0 ? "ok" : "FAIL");
}

static void test_quant() {
    std::printf("[quantization]\n");
    tt::Rng rng(1);
    const int rows = 37, cols = 256, group = 64;
    std::vector<float> w = rng.normals((size_t)rows * cols, 0.05f);
    for (Quant q : {Quant::F16, Quant::Q8, Quant::Q4}) {
        QuantizedMatrix qm = quantize(w.data(), rows, cols, q, group);
        QMatrix m = qm.view(rows, cols, q, group);
        double worst = 0.0;
        for (int r = 0; r < rows; ++r)
            for (int g = 0; g < cols / group; ++g) {
                float amax = 0.f;
                for (int i = 0; i < group; ++i) amax = std::fmax(amax, std::fabs(w[(size_t)r * cols + g * group + i]));
                for (int i = 0; i < group; ++i) {
                    int c = g * group + i;
                    double err = std::fabs(dequant_at(m, r, c) - w[(size_t)r * cols + c]);
                    double bound = q == Quant::F16 ? 1e-3 * amax : amax / (q == Quant::Q8 ? 127.0 : 7.0) * 0.5 * 1.01;
                    worst = std::fmax(worst, err / std::fmax(bound, 1e-12));
                }
            }
        std::printf("  %-4s max error / bound = %.3f\n", quant_name(q), worst);
        TT_CHECK(worst <= 1.0);

        // matvec vs naive dequantized product
        std::vector<float> x = rng.normals(cols), y(rows), ref(rows, 0.f);
        cpu::matvec(m, x.data(), y.data(), false);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) ref[r] += dequant_at(m, r, c) * x[c];
        tt::expect_close((std::string("matvec ") + quant_name(q)).c_str(), y.data(), ref.data(), rows, 1e-5);
    }
}

static TokenizerData make_tok() {
    // vocab: <unk> <s> </s> <0x00>..<0xFF>, then normal pieces
    TokenizerData d;
    d.bos = 1;
    d.eos = 2;
    d.unk = 0;
    d.add_prefix = true;
    auto add = [&](const std::string& s, uint8_t type) {
        d.pieces.push_back(s);
        d.types.push_back(type);
        return (int)d.pieces.size() - 1;
    };
    add("<unk>", TOK_CONTROL);
    add("<s>", TOK_CONTROL);
    add("</s>", TOK_CONTROL);
    for (int b = 0; b < 256; ++b) d.byte_ids[b] = add(std::string(1, (char)b), TOK_BYTE);
    const char* normal[] = {" ", "h", "e", "l", "o", "w", "r", "d", "he", "ll", "llo", "hello", " hello", " w", "or",
                            " wor", " world", "ld"};
    std::vector<int> ids;
    for (const char* s : normal) ids.push_back(add(s, TOK_NORMAL));
    auto id = [&](const std::string& s) {
        for (size_t i = 0; i < d.pieces.size(); ++i)
            if (d.types[i] == TOK_NORMAL && d.pieces[i] == s) return (int)i;
        return -1;
    };
    auto merge = [&](const std::string& a, const std::string& b) {
        d.merges.push_back(id(a));
        d.merges.push_back(id(b));
        d.merges.push_back(id(a + b));
    };
    merge("h", "e");      // rank 0
    merge("l", "l");      // rank 1
    merge("ll", "o");     // rank 2
    merge("he", "llo");   // rank 3
    merge(" ", "hello");  // rank 4
    merge(" ", "w");      // rank 5
    merge("o", "r");      // rank 6
    merge(" w", "or");    // rank 7
    merge("l", "d");      // rank 8
    merge(" wor", "ld");  // rank 9
    return d;
}

static void test_tokenizer() {
    std::printf("[tokenizer]\n");
    TokenizerData d = make_tok();
    Tokenizer tok(d);
    auto find = [&](const std::string& s) {
        for (size_t i = 0; i < d.pieces.size(); ++i)
            if (d.types[i] == TOK_NORMAL && d.pieces[i] == s) return (int)i;
        return -1;
    };
    std::vector<int> ids = tok.encode("hello world");
    TT_CHECK(ids.size() == 2);
    if (ids.size() == 2) {
        TT_CHECK(ids[0] == find(" hello"));
        TT_CHECK(ids[1] == find(" world"));
    }
    // no prefix
    ids = tok.encode("hello", false);
    TT_CHECK(ids.size() == 1 && ids[0] == find("hello"));
    // byte fallback for a character not in the vocab (U+00E9 = C3 A9)
    ids = tok.encode("h\xC3\xA9", false);
    TT_CHECK(ids.size() == 3 && ids[0] == find("h") && ids[1] == d.byte_ids[0xC3] && ids[2] == d.byte_ids[0xA9]);
    // decode round trip (BOS strips the leading space)
    ids = tok.encode("hello world");
    std::string text;
    int prev = tok.bos();
    for (int t : ids) {
        text += tok.decode(prev, t);
        prev = t;
    }
    TT_CHECK(text == "hello world");
    TT_CHECK(tok.decode(5, tok.eos()).empty());
    std::printf("  merges / byte fallback / decode              %s\n", tt::failures() ? "FAIL" : "ok");
}

static void test_sampler() {
    std::printf("[sampler]\n");
    std::vector<float> logits(100, 0.f);
    logits[42] = 10.f;
    Sampler greedy(100, 0.f, 0.9f, 1);
    TT_CHECK(greedy.sample(logits.data()) == 42);
    Sampler nucleus(100, 1.0f, 0.5f, 7);
    int hits = 0;
    for (int i = 0; i < 200; ++i) hits += nucleus.sample(logits.data()) == 42;
    TT_CHECK(hits == 200);  // p(42) ~ 0.995 > top_p -> nucleus is {42}
    std::vector<float> flat(4, 0.f);
    Sampler s(4, 1.0f, 1.0f, 3);
    int counts[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4000; ++i) counts[s.sample(flat.data())]++;
    for (int c : counts) TT_CHECK(c > 850 && c < 1150);
    std::printf("  greedy / nucleus / uniform                   %s\n", tt::failures() ? "FAIL" : "ok");
}

static void test_ops() {
    std::printf("[reference ops]\n");
    tt::Rng rng(5);
    // rope: pos 0 is identity, rotation preserves pair norms
    std::vector<float> x = rng.normals(4 * 64), y = x;
    cpu::rope(y.data(), 4, 64, 0, 10000.f);
    tt::expect_close("rope pos=0 identity", y.data(), x.data(), x.size(), 0.0);
    y = x;
    cpu::rope(y.data(), 4, 64, 123, 10000.f);
    double dn = 0;
    for (int h = 0; h < 4; ++h)
        for (int i = 0; i < 32; ++i) {
            const float* a = &x[h * 64];
            const float* b = &y[h * 64];
            dn = std::fmax(dn, std::fabs(std::hypot(a[i], a[i + 32]) - std::hypot(b[i], b[i + 32])));
        }
    TT_CHECK(dn < 1e-5);
    // attention at pos 0 returns v0
    const int H = 4, KV = 2, D = 64;
    std::vector<float> q = rng.normals(H * D), out(H * D), expect(H * D);
    std::vector<uint16_t> kc(KV * D), vc(KV * D);
    for (int i = 0; i < KV * D; ++i) {
        kc[i] = float_to_half(rng.normal());
        vc[i] = float_to_half(rng.normal());
    }
    cpu::attention(out.data(), q.data(), kc.data(), vc.data(), 0, H, KV, D);
    for (int h = 0; h < H; ++h)
        for (int d = 0; d < D; ++d) expect[h * D + d] = half_to_float(vc[(h / (H / KV)) * D + d]);
    tt::expect_close("attention pos=0 == v", out.data(), expect.data(), out.size(), 1e-6);
}

int main() {
    test_half();
    test_quant();
    test_tokenizer();
    test_sampler();
    test_ops();
    return tt::finish("tinf_cpu_tests");
}
