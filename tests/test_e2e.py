#!/usr/bin/env python3
"""
End-to-end correctness test against HuggingFace transformers.

  1. trains a tiny Llama-style SentencePiece-BPE tokenizer (byte fallback) with `tokenizers`
  2. builds a tiny random LlamaForCausalLM (GQA, 2 layers) and saves it as safetensors
  3. converts it with tools/convert_hf.py to f16 / q8 / q4 .tinf files
  4. runs `tinyinfer --dump-logits` and compares every position's logits with transformers,
     using the *same* (de)quantized weights on the HF side so the comparison is exact
  5. compares tokenizer output with HF `tokenizers` on tricky strings

usage: python tests/test_e2e.py --bin build/bin/tinyinfer [--gpu]
requires: torch, transformers, tokenizers, numpy
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import convert_hf as conv  # noqa: E402

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "CUDA kernels stream weights from memory, so decoding is bandwidth bound.",
    "Hello world! This is tinyinfer, a tiny inference engine written from scratch.",
    "Attention is all you need; rotary embeddings encode position.",
    "    indented code(x) -> return x * 2  # comment",
    "Numbers: 0 1 2 3 42 1024 3.14159 and symbols @#$%^&*()",
    "the theory of the thing that they thought there",
]

TOKENIZER_TESTS = [
    "Hello world",
    "hello  world   with   spaces",
    "The quick brown fox",
    "  leading spaces",
    "trailing spaces   ",
    "newlines\nand\ttabs\n\n",
    "café naïve 日本語 \U0001F600 emoji",
    "CUDA kernels stream weights",
    "a",
    "x" * 50,
    "less < than > greater, a<b and c>d",
]
# note: literal "<s>"/"</s>" in user text are intentionally NOT turned into control tokens
# (HF does that; tinyinfer treats user text as plain text to prevent control-token injection)


def build_tokenizer(out_dir):
    from tokenizers import Tokenizer, decoders, models, normalizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True, fuse_unk=True))
    tok.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    tok.decoder = decoders.Sequence([
        decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(" ", 1, 0)])
    specials = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
    trainer = trainers.BpeTrainer(vocab_size=700, special_tokens=specials, show_progress=False)
    rng = np.random.default_rng(0)
    words = " ".join(CORPUS).split()
    lines = list(CORPUS) + [" ".join(rng.choice(words, size=12)) for _ in range(400)]
    tok.train_from_iterator(lines, trainer)
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"chat_template": "{% for m in messages %}<|user|>\n{{ m['content'] }}</s>{% endfor %}"}, f)
    return tok


def build_model(out_dir, vocab_size, dtype):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=vocab_size, hidden_size=256, intermediate_size=640, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        rms_norm_eps=1e-5, rope_theta=10000.0, tie_word_embeddings=False, initializer_range=0.08,
        bos_token_id=1, eos_token_id=2,
    )
    model = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():  # non-trivial norm weights
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
    model.to(dtype).save_pretrained(out_dir, safe_serialization=True)
    return cfg


def hf_reference_logits(model_dir, quant, group, ids):
    """Logits from transformers with weights replaced by exactly what tinyinfer stores."""
    import torch
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            w = p.detach().numpy().astype(np.float32)
            if name.endswith("embed_tokens.weight"):
                w = w.astype(np.float16).astype(np.float32)  # embeddings are stored as fp16
            elif w.ndim == 2:
                w = conv.dequantize(conv.quantize(w, quant, group), quant, group, w.shape)
            p.copy_(torch.from_numpy(w))
        out = model(torch.tensor([ids])).logits[0].numpy()
    return out


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="path to the tinyinfer executable")
    ap.add_argument("--gpu", action="store_true", help="test the CUDA engine instead of the CPU engine")
    args = ap.parse_args()
    engine_flag = [] if args.gpu else ["--cpu"]
    failures = 0

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        hf_tok = build_tokenizer(tmp)
        vocab = hf_tok.get_vocab_size()
        build_model(tmp, vocab, __import__("torch").float32)

        rng = np.random.default_rng(1)
        ids = [1] + rng.integers(3, vocab, size=63).tolist()

        print(f"tiny llama: vocab {vocab}, dim 256, 2 layers, GQA 4/2, engine {'cuda' if args.gpu else 'cpu'}")
        for qname, tol in (("f16", 2e-3), ("q8", 2e-3), ("q4", 2e-3)):
            q = conv.QUANT_NAMES[qname]
            path = os.path.join(tmp, f"tiny-{qname}.tinf")
            conv.convert(tmp, path, q, 64, verbose=False)
            dump = os.path.join(tmp, f"logits-{qname}.bin")
            run([args.bin, "-m", path, "-q", *engine_flag, "--tokens", ",".join(map(str, ids)), "--dump-logits", dump])
            got = np.fromfile(dump, dtype=np.float32).reshape(len(ids), vocab)
            ref = hf_reference_logits(tmp, q, 64, ids)
            rel = np.abs(got - ref).max() / np.abs(ref).max()
            agree = (got.argmax(-1) == ref.argmax(-1)).mean()
            ok = rel < tol and agree >= 0.95
            failures += not ok
            print(f"  {qname}: max rel logit error {rel:.2e}, argmax agreement {agree * 100:.1f}%  "
                  f"{'ok' if ok else 'FAIL'}")

        # bf16 checkpoint path (most HF models ship bf16)
        bf_dir = os.path.join(tmp, "bf16")
        os.makedirs(bf_dir)
        for fn in ("tokenizer.json", "tokenizer_config.json"):
            with open(os.path.join(tmp, fn), "rb") as a, open(os.path.join(bf_dir, fn), "wb") as b:
                b.write(a.read())
        build_model(bf_dir, vocab, __import__("torch").bfloat16)
        path = os.path.join(tmp, "tiny-bf16.tinf")
        conv.convert(bf_dir, path, conv.Q_F16, 64, verbose=False)
        dump = os.path.join(tmp, "logits-bf16.bin")
        run([args.bin, "-m", path, "-q", *engine_flag, "--tokens", ",".join(map(str, ids)), "--dump-logits", dump])
        got = np.fromfile(dump, dtype=np.float32).reshape(len(ids), vocab)
        ref = hf_reference_logits(bf_dir, conv.Q_F16, 64, ids)
        rel = np.abs(got - ref).max() / np.abs(ref).max()
        ok = rel < 2e-3
        failures += not ok
        print(f"  bf16 checkpoint -> f16: max rel logit error {rel:.2e}  {'ok' if ok else 'FAIL'}")

        # tokenizer parity
        path = os.path.join(tmp, "tiny-f16.tinf")
        bad = 0
        for text in TOKENIZER_TESTS:
            want = hf_tok.encode(text, add_special_tokens=False).ids
            pf = os.path.join(tmp, "prompt.txt")
            with open(pf, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            got = [int(t) for t in run([args.bin, "-m", path, "-q", "--tokenize", "-f", pf]).split()]
            if got != want:
                bad += 1
                print(f"  tokenizer mismatch for {text!r}:\n    hf  {want}\n    got {got}")
        failures += bad > 0
        print(f"  tokenizer parity with HF tokenizers: {len(TOKENIZER_TESTS) - bad}/{len(TOKENIZER_TESTS)}  "
              f"{'ok' if bad == 0 else 'FAIL'}")

        # generation smoke test (greedy, chat template)
        out = run([args.bin, "-m", path, "-q", *engine_flag, "-p", "hello", "-t", "0", "-n", "8"])
        print(f"  generation smoke test: {len(out.strip())} chars  ok")

    print("\nE2E: " + ("all tests passed" if failures == 0 else f"{failures} failure(s)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
