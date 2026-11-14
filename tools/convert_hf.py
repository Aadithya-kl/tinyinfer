#!/usr/bin/env python3
"""
Convert a HuggingFace Llama-architecture checkpoint into a single tinyinfer `.tinf` file.

    python tools/convert_hf.py --hf TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quant q4 --out models/tinyllama-q4.tinf
    python tools/convert_hf.py --model-dir ./my_llama --quant f16 --out models/my-f16.tinf

Only numpy is required (plus huggingface_hub when using --hf). Weights are read straight from
.safetensors files (F32 / F16 / BF16) without PyTorch.

The file layout is documented in docs/FORMAT.md.
"""
import argparse
import json
import os
import re
import struct
import sys

import numpy as np

MAGIC = b"TINF"
VERSION = 1
ALIGN = 256
HEADER_SIZE = 128

DT_F32, DT_F16, DT_I8, DT_U4 = 0, 1, 2, 3
Q_F16, Q_Q8, Q_Q4 = 0, 1, 2
QUANT_NAMES = {"f16": Q_F16, "q8": Q_Q8, "q4": Q_Q4}
TEMPLATE_NONE, TEMPLATE_ZEPHYR, TEMPLATE_LLAMA2 = 0, 1, 2
TOK_NORMAL, TOK_BYTE, TOK_CONTROL = 0, 1, 2

SPIECE_UNDERLINE = "▁"  # the SentencePiece "▁" meta-space


# --------------------------------------------------------------------------------------
# safetensors reader (no torch)
# --------------------------------------------------------------------------------------
class SafeTensors:
    def __init__(self, model_dir):
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, encoding="utf-8") as f:
                files = sorted(set(json.load(f)["weight_map"].values()))
        else:
            files = sorted(fn for fn in os.listdir(model_dir) if fn.endswith(".safetensors"))
        if not files:
            raise SystemExit(f"error: no .safetensors files found in {model_dir}")
        self.index = {}
        for fn in files:
            path = os.path.join(model_dir, fn)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            mm = np.memmap(path, dtype=np.uint8, mode="r")
            base = 8 + n
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                s, e = info["data_offsets"]
                self.index[name] = (mm, base + s, base + e, info["dtype"], info["shape"])

    def has(self, name):
        return name in self.index

    def get(self, name):
        """Return the tensor as a contiguous float32 ndarray."""
        if name not in self.index:
            raise SystemExit(f"error: tensor '{name}' not found in checkpoint")
        mm, s, e, dt, shape = self.index[name]
        raw = np.asarray(mm[s:e])
        if dt == "F32":
            a = raw.view(np.float32)
        elif dt == "F16":
            a = raw.view(np.float16).astype(np.float32)
        elif dt == "BF16":
            a = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            raise SystemExit(f"error: unsupported dtype {dt} for tensor {name}")
        return np.ascontiguousarray(a, dtype=np.float32).reshape(shape)


# --------------------------------------------------------------------------------------
# quantization (must match include/tinf/quant.h and the CUDA kernels)
# --------------------------------------------------------------------------------------
def quantize(w, quant, group):
    """Quantize a [rows, cols] float32 matrix.

    Returns a list of (dtype, ndarray) pairs: the weight payload and, for q8/q4, the fp16
    per-group scales of shape [rows, cols // group].

    q8: w ~= q * s,        q in [-127, 127] stored as int8,      s = amax / 127
    q4: w ~= (n - 8) * s,  n in [1, 15] stored as packed nibbles, s = amax / 7
        byte j of a row holds element 2j in its low nibble and element 2j+1 in its high nibble.
    """
    rows, cols = w.shape
    if quant == Q_F16:
        return [(DT_F16, w.astype(np.float16))]
    if cols % group:
        raise SystemExit(f"error: matrix width {cols} is not a multiple of group size {group}")
    g = w.reshape(rows, cols // group, group)
    amax = np.abs(g).max(axis=2, keepdims=True)
    qmax = 127.0 if quant == Q_Q8 else 7.0
    # round the scale to fp16 *first* so that quantization uses exactly the scale the kernels see
    scale = (amax / qmax).astype(np.float16).astype(np.float32)
    inv = np.divide(1.0, scale, out=np.zeros_like(scale), where=scale > 0)
    q = np.clip(np.rint(g * inv), -qmax, qmax).astype(np.int32).reshape(rows, cols)
    scales = scale.reshape(rows, cols // group).astype(np.float16)
    if quant == Q_Q8:
        return [(DT_I8, q.astype(np.int8)), (DT_F16, scales)]
    n = (q + 8).astype(np.uint8)
    packed = (n[:, 0::2] | (n[:, 1::2] << 4)).astype(np.uint8)
    return [(DT_U4, packed), (DT_F16, scales)]


def dequantize(parts, quant, group, shape):
    """Inverse of quantize(); used by tests to build exact references."""
    rows, cols = shape
    if quant == Q_F16:
        return parts[0][1].astype(np.float32)
    payload, scales = parts[0][1], parts[1][1].astype(np.float32)
    if quant == Q_Q8:
        q = payload.astype(np.float32)
    else:
        q = np.empty((rows, cols), dtype=np.float32)
        q[:, 0::2] = (payload & 15).astype(np.float32) - 8
        q[:, 1::2] = (payload >> 4).astype(np.float32) - 8
    return (q.reshape(rows, cols // group, group) * scales[:, :, None]).reshape(rows, cols)


def quant_entries(name, shape, quant, group):
    rows, cols = shape
    if quant == Q_F16:
        return [(name, DT_F16, (rows, cols))]
    payload_dt = DT_I8 if quant == Q_Q8 else DT_U4
    return [(name, payload_dt, (rows, cols)), (name + ".scales", DT_F16, (rows, cols // group))]


def entry_nbytes(dtype, shape):
    n = 1
    for d in shape:
        n *= d
    return {DT_F32: 4 * n, DT_F16: 2 * n, DT_I8: n, DT_U4: n // 2}[dtype]


# --------------------------------------------------------------------------------------
# tokenizer export
# --------------------------------------------------------------------------------------
def _walk(node, key):
    if not node:
        return
    yield node
    if node.get("type") == "Sequence":
        for child in node.get(key, []):
            yield from _walk(child, key)


def load_tokenizer(model_dir, config):
    path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(path):
        raise SystemExit("error: tokenizer.json not found (a HuggingFace 'fast' tokenizer file is required)")
    with open(path, encoding="utf-8") as f:
        tj = json.load(f)
    model = tj["model"]
    if model.get("type") != "BPE" or not model.get("byte_fallback", False):
        raise SystemExit(
            "error: only SentencePiece-style BPE tokenizers with byte_fallback are supported "
            "(Llama-2 family: TinyLlama, Llama-2, Vicuna, ...). Byte-level BPE (Llama-3, Qwen) is on the roadmap."
        )

    vocab = dict(model["vocab"])
    added = tj.get("added_tokens", [])
    for at in added:
        vocab.setdefault(at["content"], at["id"])
    n_vocab = max(vocab.values()) + 1
    pieces = [None] * n_vocab
    for piece, idx in vocab.items():
        pieces[idx] = piece

    special = {at["content"] for at in added if at.get("special", False)}
    unk_piece = model.get("unk_token") or "<unk>"
    special.update(p for p in ("<unk>", "<s>", "</s>", unk_piece) if p in vocab)

    # does the tokenizer prepend a "▁" to the input? (legacy normalizer or Metaspace pre-tokenizer)
    add_prefix = False
    for nd in _walk(tj.get("normalizer"), "normalizers"):
        if nd.get("type") == "Prepend":
            add_prefix = True
    for nd in _walk(tj.get("pre_tokenizer"), "pretokenizers"):
        if nd.get("type") == "Metaspace":
            scheme = nd.get("prepend_scheme", "always" if nd.get("add_prefix_space", True) else "never")
            add_prefix = scheme != "never"
            if nd.get("split", False):
                print("warning: Metaspace(split=true) tokenizer; encoding may differ slightly from HF", file=sys.stderr)

    byte_re = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
    byte_ids = [-1] * 256
    types, texts = [], []
    for idx, piece in enumerate(pieces):
        if piece is None:
            types.append(TOK_CONTROL)
            texts.append(b"")
            continue
        m = byte_re.match(piece)
        if m:
            b = int(m.group(1), 16)
            byte_ids[b] = idx
            types.append(TOK_BYTE)
            texts.append(bytes([b]))
        elif piece in special:
            types.append(TOK_CONTROL)
            texts.append(piece.encode("utf-8"))
        else:
            types.append(TOK_NORMAL)
            texts.append(piece.replace(SPIECE_UNDERLINE, " ").encode("utf-8"))

    merges = []
    for rank, m in enumerate(model.get("merges", [])):
        a, b = m.split(" ", 1) if isinstance(m, str) else m
        if a not in vocab or b not in vocab or (a + b) not in vocab:
            continue  # unreachable merge, HF would skip it as well
        merges.append((vocab[a], vocab[b], vocab[a + b]))

    def tok_id(key, fallback):
        v = config.get(key, fallback)
        if isinstance(v, list):
            v = v[0]
        return int(v) if v is not None else -1

    bos = tok_id("bos_token_id", vocab.get("<s>", -1))
    eos = tok_id("eos_token_id", vocab.get("</s>", -1))
    unk = vocab.get(unk_piece, -1)

    template = TEMPLATE_NONE
    tc_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(tc_path):
        with open(tc_path, encoding="utf-8") as f:
            ct = json.load(f).get("chat_template") or ""
        if isinstance(ct, list):  # list of named templates
            ct = " ".join(t.get("template", "") for t in ct)
        if "<|user|>" in ct:
            template = TEMPLATE_ZEPHYR
        elif "[INST]" in ct:
            template = TEMPLATE_LLAMA2

    return {
        "n_vocab": n_vocab, "types": types, "texts": texts, "merges": merges,
        "bos": bos, "eos": eos, "unk": unk, "add_prefix": add_prefix,
        "byte_ids": byte_ids, "template": template,
    }


def serialize_tokenizer(tok):
    out = bytearray()
    out += struct.pack("<IIiiiI", tok["n_vocab"], len(tok["merges"]), tok["bos"], tok["eos"], tok["unk"],
                       1 if tok["add_prefix"] else 0)
    out += struct.pack("<256i", *tok["byte_ids"])
    for t, text in zip(tok["types"], tok["texts"]):
        out += struct.pack("<BI", t, len(text))
        out += text
    for a, b, c in tok["merges"]:
        out += struct.pack("<iii", a, b, c)
    return bytes(out)


# --------------------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------------------
def read_config(model_dir):
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    archs = cfg.get("architectures") or ["LlamaForCausalLM"]
    if archs[0] not in ("LlamaForCausalLM",):
        raise SystemExit(f"error: unsupported architecture {archs[0]} (supported: LlamaForCausalLM)")
    if cfg.get("rope_scaling"):
        raise SystemExit("error: rope_scaling is not supported yet")
    if cfg.get("attention_bias") or cfg.get("mlp_bias"):
        raise SystemExit("error: attention/mlp biases are not supported")
    if cfg.get("hidden_act", "silu") != "silu":
        raise SystemExit(f"error: unsupported activation {cfg.get('hidden_act')}")
    return cfg


def align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def convert(model_dir, out_path, quant, group, max_seq_len=None, verbose=True):
    cfg = read_config(model_dir)
    st = SafeTensors(model_dir)
    tok = load_tokenizer(model_dir, cfg)

    dim = cfg["hidden_size"]
    hidden = cfg["intermediate_size"]
    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads") or n_heads
    head_dim = cfg.get("head_dim") or dim // n_heads
    vocab_size = cfg["vocab_size"]
    seq = int(max_seq_len or cfg.get("max_position_embeddings", 2048))
    theta = float(cfg.get("rope_theta", 10000.0))
    eps = float(cfg.get("rms_norm_eps", 1e-5))
    q_dim, kv_dim = n_heads * head_dim, n_kv * head_dim

    if head_dim not in (64, 128, 256):
        raise SystemExit(f"error: head_dim {head_dim} unsupported (64/128/256)")
    if n_heads % n_kv:
        raise SystemExit("error: num_attention_heads must be a multiple of num_key_value_heads")
    for name, width in (("hidden_size", dim), ("q_dim", q_dim), ("intermediate_size", hidden)):
        if width % 32 or (quant != Q_F16 and width % group):
            raise SystemExit(f"error: {name}={width} must be a multiple of 32 and of the group size {group}")
    if tok["n_vocab"] > vocab_size:
        raise SystemExit(f"error: tokenizer has {tok['n_vocab']} tokens but model vocab_size is {vocab_size}")
    # pad tokenizer to the model vocabulary
    while tok["n_vocab"] < vocab_size:
        tok["types"].append(TOK_CONTROL)
        tok["texts"].append(b"")
        tok["n_vocab"] += 1

    tied = not st.has("lm_head.weight")

    # ---- plan: list of jobs, each produces one or more directory entries
    jobs = []

    def add_matrix(name, hf_name, shape):
        def produce(hf_name=hf_name, shape=shape):
            w = st.get(hf_name)
            assert tuple(w.shape) == tuple(shape), f"{hf_name}: expected {shape}, got {w.shape}"
            return [a for _, a in quantize(w, quant, group)]
        jobs.append((quant_entries(name, shape, quant, group), produce))

    def add_vector(name, hf_name, n):
        jobs.append(([(name, DT_F32, (n,))], lambda hf_name=hf_name: [st.get(hf_name).astype(np.float32)]))

    jobs.append(([("tok_embeddings", DT_F16, (vocab_size, dim))],
                 lambda: [st.get("model.embed_tokens.weight").astype(np.float16)]))
    for i in range(n_layers):
        p = f"model.layers.{i}."
        L = f"layers.{i}."
        add_vector(L + "attention_norm", p + "input_layernorm.weight", dim)
        add_matrix(L + "wq", p + "self_attn.q_proj.weight", (q_dim, dim))
        add_matrix(L + "wk", p + "self_attn.k_proj.weight", (kv_dim, dim))
        add_matrix(L + "wv", p + "self_attn.v_proj.weight", (kv_dim, dim))
        add_matrix(L + "wo", p + "self_attn.o_proj.weight", (dim, q_dim))
        add_vector(L + "ffn_norm", p + "post_attention_layernorm.weight", dim)
        add_matrix(L + "w1", p + "mlp.gate_proj.weight", (hidden, dim))
        add_matrix(L + "w3", p + "mlp.up_proj.weight", (hidden, dim))
        add_matrix(L + "w2", p + "mlp.down_proj.weight", (dim, hidden))
    add_vector("norm", "model.norm.weight", dim)
    add_matrix("output", "model.embed_tokens.weight" if tied else "lm_head.weight", (vocab_size, dim))

    entries = [e for es, _ in jobs for e in es]

    # ---- sizes / offsets
    tok_blob = serialize_tokenizer(tok)
    dir_blob_size = 0
    for name, dt, shape in entries:
        dir_blob_size += 2 + len(name.encode()) + 4 + 4 + 16 + 8 + 8
    tok_offset = HEADER_SIZE
    dir_offset = tok_offset + len(tok_blob)
    cursor = align(dir_offset + dir_blob_size)
    offsets = []
    for name, dt, shape in entries:
        offsets.append(cursor)
        cursor = align(cursor + entry_nbytes(dt, shape))
    total = cursor

    header = struct.pack(
        "<4sI8I2f3IQQQI",
        MAGIC, VERSION,
        dim, hidden, n_layers, n_heads, n_kv, head_dim, vocab_size, seq,
        theta, eps,
        quant, group, tok["template"],
        tok_offset, len(tok_blob), dir_offset, len(entries),
    )
    header = header + b"\0" * (HEADER_SIZE - len(header))

    directory = bytearray()
    for (name, dt, shape), off in zip(entries, offsets):
        nb = name.encode()
        s = list(shape) + [1] * (2 - len(shape))
        directory += struct.pack("<H", len(nb)) + nb
        directory += struct.pack("<IIQQQQ", dt, len(shape), s[0], s[1], off, entry_nbytes(dt, shape))

    if verbose:
        qn = {v: k for k, v in QUANT_NAMES.items()}[quant]
        print(f"model: dim={dim} hidden={hidden} layers={n_layers} heads={n_heads}/{n_kv} head_dim={head_dim} "
              f"vocab={vocab_size} ctx={seq} theta={theta:g} tied={tied}")
        print(f"quant: {qn}" + (f" (group {group})" if quant != Q_F16 else "") +
              f"  ->  {out_path}  ({total / 2**20:.1f} MiB)")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(header)
        f.write(tok_blob)
        f.write(directory)
        k = 0
        for j, (es, produce) in enumerate(jobs):
            arrays = produce()
            assert len(arrays) == len(es)
            for (name, dt, shape), arr in zip(es, arrays):
                data = np.ascontiguousarray(arr).tobytes()
                assert len(data) == entry_nbytes(dt, shape), name
                f.seek(offsets[k])
                f.write(data)
                k += 1
            if verbose:
                print(f"\r  [{j + 1}/{len(jobs)}] tensors written", end="", flush=True)
        f.seek(total - 1)
        f.write(b"\0")
    os.replace(tmp_path, out_path)
    if verbose:
        print("\ndone.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", metavar="REPO_ID", help="HuggingFace repo id to download, e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    src.add_argument("--model-dir", help="local directory with config.json, tokenizer.json and *.safetensors")
    ap.add_argument("--quant", choices=sorted(QUANT_NAMES), default="q4", help="weight format (default: q4)")
    ap.add_argument("--group", type=int, default=64, help="quantization group size (default: 64)")
    ap.add_argument("--max-seq-len", type=int, default=None, help="override max context length stored in the file")
    ap.add_argument("--out", required=True, help="output .tinf path")
    args = ap.parse_args()

    if args.group % 32 or args.group <= 0:
        raise SystemExit("error: --group must be a positive multiple of 32")

    model_dir = args.model_dir
    if args.hf:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise SystemExit("error: pip install huggingface_hub   (needed for --hf)")
        print(f"downloading {args.hf} ...")
        model_dir = snapshot_download(args.hf, allow_patterns=["*.json", "*.safetensors"])
    convert(model_dir, args.out, QUANT_NAMES[args.quant], args.group, args.max_seq_len)


if __name__ == "__main__":
    main()
