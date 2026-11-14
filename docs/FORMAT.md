# The `.tinf` file format (version 1)

A single little-endian file holding the model config, the tokenizer and every weight tensor.
Written by `tools/convert_hf.py`, read by `src/model_file.cpp`.

```
offset 0      header            128 bytes
tok_offset    tokenizer         tok_size bytes
dir_offset    tensor directory  n_tensors entries
...           tensor data       each tensor starts on a 256-byte boundary
```

## Header

| field | type | notes |
|---|---|---|
| magic | `char[4]` | `"TINF"` |
| version | u32 | `1` |
| dim, hidden_dim, n_layers, n_heads, n_kv_heads, head_dim, vocab_size, max_seq_len | u32 × 8 | |
| rope_theta, norm_eps | f32 × 2 | |
| quant | u32 | 0 = f16, 1 = q8, 2 = q4 |
| group_size | u32 | weights per quantization scale |
| chat_template | u32 | 0 = none, 1 = zephyr, 2 = llama2 |
| tok_offset, tok_size, dir_offset | u64 × 3 | |
| n_tensors | u32 | |
| padding | | zero-filled up to 128 bytes |

## Tokenizer

```
u32 n_vocab, u32 n_merges, i32 bos, i32 eos, i32 unk, u32 add_prefix
i32 byte_ids[256]                    token id of <0xXX>, or -1
n_vocab  × { u8 type, u32 len, u8 text[len] }   type: 0 normal, 1 byte, 2 control
n_merges × { i32 left, i32 right, i32 merged }  in BPE rank order
```

Normal pieces have the SentencePiece `▁` already replaced by a space; byte tokens store the raw byte.

## Tensor directory entry

```
u16 name_len, char name[name_len], u32 dtype, u32 ndim, u64 shape[2], u64 offset, u64 nbytes
```

dtype: 0 = f32, 1 = f16, 2 = int8, 3 = packed uint4 (shape is the logical `[rows, cols]`).

## Tensors

| name | dtype | shape |
|---|---|---|
| `tok_embeddings` | f16 | `[vocab, dim]` |
| `layers.{i}.attention_norm`, `layers.{i}.ffn_norm` | f32 | `[dim]` |
| `layers.{i}.wq` | W | `[n_heads·head_dim, dim]` |
| `layers.{i}.wk`, `layers.{i}.wv` | W | `[n_kv_heads·head_dim, dim]` |
| `layers.{i}.wo` | W | `[dim, n_heads·head_dim]` |
| `layers.{i}.w1` (gate), `layers.{i}.w3` (up) | W | `[hidden, dim]` |
| `layers.{i}.w2` (down) | W | `[dim, hidden]` |
| `norm` | f32 | `[dim]` |
| `output` | W | `[vocab, dim]` (copied from embeddings if tied) |

`W` is f16 for `quant = f16`. For q8/q4 the payload is int8/uint4 and a companion
`<name>.scales` tensor (f16, `[rows, cols / group_size]`) holds one scale per group:

* **q8**: `w = q · s`, `q ∈ [-127, 127]`, `s = absmax / 127`
* **q4**: `w = (n − 8) · s`, `n ∈ [1, 15]`, `s = absmax / 7`. Byte `j` of a row stores element
  `2j` in its low nibble and `2j+1` in its high nibble.

The scale is rounded to fp16 before quantizing, so the quantized weights match exactly what the
kernels reconstruct.

Q/K projections are kept in HuggingFace layout and RoPE uses HF's `rotate_half` convention, so
no weight permutation is needed.
