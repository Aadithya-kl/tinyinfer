# Changelog

## v0.4.0

- `--min-p F`: min-p sampling. Drops tokens whose probability is below F times that of the top token.
- `--repeat-penalty F` and `--repeat-last-n N`: repetition penalty over the last N tokens (prompt and generated). It also applies to greedy decoding.
- `--bench N --json`: benchmark result as one JSON object, for collecting numbers across GPUs.
- Unit tests for the new sampling options; the end-to-end test runs them against a tiny model.

## v0.3.1

- Split-K attention only for grouped-query models; Llama-2-7B style models use the per-head kernel.
- README with measured v0.3 numbers, terminal demo, quick start.

## v0.3

- Shared-memory tiled split-K attention: 3.6x faster than v0.1 at 2048 context.
- Fused QKV and gate/up GEMVs. 235 tok/s on an RTX 3050 Laptop (77% of peak bandwidth).

## v0.2

- Autotuned rows-per-warp GEMV.
- GQA-aware split-K flash-decoding. 212 tok/s.

## v0.1

- First release: hand-written CUDA kernels (f16/q8/q4 GEMV, RMSNorm, RoPE, attention, SwiGLU, argmax), CUDA graphs, CPU reference engine, HF converter. 200 tok/s.
