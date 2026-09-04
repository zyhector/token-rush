# Baselines

Rival measurements on our own hardware. Every number here is first-party — run
on vast machine 25132 (RTX 5090), described in `docs/environment.md`.

Re-measure and re-date whenever a rival version changes. Rivals improve; a
baseline with no date and no commit hash is not evidence.

## Method

The primary metric is **% of the memory-bandwidth roofline**, not raw tok/s.
Raw tok/s is not comparable across engines that run different quantizations —
a more aggressive quant reads fewer bytes per token and wins on tok/s without
the engine being any faster. Percentage of roofline is immune to that.

For each engine: measure bytes actually read per decode step, divide the
measured read bandwidth (**1615 GB/s**) by it to get the ceiling, then express
the achieved tok/s as a fraction of that ceiling.

## llama.cpp

Measured 2026-09-04. Upstream `ggml-org/llama.cpp` at commit `6703d78`, built
with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`. Weights
`unsloth/Qwen3.8-27B-GGUF`, `UD-Q4_K_M`.

Upstream supports this model natively — `LLM_ARCH_QWEN35`, with the MTP graph in
`src/models/qwen35.cpp`. No fork is required.

### Bytes per token

| | |
|---|---|
| GGUF file | 16.464 GB |
| MTP head (`blk.64.nextn.*`) | 0.351 GB |
| **Text path, read per raw decode step** | **16.102 GB** |
| Effective bpw over the 26.90B text path | 4.79 |

The conversion already drops the vision tower — 0 vision tensors in the file.

### Results

| Test | tok/s | Notes |
|---|---|---|
| `llama-bench` tg128 | **80.86 ± 0.22** | standard method, the number to cite |
| `llama-cli` raw | 78.0 | 77.3–78.3 over 3 runs; includes prompt + thinking |
| `llama-cli` + `--spec-type draft-mtp` | 91.1 | 88.5–93.5 over 3 runs |

```
ceiling  = 1615 GB/s / 16.102 GB = 100.3 tok/s
llama.cpp raw = 80.86 tok/s      =  80.6% of the wall
MTP gain      = 91.1 / 78.0      =  +17%
```

### Commands

```bash
cd /workspace/rivals/llama.cpp
M=/workspace/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf

./build/bin/llama-bench -m $M -ngl 999 -p 512 -n 128 -r 3

./build/bin/llama-cli -m $M -ngl 999 -c 4096 -n 200 --temp 0 \
  -p "<prompt>" --single-turn --no-warmup [--spec-type draft-mtp]
```

## What these numbers change

**llama.cpp is a harder baseline than the plan assumed.** At 80.6% of the wall it
is much closer to the roofline than the 65% that public mobile-5090 figures
suggest. The consequence lands on the target in `CLAUDE.md`: "+30% over
llama.cpp raw" means 105 tok/s, which at 4.0 bpw (ceiling 120) is **88% of the
wall** — the top of the 85–90% band, with no margin. That target now requires
both an aggressive quantization *and* near-perfect bandwidth utilisation.

**MTP speculation is leaving room on the table, which is the opening.** +17% here
against a public figure of +39% elsewhere. Whatever the cause, an engine holding
its own MTP head to +17% at bs=1 is direct evidence for headroom argument #2:
speculation tuned for a batched design point is not tuned for this one. Against
91 tok/s, the 220–280 tok/s effective target is 2.4–3.1x — a wider margin than
the raw-decode target has.

**Quantization parity has to be handled explicitly.** llama.cpp runs at 4.79 bpw;
our target is 4.0. Comparing raw tok/s across that gap hands us ~20% for free and
a reviewer will say so. Report % of roofline as the headline, and produce a
matched-bpw GGUF with `llama-quantize` for a supporting comparison.

## Still to measure

- vLLM at bs=1 — general-engine tax reference
- SGLang + DSpark — the real opponent
- ExLlamaV3 — peer specialist

Note that llama.cpp's `--spec-type` now also offers `draft-dspark` and
`draft-eagle3`, so some of the DSpark comparison can be made within one engine,
holding everything else constant.
