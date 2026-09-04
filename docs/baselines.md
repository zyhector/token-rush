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

### Reproducing

From nothing, on a fresh instance. `/workspace` survives a recycle but not a
destroy, so this is the recovery path as much as the audit trail.

```bash
# 1. build — the arch flag is required; a default build has no sm_120 kernels
git clone https://github.com/ggml-org/llama.cpp /workspace/rivals/llama.cpp
cd /workspace/rivals/llama.cpp && git checkout 6703d78
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON
cmake --build build --config Release -j 32      # ~25 min, CUDA templates dominate

# 2. weights (16.5 GB, ~2 min at 131 MB/s)
hf download unsloth/Qwen3.8-27B-GGUF --include "Qwen3.8-27B-UD-Q4_K_M.gguf" \
   --local-dir /workspace/models/Qwen3.8-27B-GGUF

# 3. measure
M=/workspace/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
./build/bin/llama-bench -m $M -ngl 999 -p 512 -n 128 -r 3

./build/bin/llama-cli -m $M -ngl 999 -c 4096 -n 200 --temp 0 \
  -p "<prompt>" --single-turn --no-warmup [--spec-type draft-mtp]
```

Stop any local server before measuring (`pkill -f llama-server`) — a resident
model competes for the bandwidth being measured.

### Decode vs. context length

Speculation off, `q8_0` KV, flash attention on. Bytes/token is weights (16.102
GB) plus the KV actually re-read each step (32 KB per token of context, over the
16 attention layers). The ceiling therefore already prices in the longer cache —
an engine holding a constant fraction of the wall would track it.

| Context | decode | bytes/token | ceiling | % of wall |
|---|---|---|---|---|
| ~0 | 77.0 tok/s | 16.10 GB | 100.3 | **76.7%** |
| 22k | 70.5 tok/s | 16.82 GB | 96.0 | 73.4% |
| 90k | 54.1 tok/s | 18.35 GB | 88.0 | 61.5% |
| 200k | 40.6 tok/s | 19.72 GB | 81.9 | **49.6%** |

llama.cpp does not track it — it gives up 27 points of roofline between short
context and 200k. Physics accounts for the ceiling falling 100 -> 82; it does not
account for the engine falling 77% -> 50% of that ceiling. At 200k more than half
the available bandwidth goes unused.

Prompt processing over the same range: 2974 tok/s at 22k, 1833 tok/s at 90k.

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

**Long context is where the gap is widest.** The roofline fraction llama.cpp
holds collapses as the KV cache grows, and only 16 of 64 layers even produce KV —
the other 48 carry constant-size recurrent state. An engine whose attention decode
keeps memory-level parallelism up as KV grows should degrade far less. This is a
larger and better-defended margin than the short-context raw-decode target.

**Quantization parity has to be handled explicitly.** llama.cpp runs at 4.79 bpw;
our target is 4.0. Comparing raw tok/s across that gap hands us ~20% for free and
a reviewer will say so. Report % of roofline as the headline, and produce a
matched-bpw GGUF with `llama-quantize` for a supporting comparison.

## SGLang + DSpark — prepared, not yet measured

Cannot run on this machine: DSpark exists only in SGLang 0.5.16+, and every
SGLang from 0.5.11 onward requires `cuda-python>=13.0` while torch cu128 requires
`cuda-bindings<13`. The last CUDA-12-compatible release, 0.5.10, has no DSpark.
See `docs/handoff.md`.

The checkpoint is chosen and its bytes are already accounted for.
`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` is genuine `nvfp4-pack-quantized` — 4-bit,
group 16, FP8 scales. (`unsloth/Qwen3.8-27B-NVFP4` is **not** NVFP4 despite the
name; its config is `float-quantized` at 8 bits.)

| Component | Bytes |
|---|---|
| Text body | 16.245 GB |
| `lm_head` (left at BF16) | 2.543 GB |
| **Text path, read per decode step** | **18.788 GB** |
| MTP head | 0.849 GB |
| Vision tower | 0.921 GB |

Ceiling at 1615 GB/s: **86.0 tok/s**, against llama.cpp's 100.3. This checkpoint
reads *more* per token than Q4_K_M despite being "4-bit", because `lm_head` is
unquantized. Raw tok/s will therefore flatter llama.cpp by roughly 15% on byte
count alone — another reason % of roofline is the only honest headline.

## Still to measure

- SGLang + DSpark — the real opponent; needs a CUDA 13 host
- vLLM at bs=1 — general-engine tax reference
- ExLlamaV3 — peer specialist

Note that llama.cpp's `--spec-type` now also offers `draft-dspark` and
`draft-eagle3`, so some of the DSpark comparison can be made within one engine,
holding everything else constant.
