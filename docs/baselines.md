# Baselines

Rival measurements on our own hardware, first-party. **Every number below was
measured on 2026-09-09/10 on vast machine 36542 (RTX 5090), in one sitting,
alongside the engine** — the Phase 4 run (`docs/environment.md`). The raw
logs are in `results/2026-09-09-machine-36542/`. Where it helps the reader,
the Phase 0 figure from machine 94372 (2026-09-04, `results/2026-09-04-machine-94372/`)
is given in parentheses; the two machines agree to within a point or two on
every GPU-bound row, and differ where a rival's loop is host-bound (see
ollama, and llama.cpp's external drafts).

Re-measure and re-date whenever a rival version changes. Rivals improve; a
baseline with no date and no commit hash is not evidence.

## The matrix

Same day, same card, same prompts (`scripts/rivals/*_bench.py`: essay /
code / math), greedy. "% of wall" is tok/s over the engine's own byte count
per step at 1701 GB/s (method below). Best config per rival.

| Engine (version, bytes/step) | raw, short | % wall | raw at 200k | % wall | speculative, essay / code / math | best draft |
|---|---|---|---|---|---|---|
| **Token Rush** (int4 g128 GPTQ, 13.65 GB) | **100.9** (`--backend triton`) / 97.5 (Marlin layout) | **81.0** / 78.2 | **71.6** / 69.8 | **85.5** / 83.3 | **228 / 357 / 378** | DFlash2 in-graph, K=7 |
| llama.cpp `434ddbbc0` (UD-Q4_K_M, 15.39 GB) | 82.8 | 74.9 | 44.6 | 58.6 | 130 / 121 / 169 | built-in MTP, 1 token |
| ollama 0.33.3 (Q4_K_M, 15.83 GB) | — (no raw mode) | — | — | — | 136 / 144 / 175 | its default: MTP n-max 4 |
| vLLM 0.29.0 (NVFP4, 16.25 GB) | 78.1 | 74.6 | 59.9 | 80.3 | 65.6 / 65.1 / 65.1 (its raw is faster) | MTP; DSpark and DFlash2 are 53 / 52 |
| SGLang 0.5.19 (NVFP4, 16.25 GB) | 62.7 | 59.9 | 49.7 | 66.6 | **106 / 138 / 207** | DSpark, block 7 |
| ExLlamaV3 1.4.6 (EXL3 4.00bpw, 13.18 GB) | 77.6 | 60.1 | 48.0 | 74.2 | 132 / 141 / 166 | MTP, 2 draft tokens |

The margins the project claims, measured on this table: **2.15x / 2.6x /
1.8x** over SGLang + DSpark on prose / code / math; **1.75x / 2.95x / 2.2x**
over llama.cpp + MTP (1.67x / 2.5x / 2.2x over ollama, which is the fastest
llama.cpp-family number on this host); **2.9x / 4.6x / 4.8x** over vLLM's
best config, which is raw; 1.7x / 2.5x / 2.3x over ExLlamaV3's chained
MTP. Raw decode: +22% over llama.cpp on 11% fewer bytes, and at matched
bytes the engine is worth **+6 points of the wall over vLLM** (81.0 vs 74.6),
+3.9 on the Marlin layout. At 200k, 85.5% of the wall against vLLM's 80.3%
— and 256k-usable (needle retrieved at 262k tokens, `results/.../engine/needle.log`).

## Method

The primary metric is **% of the memory-bandwidth roofline**, not raw tok/s.
Raw tok/s is not comparable across engines that run different quantizations —
a more aggressive quant reads fewer bytes per token and wins on tok/s without
the engine being any faster. Percentage of roofline is immune to that.

For each engine: count the bytes actually read per decode step, divide the
measured read bandwidth (**1701 GB/s**; this machine measured 1702) by it to
get the ceiling, then express the achieved tok/s as a fraction of that ceiling.

Bytes per step are the weights of the text path **except the embedding table**
plus the KV cache re-read at that context length. Only the 16 attention layers
hold KV: 16 layers x 2 (K, V) x 4 heads x 256 dims = 32768 values per token,
i.e. **32 KB/token at FP8** (32768 B), **64 KB at FP16** and **34 KB at
llama.cpp's `q8_0`** (34816 B: 34 bytes per 32 values). The context tables
use these exact figures.

> **Byte counts are from the checkpoint headers, never file sizes**, and
> exclude the embedding table (a decode step reads one row of it). This was
> corrected on 2026-09-09 (`docs/progress.md` step 31) — the original Phase 0
> tables had the embedding in every rival's count, which put vLLM at a
> fictitious 88% of the wall. `bench/decode.py` counts ours this way;
> `bench/quality_sources.py::streamed_bpw` does it for the others.

## llama.cpp

Measured 2026-09-09. Upstream `ggml-org/llama.cpp` at commit `434ddbbc0`
(2026-09-09), built with CUDA 13.3, `-DGGML_CUDA=ON
-DCMAKE_CUDA_ARCHITECTURES=120`. Weights `unsloth/Qwen3.8-27B-GGUF`,
`UD-Q4_K_M`.

Upstream supports this model natively — `LLM_ARCH_QWEN35`, with the MTP graph in
`src/models/qwen35.cpp`. `--spec-type` also offers `draft-dspark`,
`draft-eagle3` and `draft-dflash`, so the external drafts can be compared
within one engine, holding everything else constant.

### Bytes per token

| | |
|---|---|
| GGUF file | 16.464 GB |
| MTP head (`blk.64.nextn.*`) | 0.351 GB |
| Text path, total | 16.102 GB |
| `token_embd` (Q4_K; one row read per token, not streamed) | 0.715 GB |
| **Read per raw decode step** | **15.387 GB** |
| Effective bpw over the 25.63B streamed weights | 4.80 |

The conversion already drops the vision tower — 0 vision tensors in the file.

### Results

Three runs per prompt; the spread is in the logs (`llama_cli_*_r{1,2,3}.log`)
and is ≤0.2 tok/s except where noted.

| Test | tok/s | Phase 0 (94372) | Notes |
|---|---|---|---|
| `llama-bench` tg128 | **82.79 ± 0.17** | 82.84 | standard method, the number to cite |
| `llama-cli` raw, essay / code / math | 81.6 / 81.6 / 81.6 | 80.6 | includes prompt + thinking |
| `llama-cli` + `--spec-type draft-mtp`, essay / code / math | **130.3 / 120.9 / 168.5** | 129.6 / 128.4 / 162.2 | code: 120.9, 120.9, 114.9 over the three runs |
| `llama-cli` + `draft-mtp --spec-draft-n-max 4` | 122.4 / 129.1 / 169.7 | 116.3 / 107.4 / 165.8 | the chained-MTP setting ollama uses |
| `llama-cli` + `--spec-type draft-dspark` | 100.4 / 104.2 / 145.9 | 93.6 / 92.0 / 131.6 | draft `erlidev/Qwen3.8-27B-DSpark-GGUF` BF16 (2.7 GB), `-ngld 999` |
| `llama-cli` + `--spec-type draft-dflash` | 131.2 / 114.3 / 164.0 | 110.7 / 111.4 / 146.6 | draft `z-lab/Qwen3.8-27B-DFlash2-GGUF` BF16 (3.9 GB) |

```
ceiling  = 1701 GB/s / 15.387 GB = 110.5 tok/s
llama.cpp raw = 82.79 tok/s      =  74.9% of the wall
MTP gain      = 130.3 / 81.6     =  +60%  (essay; +48% code, +106% math)
```

The built-in one-token MTP is the best llama.cpp configuration on prose and
math; on code the 4-token chain and the DFlash2 draft are within a few tok/s
of each other and of it. The external drafts moved between machines while
`llama-bench` did not: DSpark +7–14 tok/s and DFlash2 +3–20 here on a Zen 5
desktop core against the Phase 0 EPYC — the speculative loop has host-side
work between small kernels (`docs/environment.md`, launch-tax measurement),
which is why every number here names its machine.

### Reproducing

From nothing, on a fresh instance. Nothing on a vast machine survives a
recycle, so this is the recovery path as much as the audit trail.

```bash
# 1. build — the arch flag is required; a default build has no sm_120 kernels
git clone https://github.com/ggml-org/llama.cpp /workspace/rivals/llama.cpp
cd /workspace/rivals/llama.cpp
export PATH=/usr/local/cuda-13.3/bin:$PATH CUDACXX=/usr/local/cuda-13.3/bin/nvcc
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON
cmake --build build --config Release -j 24      # ~10 min on 16 cores

# 2. weights (16.5 GB, ~1 min)
hf download unsloth/Qwen3.8-27B-GGUF --include "Qwen3.8-27B-UD-Q4_K_M.gguf" \
   --local-dir /workspace/models/Qwen3.8-27B-GGUF

# 3. measure
M=/workspace/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
./build/bin/llama-bench -m $M -ngl 999 -p 512 -n 128 -r 3

./build/bin/llama-cli -m $M -ngl 999 -c 4096 -n 200 --temp 0 \
  -p "<prompt>" --single-turn --no-warmup [--spec-type draft-mtp]
# current llama-cli prints "[ Prompt: N t/s | Generation: N t/s ]" at exit

# 4. decode vs. context
./build/bin/llama-bench -m $M -ngl 999 -fa 1 -ctk q8_0 -ctv q8_0 \
  -p 0 -n 128 -d 0,22000,90000,200000 -r 2
```

Stop any local server before measuring (`pkill -x llama-server`) — a resident
model competes for the bandwidth being measured.

### Decode vs. context length

`llama-bench` with `-d` (tokens already in the KV cache), speculation off,
`q8_0` KV, flash attention on. Bytes/token is weights (15.387 GB) plus the KV
re-read each step at 34816 B per token of context. The ceiling therefore
already prices in the longer cache — an engine holding a constant fraction of
the wall would track it.

| Context | decode | bytes/token | ceiling | % of wall | Phase 0 |
|---|---|---|---|---|---|
| 0 | 81.5 tok/s | 15.39 GB | 110.5 | **73.7%** | 81.5 |
| 22k | 75.0 tok/s | 16.15 GB | 105.3 | 71.2% | 75.1 |
| 90k | 60.4 tok/s | 18.52 GB | 91.8 | 65.7% | 59.7 |
| 200k | 44.6 tok/s | 22.35 GB | 76.1 | **58.6%** | 44.3 |

llama.cpp gives up 15 points of roofline between short context and 200k.
Physics accounts for the ceiling falling 110 -> 76; it does not account for the
engine falling from 74% to 59% of that ceiling. At 200k, 41% of the available
bandwidth goes unused.

## SGLang

Measured 2026-09-10. SGLang **0.5.19** (`sglang[all]` from PyPI, torch
2.13.0+cu130) in `/workspace/venvs/sglang`. Weights
`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` — genuine `nvfp4-pack-quantized`:
4-bit, group 16, FP8 scales. (`unsloth/Qwen3.8-27B-NVFP4` is FP8 despite the
name, and `RadixArk/Qwen3.8-27B-NVFP4` is a mixed FP8/FP4 config.)

### Bytes per token

| Component | Bytes |
|---|---|
| Text body (including `embed_tokens`) | 16.245 GB |
| `lm_head` (left at BF16) | 2.543 GB |
| Text path, total | 18.788 GB |
| `embed_tokens` (BF16; one row read per token, not streamed) | 2.543 GB |
| **Read per decode step** | **16.245 GB** |
| MTP head | 0.849 GB |
| Vision tower | 0.921 GB |

(The streamed total coincidentally equals the "text body" line: the embedding
that comes out is the same size as the BF16 `lm_head` that goes in.)

Ceiling at 1701 GB/s: **104.7 tok/s**, against llama.cpp's 110.5. This
checkpoint reads 5.6% *more* per token than Q4_K_M despite being "4-bit",
because `lm_head` is unquantized — another reason % of roofline is the only
honest headline.

### Results

Raw decode (no speculation), three short prompts, streamed, best of 3:

| Prompt | tok/s | % of wall | Phase 0 |
|---|---|---|---|
| essay | 62.7 | 59.9% | 63.2 |
| code | 62.6 | 59.8% | 63.1 |
| math | 62.6 | 59.8% | 63.1 |

Server: `--kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype float32
--cuda-graph-max-bs 1`, decode CUDA graphs on, Triton GDN kernels, flashinfer
attention.

### Decode vs. context length

Same server (with the 4096-token prefill chunk and a bounded token pool, see
Reproducing), random-token prompts of the given length already in the KV
cache, 256 decode tokens, best of 2. Bytes/token is weights (16.245 GB) plus
32768 B per token of FP8 KV.

| Context | decode | bytes/token | ceiling | % of wall | Phase 0 |
|---|---|---|---|---|---|
| 0 | 62.7 tok/s | 16.25 GB | 104.7 | **59.9%** | 63.1 |
| 22k | 60.5 tok/s | 16.97 GB | 100.3 | 60.3% | 60.9 |
| 90k | 55.8 tok/s | 19.19 GB | 88.6 | 63.0% | 56.2 |
| 200k | 49.7 tok/s | 22.80 GB | 74.6 | **66.6%** | 49.9 |

**SGLang does not collapse.** Its fraction of the wall is flat — slightly
rising — from empty context to 200k: the attention-decode kernel keeps up
with the growing KV, and the 48 GDN layers cost the same at every length.
llama.cpp loses 15 points over the same range. At 200k SGLang is *faster than
llama.cpp in absolute terms* (49.7 vs 44.6 tok/s) despite reading 0.45 GB more
per token.

What SGLang does lose is the 40% it never had: 60–67% of the wall at every
length, against llama.cpp's 74% at short context and the 96.4% a pure GEMV
stream achieves. That is the general-engine tax at bs=1 — scheduler,
per-layer dispatch, unfused GDN chain — and it is constant, not
context-dependent.

### DSpark speculation

Draft `RadixArk/Qwen3.8-27B-DSpark` (1.86B, bf16, unquantized), block size
7 (gamma 7, verify width 8), one speculative step, `SGLANG_RAGGED_VERIFY_MODE=static`
— the configuration RadixArk publishes. Greedy, natural stopping, 1024 tokens
max, streamed. Mean accepted length is `completion_tokens / spec_verify_ct`
from the response metadata, i.e. tokens committed per verify step including
the bonus token.

| Prompt | tok/s | mean accepted length | vs. raw 62.7 | Phase 0 |
|---|---|---|---|---|
| essay | **105.9** | 2.39 | 1.69x | 104.4 |
| code | **138.1** | 3.11 | 2.20x | 137.4 |
| math | **207.3** | 4.68 | 3.31x | 205.2 |

Effective tok/s per accepted token is 44.3 / 44.4 / 44.3 across the three —
the verify step costs the same regardless of content (about 22.6 ms, vs.
16.0 ms for a raw decode step: **1.41x**), so the whole spread is acceptance.
RadixArk's own bs=1 figure of 3.16x is GSM8K, which is the math row here.
Two machines, five days and one minor version apart: every row within 1.5%.

The draft costs 3.6 GB of VRAM at bf16, which on a 32 GB card shrinks the KV
budget to a few tens of thousands of tokens next to the 22 GB of target and
draft weights (`--max-total-tokens 30000` here). Long context and DSpark do
not currently fit together on this card in SGLang.

### Reproducing

```bash
uv venv /workspace/venvs/sglang --python 3.12
VIRTUAL_ENV=/workspace/venvs/sglang uv pip install "sglang[all]"
hf download QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 --local-dir /workspace/models/Qwen3.8-27B-NVFP4
hf download RadixArk/Qwen3.8-27B-DSpark --local-dir /workspace/models/Qwen3.8-27B-DSpark

# raw decode + the context sweep (the 4096 chunk and the bounded pool are
# what lets a 200k prompt in next to a 256k KV pool: docs/traps.md)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/workspace/venvs/sglang/bin/python -m sglang.launch_server \
  --model-path /workspace/models/Qwen3.8-27B-NVFP4 --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype float32 --context-length 262144 \
  --chunked-prefill-size 4096 --max-prefill-tokens 4096 --max-total-tokens 215000 \
  --mem-fraction-static 0.88 --max-running-requests 1 --max-mamba-cache-size 6 \
  --cuda-graph-max-bs 1 --disable-prefill-cuda-graph --port 30000
scripts/rivals/sglang_bench.py raw
scripts/rivals/sglang_bench.py context 0 22000 90000 200000

# DSpark: --context-length 32768 --max-total-tokens 30000 --mem-fraction-static 0.92
# (the bf16 draft takes 3.6 GB), plus, with SGLANG_RAGGED_VERIFY_MODE=static set:
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path /workspace/models/Qwen3.8-27B-DSpark \
  --speculative-draft-model-quantization unquant \
  --speculative-draft-attention-backend flashinfer \
  --speculative-dspark-block-size 7 --speculative-num-steps 1 --speculative-eagle-topk 1
scripts/rivals/sglang_bench.py spec --n 1024
```

`--max-mamba-cache-size` must be at least 5 (the hybrid state cache is sized
in units of `mamba_ratio=5` per request); `--disable-prefill-cuda-graph`
because capturing prefill graphs up to 32k tokens runs the card out of memory
next to a 256k KV pool; a 16k prefill chunk OOMs on the first 22k prompt
(this run's `server_raw.log`), 4096 does not. The stage scripts that drove
the run are in `results/2026-09-09-machine-36542/scripts/`.

## vLLM

Measured 2026-09-09. vLLM **0.29.0** (PyPI, torch 2.13.0+cu130) in
`/workspace/venvs/vllm`, same `QUASAR-QAT` NVFP4 checkpoint as SGLang, so the
same 16.245 GB per step and the same 104.7 tok/s ceiling. FlashInfer attention,
FP8 KV, `--max-num-seqs 1`, torch.compile and full CUDA graphs at their
defaults (startup takes about four minutes of compilation).

### Results

Raw decode, three short prompts, streamed, best of 3
(`results/.../vllm/results_raw.log`):

| Prompt | tok/s | % of wall | Phase 0 |
|---|---|---|---|
| essay | 78.1 | 74.6% | 79.4 |
| code | 78.1 | 74.6% | 79.5 |
| math | 78.1 | 74.6% | 79.4 |

### Decode vs. context length

Same server at `--max-model-len 220000`, random-token prompts already in the
KV cache, 256 decode tokens, best of 2. Bytes/token as for SGLang.

| Context | decode | bytes/token | ceiling | % of wall | Phase 0 |
|---|---|---|---|---|---|
| 0 | 78.1 tok/s | 16.25 GB | 104.7 | **74.6%** | 79.7 |
| 22k | 75.9 tok/s | 16.97 GB | 100.3 | 75.7% | 77.7 |
| 90k | 69.3 tok/s | 19.19 GB | 88.6 | 78.2% | 70.6 |
| 200k | 59.9 tok/s | 22.80 GB | 74.6 | **80.3%** | 61.0 |

vLLM holds 75–80% of the wall from empty context to 200k, rising slightly with
context. It is the strongest rival engine by this measure at every length, and
at 200k it decodes at 60 tok/s against SGLang's 50 and llama.cpp's 45 on
fewer bytes per token — but it is not at the roofline: 20 to 25 points of the
wall go unused. (0.29.0 is 1–2% slower than 0.28.0 was on the other machine;
same order as the machine difference, not attributed.)

### MTP speculation

`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`, using
the MTP head shipped inside the checkpoint (bf16, `mtp.*`), same server
otherwise. FULL decode CUDA graphs are still captured.

| Prompt | tok/s | vs. raw 78.1 | Phase 0 |
|---|---|---|---|
| essay | 65.6 | 0.84x | 63.0 |
| code | 65.1 | 0.83x | 63.0 |
| math | 65.1 | 0.83x | 62.6 |

Mean acceptance length 1.85 per step (vLLM's own `SpecDecoding metrics`),
and it is still **slower than not speculating**: a draft-plus-verify step
costs about 28 ms against 12.8 ms for a raw decode step.

### DSpark and DFlash2 drafts

`--speculative-config '{"method":"dspark","model":<RadixArk draft>,"num_speculative_tokens":7}'`
at `--max-model-len 8192` (the bf16 draft costs 3.6 GB of KV budget). Two
things have to be patched into a copy of the draft's `config.json` to get it
to load, in 0.29.0 as in 0.28.0: `architectures` set to `Qwen3DSparkModel`
(vLLM maps the checkpoint's `DSparkDraftModel` to its DeepSeek-V4 class and
fails on `hc_mult`), and a no-op `compressed-tensors` `quantization_config`
(vLLM inherits the target's quantization for the draft and then cannot find a
quant config in a bf16 checkpoint). The patched copy is
`results/2026-09-09-machine-36542/vllm/dspark_ct_config.json`.

| Draft | Prompt | tok/s | mean accepted / step | vs. raw 78.1 | Phase 0 |
|---|---|---|---|---|---|
| DSpark, gamma 7 | essay / code / math | 53.0 / 53.6 / 52.9 | 2.95 (server mean) | **0.68x** | 51.4 / 52.0 / 52.2 |
| DFlash2, block 8 | essay / code / math | 51.7 / 51.7 / 51.7 | 4.03 (server mean) | **0.66x** | 51.9 / 51.8 / 51.6 |

Both drafts land at the same 52–54 tok/s, *a third slower than raw*: a step
costs about 70 ms whichever draft is used, so the cost is the speculative
path, not the draft model. DFlash2 accepts 4.0 tokens per step here — the
same draft accepts 3.2–5.2 in our graph at a 13.8 ms step — and vLLM still
loses to its own raw decode with it. The draft model, the verify and the
bookkeeping between them run outside the decode graph; nearly-free
verification is exactly what a bs=1 engine gets from idle tensor cores, and
vLLM's speculative path does not get it. This is headroom argument #2 in
`CLAUDE.md` measured on the strongest raw engine: its raw decode is at 75%
of the wall, and its speculation loses 16–34% from there.

### Reproducing

```bash
uv venv /workspace/venvs/vllm --python 3.12
VIRTUAL_ENV=/workspace/venvs/vllm uv pip install vllm
/workspace/venvs/vllm/bin/vllm serve /workspace/models/Qwen3.8-27B-NVFP4 \
  --trust-remote-code --kv-cache-dtype fp8 --max-model-len 220000 \
  --max-num-seqs 1 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92 --port 8000
# 262144 needs 8.3 GB of KV and 0.92 leaves 7.3; 220k is enough for the 200k row
scripts/rivals/vllm_bench.py raw
scripts/rivals/vllm_bench.py context 0 22000 90000 200000
# MTP: --max-model-len 32768 --gpu-memory-utilization 0.9 \
#      --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
# DSpark / DFlash2: --max-model-len 8192 and the speculative-config lines above
```

## ExLlamaV3

Measured 2026-09-09. ExLlamaV3 **1.4.6** (release wheel `cu132.torch2.11.0`,
torch 2.11.0+cu130) in `/workspace/venvs/exl3`, in-process through its
`Generator`/`Job` API (`scripts/rivals/exl3_bench.py`). Weights
`turboderp/Qwen3.8-27B-exl3`, revision `4.00bpw` (4-bit trellis body, 6-bit
head, the MTP head quantized to 4 bits alongside) — the only rival at the
project's own target bpw, so this is the matched-quantization row.

### Bytes per token

From the safetensors headers:

| Component | Bytes |
|---|---|
| Text body (64 layers) | 12.230 GB |
| `lm_head` (6-bit) | 0.954 GB |
| **Text path, read per decode step** | **13.184 GB = 3.92 bpw** |
| `embed_tokens` (bf16, one row read) | 2.543 GB |
| MTP head (4-bit) | 0.213 GB |
| Vision tower | 0.921 GB |

Ceiling at 1701 GB/s: **129.0 tok/s** — the highest of any rival, because it
reads the fewest bytes.

### Results

Raw decode, three short prompts, FP16 cache, greedy, 256 tokens, best of 3,
from the generator's own per-job `time_generate`:

| Prompt | tok/s | % of wall | Phase 0 |
|---|---|---|---|
| essay | 77.5 | 60.1% | 76.8 |
| code | 77.6 | 60.1% | 77.0 |
| math | 77.5 | 60.1% | 76.9 |

### Decode vs. context length

FP16 cache (ExLlamaV3's default; 65536 B per token of context over the 16
attention layers), random-token prompts already in the cache, 256 decode
tokens, best of 2. At 200k the cache alone is 13 GB and the process sits at
29 GB — it fits, barely.

| Context | decode | bytes/token | ceiling | % of wall | Phase 0 |
|---|---|---|---|---|---|
| 0 | 77.6 tok/s | 13.18 GB | 129.0 | **60.1%** | 76.5 |
| 22k | 72.0 tok/s | 14.63 GB | 116.3 | 61.9% | 71.4 |
| 90k | 60.4 tok/s | 19.08 GB | 89.1 | 67.8% | 59.8 |
| 200k | 48.0 tok/s | 26.29 GB | 64.7 | **74.2%** | 47.7 |

The fraction *rises* with context, which says the attention decode kernel is
efficient and the short-context deficit is a constant per-step cost —
Python-side generator overhead and per-layer dispatch — that shrinks in
relative terms as the KV read grows. Absolute tok/s at 200k is close to
SGLang's (48.0 vs 49.7) on twice the KV bytes.

### MTP speculation

The MTP head shipped in the checkpoint (quantized to 4 bits alongside the
body), loaded as ExLlamaV3's `mtp` component and chained for one or two draft
tokens per step. Greedy, 256 tokens, same prompts. "Per step" is tokens
committed per verify step including the bonus token.

| Prompt | 1 draft token | per step | 2 draft tokens | per step | Phase 0 (2 tokens) |
|---|---|---|---|---|---|
| essay | 119.6 tok/s (1.54x) | 1.80 | 131.8 tok/s (1.70x) | 2.24 | 126.7 |
| code | 121.4 tok/s (1.56x) | 1.81 | 140.7 tok/s (1.81x) | 2.39 | 142.1 |
| math | 130.9 tok/s (1.69x) | 1.97 | 166.1 tok/s (2.14x) | 2.81 | 160.3 |

This is the best-integrated MTP path of any rival: the second draft token is
a net gain everywhere, and 2.8 committed tokens per step on math from a
one-layer head is close to what DSpark's 1.86B draft achieves in llama.cpp.
It still runs on a raw decode that holds only 60% of the wall, so the
absolute numbers land where llama.cpp's MTP does.

### Reproducing

```bash
uv venv /workspace/venvs/exl3 --python 3.12
VIRTUAL_ENV=/workspace/venvs/exl3 uv pip install "torch==2.11.0" --torch-backend=cu130
VIRTUAL_ENV=/workspace/venvs/exl3 uv pip install \
  https://github.com/turboderp-org/exllamav3/releases/download/v1.4.6/exllamav3-1.4.6+cu132.torch2.11.0-cp312-cp312-linux_x86_64.whl
hf download turboderp/Qwen3.8-27B-exl3 --revision 4.00bpw --local-dir /workspace/models/Qwen3.8-27B-exl3-4.0
/workspace/venvs/exl3/bin/python scripts/rivals/exl3_bench.py /workspace/models/Qwen3.8-27B-exl3-4.0 raw
/workspace/venvs/exl3/bin/python scripts/rivals/exl3_bench.py /workspace/models/Qwen3.8-27B-exl3-4.0 context 0 22000 90000 200000
/workspace/venvs/exl3/bin/python scripts/rivals/exl3_bench.py /workspace/models/Qwen3.8-27B-exl3-4.0 mtp --draft-tokens 2
```

## ollama

Measured 2026-09-10. ollama **0.33.3**, `ollama pull qwen3.8:27b` — ollama's
own Q4_K_M conversion (16.80 GB of tensors, Q4_K body with Q6_K/Q5_K in a few
layers, the MTP head as `blk.64.*`, plus a 0.93 GB vision projector loaded
alongside; blob `f5f1dd89…`, the same file as on 94372). ollama runs its
bundled `llama-server` with `--spec-type draft-mtp --spec-draft-n-max 4
--spec-draft-backend-sampling`, 32k context, all 66 layers on the GPU — so
its number is llama.cpp with a 4-token chained MTP draft on by default, and
there is no switch to turn speculation off.

Bytes per raw step by the same convention as llama.cpp (tensors minus MTP,
minus the embedding table): 15.83 GB, ceiling 107.5 tok/s.

| Prompt | tok/s | Phase 0 (94372) | notes |
|---|---|---|---|
| essay | **136.4** | 66.9 | `eval_count / eval_duration` from the API, 256 tokens |
| code | **143.8** | 68.3 | |
| math | **175.3** | 82.0 | |

**This row doubled between machines, and the reason is instructive.** The
runner command line is byte-for-byte the same on both hosts, the draft
acceptance is identical (188 of 266 drafts accepted, mean length 3.81 —
same greedy tokens), and the per-token eval time is 5.7 ms here against
12.6 ms on 94372. ollama's serving path — `llama-server` with backend
sampling, the MTP loop, `-b 1024 -ub 1024`, its request handling — is bound
by the host CPU, and this host's Zen 5 core issues work about twice as fast
as the Phase 0 EPYC core (`docs/environment.md`: the eager GDN chain, 4.4 vs
10.5 ms). So Phase 0's reading — "ollama's serving path is slow" — was a
property of the machine it ran on. On this machine ollama's default is the
**fastest llama.cpp-family number**: above `llama-cli`'s own one-token MTP
on every prompt and level with its 4-token chain. It is listed as the sanity
floor and is no longer one; the comparison rows in this file and in
`CLAUDE.md` use it as the best llama.cpp-family figure where it is.

## What these numbers change

**The general-engine tax at bs=1 is engine-specific, and everyone still pays
it.** On the same NVFP4 bytes, SGLang holds 60% of the wall and vLLM **75%**;
llama.cpp holds 75% on its own bytes; ExLlamaV3 60%. A bare GEMV stream
reaches 96.4%. So the best rival leaves 21 points on the table and the worst
40. Our own engine sits at 81% on 13.65 GB per step (`--backend triton`;
78% on the Marlin layout that serves the speculative step) — 6 points above
vLLM and on 2.6 GB fewer bytes. The raw-decode margin over the best rival is
real but modest in *efficiency*; most of the tok/s margin is the byte count.

**The long-context collapse is llama.cpp's, not the field's.** SGLang's and
vLLM's fractions of the wall both *rise* slightly to 200k (60 -> 67% and
75 -> 80%): their attention decode scales, and on this hybrid model only 16
layers pay for length. llama.cpp is the one that falls, 74% -> 59%. So the
project's long-context target is "hold a constant fraction", which is
achievable rather than heroic, and the margin at 200k is real: our 85.5% of
the wall against vLLM's 80.3%, ExLlamaV3's 74.2%, SGLang's 66.6% and
llama.cpp's 58.6% — 71.6 tok/s against 59.9 / 48.0 / 49.7 / 44.6.

**Speculation numbers are measured, not assumed, and they reproduce.**
DSpark's mean accepted length on this model is 2.4 on prose, 3.1 on code,
4.7 on math at gamma 7 — identical on two machines; llama.cpp's MTP head
gets +60% / +48% / +106% with a single draft token. What the rivals leave on
the table is the verify cost: SGLang's verify step is 1.41x a raw decode
step for 8 tokens of verification, where ours is 1.13x for the same width
(`docs/progress.md` step 28), and llama.cpp's MTP gain per accepted token is
far below what a fused, graph-captured verify gives.

**Speculation is where every rival is weak, each in its own way.** vLLM's
MTP path is slower than its raw decode, and so are DSpark and DFlash2 in
vLLM at 3–4 accepted tokens per step (52–54 tok/s against 78 raw). llama.cpp's
DSpark integration gets a third of what the same draft gives SGLang. SGLang's
verify step costs 1.41x a raw step. ExLlamaV3 chains the MTP head well
(2.8 tokens per step on math) but on a 60%-of-wall raw decode. Nobody has both
a raw decode near the wall and a speculative loop built for one stream with
idle tensor cores — which is headroom argument #2 measured four times over.

**Raw decode is not where the win is.** "+30% over llama.cpp raw" measured
+22% (100.9 vs 82.8) on 11% fewer bytes; held to vLLM's bytes the engine is
worth +6 points of the wall. Raw decode near the wall is table stakes,
evidence of competence, and nothing more. The headline is speculation:
228 / 357 / 378 against the best rival per family, 136 / 144 / 207.

**Speculative speedups are host-dependent for every engine whose loop
touches the host, and ours is not one of them.** llama.cpp's external drafts
and ollama moved by 10–100% between an EPYC and a Zen 5 host; SGLang's
DSpark moved 1.5%, vLLM's MTP 4%, and our in-graph step times (13.8 ms
DFlash, 12.4 ms MTP) are identical to the development machine's to the
tenth of a millisecond. The one-graph-per-step design is what makes the
number portable.

**Quantization parity has to be handled explicitly.** llama.cpp runs at 4.80
bpw over its streamed weights, we at 4.25 (KL to bf16 0.0232 against
UD-Q4_K_M's 0.0093 and EXL3's 0.0128 — `docs/quantization.md`). Comparing
raw tok/s across that gap hands us ~11% for free and a reviewer will say so.
Report % of roofline as the headline; the quality row stays reported as
measured, half met.

## Not measured

Every rival in the table has a raw, a long-context and a speculative number.
What is still missing is second-order:

- speculation *at long context* for any rival (all speculative runs are at
  short context; on this card the drafts and the KV pool compete for memory)
- ExLlamaV3 with a quantized cache, and llama.cpp at a matched 4.25 bpw
  (`llama-quantize`) for the fair-comparison row
- vLLM's long-context decode with FP16 rather than FP8 KV, to separate the
  attention kernel from the cache format
