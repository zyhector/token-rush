# Baselines

Rival measurements on our own hardware. Every number here is first-party — run
on vast machine 94372 (RTX 5090), described in `docs/environment.md`.

Re-measure and re-date whenever a rival version changes. Rivals improve; a
baseline with no date and no commit hash is not evidence.

## Method

The primary metric is **% of the memory-bandwidth roofline**, not raw tok/s.
Raw tok/s is not comparable across engines that run different quantizations —
a more aggressive quant reads fewer bytes per token and wins on tok/s without
the engine being any faster. Percentage of roofline is immune to that.

For each engine: count the bytes actually read per decode step, divide the
measured read bandwidth (**1701 GB/s**) by it to get the ceiling, then express
the achieved tok/s as a fraction of that ceiling.

Bytes per step are the weights of the text path plus the KV cache re-read at
that context length. Only the 16 attention layers hold KV: 16 layers x 2 (K, V)
x 4 heads x 256 dims = 32768 values per token, i.e. **32 KB/token at FP8** and
**34.8 KB/token at llama.cpp's `q8_0`** (34 bytes per 32 values). Bytes that
are the same in every row are not an approximation; the context rows below use
these exact figures.

## llama.cpp

Measured 2026-09-04. Upstream `ggml-org/llama.cpp` at commit `24f5bf8a4`
(2026-09-04), built with CUDA 13.3, `-DGGML_CUDA=ON
-DCMAKE_CUDA_ARCHITECTURES=120`. Weights `unsloth/Qwen3.8-27B-GGUF`,
`UD-Q4_K_M`.

Upstream supports this model natively — `LLM_ARCH_QWEN35`, with the MTP graph in
`src/models/qwen35.cpp`. No fork is required. `--spec-type` also offers
`draft-dspark`, `draft-eagle3` and `draft-dflash`, so a DSpark comparison can
be made within one engine, holding everything else constant.

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
| `llama-bench` tg128 | **82.84 ± 0.24** | standard method, the number to cite |
| `llama-cli` raw | 80.6 | 80.4–80.8 over 3 runs; includes prompt + thinking |
| `llama-cli` + `--spec-type draft-mtp`, essay prompt | 129.6 | 129.5–129.8 over 3 runs |
| same, code prompt | 128.4 | |
| same, math prompt | 162.2 | |
| `llama-cli` + `--spec-type draft-dspark`, essay / code / math | 93.6 / 92.0 / 131.6 | draft `erlidev/Qwen3.8-27B-DSpark-GGUF` BF16 (2.7 GB), `-ngld 999` |

```
ceiling  = 1701 GB/s / 16.102 GB = 105.6 tok/s
llama.cpp raw = 82.84 tok/s      =  78.4% of the wall
MTP gain      = 129.6 / 80.6     =  +61%  (essay; +59% code, +101% math)
```

The DSpark draft (RadixArk's 1.86B model, as a GGUF sidecar) is *worse* than
the built-in one-token MTP head inside llama.cpp: +16% on prose against +60%,
+63% on math against +100%. The same draft gives SGLang 1.65x / 3.25x, so
this is llama.cpp's DSpark integration, not the draft — a fresh
`--spec-type` that has not had the tuning the MTP path has. Within one engine
the comparison holds everything else constant, and it says the draft model is
not the lever; the verify loop is.

The MTP gain is content-dependent and large. It is not a property of the
llama.cpp version: commit `6703d78` (2026-09-03) built the same way on this
machine gives 83.11 tok/s raw and 129.5 with MTP, within noise of current
upstream. Other hosts measure it far lower with the same binary and weights —
the MTP loop has host-side work between small kernels, so CPU generation and
PCIe latency plausibly matter — which is why every number here names its
machine.

### Reproducing

From nothing, on a fresh instance. Nothing on this machine survives a recycle,
so this is the recovery path as much as the audit trail.

```bash
# 1. build — the arch flag is required; a default build has no sm_120 kernels
git clone https://github.com/ggml-org/llama.cpp /workspace/rivals/llama.cpp
cd /workspace/rivals/llama.cpp
export PATH=/usr/local/cuda-13.3/bin:$PATH CUDACXX=/usr/local/cuda-13.3/bin/nvcc
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON
cmake --build build --config Release -j 48      # ~6 min on 24 cores

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
`q8_0` KV, flash attention on. Bytes/token is weights (16.102 GB) plus the KV
re-read each step at 34.8 KB per token of context. The ceiling therefore
already prices in the longer cache — an engine holding a constant fraction of
the wall would track it.

| Context | decode | bytes/token | ceiling | % of wall |
|---|---|---|---|---|
| 0 | 81.54 tok/s | 16.10 GB | 105.6 | **77.2%** |
| 22k | 75.06 tok/s | 16.87 GB | 100.8 | 74.5% |
| 90k | 59.71 tok/s | 19.24 GB | 88.4 | 67.5% |
| 200k | 44.25 tok/s | 23.07 GB | 73.7 | **60.0%** |

llama.cpp gives up 17 points of roofline between short context and 200k.
Physics accounts for the ceiling falling 106 -> 74; it does not account for the
engine falling from 77% to 60% of that ceiling. At 200k, 40% of the available
bandwidth goes unused.

## SGLang

Measured 2026-09-04. SGLang **0.5.18** (`sglang[all]` from PyPI, torch
2.13.0+cu130, sgl-kernel 0.4.6.post1, flashinfer 0.6.17) in
`/workspace/venvs/sglang`. Weights `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` —
genuine `nvfp4-pack-quantized`: 4-bit, group 16, FP8 scales.
(`unsloth/Qwen3.8-27B-NVFP4` is FP8 despite the name, and
`RadixArk/Qwen3.8-27B-NVFP4` is a mixed FP8/FP4 config.)

### Bytes per token

| Component | Bytes |
|---|---|
| Text body | 16.245 GB |
| `lm_head` (left at BF16) | 2.543 GB |
| **Text path, read per decode step** | **18.788 GB** |
| MTP head | 0.849 GB |
| Vision tower | 0.921 GB |

Ceiling at 1701 GB/s: **90.5 tok/s**, against llama.cpp's 105.6. This
checkpoint reads *more* per token than Q4_K_M despite being "4-bit", because
`lm_head` is unquantized. Raw tok/s therefore flatters llama.cpp by roughly 15%
on byte count alone — another reason % of roofline is the only honest headline.

### Results

Raw decode (no speculation), three short prompts, streamed, best of 3:

| Prompt | tok/s | % of wall |
|---|---|---|
| essay | 63.2 | 69.8% |
| code | 63.1 | 69.7% |
| math | 63.1 | 69.7% |

Server: `--kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype float32
--cuda-graph-max-bs 1`, decode CUDA graphs on, Triton GDN kernels, flashinfer
attention. The server's own decode log reports the same 63.2 tok/s.

### Decode vs. context length

Same server, random-token prompts of the given length already in the KV cache,
256 decode tokens, best of 2. Bytes/token is weights (18.788 GB) plus 32 KB per
token of FP8 KV.

| Context | decode | bytes/token | ceiling | % of wall |
|---|---|---|---|---|
| 0 | 63.1 tok/s | 18.79 GB | 90.5 | **69.7%** |
| 22k | 60.9 tok/s | 19.49 GB | 87.3 | 69.8% |
| 90k | 56.2 tok/s | 21.67 GB | 78.5 | 71.6% |
| 200k | 49.9 tok/s | 25.19 GB | 67.5 | **73.9%** |

**SGLang does not collapse.** Its fraction of the wall is flat — slightly
rising — from empty context to 200k: the attention-decode kernel keeps up
with the growing KV, and the 48 GDN layers cost the same at every length.
llama.cpp loses 17 points over the same range. At 200k SGLang is *faster than
llama.cpp in absolute terms* (49.9 vs 44.3 tok/s) despite reading 2.1 GB more
per token.

What SGLang does lose is the 30% it never had: 70% of the wall at every
length, against llama.cpp's 77% at short context and the 96.6% a pure GEMV
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

| Prompt | tok/s | mean accepted length | vs. raw 63.2 |
|---|---|---|---|
| essay | 104.4 | 2.39 | 1.65x |
| code | 137.4 | 3.11 | 2.17x |
| math | 205.2 | 4.68 | 3.25x |

Effective tok/s per accepted token is 43.7 / 44.2 / 43.8 across the three —
the verify step costs the same regardless of content (about 22.8 ms, vs.
15.8 ms for a raw decode step), so the whole spread is acceptance. RadixArk's
own bs=1 figure of 3.16x is GSM8K, which is the math row here.

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

# raw decode; drop the last three lines for a plain server
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/workspace/venvs/sglang/bin/python -m sglang.launch_server \
  --model-path /workspace/models/Qwen3.8-27B-NVFP4 --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype float32 --context-length 262144 \
  --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
  --mem-fraction-static 0.88 --max-running-requests 1 --max-mamba-cache-size 6 \
  --cuda-graph-max-bs 1 --disable-prefill-cuda-graph --port 30000 \
  # DSpark: add the following and raise --mem-fraction-static to 0.92 with
  # --max-total-tokens 30000 (the bf16 draft takes 3.6 GB)
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path /workspace/models/Qwen3.8-27B-DSpark \
  --speculative-draft-model-quantization unquant \
  --speculative-draft-attention-backend flashinfer \
  --speculative-dspark-block-size 7 --speculative-num-steps 1 --speculative-eagle-topk 1
# with SGLANG_RAGGED_VERIFY_MODE=static in the environment

scripts/rivals/sglang_bench.py raw
scripts/rivals/sglang_bench.py context 0 22000 90000 200000
scripts/rivals/sglang_bench.py spec        # with the DSpark flags
```

`--max-mamba-cache-size` must be at least 5 (the hybrid state cache is sized
in units of `mamba_ratio=5` per request); `--disable-prefill-cuda-graph`
because capturing prefill graphs up to 32k tokens runs the card out of memory
next to a 256k KV pool.

## vLLM

Measured 2026-09-04. vLLM **0.28.0** (PyPI, torch 2.13.0+cu130) in
`/workspace/venvs/vllm`, same `QUASAR-QAT` NVFP4 checkpoint as SGLang, so the
same 18.788 GB per step and the same 90.5 tok/s ceiling. FlashInfer attention,
FP8 KV, `--max-num-seqs 1`, torch.compile and full CUDA graphs at their
defaults (startup takes about five minutes of compilation).

### Results

Raw decode, three short prompts, streamed, best of 3:

| Prompt | tok/s | % of wall |
|---|---|---|
| essay | 80.0 | 88.4% |
| code | 80.1 | 88.5% |
| math | 79.9 | 88.3% |

### Decode vs. context length

Same server at `--max-model-len 220000`, random-token prompts already in the
KV cache, 256 decode tokens, best of 2. Bytes/token as for SGLang.

| Context | decode | bytes/token | ceiling | % of wall |
|---|---|---|---|---|
| 0 | 79.7 tok/s | 18.79 GB | 90.5 | **88.1%** |
| 22k | 77.7 tok/s | 19.49 GB | 87.3 | 89.0% |
| 90k | 70.6 tok/s | 21.67 GB | 78.5 | 89.9% |
| 200k | 61.0 tok/s | 25.19 GB | 67.5 | **90.4%** |

vLLM holds 88–90% of the wall from empty context to 200k. At 200k it decodes
at 61 tok/s against SGLang's 50 and llama.cpp's 44, on the same or more bytes
per token. Its long-context decode is, within a few points, at the roofline.

### MTP speculation

`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`, using
the MTP head shipped inside the checkpoint (bf16, `mtp.*`), same server
otherwise. FULL decode CUDA graphs are still captured.

| Prompt | tok/s | vs. raw 80.0 |
|---|---|---|
| essay | 63.0 | 0.79x |
| code | 63.0 | 0.79x |
| math | 62.6 | 0.78x |

Mean acceptance length 1.84–1.93 per step (draft acceptance 87–93%), and it
is still **slower than not speculating**: a draft-plus-verify step costs about
30 ms against 12.5 ms for a raw decode step. Nearly-free verification is
exactly what a bs=1 engine gets from idle tensor cores, and vLLM's speculative
path does not get it — the EAGLE-style draft loop runs eager Triton kernels
between the graphs. This is headroom argument #2 in `CLAUDE.md` measured on
the strongest raw engine: its raw decode is at 88% of the wall, and its
speculation loses 21% from there.

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
```

## ExLlamaV3

Measured 2026-09-04. ExLlamaV3 **1.4.6** (release wheel `cu132.torch2.11.0`,
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

| Prompt | tok/s | % of wall |
|---|---|---|
| essay | 76.8 | 59.5% |
| code | 77.0 | 59.7% |
| math | 76.9 | 59.6% |

### Decode vs. context length

FP16 cache (ExLlamaV3's default; 64 KB per token of context over the 16
attention layers), random-token prompts already in the cache, 256 decode
tokens, best of 2. At 200k the cache alone is 13 GB and the process sits at
29 GB — it fits, barely.

| Context | decode | bytes/token | ceiling | % of wall |
|---|---|---|---|---|
| 0 | 76.5 tok/s | 13.18 GB | 129.0 | **59.3%** |
| 22k | 71.4 tok/s | 14.63 GB | 116.3 | 61.4% |
| 90k | 59.8 tok/s | 19.08 GB | 89.2 | 67.0% |
| 200k | 47.7 tok/s | 26.29 GB | 64.7 | **73.7%** |

The fraction *rises* with context, which says the attention decode kernel is
efficient and the short-context deficit is a constant per-step cost —
Python-side generator overhead and per-layer dispatch — that shrinks in
relative terms as the KV read grows. Absolute tok/s at 200k is close to
SGLang's (47.7 vs 49.9) on twice the KV bytes.

### MTP speculation

The MTP head shipped in the checkpoint (quantized to 4 bits alongside the
body), loaded as ExLlamaV3's `mtp` component and chained for one or two draft
tokens per step. Greedy, 256 tokens, same prompts. "Per step" is tokens
committed per verify step including the bonus token.

| Prompt | 1 draft token | per step | 2 draft tokens | per step |
|---|---|---|---|---|
| essay | 113.8 tok/s (1.48x) | 1.79 | 126.7 tok/s (1.65x) | 2.24 |
| code | 119.8 tok/s (1.56x) | 1.87 | 142.1 tok/s (1.85x) | 2.36 |
| math | 124.5 tok/s (1.62x) | 1.94 | 160.3 tok/s (2.08x) | 2.81 |

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
/workspace/venvs/exl3/bin/python scripts/rivals/exl3_bench.py /workspace/models/Qwen3.8-27B-exl3-4.0 mtp --draft-tokens 1
```

## What these numbers change

**The general-engine tax at bs=1 is engine-specific, and vLLM has mostly paid
it off.** On the same NVFP4 bytes, SGLang holds 70% of the wall and vLLM
**88%**; llama.cpp holds 77% on its own bytes. A bare GEMV stream reaches
96.6%. So the raw-decode headroom over the *best* rival is about 8 points,
not 30 — vLLM's torch.compile plus full-step CUDA graphs already capture most
of what headroom argument #3 in `CLAUDE.md` describes. The 30-point gap is
SGLang's, and it is what DSpark has to overcome before it starts.

**The long-context collapse is llama.cpp's, not the field's.** SGLang's
attention decode scales; its fraction of the wall rises slightly to 200k.
vLLM holds 88–90% all the way to 200k — it is already within a few points of
the wall at long context, on a hybrid model where only 16 layers pay for
length. The project's long-context target has to be "hold 92%+ of the wall at
200k", not "do not collapse": against llama.cpp that is +77%, against SGLang
+27%, against vLLM a few percent. Long context is a place to *match* the
best rival at the roofline, not a place to beat it by a margin.

**Speculation numbers are now measured, not assumed.** DSpark's mean accepted
length on this model is 2.4 on prose, 3.1 on code, 4.7 on math at gamma 7;
llama.cpp's MTP head gets +60% / +60% / +100% with a single draft token.
`feasibility.md`'s N≈2.5 is the prose number, and it is the conservative
end. What the rivals leave on the table is the verify cost: SGLang's verify
step is 1.44x a raw decode step for 8 tokens of verification, and llama.cpp's
MTP gain per accepted token is far below what a fused, graph-captured verify
would give. That, plus the ~30% raw tax, is the effective-throughput margin.


**Speculation is where every rival is weak, each in its own way.** vLLM's MTP
path is slower than its raw decode. llama.cpp's DSpark integration gets a
quarter of what the same draft gives SGLang. SGLang's verify step costs 1.45x
a raw step. ExLlamaV3 chains the MTP head well (2.8 tokens per step on math)
but on a 60%-of-wall raw decode. Nobody has both a raw decode near the wall
and a speculative loop built for one stream with idle tensor cores — which is
headroom argument #2 measured four times over.

**Raw decode is not where the win is.** "+30% over llama.cpp raw" means 108
tok/s, which at 4.0 bpw (ceiling 126.5) is 85% of the wall — the bottom of the
85–90% band, and a fraction vLLM already holds. At matched bytes the raw
margin over vLLM is 92% vs 88%: a few percent, inside the noise of a
quantization choice. Raw decode near the wall is table stakes, evidence of
competence, and nothing more.

**llama.cpp's MTP is not leaving much on the table at bs=1 on this host.**
+60% on prose and code, +100% on math, from a one-layer MTP head verifying
one draft token per step. Against 130 tok/s, the 220–280 tok/s effective
target is 1.7–2.2x — "2x or more over llama.cpp + MTP" now means 260 tok/s,
which at 4.0 bpw and 92% of the wall needs a mean accepted length of about
2.3. The margin that used to come from llama.cpp's weak MTP has to come from
deeper speculation instead.

**Long context is still where the gap is widest, but it is 17 points, not
27.** llama.cpp holds 77% of the wall at short context and 60% at 200k. Only
16 of 64 layers even produce KV — the other 48 carry constant-size recurrent
state — so an engine whose attention decode keeps memory-level parallelism up
as KV grows should degrade far less.

**Quantization parity has to be handled explicitly.** llama.cpp runs at 4.79
bpw; our target is 4.0. Comparing raw tok/s across that gap hands us ~20% for
free and a reviewer will say so. Report % of roofline as the headline, and
produce a matched-bpw GGUF with `llama-quantize` for a supporting comparison.

## Still to measure

- vLLM with a DFlash/DSpark draft (`method: dflash` / `dspark`) — only the
  built-in MTP head is measured; a Qwen DSpark draft model class does not
  exist in vLLM 0.28 (only `gemma4_dspark`)
