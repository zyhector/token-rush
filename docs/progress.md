# Phase 1 progress log

One entry per step, with the number it produced. Every speed figure here is
a **development number** on whatever instance was rented that day, quoted
against the 1701 GB/s wall recorded in `docs/environment.md`; the citable
numbers come from Phase 4. Prefill is compute-bound and not a project
target; it is logged because it is free to measure and catches regressions.

Instance for all entries so far: vast container 50295164, RTX 5090, driver
610.43.02, CUDA 13.3, torch 2.14.0+cu130, triton 3.8.0, transformers 5.16.1,
fla 0.6.0. 150 GB disk, 60 GB RAM.

## Where things stand (after step 22, 2026-09-09)

| | | |
|---|---|---|
| raw greedy decode, short context | **102 tok/s**, 9.8 ms/step, 82% of the 1701 GB/s wall (13.65 GB read/step, ceiling 124.6) | rivals: llama.cpp 83 (78%), vLLM 80 (88%, ~76% recounted), ExLlamaV3 77, SGLang 63 |
| speculative greedy (MTP chain in-graph, depth 3:4, 128k draft vocab), essay / code / math | **186 / 261 / 263 tok/s**; Chinese essay / math 162 / 279 | best rival per family: llama.cpp+MTP 130, SGLang+DSpark 137 / 205 |
| speculative sampled, T=0.7 top-p 0.9 | 167 / 237 / 239 | output distribution identical to raw sampling |
| speculative at 200k context, fp8 KV, prose / code | **195 / ~200 tok/s** (raw 71.7 = 85.6% of the wall) | vLLM 61, SGLang 50, llama.cpp 44 at 200k |
| context | 256k usable (needle at 128k and 256k), 26 GB peak | |
| correctness | bf16 path = HF on 48/48 greedy tokens; spec = raw greedy 200/200 with shared kernels; every fused kernel differential-tested; 39 tests | |
| quantization | int4 g128 RTN, uncalibrated: teacher-forced KL 0.06 vs bf16, 2–4x a calibrated quant's; **the quality gate is not met** (`docs/quality_plan.md`, deferred to a two-GPU box) | |
| phases | 0 done (frozen), 1a done, 1b half (gate done, quality owed), 2 done, 3 in progress (chain done; tree, DSpark, GEMV tuning open), 4 not started | |

Every number above is a development number on this instance; Phase 4
re-measures everything on one machine.

| Step | Date | State of the engine | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| 1. Environment and weights | 2026-09-08 | no engine yet | — | — |
| 2. Skeleton runs | 2026-09-08 | full text path, int4 g128 RTN, dequantize-then-matmul GEMV, eager | 1490 | 3.2 |
| 3. GEMV shootout, kernel swapped in | 2026-09-08 | + own Triton int4 GEMV (92% of wall inside the model), fused q\|k\|v and gate\|up projections; still eager, CPU-bound on ~2500 launches | 1500 | 36 (46 with tinygemm) |
| 4. Whole-step CUDA graph | 2026-09-08 | + one graph per context bucket, device position, in-graph argmax feeding the next step; GPU-bound at 14.6 ms/step | 1500 | **68.4** (55% of wall) |
| 5. Engine-correctness gate vs HF | 2026-09-08 | bf16 path streamed layer by layer against HF transformers 5.16.1: 48/48 greedy tokens identical, residual error <=2% with no jump; engine declared correct | — | — |
| 6. Phase 2: fused GDN step, fused add+norm | 2026-09-08 | + one Triton kernel per GDN layer (conv, gating, delta rule, gated norm, output gate), one kernel per residual+RMSNorm, silu*mul fused; ring-buffer conv state | 1500 | **83.2** (67% of wall) |
| 7. Phase 2: fused attention decode | 2026-09-08 | + prep kernel (q/k norm, rope, KV write) and a flash-decoding kernel over the live length with the output gate in its reduce; no context buckets, one graph | 1500 | **99.5** (80% of wall) |
| 8. Phase 2: b\|a folded into the GDN kernel, split-K GEMV for the hidden-sized outputs | 2026-09-08 | + no cuBLAS launches left in the step; out/o/down projections at split-K 4 write fp32 partials that the fused add+RMSNorm sums | 1500 | **102.5** (82% of wall) |
| 9. Phase 2: FP8 KV cache + own prefill attention kernel | 2026-09-08 | + e4m3 cache with per-(head, token) scales; prep/decode kernels read it; a Triton flash-attention prefill kernel replaces SDPA for fp8. **256k usable**: needle at 128k and 256k; decode at 200k = 69.2 tok/s, 82.6% of wall | 1160–1640 at 128k–256k | 101.9 short, **69.2 at 200k** |
| 10. Phase 2: sampling in graph | 2026-09-08 | + temperature / top-k / top-p on a fixed top-64 candidate set, branch-free, parameters as device tensors | 1500 | 101.9 (no change) |
| 11. Phase 2: GEMV config re-pick (L2-proof sweep) | 2026-09-08 | + configs re-picked with weights cycled through >400 MB; **no measurable change** in the step. Phase 2 closed | 1500 | 102.2 (82% of wall) |
| 12. Phase 3 step 1: MTP head acceptance rate | 2026-09-08 | + MTP head implemented eagerly (`tokenrush/mtp.py`); chained drafts teacher-forced against the target's greedy: accepted tokens per verify step at depth 3 = 2.58 prose / 3.35 code / 3.50 math | — | — |
| 13. Phase 3 steps 2–3: verify step in one graph, accept/commit on device | 2026-09-08 | + M-row kernels (tensor-core int4 GEMM, M-token GDN with per-prefix state slots, M-query attention), a graphed verify step per K with device-side accept/commit; eager MTP drafts. K=3 verify = 1.22x a raw step. Spec greedy == raw greedy 200/200 with shared numerics | — | **172 / 212 / 204 effective** (essay / code / math), raw 102 |
| 14. Phase 3 step 4: MTP draft chain inside the graph, int4 head, adaptive depth | 2026-09-08 | + one graph per depth runs the MTP batched pass over the accepted rows, the chained drafts and the verify; depth = clamp(n_prev+2, 3, 4) | — | **183 / 254 / 258 effective** (essay / code / math), raw 102 |
| 15. Speculation is the default decode | 2026-09-08 | + `run.py` decodes speculatively by default (`--no-spec` for raw; temperature > 0 falls back to raw until spec sampling lands); chunked prefill keeps every position's hidden for the MTP prompt pass; the loop stops before a step could overrun the cache; works with fp8 KV | — | 154 on a short haiku (2.1 tokens/step), 280 on a 30k needle answer |
| 16. Speculative sampling (temperature > 0) | 2026-09-08 | + the verify step draws one sample per position; a draft is accepted only when it equals the draw, the draw is what gets committed: the output distribution is exactly the target's. Chat at T=0.8: 165 tok/s | — | **167 / 237 / 239** at T=0.7 top-p 0.9 (essay / code / math), raw 102 |
| 17. Speculation vs. context length (real long text, fp8 KV) | 2026-09-08 | + `bench/spec_context.py` on WikiText-103 (prose) and torch's Python sources (code); MTP prompt pass streamed per chunk so 200k fits | — | **prose 228 / 188 / 164 / 174** and **code 250 / 201 / 251 / 197** at 0 / 22k / 90k / 200k; raw 100 / 95 / 82 / 68 |
| 18. fp8 cache for the MTP head | 2026-09-08 | + the draft head's own single-layer cache in e4m3 (follows the engine's KV dtype in `run.py`) | — | at 200k: prose 180 (was 174), code 199 (was 197); short context unchanged |
| 19. rows-GEMM config re-pick (L2-proof sweep at M=4) | 2026-09-08 | + configs re-picked; **neutral**: verify K=3 at 1.20x raw (was 1.21x). The M-row kernel's 79–84% on layer shapes is structural (bf16 dequant + dot per block), not a config matter | — | 179 / 243 / 252 (noise vs step 14) |
| 20. Truncated draft vocabulary | 2026-09-08 | + the draft chain's argmax reads the first 131072 rows of lm_head (id order = BPE merge rank, language-neutral) instead of all 248k: −0.7 ms per step, no family loses. A 64k English-corpus list was tried first and cut Chinese below raw | — | **186 / 261 / 263** (essay / code / math), Chinese essay / math 162 / 279; raw 100 |
| 21. Corpus-specific draft vocabularies (en_64k, mix_64k, mix_96k) | 2026-09-08 | + lists built from English, code and Chinese Wikipedia corpora, shipped in `tokenrush/draft_vocab/`, selectable by name; measured on six families incl. Chinese and mixed. None beats the id-order 128k default by more than noise; mix_96k is the pick for a Chinese-English daily driver | — | id_128k 186 / 261 / 263 / 162 / 279 / 216 vs mix_96k 185 / 253 / 262 / 162 / 279 / 227 (essay / code / math / zh-essay / zh-math / mixed) |
| 22. Profile of the spec step at 200k; flash-decoding pipeline depth | 2026-09-09 | + the spec step's extra growth with context attributed (M-row attention kernel under-occupied, MTP's own attention); one config change (3 pipeline stages) lifts the kernel from 1390/1147 GB/s (M=1/M=4) to 1587/1558 | — | at 200k: raw **71.7** (85.6% of wall, was 68.4 / 82.6%), spec prose **195** (was 180); short context unchanged |
| 23. Draft trees: simulated, not built | 2026-09-09 | + `bench/tree_accept.py` (teacher-forced acceptance of static trees with the eager MTP) and verify cost measured to M=8: the best 7-node tree gains +10% acceptance on prose, +3–4% on code/math, for a step ~20% dearer. **Net negative on this stack; trees dropped.** A shared-memory overflow for M >= 6 fixed on the way | — | unchanged |
| 24. Research: what is left to gain | 2026-09-09 | + surveyed 2025–26 single-stream speculation and small-M int4 GEMM work; measured z-lab's **DFlash2** block-diffusion draft teacher-forced on our target: 3.57 / 5.37 / 5.76 accepted per 7-draft step (essay / code / math) vs the MTP chain's 2.75 / 3.88 / 4.05 at K=4, from one draft forward. Projected 238 / 358 / 384 tok/s in-graph; with a Marlin-class M-row GEMM ~275 / 413 / 443 | — | unchanged |

## Step 1 — environment and weights (2026-09-08)

- `/venv/main` provisioned from the recipe in `docs/environment.md`.
- `check_stack.py`: same picture as the Phase 0 machine. Triton `sm_120`
  codegen and CUDA graph replay work; `fla`'s fused GDN step returns NaN
  heads; the chunk kernel and the minimal Triton step match the FP32
  reference over repeated calls. Its timing output was not recorded (freeze).
- `Qwen/Qwen3.8-27B` bf16, 18 shards, 55.6 GB, at `/workspace/models/Qwen3.8-27B`.
  Verified by size against the Hub listing and by parsing every header (the
  shipped `crc32.txt` does not cover the shards). Parameter split matches
  `CLAUDE.md`: text body 25.625B, `lm_head` 1.271B, MTP 0.425B, vision 0.461B.
  The vision tower shares shard 1 with 59 text tensors, so it is dropped at
  load time, not download time.
- Facts read from the config that the model code needs: 64 layers in a
  3 GDN + 1 attention pattern, attention output gate, partial rotary on 64
  of 256 dims with interleaved mRoPE (collapses to plain RoPE for text),
  swish output gate on GDN, FP32 recurrent state, untied embeddings.
- Disk reads at ~2.5 GB/s, so a pass over the bf16 checkpoint is ~25 s.

## Step 2 — skeleton runs (2026-09-08)

`tokenrush/`, ~800 lines, commit `dfd7cd8`. What it is:

- **Explicit state** (`state.py`): contiguous preallocated KV per attention
  layer `[layer, kv_head, pos, 256]`, conv state `[layer, 10240, 3]`, FP32
  recurrent state `[layer, 48, 128, 128]`, a position counter. Allocated
  once, written in place.
- **Layers as pure functions** over dataclass weight containers
  (`model.py`); activations `[T, hidden]`, bs=1 implicit; `T == 1` is the
  decode path, `T > 1` the prefill path. Chunked prefill; greedy loop.
- **Existing blocks only**: fla `chunk_gated_delta_rule` for GDN prefill,
  the verified Triton step from `check_stack.py` for GDN decode (state
  updated in place), SDPA attention with a causal-over-cache boolean mask in
  1024-query blocks, cuBLAS for everything dense.
- **Quantization**: own int4 g128 asymmetric RTN packing (4.25 bpw), bf16
  scale and minimum per group; the GEMV is dequantize-then-matmul. Packed
  checkpoint at `/workspace/models/Qwen3.8-27B-int4g128` (17 GB, packs in
  18 s, loads in 3 s). MTP head packed alongside in bf16, loaded on request.
- **Mirrors HF exactly where it matters**: RMSNorm gain is `1 + weight`,
  the GDN gated norm's is plain; `q_proj` emits query and sigmoid gate
  interleaved per head; rotary on the first 64 dims after q/k norm.

Results on the real weights (eager, `--max-len 32768`, peak 21 GB VRAM):

| Run | Outcome |
|---|---|
| "The capital of France is" | "Paris. The capital of Germany is Berlin. ..." |
| chat, thinking off, haiku + one-sentence explanation | correct, stops on `<\|im_end\|>` |
| 5451-token needle prompt, 3 prefill chunks of 2048 | retrieves the needle |

| | tok/s | note |
|---|---|---|
| prefill (5451 tokens) | 1490 | ~80 TFLOPS; compute-bound, dequant amortized over a 2048-token chunk |
| decode | 3.2 | each step expands 14 GB of int4 into 28 GB of bf16 and reads it back |

Tests (`tests/test_ops.py`, 10 passing): each op against a torch or HF
reference (exact for RMSNorm, gated norm, rotary against HF's modules; the
GDN step against an fp32 reference over repeated calls), int4 pack
round-trip, and a structural test that one-shot prefill, chunked prefill and
token-by-token decode agree. Finding from that test: the three paths differ
by ~2% in norm because SDPA tiles by sequence length and fla's chunk kernel
accumulates in bf16 — so the criterion is "bounded, and no jump after a
chunk boundary", not elementwise closeness.

## Step 3 — GEMV shootout, kernel swapped in (2026-09-08)

`scripts/gemv_shootout/` (method, full table and decisions in its README),
`bench/decode.py` (the engine's decode timer and profiler).

Six candidates at the real shapes, in-graph, packed GB/s against 1701:
bf16 cuBLAS 96–97% (the reference), dequant-then-matmul 2.6% (what step 2
ran), **our Triton int4 88% on the fused layer set and 99% on `lm_head`**,
torch's tinygemm 91% / 99%, vLLM's Marlin 88% / 97%, vLLM's NVFP4 cutlass
GEMM 52% / 58% (a GEMM, not a GEMV; the wrong tool at M=1).

Decisions: our Triton kernel is the default (one copy of the weights, prefill
dequantizes the same nibbles, ours to fuse in Phase 2); q|k|v, in_proj_qkv|z
and gate|up are fused into one matrix each at load time (+4–5 points: a
2.8 MB GEMV costs 6 µs whatever the kernel); autotune pinned to six configs.

| | tok/s | note |
|---|---|---|
| decode, triton backend | 36.0 | 27.7 ms/step: GEMVs 9.1 ms GPU (1560 GB/s = 92% of wall), other ~2500 kernels 9 ms GPU, **CPU launch time 30 ms** |
| decode, tinygemm backend | 46.4 | same GPU work; aten launches cost ~25 µs less Python each than Triton launches |
| prefill (5451 tokens) | ~1500 | unchanged: dequant + cuBLAS GEMM |

The step is now CPU-bound: the GPU finishes its 18 ms of work while the host
is still issuing launches for 30 ms. That is Phase 0's launch tax measured on
the whole model, and it is what step 4's CUDA graph removes. With the GEMVs at
92% of the wall, the graphed step should land near 9 + (fused GDN/attention
chain) ms; the eager non-GEMV GPU time of 9 ms is the next thing to shrink
after that.

## Step 4 — whole-step CUDA graph (2026-09-08)

What changed (`state.py`, `model.py`, `ops.py`): the position is a device
tensor; rotary lookup and KV writes index with it; decode attention runs over
a fixed context bucket (1k, 2k, 4k, ... , max_len) with a mask derived from
the position, so a step is shape-static; `Engine.capture()` records one graph
per bucket into a shared pool, and a step is `graph.replay()` with the argmax
written back into the step's own input token. Nothing returns to the host.
Test: replay matches the eager bucketed step bit for bit.

| | | note |
|---|---|---|
| decode | **68.4 tok/s**, 14.62 ms/step | equals the step's GPU time: CPU launch cost is gone |
| bytes read per step | 13.65 GB | 25.6B streamed params at 4.25 bpw; the 2.5 GB embedding table is not read (earlier "16.19 GB" figures counted it) |
| ceiling | 124.6 tok/s | 1701 / 13.65 |
| % of wall | 55% | |
| capture | 4 graphs, 4.8 s, +0.6 GB | max_len 8192 in the bench |

Where the 14.6 ms of GPU time goes now: GEMVs 9.06 ms (92% of the wall on
their bytes), memory-efficient SDPA over the 1024 bucket 1.18 ms for 16
layers, and **~4.4 ms in about 2200 elementwise, reduce and copy kernels**
of the GDN chain, the norms and the gating. That last item is Phase 2's
GDN-step fusion; at Phase 0's 48-layer measurement (1.4 ms graphed for the
whole chain) it is worth ~3 ms, i.e. the step goes to ~11.5 ms and ~87 tok/s
before attention or sampling are touched.

## Step 5 — engine-correctness gate against HF (2026-09-08)

Phase 1b, first half. `bench/hf_reference.py` dumps HF's residual stream
after every layer, prompt logits and 48 greedy tokens with their logits
(Qwen3_5ForCausalLM, bf16, CPU-offloaded, `fla` blocked so HF's GDN path is
pure torch; 1.9 s per decode step). `bench/gate_engine.py` runs our engine
against the dump; `StreamingBF16Engine` builds each layer's bf16 weights
from the HF checkpoint on demand since 55.6 GB does not fit the card.

**bf16 path (the gate):**

| | |
|---|---|
| residual stream vs HF, after each of 64 layers | 1.7e-3 at layer 1, growing smoothly to 2.0e-2 at layer 64; no jump |
| prompt logits | top-1 agreement 32/32, KL 1.3e-4 mean, 1.8e-3 max |
| teacher-forced decode, 48 steps | **top-1 agreement 48/48**, max logprob gap 0.000, KL 3.8e-4 mean |
| greedy tokens identical | **48 of 48** |

Passes the gate in `CLAUDE.md` with no divergence at all. The 2% residual
drift over 64 layers is bf16 accumulation through different kernels (fla's
chunk kernel and SDPA against HF's pure-torch loops), the same order as the
chunked-vs-stepped drift seen in step 2. The engine is correct; every fused
kernel from here is tested against these ops.

**int4 g128 RTN path (first quality data point, not yet the quality table):**
residual error 3.5% after layer 1 rising to 19% at layer 63; prompt top-1
96.9%; teacher-forced top-1 37/48 with HF's token always in our top-5, KL
6e-2 mean; greedy diverges at step 4. That is what 4.25-bit round-to-nearest
with no calibration costs on this model, and it is the number the Phase 1b
quality table has to beat (ExLlamaV3's 4.00 bpw KL is the bar).

## Step 6 — Phase 2 begins: fused GDN step and fused norms (2026-09-08)

`tokenrush/fused.py`, tests in `tests/test_fused.py`. Kernel anatomy of one
decode step at the start: a GDN layer ran 26 kernels (2 GEMVs, 1 small
cuBLAS GEMV for b|a, 1 delta-rule step, 22 elementwise/reduce/copy kernels),
every residual+RMSNorm 11 kernels, an attention layer 51.

Three fused kernels, each differential-tested against the ops it replaces
(which step 5 verified against HF):

- **`gdn_step_fused`**: conv step, sigmoid/softplus gating, L2 norms, delta
  rule state update, gated RMSNorm and silu output gate for one GDN layer in
  one launch, one program per V head (BV=128; splitting the head into more
  programs plus a second norm kernel was measured slower). 22 kernels -> 1.
  It needed the conv state to become a **4-column ring** (column = position
  mod 4) so a program can write the new input while others still read the
  older columns.
- **`add_rmsnorm`**: residual add (rounded to bf16, as the residual stream is)
  and RMSNorm with the (1 + w) gain in one launch. 11 kernels -> 1, 129 per step.
- **`silu_mul`**: 2 -> 1.

A finding on the way: the eager `conv_step` multiplied in bf16 and rounded
every product; HF's cuDNN conv1d accumulates in fp32 and rounds once, and so
does the fused kernel. The fused kernel matched HF exactly; the eager op was
the odd one out and was fixed. The fused path reproduces the int4 gate
numbers of step 5 to three digits (teacher-forced KL 6.04e-2 vs 6.05e-2,
same divergence step).

| | | |
|---|---|---|
| decode | **83.2 tok/s**, 12.02 ms/step | 67% of the wall, ceiling 124.6 |
| GEMVs | 9.06 ms | unchanged, 92% of the wall on their bytes |
| attention (SDPA over the 1024 bucket) | 1.19 ms | 16 layers, 74 us each |
| fused norms | 0.32 ms | 129 launches |
| fused GDN steps | 0.31 ms | 48 launches, 6.4 us each |
| b\|a GEMV (cuBLAS) | 0.20 ms | 48 launches; foldable into the fused kernel |
| attention layer small kernels | ~0.6 ms | ~48 launches per attention layer: q/k norm, rope, cat, index ops, mask, gate |

Next inside Phase 2: the attention layer (one prep kernel for q/k norm +
rope + KV write, then our own decode attention kernel reading the live
length instead of a bucket, with FP8 KV so 256k fits), then the b|a fold and
GEMV split-K for the two 5120-row shapes.

## Step 7 — fused attention decode (2026-09-08)

Three kernels in `tokenrush/fused.py` replace SDPA-over-a-bucket and the ~48
small kernels around it in each attention layer:

- **`attn_prep`**: one program per head (24 q + 4 kv): RMSNorm with the
  (1 + w) gain, rope on the first 64 dims with HF's bf16 rounding order,
  query out, key and value written into the cache at the device position.
- **`attn_split`**: flash-decoding. One program per (kv head, split of the
  key range), 32 splits; the 6 query heads of a kv head are padded to 16
  rows so the score and PV products run on tensor cores (`tl.dot`), online
  softmax in fp32, keys beyond the live length masked. The live length comes
  from `pos_t`, the grid is static, so it captures into a graph and **the
  context buckets are gone**: one graph per engine.
- **`attn_reduce`**: one program per query head combines the splits,
  normalizes, and applies the sigmoid output gate.

A 64-key block of bf16 K and V triple-buffered exceeded the 101 KB of shared
memory per block on this card; 32-key blocks with two stages fit.

| | | |
|---|---|---|
| decode | **99.5 tok/s**, 10.05 ms/step | 80% of the wall, ceiling 124.6 |
| GEMVs | 9.06 ms | 90% of the step |
| attention, 16 layers | 0.09 ms | was 1.8 ms (SDPA 1.19 + small kernels 0.6) |
| fused norms / GDN / silu / b\|a | 0.32 / 0.30 / 0.07 / 0.20 ms | |
| launches per step | ~600 | was ~2500 |

The gate through the new path: teacher-forced KL 6.1e-2 vs 6.0e-2 before,
HF's token in our top-5 48/48, same divergence step. Tests cover short and
long (2100+) contexts where all splits are active and the last block is
partial.

Decode vs. context (`bench/decode.py --context N`, bf16 KV, 32k cache;
bytes per step = weights + live KV at 64 KB/token):

| context | ms/step | tok/s | ceiling | % of wall |
|---|---|---|---|---|
| 0 | 10.05 | 99.5 | 124.6 | 79.9% |
| 22k | 10.93 | 91.5 | 112.7 | 81.2% |
| 30k | 11.26 | 88.8 | 108.9 | 81.5% |

Flat, slightly rising: the attention kernel keeps up with the KV read, as
SGLang's and vLLM's do (llama.cpp's falls 17 points over this range). Needle
retrieval passes at 5k and 30k tokens through the fused path.

The step is now the GEMVs plus 1 ms. What is left inside Phase 2: fold the
b|a projection into the GDN kernel (0.2 ms), split-K for the two 5120-row
GEMV shapes (they run at 74–81% against 90%+ for the wide ones: ~0.5 ms),
sampling in-graph beyond argmax, FP8 KV with a prefill attention kernel
for 256k.

## Step 8 — b|a into the GDN kernel, split-K for the narrow GEMVs (2026-09-08)

- The b and a gate projections (2 x 48 rows of 5120) are now two dot
  products inside `gdn_step_fused`, per program, fp32-accumulated and rounded
  once as cuBLAS did. The 48 cuBLAS `gemvx` launches are gone; the GDN kernel
  grew from 0.30 to 0.35 ms. Net -0.15 ms.
- `int4_gemv_splitk`: the three matrices whose output is the hidden size
  (`out_proj`, `o_proj`, `down_proj`; 5120 rows, so too few programs to fill
  the card) cut their K range into split-K pieces and write fp32 partials
  `[S, 5120]`; `add_rmsnorm` sums the partials as it reads them (rounding the
  sum to bf16 first, the value a plain GEMV would have produced), so the
  reduction costs no launch. Measured on the model: split-K 1 / 2 / 4 / 8 ->
  99.5 / 102.5 / 102.6 / 101.3 tok/s; 4 is the default. Those matrices went
  from ~78% to 87% of the wall; the wide ones are at 91.5%, `lm_head` at 99%.
- **A bug caught by the gate, not by the tests**: with partials in flight,
  the final norm's `h[-1:]` kept one partial row of the last MLP output.
  The text stayed coherent; the teacher-forced KL against HF went from 0.060
  to 0.196. Fixed; a regression test now checks the final-norm input
  against the traced residual. The gate is back to KL 5.97e-2, HF's token
  in our top-5 48/48. Lesson recorded in `docs/traps.md`.

| | | |
|---|---|---|
| decode | **102.5 tok/s**, 9.76 ms/step | 82.3% of the wall, ceiling 124.6 |
| GEMVs | 8.90 ms (wide 6.14, split-K 2.76) | 91% of the step |
| everything else | 0.86 ms | GDN 0.35, norms 0.34, silu 0.07, attention 0.09 |

Phase 2 remaining after step 8: FP8 KV (step 9), sampling (step 10), then
GEMV from 88–91% toward 95% on the layer matrices (the last ~0.5 ms), which
is the "GEMV last, if measurement demands it" item. The step is 91% GEMV, so from here the number moves with
bytes (quantization, Phase 1b) and with speculation (Phase 3), not with
fusion.

## Step 9 — FP8 KV cache and a prefill attention kernel: 256k usable (2026-09-08)

bf16 KV caps the context at ~32k on this card (64 KB/token next to 16 GB of
weights). Now `State(kv_dtype=torch.float8_e4m3fn)`: K and V stored as e4m3
with one fp32 scale per (head, position) (`value = code * scale`, scale =
amax/448), 32 KB/token plus 128 B of scales. Three pieces:

- `attn_prep` gets an FP8 branch: after norm and rope it computes the amax of
  the roped key, writes the scale and the quantized row; same for V.
- `attn_split` (decode) dequantizes K rows by their scale before the QK dot,
  and folds the V scales into the probabilities before the PV dot.
- **`attn_prefill_fused`**: a Triton flash-attention forward for a prefill
  chunk (one program per 64-query block and query head, causal over the
  whole cache, fp32 online softmax, tensor-core dots) since SDPA cannot read
  an fp8 cache. Tested in bf16 mode against SDPA, then used in fp8 mode.
  The prefill path writes the cache through `kv_write_prefill` with the same
  quantization convention as the prep kernel (tested: the rows agree).

Quality on the gate prompt: teacher-forced KL 5.88e-2 with fp8 KV vs 5.97e-2
with bf16 — no measurable loss; the int4 weights dominate. (Proper
measurement belongs in the Phase 1b quality table.)

| | bf16 KV | fp8 KV |
|---|---|---|
| decode, short context | 102.8 tok/s (82.5%) | 101.9 tok/s (81.8%) |
| decode at 30k | 91.6 (84.1%) | 94.7 (81.5%; reads 1 GB less) |
| decode at 200k | does not fit | **69.2 tok/s, 14.45 ms, 82.6% of the wall** (20.3 GB/step, ceiling 83.8) |
| needle at 128k / 256k | — | retrieved / retrieved |
| prefill at 128k / 256k | — | 79 s / 224 s (1640 / 1160 tok/s; attention is O(L^2)) |
| VRAM at 256k | — | state 8.9 GB, peak 26.1 GB |
| graph capture at 256k | — | one graph, 8.5 s |

The 200k row against the rivals (`docs/baselines.md`, their KV formats):
llama.cpp 44.3 tok/s (60% of its wall), SGLang 49.9 (74%), vLLM 61.0 (90%,
or ~81% with the embedding removed from its byte count), Token Rush 69.2
(82.6%). The fraction is flat from 0 to 200k; the target of 92% at 200k is
the same gap as at short context, i.e. GEMV headroom, not attention.

## Step 10 — sampling inside the graph (2026-09-08)

`tokenrush/sample.py`: temperature, top-k and top-p on a fixed top-64
candidate set, all shape-static and branch-free (greedy is selected
arithmetically when temperature is 0), parameters are device tensors read
at replay, the uniform draw comes from torch's CUDA generator which the
graph captures. About ten small torch kernels; a single fused sampler kernel
is possible later. `run.py --temperature/--top-p/--top-k/--seed`. Tests:
greedy equals argmax exactly; top-k/top-p masks hold; empirical frequencies
match the truncated softmax within 2%; in-graph replay samples valid tokens
and two seeds differ.

## Step 11 — GEMV config re-pick, and Phase 2 closed (2026-09-08)

Suspicion: Triton's autotuner times each config on the same inputs, so a
30–90 MB layer matrix sits in the 96 MB L2 during tuning and the picks are
"L2-optimal", while `lm_head` (675 MB, cannot be resident) is the one shape
at 99%. An L2-proof sweep (cycling >400 MB of weight copies per shape, 240
configs per wide shape, 54 per split-K shape) found the pinned wide-shape
configs within 0.4–1.3 points of the best and the split-K shapes 2–3 points
short. Re-pinned all of them (`_GEMV_CONFIGS`, `_SPLITK_CONFIGS`).

Whole-step result: 102.2 tok/s vs 102.5 before, 8.85 vs 8.90 ms of GEMV time
— inside noise. The isolated gains do not survive interleaving with the
rest of the step. What holds the layer matrices at 87–93% (vs 99% on
`lm_head`) is the kernel's structure — one program streams a few rows
through the whole K with a sequential loop — not its parameters. A
different design (wider per-program tiles with more loads in flight, or a
cooperative reduction over K) is the remaining GEMV headroom, worth ~0.5 ms
= 5 tok/s, and it is the "do not start by chasing the last 5%" item in
`docs/feasibility.md`. Left for after Phase 3.

### Phase 2 close-out

| | |
|---|---|
| decode, short context | **102.2 tok/s**, 9.78 ms/step, **82% of the wall** (ceiling 124.6 at 4.25 bpw) |
| decode at 200k, fp8 KV | **69.2 tok/s**, 82.6% of the wall (ceiling 83.8) |
| context | **256k usable**: needle at 128k and 256k, 26 GB peak |
| launches per step | ~600 (was ~2500): 257 GEMVs, 129 norms, 48 GDN, 64 silu, 48 attention, ~10 sampling |
| where the step goes | GEMVs 8.85 ms (91%), everything else 0.93 ms |
| correctness | bf16 path 48/48 greedy tokens vs HF; every fused kernel differential-tested; graph replay bit-exact vs eager |

Against the plan's Phase 2 list (`CLAUDE.md`): full-step graph, fused GDN
step, attention decode, fused sampling — all done; GEMV "last and only if
measurement demands it" — measured, and it demands a redesign rather than
tuning, deferred. The raw-decode target (110–120 tok/s, 90–95% of the
wall) is not met: 82%. The gap is the GEMV kernel (~5 tok/s) and, beyond
that, bytes: the quality table (Phase 1b) decides whether 4.0 bpw is
allowed, which alone is worth +7 tok/s of ceiling.

Rivals, same yardstick, Phase 0 numbers: llama.cpp 78% / 82.8 tok/s, vLLM
88% (~76% recounted) / 80.0, SGLang 70% / 63.2, ExLlamaV3 60% / 77.0; at
200k: vLLM 61.0, SGLang 49.9, llama.cpp 44.3. Token Rush raw decode is now
the fastest in absolute tok/s at short and long context, on fewer bytes;
the rivals' speculative modes (104–205 tok/s) are what Phase 3 is for.

## Step 12 — Phase 3 begins: the MTP head's acceptance rate (2026-09-08)

`tokenrush/mtp.py` implements the shipped MTP head eagerly, following
vLLM's `Qwen3_5MultiTokenPredictor`: `fc(cat(norm_e(embed(next_token)),
norm_h(target_hidden)))` into one full-attention block with its own KV cache
at next_token's position, then `lm_head(norm(.))`; `target_hidden` is the
target's post-final-norm vector (the one `lm_head` sampled next_token from),
and chaining feeds the block's own normed output. Embedding and `lm_head`
are the target's. `Engine.last_hidden` exposes the target's vector.

`bench/mtp_accept.py`: the target generates 256 greedy tokens per prompt
family (the Phase 0 prompts, chat-formatted, thinking off), then at every
position the MTP drafts a chain of 4 exactly as the engine will, each draft
compared with the true continuation.

| | essay | code | math | all |
|---|---|---|---|---|
| P(draft 1 correct) | 0.778 | 0.917 | 0.933 | 0.876 |
| P(draft 2 correct \| 1) | 0.496 | 0.794 | 0.841 | 0.710 |
| P(draft 3 correct \| 1,2) | 0.306 | 0.643 | 0.730 | 0.560 |
| P(draft 4 correct \| 1..3) | 0.187 | 0.524 | 0.639 | 0.450 |
| accepted tokens / verify step, depth 1 | 1.78 | 1.92 | 1.93 | 1.88 |
| depth 2 | 2.27 | 2.71 | 2.77 | 2.59 |
| **depth 3** | **2.58** | **3.35** | **3.50** | 3.15 |
| depth 4 | 2.77 | 3.88 | 4.14 | 3.60 |

Two checks that the semantics are right: the depth-2 numbers match
ExLlamaV3's measured chained MTP on the same model (2.24 / 2.36 / 2.81 in
`docs/baselines.md`, on its own prompts), and a wrong wiring would give
near-zero acceptance. `docs/feasibility.md` assumed 2.4 / 2.6 / 3.2 for a
three-deep chain; measured is 2.58 / 3.35 / 3.50 — the lever the whole
Phase 3 number rests on is larger than planned.

Step-cost projection with these numbers (raw step 9.8 ms; verify step taken
at 1.1x = 10.8 ms; a three-deep MTP chain in-graph ~1.6 ms if the head is
int4 and `lm_head` is read three times): ~12.4 ms per verify step ->
**~210 tok/s prose, ~270 code, ~280 math**; depth 4 adds ~0.5 ms and gives
~215 / ~300 / ~320. The prose floor of 200–240 in `CLAUDE.md` is
reachable with the shipped head alone, before any DSpark-class draft.

Cost of the MTP head: 0.85 GB bf16 (0.425B params); the eager chained draft
costs 3.5–4.7 ms today through ~150 launches, which is the Phase 3 step-2/3
work (T=K verify kernels, in-graph accept/commit).

## Step 13 — Phase 3 steps 2 and 3: the verify step in one graph (2026-09-08)

**What a verify step is.** The engine holds a committed-but-unprocessed token
t at position pos; K drafts follow. One graph replay processes all K+1 tokens
at pos..pos+K, takes the argmax at every position, counts the leading drafts
that match (n), and commits on the device: `tok` = the argmax after the last
accepted token (the correction or bonus), `pos += n+1`, recurrent state =
the snapshot after n+1 tokens. Nothing returns to the host except n.

**Every kernel got an M-row variant** (M = K+1 <= 8):

- GEMV: `_int4_gemm_rows_kernel` dequantizes each weight block once to bf16
  and multiplies all rows against it on the tensor cores (`tl.dot`), so a row
  costs a dot, not another pass over the bytes. The first version did per-row
  masked reductions inside the loop and cost 3 ms per extra row (K=3 at
  1.92x a raw step); the dot version is at **1.22x**.
- GDN: the fused step loops the M tokens inside the program; the state is
  read from slot `slot` and the state after token i is written to slot i, so
  any prefix can be committed by naming its slot. No copies: the next step
  reads from whichever slot was committed. The conv ring grew from 4 to 16
  columns, because K rejected tokens' inputs would otherwise wrap around and
  overwrite the true ones.
- Attention: the prep kernel runs (heads x M) programs; the flash-decoding
  kernel takes G x M query rows (padded to 32) with a per-row causal limit
  `pos + i`; the reduce kernel one program per (query, head).
- Norms and silu take M rows; partials are `[S, M, N]`.

| verify step, real model | ms | x raw (9.73) | tok/s if all accepted |
|---|---|---|---|
| K=0 | 9.73 | 1.00 | 103 |
| K=1 | 10.91 | 1.12 | 183 |
| K=2 | 11.41 | 1.17 | 263 |
| **K=3** | **11.87** | **1.22** | 337 |
| K=4 | 12.25 | 1.26 | 408 |

Where the K=3 step's extra 2.1 ms goes: rows GEMV +1.3 ms over the M=1
kernel (bf16 dequant + dot per block), GDN +0.6 ms (the sequential M-token
loop with the b|a dot products per token), silu +0.2 (now one launch).
`docs/feasibility.md` assumed 1.1x; 1.22x costs ~10% of the projected
speculative number.

**Exactness.** `tokenrush/spec.py` runs the loop with eager MTP drafts.
Speculative greedy vs raw greedy over 200 tokens: with K=0 identical on all
prompts (the loop logic is right); with K=3, identical only for 22 / 165 / 20
tokens — and the trace showed every accept/commit correct, the logits at the
divergence agreeing across paths, and the first divergent token a 0.125-logit
runner-up. The cause is numerics: the M-row GEMV rounds dequantized weights
to bf16 before a tensor-core dot, the single-row GEMV folds scales in fp32,
and twenty tokens of that drift flip a near-tie. With `Engine(consistent=True)`
(single-token steps on the M-row kernel too) K=3 is **identical 200/200** on
all three prompts. Raw decode on that kernel is 9% slower (92.9 vs 102.3
tok/s), so the default keeps the fast kernel for raw and accepts near-tie
disagreement; the exactness check runs with the flag. A second consistency
bug found on the way: greedy picked `topk(...)[0]` while verify used
`argmax`, and they break exact bf16 ties differently (math diverged on a
'\n\n' vs '\n' tie); greedy now uses `argmax` everywhere.

**Effective throughput, eager drafts** (3.5–4.7 ms of ~150 launches per
chain, the Phase 3 step-4 work):

| | essay | code | math |
|---|---|---|---|
| tokens per verify step | 2.63 | 3.24 | 3.12 |
| ms per step (verify + eager draft) | ~15 | 15.5 | 15.5 |
| **effective tok/s** | **172** | **212** | **204** |
| raw | 102 | 102 | 102 |

Already above every rival's speculative mode on code and math (SGLang +
DSpark 137 / 205, llama.cpp + MTP 128 / 162, ExLlamaV3 142 / 160) and above
all but llama.cpp's 130 on prose, with the draft still eager. In-graph
drafting (step 4) removes ~2 ms per step: ~220 / 275 / 265 projected.

Autotune note: the rows kernel's picks are pinned (`_ROWS_CONFIGS`); a sweep
firing on `lm_head` at M=2..4 mid-generation had cost 134 ms/step on the
first prompt.

## Step 14 — Phase 3 step 4: the draft chain inside the graph, adaptive depth (2026-09-08)

One graph per depth K now does the whole step (`Engine._spec_step`):

- **A. batched MTP pass** over K+1 rows at MTP positions pos-n .. pos-n+K,
  tokens `[d_1..d_n, tok, tok...]` (built with `where` on the device from
  the previous step's drafts, its accepted count n and the new token) paired
  with the previous verify's K+1 target hiddens. Rows <= n rewrite the
  accepted drafts' pairs with true hiddens; row n is the (tok, h_{pos-1})
  pair whose output is the first new draft; rows beyond are garbage and get
  overwritten before anything reads them.
- **B.** K-1 chained single-row MTP calls from the MTP's own hidden.
- **C.** the verify step of step 13.

The MTP block runs through the same fused kernels as the body (`MTPHead.
hidden_rows`); its projections are quantized to int4 at load (0.85 -> 0.21
GB). `lm_head` is applied only to the rows that need an argmax: K reads of
0.68 GB per step, ~1.2 ms, the dominant draft cost.

| K=3 spec step, 13.46 ms GPU | ms |
|---|---|
| body GEMVs (M=4 rows kernel) | 10.27 |
| MTP + lm_head single-row GEMVs | 1.42 |
| GDN (M=4) | 0.95 |
| norms | 0.38 |
| everything else | 0.44 |

**Depth**, 300 greedy tokens per family, int4 MTP, `dynamic=(Kmin, Kmax)`
means K = clamp(n_prev + 2, Kmin, Kmax):

| effective tok/s | essay | code | math |
|---|---|---|---|
| fixed K=3 (200 tokens) | 187 | 242 | 235 |
| fixed K=4 | 174 | 257 | 265 |
| dynamic 2:4 | 180 | 255 | 262 |
| **dynamic 3:4 (default)** | **183** | **254** | **258** |
| raw | 102 | 102 | 102 |
| tokens per step (3:4) | 2.56 | 3.61 | 3.68 |
| ms per step (3:4) | 14.0 | 14.2 | 14.3 |

A first policy of clamp(n_prev + 1, 2, 4) was too reactive (one rejection
drops to K=2): 182 / 244 / 237.

Exactness: with `Engine(consistent=True)` the in-graph chain reproduces raw
greedy 200/200 on all three prompts (step 13's argument; K=3 bf16 head:
182 / 221 / 213 in that mode). `tests/test_spec.py` checks the graphed step
against raw greedy on random weights.

**Against the rivals' speculative modes** (Phase 0 numbers, their best):
SGLang + DSpark 104 / 137 / 205, llama.cpp + MTP 130 / 128 / 162, ExLlamaV3
+ MTP 127 / 142 / 160, vLLM 80 (raw). Token Rush 183 / 254 / 258:
**1.4x / 1.9x / 1.26x the best rival** per family, 1.8x / 2.5x / 2.5x raw
Token Rush. The `CLAUDE.md` prose floor (200–240) is not met yet: essay
stops at 183 because prose accepts 2.56 tokens per step and each step costs
14 ms. What is left on the table, in order: the 1.22x verify (GDN M-loop,
rows GEMV: ~2 ms), the K lm_head reads (~1.2 ms), and a tree instead of a
chain for prose.

## Step 15 — speculation is the default decode (2026-09-08)

`python -m tokenrush.run` now loads the MTP head (int4), captures the raw
graph plus spec graphs for depths 3..4, and generates through
`generate_spec_graph`; `--no-spec` keeps the raw path, `--spec-depth
Kmin:Kmax` sets the range. Temperature > 0 falls back to the raw sampled
graph until speculative sampling exists (next step).

Plumbing that had to change: `Engine.prefill_hidden` prefills in chunks
while keeping every position's post-norm hidden (the MTP head's prompt
input) and applies `lm_head` only to the last row — the earlier one-shot
`forward(all_logits=True)` would have built 15 GB of logits for a 30k prompt;
`prime_spec` runs the MTP prompt pass in chunks too; the loop stops when a
step of up to Kmax+1 tokens would overrun the cache (checked on the real
model at max_len - 6: stops at 8190 of 8192); eos inside an accepted chain
truncates the output at the stop token.

Checked: chat streaming (haiku prompt: 154 tok/s at 2.1 accepted per step —
short, low-acceptance text), the 30k needle with fp8 KV through the spec
path (retrieved, 4.3 accepted per step on the predictable answer), sampled
fallback, and the max_len edge. 36 tests.

## Step 16 — speculative sampling (2026-09-08)

The verify step now runs the sampler on all K+1 rows (`sample()` takes M
rows, one uniform draw each, greedy when temperature is 0): a draft is
accepted only when it equals the draw at its position, and the draw itself
is committed. Each committed token is therefore a sample from the target's
own conditional at the right prefix, so the output distribution is exactly
raw sampling's — drafts change only speed. (Classic rejection sampling
accepts more of the drafts; this scheme is the simplest exact one and was
enough.) A test compares the empirical first-token distribution of the raw
sampled graph and the spec graph over 600 runs on random weights.

| effective tok/s, dynamic 3:4, int4 MTP | essay | code | math |
|---|---|---|---|
| greedy (step 14) | 183 | 254 | 258 |
| **T=0.7, top-p 0.9** | **167** (2.34/step) | **237** (3.39) | **239** (3.41) |
| T=1.0, top-p 1.0 | 164 (2.31) | 252 (3.62) | 255 (3.67) |
| raw, any temperature | 102 | 102 | 102 |

Code and math barely lose acceptance at temperature 1.0 (their next-token
distributions are peaked); prose drops from 2.56 to 2.3 accepted per step.
`run.py` no longer falls back to raw for temperature > 0; a sampled chat
run gets 165 tok/s.

## Step 17 — speculation vs. context length (2026-09-08)

`bench/spec_context.py`: the prompt is the first N tokens of a real text
(WikiText-103 train for prose, ~3 MB of torch's Python sources for code),
200 greedy tokens generated raw and speculatively (dynamic 3:4, int4 MTP),
fp8 KV, 256k cache. Two memory fixes on the way: the MTP prompt pass no
longer computes logits (2 GB per 4096-token chunk), and it now runs chunk by
chunk on each chunk's hiddens instead of keeping every position's hidden
(4 GB at 200k) — `Engine.forward_hidden` + `prime_spec`.

| context | raw tok/s (ms/step) | spec prose tok/s (tokens/step, ms/step) | spec code |
|---|---|---|---|
| 0 | 99.6 (10.0) | 228 (3.30, 14.4) | 250 (3.62, 14.5) |
| 22k | 94.6 (10.6) | 188 (2.83, 15.1) | 201 (3.05, 15.2) |
| 90k | 82.2 (12.2) | 164 (2.90, 17.6) | 251 (4.55, 18.1) |
| **200k** | 68.4 (14.6) | **174** (3.83, 22.0) | **197** (4.35, 22.1) |

Acceptance depends on the text region (WikiText at 200k happened to be
more predictable than at 90k), so the tok/s columns are not monotone; the
step cost is the clean signal. A spec step grows from 14.4 to 22.0 ms over
the range (+7.6 ms) where a raw step grows +4.6 ms: the extra is the MTP
head's own attention, which reads its full 200k cache (0.8 GB in bf16) on
every one of the ~4 draft calls per step. Giving the MTP cache fp8 (or
fewer calls) is the fix, ~2 ms at 200k.

`docs/feasibility.md` projected prose at 200k at 140–180 tok/s: measured
174, on a range that no rival reaches (SGLang + DSpark does not fit next to
a 200k cache on this card; vLLM's best at 200k is 61 raw; llama.cpp 44).

Also re-measured after vectorizing the b|a gate projections in the GDN
kernel: verify K=3 at 1.21x raw (was 1.22x) — neutral; kept for clarity.
The remaining verify overhead sits in the rows GEMV (bf16 dequant + dot per
block) and the sequential M-token recurrence itself.

## Step 18 — fp8 cache for the MTP head (2026-09-08)

`MTPHead(kv_dtype=...)`; `run.py` gives the head the engine's KV dtype. The
head's prompt pass goes through the fp8 prefill path and its draft rows
through the fused fp8 kernels, both already there. Drafts only affect
speed, so the head's cache precision never reaches the output.

| spec, dynamic 3:4 | MTP cache bf16 | MTP cache fp8 |
|---|---|---|
| prose 200k | 174 tok/s, 22.0 ms/step, 3.83/step | **180**, 21.3 ms, 3.83/step |
| code 200k | 197 tok/s, 22.1 ms/step, 4.35/step | **199**, 21.4 ms, 4.26/step |
| prose / code 90k | 164 / 251 | 167 / 255 |
| short context | 228 / 250 | 228 / 250 |

-0.7 ms per step at 200k, acceptance unchanged: the head's cache reads were
a smaller share of the step's growth with context than estimated (~2 ms).
The rest of the spec step's extra growth over raw (+3 ms at 200k after this)
is not yet attributed; a profile of the spec step at 200k is the next
diagnostic if the long-context row needs more. Kept on by default: free.

## Step 19 — rows-GEMM config re-pick (2026-09-08)

The same L2-proof sweep as step 11, on the M-row kernel at M=4 (72 configs
per shape: BLOCK_N 16–128, BLOCK_K 128–512, 4/8 warps, 2–4 stages). The
pinned configs were within 0–1.7 points of the best; three were re-pinned.
Verify cost: K=3 at 11.86 ms = 1.20x raw (was 11.97 / 1.21x), K=4 1.24x.
Effective throughput 179 / 243 / 252, noise against step 14.

What the sweep says: at M=4 the layer shapes top out at 79–84% of the wall
(lm_head 95%) against 87–93% for the single-row kernel — the M-row kernel
has more fixed work per weight block (unpack, bf16 dequantize, transpose
into the tensor-core dot) and configuration cannot buy that back. The
remaining verify overhead (~1.2x) is therefore structural on both counts
(this kernel's per-block work, and the sequential M-token recurrence in the
GDN kernel). Left as is; the draft-side costs are the cheaper target.

## Step 20 — a truncated vocabulary for drafting (2026-09-08)

**The idea.** A draft only has to be *likely*; a wrong one is rejected at no
cost to the output. So the draft chain's argmax can run over a slice of
`lm_head`, and the K reads of 0.68 GB per step shrink with the slice.

**What it sacrifices.** A token outside the slice can never be drafted. The
cost of a miss is not just a lost gain: a verify step whose drafts are all
rejected costs ~13 ms and yields one token, slower than a 9.8 ms raw step.
So the speed depends entirely on how well the slice covers the text being
generated, and a slice that misses a language turns speculation into a
loss for that language.

**First attempt, rejected.** `bench/draft_vocab.py` ordered ids by
frequency over the prose and code corpora with math-ish tokens boosted; the
first 65536 gave 192 / 256 / 267 on essay / code / math (+6–7%). It held
147 of the vocabulary's 55,328 CJK tokens:

| 300 tokens | full vocabulary | 64k English-corpus slice |
|---|---|---|
| Chinese essay | 156 tok/s (2.19/step) | **92** (1.19/step) — below raw |
| Chinese math | 269 (3.92) | 174 (2.29) |

Not a default anyone should ship. The file was removed; the script stays,
marked as corpus-specific.

**What shipped.** The tokenizer's id order is its BPE merge order, i.e. a
frequency ranking over the tokenizer's own multilingual training corpus,
with no corpus of ours involved. Measured with the first N ids:

| N by id order | CJK covered | essay | code | math | zh essay | zh math | ms/step |
|---|---|---|---|---|---|---|---|
| full (248k) | all | 179 | 243 | 252 | 156 | 269 | 14.3–14.6 |
| 65536 | 0 | 187 | 250 | 271 | 92 | 170 | 13.2 |
| 98304 | 2,275 | 188 | 264 | 266 | 126 | 209 | 13.5 |
| **131072 (default)** | 35,024 | **186** | **261** | **263** | **162** | **279** | 13.6–13.9 |

At half the vocabulary no family loses (Chinese gains slightly: the missing
tail is rare tokens the target rarely produces either) and a step is ~0.7 ms
cheaper. `run.py --draft-vocab full|128k|<file>`; `Engine.attach_mtp
(draft_vocab=...)` gathers the int4 rows. Caveat recorded: languages whose
tokens sit late in the merge order (the rarer 20k CJK, other scripts) were
not measured; `--draft-vocab full` is the safe setting for them.

## Step 21 — corpus-specific draft vocabularies, measured against the id order (2026-09-08)

Question asked: for a Chinese-English daily driver (some code, some math),
can a 64k list built from the right corpora keep the 64k slice's larger
saving without the Chinese collapse of step 20?

`bench/draft_vocab.py` now builds three lists from English Wikipedia
(WikiText-103), torch's Python sources, and a Chinese Wikipedia shard
(`wikimedia/wikipedia` 20231101.zh, 20k articles, simplified and
traditional), math-ish tokens boosted, the rest in id order:

| list | corpora | CJK tokens held (of 55,328) |
|---|---|---|
| `en_64k` | English + code | 147 |
| `mix_64k` | Chinese + English + code | 26,413 |
| `mix_96k` | same, first 98,304 | 40,923 |
| id order 128k (default) | none (tokenizer merge rank) | 35,024 |

`bench/draft_vocab_eval.py`, 300 greedy tokens per family, dynamic 3:4,
int4 MTP; the sixth family is a mixed prompt (Chinese question about
Python locks with a code example and a throughput formula):

| tok/s (accepted/step) | essay | code | math | zh-essay | zh-math | mixed |
|---|---|---|---|---|---|---|
| full (no slice) | 178 (2.53) | 249 (3.61) | 253 (3.68) | 156 (2.19) | 269 (3.92) | 214 (3.07) |
| **id order 128k** | 186 (2.53) | **261** (3.61) | 263 (3.64) | 162 (2.17) | 279 (3.87) | 216 (2.94) |
| en_64k | **193** (2.55) | 257 (3.45) | **267** (3.60) | 92 (1.19) | 174 (2.29) | 154 (2.01) |
| mix_64k | 184 (2.43) | 249 (3.33) | 264 (3.55) | **164** (2.14) | **281** (3.77) | 212 (2.81) |
| mix_96k | 185 (2.49) | 253 (3.45) | 262 (3.60) | 162 (2.16) | 279 (3.82) | **227** (3.07) |

**What the numbers say.**

- The 64k budget is too small for two languages plus code: `mix_64k`
  keeps Chinese but pays in code (3.33 vs 3.61 accepted per step) and mixed
  text (2.81 vs 3.07); the 0.35 ms it saves over 128k is eaten by the lost
  acceptance, and it lands on par with the default.
- `mix_96k` is the best on mixed text (+5% over the default) and within
  noise elsewhere. Its acceptance on code and mixed is back at the full
  head's.
- `en_64k` is the fastest on pure English (+4% over the default) and
  unusable for anything with Chinese in it (below raw decode).
- The differences among full, 128k and mix_96k are 0–5%: the draft
  vocabulary is a small lever once the language question is handled.

**Decision.** The engine default stays the id-order 128k: language-neutral,
no corpus, no file, no family loses. For this project's daily use
(Chinese-English, some code and math) `--draft-vocab mix_96k` is the
recommendation, worth about 5% on mixed text. `en_64k` stays available for
English-only deployments and as the documented example of what a
language-blind list costs. All three lists are in `tokenrush/draft_vocab/`
(1 MB), rebuildable with the script; a deployment on other languages should
rebuild from its own corpus and rerun `bench/draft_vocab_eval.py`.

## Step 22 — the spec step at 200k, profiled and tuned (2026-09-09)

`bench/spec_profile.py`: one raw step and one K=3 spec step under the
profiler at 64 and at 200k tokens of context (fp8 KV, fp8 MTP cache, 128k
draft vocabulary).

| GPU ms | raw @64 | raw @200k | spec @64 | spec @200k |
|---|---|---|---|---|
| body GEMVs | 8.9 | 8.9 | 10.1 | 10.2 |
| attention split kernel | 0.04 (16x) | 4.66 (16x) | 0.07 (19x) | 6.61 (19x) |
| everything else | 1.0 | 1.0 | 2.9 | 2.9 |
| **step** | 9.98 | 14.62 | 13.07 | 19.66 |

So the spec step's +6.6 ms from 64 to 200k against raw's +4.6 ms is all in
the attention split kernel: the 16 body launches read the same 6.7 GB of KV
but with M=4 query rows (a 32-row tile) ran slower than with one, and the
MTP head's 3 launches over its own 200k cache add ~0.9 ms. No bug — the
M-row tile was under-occupied.

A sweep of the kernel on a 200k fp8 cache (NSPLIT 32–128, BLOCK_N 32/64,
4/8 warps, 2/3 stages): `num_stages=3` alone takes it from 1390 GB/s (M=1)
and 1147 (M=4) to **1587 and 1558**, i.e. from 82%/67% of the wall to 93%/92%.
One constant changed.

| after | raw @200k | spec prose @200k |
|---|---|---|
| before | 68.4 tok/s, 14.45 ms, 82.6% of wall | 180 tok/s, 21.3 ms/step |
| **now** | **71.7 tok/s, 13.94 ms, 85.6%** | **195 tok/s, 19.7 ms/step** |

Short context is unchanged (100.9 raw, 228 spec prose). The `CLAUDE.md`
target of 92% of the wall at 200k is now 6 points away; the remaining gap
at long context is the same GEMV share as at short context.

## Step 23 — draft trees: simulated, costed, dropped (2026-09-09)

The plan's last lever for prose was a draft *tree* (top-2 branches instead
of one chain) verified in one step. Before building it — siblings share a
position, so the conv ring, the KV writes and the MTP head's own cache all
need scratch-then-commit paths, the GDN kernel needs parent-slot reads, the
attention kernel a tree mask, and the MTP has to draft every node — the
gain and the cost were measured separately.

**Acceptance**, `bench/tree_accept.py`: static trees drafted with the eager
MTP head (top-k at each node), teacher-forced against the target's greedy
continuation, 256 tokens per family; nodes as (parent, rank):

| tree | nodes | essay | code | math |
|---|---|---|---|---|
| chain, depth 3 | 3 | 2.57 | 3.34 | 3.46 |
| chain, depth 4 | 4 | 2.75 | 3.88 | 4.05 |
| 2 roots; chains 3 and 2 | 5 | 2.80 | 3.44 | 3.56 |
| 2 roots, 2 children under the first; depth 3 | 7 | 2.98 | 3.57 | 3.67 |
| **2 roots; chains 4 and 3** | 7 | **3.04** | **4.02** | **4.18** |
| 3 roots; chains 3, 1, 1 | 7 | 2.86 | 3.46 | 3.58 |

The best 7-node tree beats the 4-chain by +10.5% on prose and +3–4% on
code and math. Branching only pays where the first draft is unsure
(prose); the extra nodes mostly duplicate what a longer chain gets.

**Cost**, `bench/verify_cost.py` to M=8 (K=7), measured:

| K (M = K+1) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| ms / verify step | 10.01 | 10.98 | 11.50 | 11.86 | 12.36 | 12.71 | 13.44 | 13.51 |

A 7-node tree verify costs 13.5 ms against 12.4 for the 4-chain (+1.15 ms),
plus 7 MTP draft calls instead of 4 (+1.0 ms), plus the scratch-to-cache
commits and the parent-slot state reads (~+0.5 ms): the step goes from
~13.6 to ~16.3 ms, +20%, for +10% tokens on prose and +4% elsewhere.
**Net: about -8% on prose, -13% on code and math.** Even an unrealistically
free implementation (+1.2 ms) only breaks even on prose. The per-row cost
of verification here (~3% of a step per extra token) is higher than the
per-node acceptance gain a second branch buys with this draft head.

Dropped. The prose number stays at 186–192 short-context, below the 200
floor; what would move it is a cheaper verify row (the M-row GEMM's fixed
per-block work) rather than more rows.

Found on the way: for M >= 6 the attention kernel's 64-row query tile with
three pipeline stages needed 102,400 bytes of shared memory against a
101,376-byte limit; it now uses two stages above 32 rows. (Depths 3:4 never
reached it.)

**DSpark, assessed without building it.** SGLang's measured DSpark chain at
gamma 7 accepts 2.39 / 3.11 / 4.68 per step on essay / code / math
(`docs/baselines.md`); our MTP chain at depth 3–4 accepts 2.55–2.75 /
3.5–3.9 / 3.7–4.05. Per draft, the 1.86B DSpark head is not better than the
0.42B MTP head on prose or code, and each of its draft calls reads 4x the
weights. Only math would gain (4.68 vs 4.05), and by the tree arithmetic
above the extra draft cost eats most of it. Not built.

## Step 24 — research: what is left to gain (2026-09-09)

A survey of what single-stream inference work in 2025–26 has that this
engine does not, checked against our own profiles. Two levers are real, the
rest are not for this stack.

### 1. DFlash2: a block-diffusion draft — the big one

`z-lab/Qwen3.8-27B-DFlash2` (MIT): a 1.92B, 5-layer Qwen3-style draft that
produces a whole block of 7 draft tokens in **one forward**, conditioned on
the target's hidden states after layers 5, 19, 33, 47 and 61 (concatenated,
projected by `fc`, injected into every draft layer's K/V), with mask tokens
as the block's inputs, bidirectional attention inside the block, a sliding
window of 2048 over its own context cache, two grouped dynamic convs per
layer, and a rank-256 candidate selector that chains the top-16 candidates
per position. Phase 0 saw it in vLLM (3.69 accepted per step, but on a
70 ms step) and in llama.cpp (+37% / +37% / +82%).

Measured here (`bench/dflash_accept.py`), teacher-forced against **our int4
target's** greedy continuation, feeding the draft our engine's own hidden
states through the trace hook, 248 positions per family:

| accepted tokens per verify step | essay | code | math |
|---|---|---|---|
| MTP chain, K=4 (step 23) | 2.75 | 3.88 | 4.05 |
| best 7-node MTP tree (step 23) | 3.04 | 4.02 | 4.18 |
| DFlash2, first 3 drafts (K=3) | 2.79 | 3.47 | 3.50 |
| DFlash2, first 4 drafts (K=4) | 3.08 | 4.08 | 4.17 |
| **DFlash2, the full block (K=7)** | **3.57** | **5.37** | **5.76** |
| P(first draft correct) | 0.79 | 0.92 | 0.92 |
| P(all 7 correct) | 0.12 | 0.35 | 0.48 |

The per-draft quality is at least the MTP head's, and it comes 7 at a time
from one forward, which is exactly what this engine's verify step is good
at (a K=7 verify costs 13.5 ms, 1.35x raw, step 23). Draft cost in-graph:
5 layers of ~1 GB at int4 through the M-row kernels (~0.8 ms), `lm_head`
over 7 rows (~0.45 ms), the selector and convs (~0.2 ms): ~1.5 ms, less
than today's 3–4 chained MTP calls.

| projection | essay | code | math |
|---|---|---|---|
| today (MTP chain, dynamic 3:4) | 186 | 261 | 263 |
| DFlash2, K=7 verify (13.5 + 1.5 ms) | **238** | **358** | **384** |
| + a Marlin-class M-row GEMM (verify ~11.5 ms) | ~275 | ~413 | ~443 |

That is every effective-throughput target in `CLAUDE.md` (prose 200–240,
code 230–280, math 300–380) from the first row alone. Cost to build: the
draft's forward through our fused kernels (its attention needs the KV
injection and the in-block bidirectional mask; the convs and selector are
small), capture of the target's five layer outputs inside the spec graph
(free: the residual stream is there), int4 packing of the draft, and the
verify at K=7. Estimate 2–3 days. Its own cache is a 2048-token window, so
long context costs it nothing.

### 2. A Marlin-class M-row int4 GEMM — the second lever

The verify overhead (1.2x at K=3, 1.35x at K=7) is the M-row kernel's fixed
per-block work (step 19). Marlin (IST-DASLab, Apache-2, 822 lines of CUDA)
is built for exactly this regime: dequantization in registers overlapped
with `mma.sync`, near-full bandwidth for M up to 16–64, `sm_80+`
instructions all present on `sm_120`. Vendoring it means a small CUDA
extension built against torch 2.14 / CUDA 13.3, a repack from our packing
to its layout (symmetric int4 with fp16 group scales; our asymmetric
scale+min becomes a zero-point variant or a symmetric re-quantization —
the quality table decides), and fp16 activations (its original operand
type; Qwen activations are bf16, so the range needs checking). If it holds
~88% at M=8 as it does at M=1, a K=7 verify drops from 13.5 to ~11.5 ms:
+15% on every speculative number. 1–2 days if the build cooperates.

### Not worth it here

- **Draft trees** (step 23): +10% acceptance for +20% step cost.
- **DSpark** (step 23): a 1.86B autoregressive draft, no better per draft
  than the MTP head on prose/code; DFlash2 supersedes it.
- **Prompt-lookup / n-gram drafts**: free drafts for text that repeats the
  prompt; our code acceptance is already 3.5–5.4 per step and prose gains
  little. Could be an opportunistic add-on later.
- **NVFP4 kernels / "160 tok/s" claims**: 4.5 bpw reads 6% more bytes than
  our int4 g128, and the public 5090 figures do not say whether MTP was on.
- **EAGLE-3-style heads**: no trained head exists for this model; DFlash2
  is the trained draft that does exist.

Sources: [DFlash paper](https://arxiv.org/abs/2602.06036), [z-lab/dflash](https://github.com/z-lab/dflash),
[vLLM speculators DFlash](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/),
[DFlash & DSpark write-up](https://jianyuh.github.io/llm/inference/speculative%20decoding/2026/06/29/DFlash-DSpark-Diffusion-Speculative-Decoding.html),
[NVIDIA on DFlash](https://developer.nvidia.com/blog/boost-inference-performance-up-to-15x-on-nvidia-blackwell-using-dflash-speculative-decoding/),
[Marlin](https://github.com/IST-DASLab/marlin), [AutoAWQ+Marlin notes](https://www.emergentmind.com/topics/autoawq-marlin),
[prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding),
[EAGLE-3 overview](https://www.spheron.network/blog/eagle-3-speculative-decoding-gpu-cloud/),
[5090 NVFP4 guide](https://runaihome.com/blog/qwen36-27b-nvfp4-blackwell-2x-speed-guide-2026/).

## Next

- **Phase 3**, steps 1–4 done and spec is the default decode: 183 / 254 /
  258 tok/s effective greedy, 167 / 237 / 239 sampled at T=0.7, 180 / 199
  at 200k (195 prose after step 22); 186 / 261 / 263 with the 128k draft
  vocabulary. Trees and DSpark dropped (step 23). Step 24's research found
  two levers worth building: **DFlash2** as the draft (projected 238 / 358
  / 384) and a **Marlin-class M-row GEMM** (+15% on top). Then
  rejection-sampling acceptance for sampled decoding, and Phase 3 closes.
- **Decision 2026-09-08: Phase 2 first.** The quality table and the
  quantization choice (Phase 1b, second half) are deferred to a two-GPU box;
  the full plan, what exists to build on, and what else is owed from 1b are
  in `docs/quality_plan.md`. Until then the engine runs on an uncalibrated
  int4 RTN that is 2–4x worse in KL than it should be.
- Phase 2, in order: fused GDN step (the 4.4 ms of small kernels), attention
  decode over the live length instead of a bucket (with FP8 KV, which 256k
  needs), fused sampling.
- Phase 1b: engine-correctness gate against HF (per-layer first), quality
  table, quantization choice.
