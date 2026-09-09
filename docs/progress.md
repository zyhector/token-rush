# Phase 1 progress log

One entry per step, with the number it produced. Every speed figure here is
a **development number** on whatever instance was rented that day, quoted
against the 1701 GB/s wall recorded in `docs/environment.md`; the citable
numbers come from Phase 4. Prefill is compute-bound and not a project
target; it is logged because it is free to measure and catches regressions.

Instance for all entries so far: vast container 50295164, RTX 5090, driver
610.43.02, CUDA 13.3, torch 2.14.0+cu130, triton 3.8.0, transformers 5.16.1,
fla 0.6.0. 150 GB disk, 60 GB RAM.

## Where things stand (after step 30, 2026-09-09)

| | | |
|---|---|---|
| raw greedy decode, short context | **102 tok/s** on the Triton GEMV (`--backend triton`), 9.9 ms/step, 81% of the 1701 GB/s wall (13.65 GB read/step, ceiling 124.6); 97.6 on the default Marlin kernel, whose one layout serves the speculative step | rivals: llama.cpp 83 (78%), vLLM 80 (88%, ~76% recounted), ExLlamaV3 77, SGLang 63 |
| speculative greedy, DFlash2 draft in-graph (K=7, 128k draft vocab), essay / code / math | **224 / 373 / 373 tok/s** (MTP chain: 213 / 306 / 286); Chinese essay 179 / math 310 / mixed 251 with the MTP chain, which `--draft auto` (the default) picks for CJK prompts | best rival per family: llama.cpp+MTP 130, SGLang+DSpark 137 / 205 |
| speculative sampled, T=0.7 top-p 0.9 | **211 / 391 / 358** (DFlash2), 210 / 292 / 294 (MTP chain) | output distribution identical to raw sampling (exact rejection sampling for deterministic drafts) |
| speculative at 200k context, fp8 KV, prose / code | **211 (MTP) / 238 (DFlash) tok/s** (raw 71.7 = 85.6% of the wall on the Triton GEMV, 69.8 on Marlin) | vLLM 61, SGLang 50, llama.cpp 44 at 200k |
| context | 256k usable (needle at 128k and 256k), 26 GB peak | |
| correctness | bf16 path = HF on 48/48 greedy tokens; spec = raw greedy 200/200 with shared kernels; every fused kernel differential-tested; 76 tests | |
| quantization | **int4 g128 GPTQ + MSE range search** (step 30, `docs/quantization.md`, unchanged packing): KL to bf16 **0.0232** over 82k positions, WikiText-2 PPL 6.365 vs 6.255, top-1 0.942, GSM8K 96.5% vs bf16's 96.0% (met); RTN was 0.0546. **The KL half of the row is not met**: ExLlamaV3 4.00bpw is 0.0128 and what is left is the uniform int4 codebook, not the calibration | rivals: GGUF UD-Q4_K_M 0.0093 at 4.80 bpw, EXL3 0.0128 at 4.10, NVFP4 0.0231 at 5.07, RedHatAI INT4 0.0458 at 4.71 |
| phases | 0 done (frozen), 1a done, **1b done as measurement** (gate, quality table, GPTQ; the bar itself is not met), 2 done, 3 done, 4 not started | |

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
| 25. Phase 3b: DFlash2 draft inside the graph (first version) | 2026-09-09 | + our implementation of the DFlash2 forward (bit-identical to z-lab's reference), int4-packed, graph-capturable with a fixed context-row count and a fixed 2048-key window; the verify body captures the five feature layers; one graph = draft block + K=7 verify. Exact vs raw greedy 300/300 | — | **178 / 306 / 310** (essay / code / math) at 16.8 ms/step; MTP chain was 186 / 261 / 263 |
| 26. Phase 3b: the DFlash step tuned | 2026-09-09 | + GDN at M>=4 with two programs per head and one M-row norm launch (2.4 -> 1.4 ms); the draft's norms and grouped convs as fused kernels; the draft's attention through our windowed split kernel + a block-keys kernel (SDPA with a mask was 7x slower than unmasked). Drafts identical to z-lab's at 3000 tokens of context (window active) | — | **220 / 342 / 353** at 14.8 ms/step (v1: 178 / 306 / 310 at 16.8) |
| 27. Phase 3b: DFlash across families and context; ring cache | 2026-09-09 | + the draft's context cache is a 4096-row ring (its window is 2048), so 256k fits (29.4 GB peak); its conv/selector projections int4 too; measured on six families and at 200k. Chinese prose is the one family where the MTP chain stays better (DFlash2 accepts 1.77/step there) | — | six families: **217 / 355 / 350 / 121 / 289 / 212** (essay / code / math / zh-essay / zh-math / mixed); at 200k: prose 179, code 229 (MTP: 195 / 199) |
| 28. Phase 3c: Marlin-class int4 GEMM | 2026-09-09 | + Marlin (Apache-2) ported to bf16, our asymmetric g128 format, fp32 reduction, a lock-free partial mode for the split-K shapes, no L2 cache hints (illegal on sm_120); one kernel for M <= 16 in Marlin's weight layout, the default. Verify K=7 = **1.13x** a raw step (was 1.35x), raw step 4% dearer (the layout has no better M=1 kernel than the mma one) | — | **224 / 373 / 373** DFlash, 213 / 306 / 286 MTP (essay / code / math); zh-essay / zh-math / mixed 179 / 310 / 251 (MTP); 200k: prose 211 (MTP) / code 238 (DFlash); raw 97.6 (102 on `--backend triton`) |
| 29. Phase 3 closed: sampled decoding measured, per-content draft choice | 2026-09-09 | + the verify step's accept-if-equal-to-the-draw rule shown to *be* rejection sampling for deterministic drafts (no change needed); `--draft auto` (default) keeps both drafts resident and picks the MTP chain for prompts >= 20% CJK, DFlash2 otherwise; two shared-buffer bugs fixed on the way (a graph holds tensors by address) | — | sampled T=0.7 top-p 0.9: **211 / 391 / 358** DFlash2, 210 / 292 / 294 MTP (essay / code / math); zh-essay 171 (MTP); auto: English essay 232, Chinese essay 182 |
| 30. Phase 1b, second half: the quality table and GPTQ + MSE | 2026-09-09 | + the yardstick (KL to bf16 over 82k positions, PPL, top-1, a measured 5e-4 noise floor) for ours and four rivals; our own GPTQ with an MSE range search into the unchanged packing: **KL 0.0546 -> 0.0232** (EXL3, the bar, 0.0128; GGUF 0.0093; NVFP4 0.0231; RedHatAI 0.0458); GSM8K 96.5% vs bf16's 96.0%; speed bit-identical; the recipe and the corpora in git | 1500 | **97.4** (10.26 ms/step) |

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

## Step 25 — Phase 3b: the DFlash2 draft inside the graph, first version (2026-09-09)

`tokenrush/dflash.py` implements the draft on our ops: `fc` + plain-gain norm
of the five concatenated target residuals, five Qwen3-style layers whose
attention takes the context rows' K/V (from the features, cached in the
draft's own K/V) plus the block rows (bidirectional among themselves,
windowed at 2048), two grouped dynamic causal convs per layer, a final norm,
and the rank-256 candidate selector chaining the top-16 per position.
Norms are Qwen3's plain gain (weight * x), not Qwen3.5's (1 + weight).

Checked against z-lab's reference (`bench/dflash_check.py`): the block hidden
states are **bit-identical** on two consecutive steps with random inputs and
the draft tokens match once the selector's arithmetic is done in bf16 like
the reference's (a near-tie had flipped in fp32); the essay acceptance through
our forward is 3.60 vs the reference measurement's 3.57. With the draft's
projections int4-packed (1.92B -> ~0.5 GB): 3.64.

Graph integration (`Engine.attach_dflash`, `_spec_step_dflash`,
`capture_spec_dflash`; `spec.generate_dflash`, `prime_dflash`):

- `_body` captures the residual after layers 5, 19, 33, 47, 61 for the
  verify's 8 rows into `feat_buf` (a copy per layer, free). After a commit
  with n accepted, rows 0..n are the next context rows.
- `DFlashDraft.forward_static`: always 8 context rows (the invalid ones land
  beyond the block start and are overwritten before anything reads them) and
  a fixed window of 2048 keys ending at the block start, masked on the
  device; the draft's cache end is `pos_t - n_ctx`, all device arithmetic.
- The selector runs on the 128k draft-vocabulary slice of `lm_head`.
- Priming: the chunked prefill keeps the five layers' residuals per chunk and
  `DFlashDraft.prime` writes their K/V (no block work).
- One graph per step: draft block from the previous verify's features, then
  the K=7 verify and commit. `Engine(max_spec=7)`.

| 300 greedy tokens | essay | code | math |
|---|---|---|---|
| MTP chain, dynamic 3:4, 128k vocab (step 21) | 186 (2.53/step, 13.6 ms) | 261 (3.61) | 263 (3.64) |
| **DFlash2, K=7** | **178** (2.98/step, 16.8 ms) | **306** (5.14) | **310** (5.21) |
| identical to raw greedy (consistent kernels) | 300/300 | 300/300 | 300/300 |

Acceptance in the loop is below the teacher-forced numbers (2.98 vs 3.57 on
prose) because a real loop over-samples the positions right after a
rejection, which are the uncertain ones; the MTP chain showed the same gap.
Code and math jump by 17–18%. Prose does not, because the step costs
16.8 ms against the 15 projected: the profile (1347 launches, 16.74 ms GPU):

| | ms |
|---|---|
| body rows GEMM, M=8 | ~10.3 |
| **GDN M=8 step kernel** | **2.37** (48 launches, 49 us each: 8 sequential tokens and 8 x 64 KB snapshot writes per program) |
| draft rows GEMM (int4, 42 launches) | ~1.3 |
| draft attention (SDPA over 2056 keys, 5 launches) | 0.68 |
| draft small ops: convs, plain norms, selector, index ops (~1000 launches) | ~1.3 |
| fused norms, silu, body attention | ~0.6 |

Next inside 3b, in payoff order: the GDN kernel at M=8 (finer programs so
the snapshot writes parallelize; or fewer/cheaper snapshots), the draft's
attention through our split kernel with a window, and fusing the draft's
convs, norms and selector. Target: ~13.5 ms per step -> ~220 / 375 / 380;
then 3c (Marlin-class GEMM) on top.

## Step 26 — the DFlash step tuned (2026-09-09)

Three changes from the profile of step 25, each verified separately:

- **GDN kernel at M >= 4**: two programs per V head (`BV=64`) with the gated
  norm as one extra launch for all M rows. In-graph microbench at M=8:
  33 -> 27 us per layer; in the step 2.37 -> 1.44 ms. (M=1 keeps one program
  per head: 6.1 us, the norm folded in.) The cost at M=8 is the 8 x 64 KB
  snapshot writes per program more than the recurrence itself.
- **The draft's norms and convs as kernels**: `add_rmsnorm(plain=True)` (the
  Qwen3 gain, weight * x) replaces three torch ops per norm; `grouped_conv`
  does a dynamic causal conv over the block rows in one launch, rounding in
  the reference's order (bf16 add of the static term, fp32 addcmul of the
  dynamic term, rounded once). ~350 launches fewer per step.
- **The draft's attention through our kernels**: SDPA with any mask ran at
  141 us per call against 21 us unmasked (7x; the masked path is a slow
  kernel), 5 calls per step. Now the context keys go through the split kernel
  with two new modes — `NEWROWS=0` (the queries are not cache rows) and a
  per-row `WINDOW` lower bound — and the 8 block keys through a small
  `_attn_block_kernel` that writes one more partial; the reduce merges 15 + 1
  partials, ungated. 0.68 -> ~0.1 ms.

A finding on the way: my eager `DFlashDraft.forward` and the first unit
test both truncated the context keys at the *last* block row's window bound
for every row; z-lab's mask gives each row its own bound. Beyond 2048 tokens
of context that excluded up to seven keys per row — the kernel had it right
and the references were wrong (the split kernel alone matches an fp32
reference with z-lab's semantics at 0.2% per row). Fixed; then at 3000
tokens of context, z-lab's reference, our eager path and our static path
produce identical drafts (42/42).

| 300 greedy tokens | essay | code | math |
|---|---|---|---|
| MTP chain, dynamic 3:4 (step 21) | 186 (2.53/step, 13.6 ms) | 261 (3.61) | 263 (3.64) |
| DFlash2 v1 (step 25) | 178 (2.98, 16.8 ms) | 306 (5.14) | 310 (5.21) |
| **DFlash2 v3** | **220** (3.25, 14.8 ms) | **342** (5.05) | **353** (5.21) |

`run.py --draft dflash` is the default now (`--draft mtp` keeps the chain;
the MTP path stays for machines without the DFlash2 checkpoint). Chinese
chat through it: 243 tok/s at 3.59 accepted per step; sampled English chat
at T=0.8: 205.

The step's profile now (14.72 ms GPU, 1113 launches): body rows GEMM
~10.3, draft rows GEMM ~1.3, GDN 1.44, norms 0.38, the draft's bf16 conv
projections through cuBLAS 0.29 (11 launches at 26 us each: the small-M
cuBLAS path is slow; int4 rows would be ~0.1), attention 0.15, the rest
~0.5. The prose floor of 200 is crossed; code and math are inside their
target bands (230–280 and 300–380) and above them respectively.

## Step 27 — DFlash across families and context; the draft's ring cache (2026-09-09)

- **Ring cache.** The draft attends within a 2048-position window, so its
  context K/V is now a ring of 4096 rows (position mod 4096; the split
  kernel got a `RING` mode) instead of a full-length cache: 84 MB instead of
  5.4 GB at 256k, which had run out of memory. A test checks the window at
  position 9000 through the ring. At 256k with fp8 KV the whole thing peaks
  at 29.4 GB.
- The draft's conv-kernel and selector projections go through int4 too
  (cuBLAS's small-M bf16 path cost 26 us per call).
- Tests: 44, including a random-weight graph test of the DFlash step.

**Six families, short context** (300 greedy tokens, 128k draft vocabulary):

| tok/s (accepted/step) | essay | code | math | zh-essay | zh-math | mixed |
|---|---|---|---|---|---|---|
| MTP chain, dynamic 3:4 (step 21) | 186 (2.53) | 261 (3.61) | 263 (3.64) | **162** (2.17) | 279 (3.87) | 216 (2.94) |
| **DFlash2, K=7** | **217** (3.18) | **355** (5.19) | **350** (5.12) | 121 (1.77) | **289** (4.23) | 212 (3.10) |

DFlash2 wins everywhere except Chinese prose, where its acceptance drops
to 1.77 per step and the MTP chain's 162 tok/s stands: the draft was
evidently trained on little Chinese (Chinese math, being mostly symbols and
numbers, is fine). Mixed Chinese-English text ties. For a Chinese-prose
daily driver `--draft mtp` is the setting; a per-content switch between the
two drafts (both graphs resident) is a possible later refinement.

**Long context, real text, fp8 KV** (200 tokens):

| | prose @64 | prose @200k | code @64 | code @200k |
|---|---|---|---|---|
| MTP chain (steps 17–22) | 228 | 195 (3.83/step, 19.7 ms) | 250 | ~200 |
| **DFlash2** | **249** (3.65) | 179 (4.30/step, 23.9 ms) | **299** (4.39) | **229** (5.49, 24.0 ms) |

At 200k DFlash accepts more per step but the step is 4 ms dearer than the
MTP chain's: the body's attention over the 6.7 GB cache with 8 query rows
is a 64-row tile, which runs at ~700 GB/s against ~1550 for the 32-row tile
(register pressure; block size, stages and warps do not help — measured).
Two passes of 4 rows would read the cache twice at full speed, a wash; a
kernel that splits the head dimension is the real fix. Recorded as an item;
prose at 200k therefore stays with the MTP number (195) as the best
measured, code at 200k improves to 229.

## Step 28 — Phase 3c: a Marlin-class int4 GEMM for the M-row step (2026-09-09)

**What.** Steps 19 and 23 left the verify step's overhead in one place: the
Triton M-row kernel dequantizes a block to bf16 and dots it, and that
per-block work put it at 79–84% of the wall at M = 4 against the single-row
GEMV's 92%; K = 7 cost 1.35x a raw step. Marlin (IST-DASLab, Apache-2,
`/workspace/marlin`) is the kernel built for exactly this: 16-row tiles on
`mma.sync`, a 4-stage `cp.async` pipeline into shared memory, the int4 -> fp16
conversion done with two `lop3` per pair, the weight pre-permuted into the
fragment order so every thread's 32-bit word is its own B fragment.

**Port** (`tokenrush/csrc/marlin_bf16.cu`, `tokenrush/marlin.py`, built on
first import with `torch.utils.cpp_extension.load`, ~10 s):

- **bf16 activations** (the `.bf16` mma; the magic-number dequant becomes
  OR-into-the-mantissa-of-128 then subtract 128, which yields the code
  exactly);
- **our asymmetric format**: w = q * s + m as one `hfma2` with a second
  fragment of per-group minimums fetched beside the scales, so the packed
  checkpoint (`quantize.py`) is unchanged and `pack`/`unpack` convert to and
  from Marlin's layout at load (5 s for the model; `unpack` also serves
  prefill's dequant path and the draft head's row selection);
- **fp32 cross-block reduction** through a workspace instead of Marlin's
  bf16 round trip through the output (K = 17408 costs bits otherwise);
- **no L2 cache hint**: stock Marlin dies with `cudaErrorIllegalInstruction`
  on sm_120 — the `createpolicy … L2::evict_first` + `cp.async …
  L2::cache_hint` pair is not executable there (`docs/environment.md`);
  a plain `cp.async.cg` costs nothing measurable;
- **a lock-free partial mode** (below); M <= 16 only (one row tile), the
  two grouped configs Marlin uses for small M, exact shared-memory size
  (50 KB instead of Marlin's 96 KB request).

**Measured, in-graph, weights cycled through 400 MB** (GB/s; "ours" is the
Triton GEMV at M = 1 and the M-row kernel at M = 8, split-K 4 on the narrow
shapes):

| shape | ours M=1 | ours M=8 | Marlin M=1 | Marlin M=8 |
|---|---|---|---|---|
| in_qkvz 16384x5120 | 1454 | 1350 | 1537 | 1538 |
| qkv 14336x5120 | 1478 | 1257 | 1376 | 1377 |
| gate_up 34816x5120 | 1601 | 1425 | 1569 | 1564 |
| out 5120x6144 | 1364 | 1282 | 1325 (chained: 1070) | 1318 (1093) |
| down 5120x17408 | 1602 | 1414 | 1580 (1457) | 1578 (1462) |
| lm_head 248320x5120 | 1687 | 1610 | 1591 | 1580 |

Marlin is flat in M — M = 16 costs what M = 1 costs — where the Triton
M-row kernel had lost 15–20% by M = 8. Its M = 1 is 2–7% behind the
folding GEMV on most shapes (a 16-row tile's ramp and tail per block, one
block per SM); a Triton GEMV reading the Marlin layout was written and
measured too (the layout is regular enough: per 64-column chunk, word 4i+j
is thread i's fragment of n-tile j) and came out 10–25% *slower* than the
mma kernel on every shape but lm_head, so it was dropped — one layout, one
kernel. Tile configs (128x128 vs 64x256), pipeline depth (2, 6, 8) and
more blocks per SM (340–680) were all measured: Marlin's defaults win.

**The narrow shapes and the chain.** On N = 5120 Marlin's stripe partition
puts 4–6 blocks on each column slice and reduces them *serially* through a
lock chain; on a 19 MB matrix that chain is a quarter of the kernel
(1070 GB/s). Those two projections already feed `add_rmsnorm`, which sums
split-K partials, so the port has a partial mode: every block writes its own
fp32 slot `[S, M, N]` (the last block of a slice zero-fills the slots its
column does not use; `partial_slots()` replays the partition on the host to
size S = 5–6), no locks, and the consumer sums. 1070 -> 1325 and 1457 ->
1580 GB/s. (Partial mode is also 2% faster than the locked mode on the wide
shapes; their consumers would need to sum slots — an item.)

**In the model.**

| | Triton kernels (step 27) | Marlin |
|---|---|---|
| raw step | 9.88 ms, 101.2 tok/s (81.3%) | 10.25 ms, 97.6 tok/s (78.3%) |
| raw step, kernel totals | wide 6.12 + narrow 2.74 + norm 0.34 ms | 6.29 + 2.86 + 0.41 ms |
| verify K=3 | 1.20x | **1.04x** |
| verify K=7 | 1.35x (13.2 ms) | **1.13x** (11.6 ms) |
| DFlash step | 14.7 ms | **13.7 ms** |
| MTP step (3:4) | ~13.5 ms | **12.4 ms** |

Six families, 300 greedy tokens, 128k draft vocabulary (`bench/families.py`):

| tok/s (accepted/step) | essay | code | math | zh-essay | zh-math | mixed |
|---|---|---|---|---|---|---|
| DFlash2, step 27 | 217 | 355 | 350 | 121 | 289 | 212 |
| **DFlash2, Marlin** | **224** (3.08) | **373** (5.12) | **373** (5.12) | 129 (1.78) | 301 (4.14) | 216 (2.96) |
| MTP chain, step 27 | 186 | 261 | 263 | 162 | 279 | 216 |
| **MTP chain, Marlin** | 213 (2.65) | 306 (3.86) | 286 (3.60) | **179** (2.21) | **310** (3.92) | **251** (3.14) |

The MTP chain gains more (+15%) than DFlash (+4–5%): its step is M-row
GEMM for the verify *and* for the chain's draft passes, DFlash's step has
the 1.9B draft forward and the eight-row attention in it. Consequence: the
MTP chain now wins every family with Chinese in it, DFlash the three
English ones; `--draft mtp` is the setting for a Chinese-heavy daily driver,
and a per-content switch with both graphs resident is the obvious next
refinement.

Long context, real text, fp8 KV, 200 tokens (`bench/spec_context.py --draft`):

| | prose @64 | prose @200k | code @64 | code @200k |
|---|---|---|---|---|
| MTP chain, step 27 | 228 | 195 | 250 | ~200 |
| **MTP chain, Marlin** | 254 | **211** (3.74/step, 17.7 ms) | 288 | 240 (4.26, 17.8 ms) |
| DFlash2, step 27 | 249 | 179 | 299 | 229 |
| **DFlash2, Marlin** | **281** | 187 (4.30, 23.0 ms) | 285 | **238** (5.49, 23.0 ms) |

**Cost.** Raw decode on the Marlin layout is 4% slower than on the Triton
GEMV (97.6 vs 101–102 tok/s): the same kernel serves M = 1, and its per-block
ramp shows there. Both layouts cannot be resident (13.5 GB each), so the
default is the layout that makes the default decode — speculative — fastest;
`--backend triton` keeps the old kernels for a raw-decode measurement.
Prefill is unchanged (dequant + cuBLAS, now via `unpack`: 1500 tok/s).

**Correctness.** 32 new tests (`tests/test_marlin.py`): pack/unpack
round-trip exact; the product against the dequantized reference at every
layer shape for M = 1, 3, 8, 16, with error no worse than the Triton
kernel's; row results bit-identical whatever M is (M = 1 and M = 8 share
tiles, partition and reduction order, so raw and speculative steps agree
without the `consistent` flag); partial slots sum to the product with every
slot written; graph replay equals eager. The engine suite (random-weight
DFlash and MTP spec graphs vs raw greedy) runs on the new backend: 76
tests.

**State of the plan.** 3c closes the Phase 3 kernel work: the verify step
is 1.13x a raw step at 8 positions (SGLang's is 1.45x). Left in Phase 3:
rejection-sampling acceptance for sampled decoding, the per-content draft
switch, and the two long-context items (the 64-row attention tile, wide-shape
partial mode).

## Step 29 — Phase 3 closed: sampled decoding, per-content draft choice (2026-09-09)

**Rejection sampling was already there.** The item "rejection-sampling
acceptance for sampled decoding" turned out to be a no-op. The verify step
draws y ~ p (the target's distribution after temperature / top-k / top-p) at
each position, accepts the draft x when y == x and otherwise commits y. For
a *deterministic* draft — and both of ours are: the MTP chain drafts its
argmax, DFlash2 its per-position argmax — the draft distribution q is a
point mass at x, so textbook speculative sampling accepts x with probability
min(1, p(x)/q(x)) = p(x) and on rejection samples from norm(max(0, p − q)) =
p(·|≠x). The draw rule does exactly that: P(y = x) = p(x), and y | y ≠ x is
p(·|≠x). Same acceptance, same output distribution, one draw per position.
What *would* change acceptance is a stochastic draft (sample the chain
instead of taking its argmax and keep its distribution for the residual):
expected acceptance becomes Σ min(p, q) instead of p(argmax q), which wins
when p is flat (creative prose at high temperature) and loses nothing else.
Not built; the sampled numbers below are within a few percent of greedy, so
the ceiling is small at T=0.7.

Measured on the Marlin stack (T=0.7, top-p 0.9, `bench/families.py --temperature 0.7 --top-p 0.9`):

| tok/s (accepted/step) | essay | code | math | zh-essay | zh-math | mixed |
|---|---|---|---|---|---|---|
| DFlash2 | 211 (2.89) | 391 (5.37) | 358 (4.92) | 129 (1.78) | 305 (4.19) | 216 (2.97) |
| MTP chain | 210 (2.60) | 292 (3.68) | 294 (3.70) | 171 (2.10) | 299 (3.76) | 223 (2.78) |
| step 16 (MTP, before 3b/3c) | 167 | 237 | 239 | — | — | — |

**Per-content draft, `--draft auto` (the default).** Both drafts are
resident (DFlash2 int4 1.0 GB + the MTP head 0.2 GB + both graphs; 21.8 GB
allocated at a 32k bf16 context, and at 256k fp8 the MTP head's own cache
adds 0.5 GB to the 29.4 GB peak, still under 31.4). The choice is made per
prompt from its script: the MTP chain when >= 20% of the non-blank characters
are CJK (`spec.pick_draft`), DFlash2 otherwise — the cheapest proxy for the
one thing that decides it, DFlash2's 1.8 accepted per step on Chinese prose
against the chain's 2.2 at a cheaper step. Chinese math is a tie either way,
mixed text goes to the chain (251 vs 216), which the heuristic also picks.
Measured through `run.py --chat`: the Roman-empire essay in English 232
tok/s (DFlash2, 3.19/step), in Chinese 182 tok/s (MTP, 2.25/step); before,
the Chinese prompt ran DFlash2 at 129.

Two bugs when both drafts attach to one engine, both of the same kind: a
captured graph holds its tensors by address, so an attribute that a second
`attach_*` re-allocated (the shared `drafts` buffer, and the 128k-row draft
head, whose old copy was freed under the DFlash graph and read back as
garbage token ids -> a device-side assert) has to be allocated once and
shared. `docs/traps.md`.

**Phase 3 closes here.** Its output, the effective-throughput headline, on
this instance: 224 / 373 / 373 greedy and 211 / 391 / 358 sampled on prose /
code / math, 179 / 310 / 251 on the Chinese families, 211 / 238 at 200k, with
the verify step at 1.13x a raw step. Left as optional items, none blocking:
stochastic drafts with residual sampling; the 64-row attention tile at long
context (M=8 body attention at ~700 GB/s over a 6.7 GB cache); partial-mode
Marlin on the wide shapes (+2%); a Marlin-layout kernel that matches the
Triton GEMV at M=1 (would recover raw decode's 4%).

## Step 30 — Phase 1b, second half: the quantization quality table (2026-09-09)

The yardstick owed since step 5, measured on the two-GPU box (two RTX 5090s,
vast instance of 2026-09-09; torch 2.14.0+cu130, triton 3.8.0, transformers
5.17.0, fla 0.6.0). Part 1 of `docs/quality_plan.md`: every quantization of
Qwen3.8-27B that competes with ours, through the same bf16 forward, against
the same bf16 logits. Nothing about the engine or its weights changed in this
step; it produces numbers.

**Method.** `bench/quality_corpus.py` cuts three corpora into independent
4096-token chunks: WikiText-2 test (raw, the first 65,536 of its 297k tokens,
16 chunks), code (torch's `nn/modules/*.py`, 2 chunks), math (GSM8K *train*
questions with their worked solutions, 2 chunks; the accuracy task uses the
disjoint test split). `bench/quality_logits.py` runs the bf16 checkpoint
through the engine's eager prefill path (fla chunk kernel + SDPA — the path
the engine-correctness gate verified against HF in step 5), one layer built
from the checkpoint at a time, and keeps the full-vocabulary bf16 logits of
every position (81,920 positions, 41 GB). Each candidate is then a tensor
source (`bench/quality_sources.py`) that presents its quantized matrices
dequantized to bf16 and everything else from the HF checkpoint, run through
the identical forward: KL(bf16 ‖ candidate) over the full vocabulary in fp32
per position, top-1 agreement, and the next-token NLL for perplexity.

The noise floor of the yardstick — HF transformers' own bf16 forward
(`bench/quality_hf.py --logits-check`, fla blocked, two cards) against our
bf16 reference on two chunks per corpus: KL 2.5e-4 to 6.2e-4 mean, p99
3e-3 to 8e-3, top-1 agreement 0.988–0.996, perplexity within ±0.005. Every
number below is at least 15x above that floor.

**What each rival's format needed.** llama.cpp's converter reorders the 48
GDN value heads (its head j is HF's head 3·(j mod 16) + j div 16; q/k keep
their order) — found by matching rows against bf16, since the first attempt
gave a relative error of 1.0–1.4 on exactly the four GDN projections; after
the un-permutation every tensor is within its type's expected error
(Q4_K 0.077, Q5_K 0.039, Q6_K 0.020, Q8_0 0.006 relative). The GGUF also
quantizes the embedding (Q4_K) and the output head (Q6_K), and UD-Q4_K_M is
mixed per tensor (131 Q5_K, 117 IQ4_XS, 104 Q4_K, 29 Q6_K, 7 Q3_K, 7 IQ4_NL,
4 IQ3_S, 106 Q8_0), so its 4.80 bits/weight is an average. NVFP4
(compressed-tensors `nvfp4-pack-quantized`): e2m1 codes × e4m3 block-16
scale ÷ fp32 global scale; its `lm_head` is bf16 and the QAT left the
embedding, norms, conv and A/dt exactly at the original values (relative
error 0.000). ExLlamaV3's trellis format is reconstructed by its own code
(`bench/exl3_export.py`, in the exl3 venv: `get_weight_tensor`, the
`mul1` codebook, both Hadamards and the sign vectors) into bf16 shards, 56 s
for the model; 4.005 bits/weight on the body, a 6-bit head, 4.10 overall.

**The table.** 81,920 positions pooled; bits/weight over the 25.60B weights
the engine streams per token (the 401 matrices we quantize; the embedding
is read one row per token and does not count).

| candidate | bits/weight | KL mean | KL p99 | top-1 | WikiText-2 PPL (Δ vs bf16 6.255) | KL wiki / code / math |
|---|---|---|---|---|---|---|
| bf16 | 16 | — | — | — | 6.255 | — |
| **ours, int4 g128 RTN, no calibration** | **4.25** | **0.0546** | 0.549 | 0.905 | 6.495 (+0.240) | 0.054 / 0.034 / 0.084 |
| ExLlamaV3 4.00 bpw, 6-bit head (the bar) | 4.10 | 0.0128 | 0.158 | 0.960 | 6.274 (+0.019) | 0.011 / 0.011 / 0.028 |
| llama.cpp UD-Q4_K_M (imatrix) | 4.80 | 0.0093 | 0.109 | 0.966 | 6.270 (+0.015) | 0.007 / 0.008 / 0.025 |
| NVFP4, QUASAR-QAT (bf16 head) | 5.07 (body 4.5) | 0.0231 | 0.273 | 0.943 | 6.369 (+0.114) | 0.022 / 0.016 / 0.041 |
| RedHatAI INT4 (AWQ+GPTQ, bf16 head) | 4.71 (body 4.125) | 0.0458 | 0.597 | 0.924 | 6.460 (+0.205) | 0.045 / 0.036 / 0.062 |
| **ours, int4 g128 GPTQ** (Part 2, first version) | **4.25** | **0.0265** | 0.320 | 0.937 | 6.353 (+0.098) | 0.025 / 0.021 / 0.047 |
| noise floor (HF bf16 vs ours) | 16 | 0.0005 | 0.005 | 0.992 | ±0.005 | — |

**Speed did not move**, which is the whole reason the format was held fixed.
Raw decode, one card, nothing else on the box: the adopted MSE checkpoint
**97.4 tok/s** at 10.26 ms/step, against 97.6 recorded on the previous
instance — the packing, the Marlin kernel and the graphs are the same, so the
step reads the same bytes. (RTN and plain GPTQ were measured at 95.7 each
while the other card was running a quantization; that pair is a valid
comparison with each other, identical to the hundredth of a millisecond, but
not with the 97.4 above.) Six families, 300 tokens, DFlash2 / MTP chain:

| checkpoint | essay | code | math | zh-essay | zh-math | mixed | ms/step |
|---|---|---|---|---|---|---|---|
| GPTQ + MSE, DFlash2 | 227 | 354 | 376 | 124 | 294 | 219 | 13.8–13.9 |
| GPTQ + MSE, MTP chain | 204 | 255 | 282 | 160 | 278 | 225 | 12.4–12.7 |
| GPTQ, DFlash2 | 229 | 366 | 368 | 120 | 276 | 226 | 14.1 |
| RTN, DFlash2 | 218 | 368 | 376 | 117 | 291 | 218 | 14.1 |

The spread across checkpoints is draft acceptance on different text, not
step cost. All 76 tests pass unchanged; one test's HF rotary call needed the
(3, bs, T) position ids transformers 5.17 expects.

**GSM8K**, 200 test problems, greedy, chat template with thinking off, 1024
new tokens, the answer read from the last `\boxed{}`: the MSE checkpoint
**193/200 = 96.5%** through the engine (plain GPTQ also 193/200), against the
bf16 reference through HF transformers (both cards, 33 min): **192/200 =
96.0%**. Paired per problem: bf16 right where we are wrong 1, we are right
where bf16 is wrong 2, both wrong 6 — a net difference of one problem,
inside the ±3.5-point noise band of a 200-problem run. **The GSM8K half of
the quality row is met**, against bf16, which is a stronger anchor than the
llama.cpp Q4_K_M the row names (that quant is itself 0.0093 from bf16). A
512-token limit had truncated a third of the answers on the first tries and
read as 65% accuracy — the limit, not the quant.

**Reading it.**

- **The quality row in `CLAUDE.md` is not met, by a wide margin.** The
  target is mean KL no worse than ExLlamaV3's 4.00 bpw: 0.0128. Ours is
  0.0546, **4.3x**, at more bits (4.25 vs 4.10). Against the GGUF it is 5.9x
  at 0.55 fewer bits. Perplexity says the same: +0.24 on WikiText-2 where
  the calibrated 4-bit formats lose +0.02. Step 5's one-prompt estimate
  (0.06, "2–4x a calibrated quant") was right in kind and slightly kind in
  degree.
- **Calibration is the whole gap.** The three rivals are all calibrated
  (imatrix, EXL3's 250×2048 calibration rows, QAT); ours rounds to nearest
  with per-group min/max. Nothing in the table separates 4.0 from 4.8 bits
  the way calibration separates 0.05 from 0.01.
- **Math is the hardest corpus for every quant** (KL 2–3x the prose value
  for all four), and prose the easiest by p99; code has the lowest median
  (most positions near-deterministic) and a heavy tail.
- **NVFP4 QAT at 4.5 bits on the body is 2x worse than the calibrated int4
  formats**, with the head in bf16 — the format, not the training, is what
  its rivals' vLLM/SGLang numbers in `docs/baselines.md` run on.
- **Where our RTN loss sits** (`--keep-bf16`, the candidate with a tensor
  set taken from bf16 instead): body int4 + head bf16 gives KL 0.0469; body
  bf16 + head int4 gives 0.0075; the full quant 0.0546 — the two parts add.
  The body is 86% of the loss, so a higher-precision head alone cannot reach
  the bar; but the RTN head by itself (0.0075) already spends 60% of the
  EXL3 budget, so once the body is calibrated the head has to be too.

**The Hub's int4 g128 checkpoints, and why they do not help.** A search of
the Hub (1,993 repos matching Qwen3.8; 516 tagged as quantizations of the
27B: 211 MLX, 124 GGUF, 79 compressed-tensors, 45 FP8, 30 GPTQ, 23 AWQ, 20
EXL3, 12 AutoRound) found no official int4 and ~30 community int4 g128
checkpoints, all symmetric, all keeping `lm_head` in bf16 (400 of our 401
matrices). Their format is a re-pack away from ours — the codes are the
same nibbles, `mn = -8·scale` is exact in bf16, no column permutation
(`tokenrush/convert.py` does it; `bench/quality_sources.py` reads the
compressed-tensors layout). The strongest candidate by evidence,
`RedHatAI/Qwen3.8-27B-INT4` (AWQ smoothing + GPTQ, 512 calibration samples,
published bf16-relative evals at 99–101% recovery, 182k downloads), measures
**0.0458** on this yardstick with its bf16 head — the level of our
uncalibrated RTN body (0.0469). Reading it right took one care: its AWQ
smoothing folds per-channel scales into the layer norms (Qwen's norms are
`1 + w`) and the small projections, so every non-quantized tensor must come
from that checkpoint too; undoing the smoothing puts its unquantized
`in_proj_a` within 0.3% of the original (bf16 rounding) and its quantized
matrices at 0.14–0.16 relative error, the GPTQ signature (weight error above
RTN's 0.10, output error below). On layer 0's real input its `in_proj_qkv`
is only a little better than RTN (output error 0.024 vs 0.032; our GPTQ
0.018) and its `gate_proj` no better (0.092 vs 0.091; ours 0.052).
Task accuracies at 99–101% recovery do not see a KL of 0.046; that is why
the project measures KL.

**Part 2, first version: GPTQ into the same packing** (`tokenrush/gptq.py`,
~200 lines, no tool). Plain GPTQ: per linear, the Hessian of its input over
256 calibration sequences × 2048 tokens (`bench/quality_calib.py`: 160
WikiText-103 train, 64 torch sources outside `nn/modules`, 32 GSM8K train
rows ≥ 2000 — disjoint from the table's corpora), columns quantized in
blocks of 128 with the OBS error update through H⁻¹ (damp 1%), group
parameters from the error-updated weights rounded to bf16 exactly as
`quantize_int4` does, layers in sequence with each layer's calibration input
computed through the already-quantized layers, `lm_head` last on the
final-normed hidden states. The output is `pack_checkpoint`'s format with
the codes chosen differently, so the Marlin kernel, the graphs, the drafts
and the tests carry over untouched. 17 min on one card. Differential checks:
with an identity Hessian it reproduces RTN code for code; with correlated
inputs its output error is 0.070 vs RTN's 0.100 while its weight error rises
to 0.157 — the trade GPTQ makes. On the model: **KL 0.0546 → 0.0265**
(2.06x), top-1 0.905 → 0.937, WikiText-2 PPL +0.240 → +0.098, code 0.034
→ 0.021, math 0.084 → 0.047. Halfway to the bar in log terms: EXL3 is
2.07x lower still.

**What the remaining levers are worth** (each a 20-minute run; the
attribution again with `--keep-bf16`):

| variant | bits/weight | KL pooled | note |
|---|---|---|---|
| GPTQ, plain (the checkpoint) | 4.25 | 0.0265 | head bf16 → 0.0240, body bf16 → 0.0028: the body is 90% of it |
| + act-order, static groups | 4.25 | 0.0254 | within noise; WikiText PPL worse (+0.121 vs +0.098); not adopted |
| + MSE range search for the group parameters | 4.25 | **0.0232** | and the worst case: max KL 18.3 -> 6.1, math 0.047 -> 0.037; WikiText PPL slightly worse (+0.111 vs +0.098) |
| group 64 (4.5 bits/weight) | 4.50 | 0.0210 | what 0.25 more bits buy; would run on the Triton kernels (Marlin is g128) |
| an 8-bit head (not built) | 4.4 | ≈ 0.024 | bounded by the head-in-bf16 attribution: −0.0025 at most |

**The MSE range search is adopted; the default checkpoint is
`Qwen3.8-27B-int4g128-gptq-mse`.** It wins the metric the target row is
written in (KL 0.0232 vs 0.0265) at the same bits and the same speed, and it
wins the tail decisively (max KL 18.3 -> 6.1, math 0.047 -> 0.037). It loses
a little on WikiText-2 perplexity (6.365 vs 6.353 against bf16's 6.255): the
two metrics disagree, and the choice went to KL and top-1 because they read
the whole distribution while perplexity reads only the correct token's
probability. Recorded here because a disagreement between metrics is worth
keeping visible.

The bar (0.0128) is 1.8x below it, and nothing in the table closes 1.8x:
the uniform int4 grid is the limit, not the calibration. ExLlamaV3 sits
there at 4.00 bits with a trellis (non-uniform) codebook; llama.cpp's
Q4_K_M at 4.80 bits with mixed types and an importance matrix. Reaching it
at <= 4.25 bits means a different codebook, which is a kernel and a format,
not a quantizer setting. That is the decision left open at the end of this
step, and the argument for both sides is in `docs/quantization.md`.

**Reproducibility.** Nothing on a vast instance survives, so the recipe is
in git: `scripts/quantize/build.sh` goes from the public bf16 weights to the
checkpoint on one card in ~25 minutes, `measure.sh` rebuilds the bf16
reference (41 GB, ~2 min on one card) and measures a candidate. The
calibration ids and the evaluation corpora are committed as
`data/quality/*.npz` (0.8 MB) rather than rebuilt, because their code
portion is read from the installed torch and is not reproducible across
torch versions; their hashes are in `docs/quantization.md`.
`results/quality/*.json` hold every candidate's per-corpus and pooled
metrics and every GSM8K answer. The rival checkpoints were evaluated and
deleted, one at a time.

**Not changed in this step**: the packing (every checkpoint here loads in
the same engine, on the same kernels and graphs, with the same 76 tests),
and the Targets table's verdict in `CLAUDE.md` — the bar is still not met.

**The whole thread, by subject rather than by step** — why round-to-nearest
on day 2, why the yardstick was built before any quantizer was chosen, what
the Hub survey found, why GPTQ, what each refinement was worth, and the two
honest positions left — is in `docs/quantization.md`.

## Next

- **Phase 1b measured** (step 30): the quality table, and GPTQ in the same
  packing at KL 0.0265 (RTN 0.0546); the bar, ExLlamaV3's 0.0128, is 2x
  away and no GPTQ setting closes it. **Open decision**: change the codebook
  (a non-uniform grid means a kernel and a format) or keep uniform int4 at
  ~0.026 and restate the quality row with the measured number. The GPTQ
  checkpoint (`/workspace/models/Qwen3.8-27B-int4g128-gptq`, `python -m
  tokenrush.gptq`) is the one to use either way. Then **Phase 4** on one
  card. Optional Phase 3 items are listed at the end of step 29.
- **Phase 3 done** (steps 12–29): 224 / 373 / 373 greedy, 211 / 391 / 358
  sampled (DFlash2, essay / code / math), 179 / 310 / 251 on the Chinese
  families (MTP chain, chosen automatically), 211 / 238 at 200k, verify K=7 at
  1.13x a raw step.
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
