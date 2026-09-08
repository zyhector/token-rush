# Phase 1 progress log

One entry per step, with the number it produced. Every speed figure here is
a **development number** on whatever instance was rented that day, quoted
against the 1701 GB/s wall recorded in `docs/environment.md`; the citable
numbers come from Phase 4. Prefill is compute-bound and not a project
target; it is logged because it is free to measure and catches regressions.

Instance for all entries so far: vast container 50295164, RTX 5090, driver
610.43.02, CUDA 13.3, torch 2.14.0+cu130, triton 3.8.0, transformers 5.16.1,
fla 0.6.0. 150 GB disk, 60 GB RAM.

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

## Next

- **Phase 3**, steps 1–3 done. Step 4: the MTP draft chain inside the graph
  (its attention block through the fused kernels, its own K-row cache
  writes, the chain of K MTP calls and the verify in one replay; int4 head
  optional), then dynamic depth per content, then the effective-throughput
  table on the three families with the draft in-graph.
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
