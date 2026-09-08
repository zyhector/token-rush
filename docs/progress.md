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

## Next

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
