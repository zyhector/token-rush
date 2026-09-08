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

## Next

- Step 4: first CUDA graph over a whole decode step (positions as device
  tensors, attention over a fixed-length masked cache or a bucketed set of
  graphs, argmax in-graph). Expect the 30 ms of CPU launch time to vanish
  and the step to approach its 18 ms of GPU time, then the ~9 ms of
  non-GEMV GPU time to become the target (GDN chain fusion, Phase 2).
- Phase 1b: engine-correctness gate against HF (per-layer first), quality
  table, quantization choice.
