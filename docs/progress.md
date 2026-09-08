# Progress log

One row per step: what changed and the decode speed it produced. The same
rows are in `progress.csv` for plotting (x = step, y = decode tok/s). All
numbers are **development numbers** on the instance of the day, quoted
against the 1701 GB/s wall of `docs/environment.md`; Phase 4 produces the
citable ones. Decode is short-context, greedy, `bench/decode.py`. The wall
fraction uses bytes actually streamed per step (13.65 GB: the embedding
table is not read). Ceiling at this quantization: 124.6 tok/s.

| step | what changed | decode tok/s | % of wall | ms/step |
|---|---|---|---|---|
| 1 | environment + weights | — | — | — |
| 2 | skeleton runs: int4 RTN, dequantize-then-matmul, eager | 3.2 | 2.6 | 312 |
| 3 | own Triton int4 GEMV, fused q\|k\|v and gate\|up | 36.0 | 29 | 27.7 |
| 4 | whole-step CUDA graph | 68.4 | 55 | 14.6 |
| 5 | engine-correctness gate vs HF (no speed change) | 68.4 | 55 | 14.6 |
| 6 | fused GDN step, fused add+RMSNorm | 83.2 | 67 | 12.0 |
| 7 | fused attention decode (flash-decoding), no buckets | **99.5** | **80** | 10.05 |

Rivals on the same yardstick (Phase 0, `docs/baselines.md`): llama.cpp 78%,
vLLM 88% (likely ~76% once the embedding is removed from its byte count),
SGLang 70%, ExLlamaV3 60%. Prefill is ~1500 tok/s throughout (compute-bound,
not a target).

Decode vs. context after step 7 (bf16 KV): 0 / 22k / 30k tokens ->
99.5 / 91.5 / 88.8 tok/s = 79.9 / 81.2 / 81.5% of the wall. Flat, as the
attention kernel keeps up with the KV read. Needle retrieval passes at 5k
and 30k.

## Notes per step

All on 2026-09-08, vast container 50295164 (RTX 5090, driver 610.43.02,
CUDA 13.3, torch 2.14.0+cu130, triton 3.8.0, transformers 5.16.1, fla 0.6.0;
150 GB disk, 60 GB RAM).

**1. Environment and weights.** Recipe in `docs/environment.md`.
`check_stack.py` same picture as the Phase 0 machine (fla's fused GDN step
NaN; chunk kernel and minimal Triton step correct). `Qwen/Qwen3.8-27B` bf16
verified by size and header; the vision tower shares shard 1 with text
tensors, dropped at load. Disk reads ~2.5 GB/s.

**2. Skeleton runs.** `tokenrush/`, ~800 lines: explicit preallocated state,
layers as pure functions, chunked prefill, greedy loop; fla chunk kernel for
GDN prefill, verified Triton step for decode, SDPA attention; own int4 g128
RTN packing. Coherent text on raw, chat and a 5k needle prompt. Chunked vs
one-shot vs stepped paths agree to ~2% (bf16 kernel noise, no jump at
boundaries).

**3. GEMV shootout.** `scripts/gemv_shootout/README.md`. Our Triton int4 88%
of the wall on the fused layer set, 99% on lm_head; tinygemm 91/99, Marlin
88/97, NVFP4 cutlass 52 (a GEMM, wrong tool at M=1). Ours is the default;
projections fused at load (+4–5 points). Step still CPU-bound: 30 ms of
launches for 18 ms of GPU work.

**4. CUDA graph.** Device position tensor, indexed KV writes and rope,
attention over context buckets with a mask, one graph per bucket, argmax
in-graph feeding the next step. Replay matches eager bit for bit. Step time
= GPU time. Byte accounting corrected (embedding not streamed).

**5. Engine-correctness gate.** `bench/hf_reference.py` + `bench/gate_engine.py`.
bf16 path streamed layer by layer vs HF (fla blocked): residual drift
2e-3 -> 2e-2 over 64 layers with no jump, prompt top-1 32/32, teacher-forced
top-1 48/48, greedy identical 48/48. int4 RTN: KL 6e-2, diverges at step 4,
HF's token always in top-5 — the quality baseline to beat
(`docs/quality_plan.md`).

**6. Fused GDN step and norms.** `tokenrush/fused.py`. One kernel per GDN
layer (conv on a 4-column ring state, gating, delta rule, gated norm, output
gate: 22 -> 1), one per residual+RMSNorm (11 -> 1), silu*mul. Found and fixed
the eager conv rounding each product in bf16 where HF's cuDNN conv rounds
once. Gate numbers unchanged.

**7. Fused attention decode.** Prep kernel (norm, rope, KV write), flash-
decoding split kernel over the live length (32 splits per kv head, tensor-core
dots, fp32 online softmax), reduce kernel with the output gate. Static grid,
so context buckets are gone. Attention 1.8 -> 0.09 ms; ~600 launches per
step, GEMVs 90% of it.

## Next

- Phase 2 remaining: fold b|a into the GDN kernel (0.2 ms); split-K for the
  two 5120-row GEMV shapes (74–81% -> 90%+, ~0.5 ms); sampling in-graph
  beyond argmax; FP8 KV with a prefill attention kernel for 256k.
- Phase 1b second half (quality table, quantization choice) deferred to a
  two-GPU box: `docs/quality_plan.md`.
- Decision 2026-09-08: Phase 2 first; Phase 0 numbers frozen until Phase 4.
