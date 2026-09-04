# Token Rush

A single-stream inference engine for **Qwen3.8-27B on one RTX 5090**. One machine,
one GPU, one user, one stream. Everything that exists in general engines to serve
concurrent traffic is deliberately thrown away in exchange for latency.

The first user is the author: the deliverable is not a demo, it is the local model
he opens every day.

## The claim

General serving engines structurally leave single-stream (bs=1) headroom on the
table. That is not an accident they will patch — it follows from their design point
(throughput under batching) and their business constraints. Token Rush takes that
headroom.

**Speed is the only objective.** Not generality, not features, not serving.

## Targets

| | Target |
|---|---|
| Raw decode | 95–105 tok/s at ≤4.25 bpw, i.e. 85–90% of the bandwidth wall |
| vs llama.cpp raw | +30% or more |
| With fused speculative decoding | 220–280 tok/s effective |
| vs llama.cpp + MTP | 2x or more |
| vs SGLang + DSpark | +20–40% |

**% of the memory-bandwidth roofline** is the primary metric — unlike a margin over
a rival, it does not move as rivals mature.

## The physics

**RTX 5090**: 32 GB GDDR7, 1792 GB/s on the spec sheet but **1605 GB/s of
measured read bandwidth** — that measured figure is the wall, and every "% of
roofline" number in this project is a fraction of it. 170 SMs, SM120 (consumer
Blackwell — `mma.sync` lineage, no Hopper `wgmma`, no datacenter-Blackwell
`tcgen05`; adds FP4 tensor cores). Machine details in `docs/environment.md`.

**Qwen3.8-27B**: 64 layers = 16 x (3 Gated DeltaNet + 1 gated attention).
hidden 5120, FFN 17408, vocab 248320. GDN layers: 48 V heads / 16 QK heads, head
dim 128, conv kernel 4. Attention layers: 24 Q heads / 4 KV heads, head dim 256.
Ships with an MTP head (`mtp.*`, 0.425B — one attention layer plus `fc`), so
Phase 3 trains nothing. Natively multimodal — **this engine handles the text path
only; the 0.461B vision tower is discarded.**

**The text path we serve is 26.90B params** (25.63B body + 1.27B `lm_head`), out
of 27.78B in the repo. Weight bytes are computed against 26.90B, not 27B.

**Memory budget** (~16–17 GB, comfortable on 31.4 GB usable):

- weights at 4.0–4.5 bpw: 13.5–15.1 GB
- KV cache: only the 16 attention layers produce it — 64 KB/token FP16,
  32 KB/token FP8, so 32k context is ~1 GB
- GDN recurrent state: ~75 MB for the whole model, ~151 MB stored FP32,
  independent of context length
- draft model (MTP head or 4-bit DSpark): ~1 GB

**The bandwidth wall**: single-stream decode tok/s ~= 1605 / bytes-read-per-token.

| bpw | weights | ceiling, empty KV | at 32k FP8 KV | 85–90% of wall |
|---|---|---|---|---|
| 4.0 | 13.45 GB | 119 tok/s | ~112 | **95–101** |
| 4.5 | 15.13 GB | 106 tok/s | ~100 | 85–90 |
| 5.0 | 16.81 GB | 96 tok/s | ~90 | 77–81 |

Quantization is therefore the single largest lever on the headline number — it
moves the ceiling by ~24 tok/s across that range, and **the 95–105 tok/s target
is only reachable at or below ~4.25 bpw**. Pick accordingly in Phase 1.

Raw decode headroom equals the distance rivals sit from this wall.
**The only way through the wall is speculative decoding** — and at bs=1 the 5090's
tensor cores are idle, so verification FLOPs are free budget.

Be honest about long context: within 64k, FP8 KV costs ~1 ms per step. At 262k the
16 attention layers' KV reads dominate. That is physics, not an engineering defect.

## Where the headroom comes from

1. **Design-point mismatch.** Continuous batching schedulers, PagedAttention block
   tables, batch-vectorized sampling, multi-process API servers, prefix-cache
   bookkeeping — all pure tax at bs=1. Replaced by contiguous preallocated KV,
   in-process execution, fused on-GPU sampling, one quantization format with
   offline weight reordering.
2. **Speculation conservatism is structural** (most important). Under batching,
   verification FLOPs are not free, so aggressive speculation hurts throughput and
   general engines must stay conservative. At bs=1 aggressive tree speculation is
   nearly free. Concretely: llama.cpp's MTP path is single-slot with a
   device-to-host embedding round trip during prompt; SGLang's DSpark speedup falls
   from 3.16x to 2.48x at concurrency 8 — its tuning point is not the bs=1 limit.
3. **Full-step CUDA graph capture needs fixed shapes.** The hybrid architecture
   amplifies this: 48 GDN layers are each a chain of small ops (conv1d, norm,
   gating, delta-rule state update, output gate) launched individually elsewhere.
   At bs=1 the entire decode step — embedding, 64 layers, lm_head, sampling, token
   staying resident for the next step — records into one graph, never returning to
   the host. Engines supporting dynamic batching structurally cannot do this.
   **Measured, and this is the biggest single item**: a 48-layer GDN chain costs
   4.39 ms/token launched eagerly and 0.15 ms/token replayed from one graph. The
   4.2 ms of pure launch tax is ~42% of the entire 10 ms budget for 100 tok/s.
4. **Nobody feeds Blackwell's bandwidth.** 1605 GB/s needs far more memory-level
   parallelism than Ada-era kernels were tuned for. GDN kernels are young in every
   engine. And consumer cards are second-class citizens to vLLM/SGLang, whose main
   theater is H100/B200 — cuBLAS still dispatches Ampere-lineage
   `cutlass_80_tensorop_*` kernels for bf16 matmul on this card.

**But not from GEMV.** An untuned Triton bf16 GEMV already reaches 1569 GB/s,
97.8% of the wall. Dense weight streaming is close to solved before we start; the
win is concentrated in 1–3. Do not spend Phase 2 hand-writing GEMV.

## Who we are measured against

| Rival | Role |
|---|---|
| ollama | sanity floor only, no credibility value |
| llama.cpp (+MTP) | raw-decode reference |
| vLLM (bs=1) | general-engine tax reference |
| **SGLang + DSpark** | **the real opponent** |
| ExLlamaV3 | peer specialist; included proactively because reviewers will ask |

## The plan

Each phase is independently deliverable — there is no "nothing to show until it all
works" risk.

| Phase | Content | Output |
|---|---|---|
| 0 (1 wk) | Run llama.cpp (+MTP), SGLang (int4 + DSpark), vLLM on Qwen3.8-27B; slice per-token timeline with `nsys`; read GDN and MTP structure | Baseline report + overhead breakdown |
| 1 (2 wk) | Clean PyTorch reference for the text path (GDN via `fla`), token-exact against HF; pick quantization at ≤4.25 bpw (NVFP4 or int4 groupwise) | Correctness baseline + first speed number |
| 2 (3–4 wk) | Own kernels, in payoff order: full-step CUDA graph first, then fused GDN single step, attention decode, fused sampling; GEMV last and only if measurement demands it | Raw decode near the wall |
| 3 (2–3 wk) | Speculation fused into the graph: MTP chain/tree vs 4-bit DSpark, acceptance-driven dynamic depth | Effective-throughput headline |
| 4 | Fair benchmark matrix + writeup | Report and blog post |

## Scope

One model, one quantization, one card, bs=1, greedy/top-p, text path. Nothing else.

Environment is provisioned and validated — RTX 5090 (vast machine 25132), CUDA
12.8, torch 2.11+cu128, `fla` GDN decode and CUDA graph capture both confirmed on
`sm_120`. One constraint to plan around: **`ncu` hardware counters are blocked on
this host** (`ERR_NVGPUCTRPERM`, unfixable in-container), so kernel-level
occupancy and DRAM-throughput tuning must come from wall-clock timing against
known byte counts. `nsys` timelines work. See `docs/environment.md`.

- **Correctness is a gate**: token-exact against the HF reference. GDN decode has no
  mature single-stream reference to copy, so correctness comes from differential
  testing. Store recurrent state in FP32 to avoid long-sequence drift.
- **Time-box**: resume-ready milestone by **end of October 2026**.

Longer-form argument, precedents and sources: `docs/feasibility.md`.
