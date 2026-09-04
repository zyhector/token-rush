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
| Raw decode | 95–105 tok/s, i.e. 85–90% of the bandwidth wall |
| vs llama.cpp raw | +30% or more |
| With fused speculative decoding | 220–280 tok/s effective |
| vs llama.cpp + MTP | 2x or more |
| vs SGLang + DSpark | +20–40% |

**% of the memory-bandwidth roofline** is the primary metric — unlike a margin over
a rival, it does not move as rivals mature.

## The physics

**RTX 5090**: 32 GB GDDR7, 1792 GB/s, SM120 (consumer Blackwell — `mma.sync`
lineage, no Hopper `wgmma`, no datacenter-Blackwell `tcgen05`; adds FP4 tensor
cores).

**Qwen3.8-27B**: 64 layers = 16 x (3 Gated DeltaNet + 1 gated attention).
hidden 5120, FFN 17408, vocab 248k. GDN layers: 48 V heads / 16 QK heads, head
dim 128. Attention layers: 24 Q heads / 4 KV heads, head dim 256. Ships with an
MTP (nextn) head. Natively multimodal — **this engine handles the text path only;
the vision tower is discarded.**

**Memory budget** (~20 GB, comfortable on 32 GB):

- weights at 4.5–5 bpw: 15–16.5 GB
- KV cache: only the 16 attention layers produce it — 64 KB/token FP16,
  32 KB/token FP8, so 32k context is ~1 GB
- GDN recurrent state: ~72 MB for the whole model, ~144 MB stored FP32,
  independent of context length
- draft model (MTP head or 4-bit DSpark): ~1 GB

**The bandwidth wall**: single-stream decode tok/s ~= 1792 / bytes-read-per-token.
At 16 GB of weights that is ~112 tok/s; at 32k context KV reads take another ~6%,
so ~100 tok/s. Raw decode headroom equals the distance rivals sit from this wall.
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
4. **Nobody feeds Blackwell's bandwidth.** 1792 GB/s needs far more memory-level
   parallelism than Ada-era kernels were tuned for. GDN kernels are young in every
   engine. And consumer cards are second-class citizens to vLLM/SGLang, whose main
   theater is H100/B200.

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
| 0 (1 wk) | Rent 5090; run llama.cpp (+MTP), SGLang (int4 + DSpark), vLLM on Qwen3.8-27B; compute roofline; slice per-token timeline with Nsight; read GDN and MTP structure | Baseline report + overhead breakdown |
| 1 (2 wk) | Clean PyTorch reference for the text path (GDN via `fla`), token-exact against HF; pick quantization (NVFP4 or int4 groupwise) | Correctness baseline + first speed number |
| 2 (3–4 wk) | Own kernels: GDDR7-saturating GEMV, fused GDN single step, attention decode, fused sampling, full-step CUDA graph | Raw decode near the wall |
| 3 (2–3 wk) | Speculation fused into the graph: MTP chain/tree vs 4-bit DSpark, acceptance-driven dynamic depth | Effective-throughput headline |
| 4 | Fair benchmark matrix + writeup | Report and blog post |

## Scope

One model, one quantization, one card, bs=1, greedy/top-p, text path. Nothing else.

- **Correctness is a gate**: token-exact against the HF reference. GDN decode has no
  mature single-stream reference to copy, so correctness comes from differential
  testing. Store recurrent state in FP32 to avoid long-sequence drift.
- **Time-box**: resume-ready milestone by **end of October 2026**.

Longer-form argument, precedents and sources: `docs/feasibility.md`.
