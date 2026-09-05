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
| Raw decode | 110–120 tok/s at ≤4.25 bpw, i.e. 90–95% of the bandwidth wall |
| vs llama.cpp raw | +30% or more; at matched bytes the engine itself is worth +5% over vLLM — concede that row up front |
| With fused speculative decoding | 200–240 tok/s effective on prose (the floor); 230–280 code, 300–380 math |
| vs SGLang + DSpark | **2x on prose** (DSpark: 104 tok/s), +50% or more on math (205) — the durable headline |
| vs vLLM (its best config is raw, 80 tok/s) | 2.5–3x effective; true today, perishable if vLLM fixes its speculative path |
| vs llama.cpp + MTP | 1.5–1.85x on prose (130), 2x on code and math |
| Context | **256k usable**, not merely loadable — the model's native maximum |
| Decode at 200k | hold **≥92% of the wall**, the same fraction as at short context (llama.cpp holds 60%, SGLang 74%, vLLM 90%) |
| Quantization quality | at ≤4.25 bpw, mean KL divergence to bf16 **no worse than ExLlamaV3's 4.00 bpw** (the one rival at our bpw); GSM8K within noise of llama.cpp's Q4_K_M — without this row the byte advantage does not count |

**% of the memory-bandwidth roofline** is the primary metric — unlike a margin over
a rival, it does not move as rivals mature. The projection behind these rows,
the step-cost model and the reasons the project is worth its 6–7 weeks are in
`docs/feasibility.md`; the numbers there are extrapolated from component
measurements, not from a working engine.

## The physics

**RTX 5090**: 32 GB GDDR7, 1792 GB/s on the spec sheet but **1701 GB/s of
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
  32 KB/token FP8, so 32k is ~1 GB and **256k is ~8.4 GB**. At 4.0 bpw the full
  256k window costs 13.45 + 8.4 = **21.9 GB**, comfortable on 31.4 GB
- GDN recurrent state: ~75 MB for the whole model, ~151 MB stored FP32,
  independent of context length
- draft model (MTP head or 4-bit DSpark): ~1 GB

**The bandwidth wall**: single-stream decode tok/s ~= 1701 / bytes-read-per-token.

| bpw | weights | ceiling, empty KV | at 32k FP8 KV | 85–90% of wall |
|---|---|---|---|---|
| 4.0 | 13.45 GB | 126 tok/s | ~117 | **108–114** |
| 4.5 | 15.13 GB | 112 tok/s | ~105 | 96–101 |
| 5.0 | 16.81 GB | 101 tok/s | ~95 | 86–91 |

Quantization is therefore the single largest lever on the headline number — it
moves the ceiling by ~25 tok/s across that range, and **the 110–120 tok/s target
is only reachable at or below ~4.25 bpw**. Pick accordingly in Phase 1.

Raw decode headroom equals the distance rivals sit from this wall.
**The only way through the wall is speculative decoding** — and at bs=1 the 5090's
tensor cores are idle, so verification FLOPs are free budget.

Long context is a **target, not a caveat**. Growing KV reads do lower the ceiling —
at 200k they add 6.4 GB/token at FP8, pulling the 4.0 bpw wall from 126 to 86
tok/s. That part is physics. But the ceiling already prices that in, so an
engine holding a constant fraction of the wall would degrade only that much,
and llama.cpp does not: it falls from 77% of the wall at short context to
**60% at 200k**. SGLang holds a flat ~70%, and vLLM a flat **88–90%**, from
empty context to 200k (`docs/baselines.md`). So the collapse is an engine
property, not a property of the problem — and at long context the bar is
vLLM, which is already near the roofline.

The hybrid architecture is why this is winnable. **48 of 64 layers are GDN, whose
recurrent state is constant-size regardless of context** — only the 16 attention
layers pay for length. An engine that exploits that should barely degrade.

## Where the headroom comes from

1. **Design-point mismatch.** Continuous batching schedulers, PagedAttention block
   tables, batch-vectorized sampling, multi-process API servers, prefix-cache
   bookkeeping — all pure tax at bs=1. Replaced by contiguous preallocated KV,
   in-process execution, fused on-GPU sampling, one quantization format with
   offline weight reordering. **Measured, this tax is engine-specific**: SGLang
   pays 30 points of the wall for it, llama.cpp 23, and vLLM — with
   torch.compile and full-step CUDA graphs — only 12. Raw decode is not where
   the win over vLLM is.
2. **Speculation conservatism is structural** (most important). Under batching,
   verification FLOPs are not free, so aggressive speculation hurts throughput and
   general engines must stay conservative. At bs=1 aggressive tree speculation is
   nearly free. Concretely: llama.cpp's MTP path is single-slot with a
   device-to-host embedding round trip during prompt; SGLang's DSpark speedup falls
   from 3.16x to 2.48x at concurrency 8 — its tuning point is not the bs=1 limit —
   and its verify step costs 1.45x a raw decode step for 8 verified positions,
   with tensor cores that are otherwise idle at bs=1.
3. **Full-step CUDA graph capture needs fixed shapes.** The hybrid architecture
   amplifies this: 48 GDN layers are each a chain of small ops (conv1d, norm,
   gating, delta-rule state update, output gate) launched individually elsewhere.
   At bs=1 the entire decode step — embedding, 64 layers, lm_head, sampling, token
   staying resident for the next step — records into one graph, never returning to
   the host. Engines supporting dynamic batching structurally cannot do this.
   **Measured, and this is the biggest single item**: a 48-layer GDN chain
   (projections excluded) costs 10.6 ms/token launched eagerly and
   1.4 ms/token replayed from one graph. The 9.1 ms of pure
   launch tax is ~90% of the entire 10 ms budget for 100 tok/s.
4. **Nobody feeds Blackwell's bandwidth.** 1701 GB/s needs far more memory-level
   parallelism than Ada-era kernels were tuned for. GDN kernels are young in every
   engine — young enough that `fla`'s fused single-step kernel, the one every
   PyTorch-based engine would reach for, is miscompiled on `sm_120`
   (`docs/environment.md`). And consumer cards are second-class citizens to
   vLLM/SGLang, whose main theater is H100/B200.

5. **Long-context decode is where the gap over llama.cpp is widest, and
   where vLLM is already at the wall.** llama.cpp goes 77% -> 60% of the wall
   between short context and 200k; SGLang holds ~70% flat; vLLM holds 88–90%
   flat and decodes at 61 tok/s at 200k against llama.cpp's 44. Since 48 of 64
   layers carry constant-size recurrent state, only 16 pay for length, so
   degradation is not a hard limit — vLLM proves it. Long context is therefore
   a place to match vLLM at the roofline while being 256k-*usable* (llama.cpp
   and SGLang lose a third of their short-context speed there), not a place
   for a headline margin. The headline margin is speculation.

**But not from GEMV.** cuBLAS bf16 GEMVs at the real layer shapes, replayed
from one CUDA graph, already stream at 1643 GB/s, 96.6% of the wall. Dense
weight streaming is close to solved before we start; the win is concentrated
in 1–3. Do not spend Phase 2 hand-writing GEMV.

## Who we are measured against

| Rival | Role |
|---|---|
| ollama | sanity floor only — 67–82 tok/s with its default 4-token MTP chain, slower than llama.cpp raw on prose, `docs/baselines.md` |
| llama.cpp (+MTP) | raw-decode reference — measured at 78.4% of the wall, MTP +60%; external drafts do worse (DFlash2 +37%, DSpark +16%), `docs/baselines.md` |
| vLLM (bs=1) | **the strongest raw engine** — 88% of the wall at bs=1 with torch.compile + full CUDA graphs, 90% at 200k; every speculative path it has (MTP, DSpark, DFlash2) is *slower* than its raw decode, `docs/baselines.md` |
| **SGLang + DSpark** | **the real opponent** — 70% of the wall raw, flat to 200k; 104 / 137 / 205 tok/s with DSpark on prose / code / math, `docs/baselines.md` |
| ExLlamaV3 | peer specialist, and the only rival at our bpw — 77 tok/s = 60% of the wall at 3.92 bpw; its chained MTP reaches 127–160 tok/s, `docs/baselines.md` |

## The plan

Each phase is independently deliverable — there is no "nothing to show until it all
works" risk.

| Phase | Content | Output |
|---|---|---|
| 0 (1 wk) | Run llama.cpp (+MTP), SGLang (int4 + DSpark), vLLM on Qwen3.8-27B; slice per-token timeline with `nsys`; read GDN and MTP structure | Baseline report + overhead breakdown |
| 1 (2 wk) | Clean PyTorch reference for the text path (GDN via `fla`), bf16 path passing the engine-correctness gate against HF; pick quantization at ≤4.25 bpw (NVFP4 or int4 groupwise) and pass the quantization-quality gate | Correctness baseline + quality table + first speed number |
| 2 (3–4 wk) | Own kernels, in payoff order: full-step CUDA graph first, then fused GDN single step, attention decode, fused sampling; GEMV last and only if measurement demands it | Raw decode near the wall |
| 3 (2–3 wk) | Speculation fused into the graph: MTP chain/tree vs 4-bit DSpark, acceptance-driven dynamic depth; greedy speculative output identical to greedy raw output | Effective-throughput headline |
| 4 | Fair benchmark matrix + writeup | Report and blog post |

## Scope

One model, one quantization, one card, bs=1, greedy/top-p, text path. Nothing else.

Environment is provisioned and validated — RTX 5090 (vast machine 94372), driver
610 / CUDA 13.3, torch 2.14+cu130, CUDA graph capture confirmed on `sm_120`.
Two constraints to plan around: **`fla`'s fused GDN decode kernel is
miscompiled here** (the chunk kernel is correct and is the Phase 1 reference
path; the Phase 2 fused step is ours anyway), and **`ncu` hardware counters
are blocked on this host** (`ERR_NVGPUCTRPERM`, unfixable in-container), so
kernel-level occupancy and DRAM-throughput tuning must come from wall-clock
timing against known byte counts. `nsys` timelines work. See
`docs/environment.md`.

- **Correctness is two gates, answering two different questions.**
  *Engine correctness* — is the implementation right — compares our bf16 path
  with HF `transformers` on the same weights: greedy tokens identical up to
  the first divergence, and at the divergence the reference's token within our
  top-5 with the logprob gap inside bf16 noise (the vLLM test-suite
  standard; HF runs CPU-offloaded, since 55.6 GB of bf16 does not fit the
  card). Under it, every custom kernel is differential-tested against a torch
  reference over repeated calls, CUDA-graph replay must match eager
  bit-for-bit, and greedy speculative output must equal greedy raw output
  exactly — the one check that is truly exact. Long context is verified
  functionally (needle retrieval at 128k and 256k). GDN decode has no mature
  single-stream reference to copy, so this is where correctness comes from.
  Store recurrent state in FP32 to avoid long-sequence drift.
  *Quantization quality* — how much the 4-bit weights lose — compares our
  quant with bf16 and with the rivals' quants on the same text with the same
  script: KL divergence (mean, p99), WikiText-2 perplexity delta, top-1
  agreement, and one downstream task; bf16 logits computed once on CPU and
  reused. Threshold in the Targets table. Token-exactness against HF was
  never a quantization metric and is not a goal.
- **Time-box**: resume-ready milestone by **end of October 2026**.

Longer-form argument, precedents and sources: `docs/feasibility.md`.
