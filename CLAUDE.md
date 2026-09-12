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

**Final measurement 2026-09-09/10, one machine (vast 36542), one sitting**
(`docs/progress.md` step 32, `docs/baselines.md`, `results/2026-09-09-machine-36542/`).
Every rival re-run the same day. The rows below carry the target and the result.

| | Target | **Measured** |
|---|---|---|
| Raw decode | 110–120 tok/s at ≤4.25 bpw, i.e. 90–95% of the bandwidth wall | **100.9 tok/s = 81.0% of the wall** (`--backend triton`; 97.5 = 78.2% on the Marlin layout that serves the speculative step) — **not met**; the last ~10 points are the int4 GEMV, deferred at Phase 2 close |
| vs llama.cpp raw | +30% or more; at matched bytes the engine itself is worth +5% over vLLM — concede that row up front | **+22%** (100.9 vs 82.8) on 11% fewer bytes — not met; at matched bytes **+6 points of the wall over vLLM** (81.0 vs 74.6) — met |
| With fused speculative decoding | 200–240 tok/s effective on prose (the floor); 230–280 code, 300–380 math | **228 / 357 / 378** greedy (DFlash2 draft in-graph); 222 / 300 / 403 sampled at T=0.7 — **met**, code above its band |
| vs SGLang + DSpark | **2x on prose** (DSpark: 104 tok/s), +50% or more on math (205) — the durable headline | SGLang+DSpark re-measured 106 / 138 / 207: **2.15x / 2.6x / 1.8x** — **met** |
| vs vLLM (its best config is raw, 80 tok/s) | 2.5–3x effective; true today, perishable if vLLM fixes its speculative path | vLLM 0.29 raw 78.1, every speculative path of its own slower: **2.9x / 4.6x / 4.8x** — met |
| vs llama.cpp + MTP | 1.5–1.85x on prose (130), 2x on code and math | llama.cpp+MTP 130 / 121 / 169, ollama's default MTP chain 136 / 144 / 175 on this host: **1.67–1.75x / 2.5–2.95x / 2.2x** — met |
| Context | **256k usable**, not merely loadable — the model's native maximum | needle retrieved at 131k and **262k** tokens, 26.4 GB peak, 64 tok/s raw at full context — met |
| Decode at 200k | hold **≥92% of the wall**, the same fraction as at short context | **85.5%** (71.6 tok/s; 83.3% on Marlin) against vLLM 80.3%, ExLlamaV3 74.2%, SGLang 66.6%, llama.cpp 58.6% — **not met** as a fraction, ahead of every rival; +4.5 points over short context |
| Quantization quality | at ≤4.25 bpw, mean KL divergence to bf16 **no worse than ExLlamaV3's 4.00 bpw** (the one rival at our bpw); GSM8K within noise of llama.cpp's Q4_K_M — without this row the byte advantage does not count | ExLlamaV3 0.0128; ours 0.0546 uncalibrated, **0.0232 with GPTQ + an MSE range search** at 4.25 bpw (`docs/quantization.md`, step 30) — **not met**, 1.8x away, and what is left is the uniform int4 codebook, not the calibration. GSM8K **97.0%** in Phase 4 (96.5% in step 30) vs bf16's 96.0% — met |

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
is only reachable at or below ~4.25 bpw**. Picked in Phase 1b: int4 g128 GPTQ
with an MSE range search, 4.25 bpw (`docs/quantization.md`); the checkpoint is
published at `zyhector/Qwen3.8-27B-TokenRush-int4g128`.

Raw decode headroom equals the distance rivals sit from this wall.
**The only way through the wall is speculative decoding** — and at bs=1 the 5090's
tensor cores are idle, so verification FLOPs are free budget.

Long context is a **target, not a caveat**. Growing KV reads do lower the ceiling —
at 200k they add 6.4 GB/token at FP8, pulling the 4.0 bpw wall from 126 to 86
tok/s. That part is physics. But the ceiling already prices that in, so an
engine holding a constant fraction of the wall would degrade only that much,
and llama.cpp does not: it falls from 74% of the wall at short context to
**59% at 200k**. SGLang holds 60–67% and vLLM 75–80%, both rising slightly
from empty context to 200k (`docs/baselines.md`, Phase 4 numbers; byte
counts exclude the embedding table, which is not streamed). So the collapse
is an engine property, not a property of the problem — and at long context
the bar is vLLM, which is the best of them and still 20 points off the wall.
We hold 85.5% there.

The hybrid architecture is why this is winnable. **48 of 64 layers are GDN, whose
recurrent state is constant-size regardless of context** — only the 16 attention
layers pay for length. An engine that exploits that should barely degrade.

## Where the headroom comes from

1. **Design-point mismatch.** Continuous batching schedulers, PagedAttention block
   tables, batch-vectorized sampling, multi-process API servers, prefix-cache
   bookkeeping — all pure tax at bs=1. Replaced by contiguous preallocated KV,
   in-process execution, fused on-GPU sampling, one quantization format with
   offline weight reordering. **Measured, this tax is engine-specific** (Phase 4,
   same day, same card): SGLang pays 40 points of the wall for it, ExLlamaV3
   40, llama.cpp 25, and vLLM — with torch.compile and full-step CUDA graphs
   — 25. We pay 19 on the Triton GEMV, 22 on the Marlin layout. Raw decode
   is not where the win over vLLM is: 6 points of the wall are the engine,
   the rest of the tok/s margin is reading fewer bytes.
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

5. **Long-context decode is where the gap over llama.cpp is widest.**
   llama.cpp goes 74% -> 59% of the wall between short context and 200k;
   SGLang rises 60% -> 67%, vLLM 75% -> 80%, decoding 60 tok/s at 200k
   against llama.cpp's 45. Since 48 of 64 layers carry constant-size
   recurrent state, only 16 pay for length, so degradation is not a hard
   limit — SGLang and vLLM prove it. Measured on the Phase 4 machine, our
   engine holds **85.5% at 200k** (71.6 tok/s) against vLLM's 80.3% (59.9),
   ExLlamaV3's 74.2%, SGLang's 66.6% and llama.cpp's 58.6% — ahead of all of
   them, by 5 points over the best, while being 256k-*usable* where llama.cpp
   and SGLang lose a third of their short-context speed. The headline margin
   is speculation, not this.

**But not from GEMV.** cuBLAS bf16 GEMVs at the real layer shapes, replayed
from one CUDA graph, already stream at 1643 GB/s, 96.6% of the wall. Dense
weight streaming is close to solved before we start; the win is concentrated
in 1–3. Do not spend Phase 2 hand-writing GEMV.

## Who we are measured against

| Rival | Role |
|---|---|
| ollama | listed as the sanity floor; **on the Phase 4 host it is the fastest llama.cpp-family number** — 136 / 144 / 175 tok/s with its default 4-token MTP chain. Its serving path is host-CPU-bound: 67–82 on the Phase 0 EPYC with byte-identical runner and acceptance, `docs/baselines.md` |
| llama.cpp (+MTP) | raw-decode reference — 74.9% of the wall, MTP +60% on prose (130 / 121 / 169); external drafts do about as well on this host (DFlash2 131 / 114 / 164, DSpark 100 / 104 / 146), `docs/baselines.md` |
| vLLM (bs=1) | **the strongest raw engine** — 74.6% of the wall at bs=1 with torch.compile + full CUDA graphs, 80.3% at 200k; every speculative path it has (MTP 0.84x, DSpark 0.68x, DFlash2 0.66x) is *slower* than its raw decode, `docs/baselines.md` |
| **SGLang + DSpark** | **the real opponent** — 60% of the wall raw, rising to 67% at 200k; **106 / 138 / 207** tok/s with DSpark on prose / code / math (reproduced within 1.5% across two machines), `docs/baselines.md` |
| ExLlamaV3 | peer specialist, and the only rival at our bpw — 77.6 tok/s = 60% of the wall at 3.92 bpw; its chained MTP reaches 132 / 141 / 166, `docs/baselines.md` |

## The plan

Each phase is independently deliverable — there is no "nothing to show until it all
works" risk.

| Phase | Content | Output |
|---|---|---|
| 0 (1 wk) | Run llama.cpp (+MTP), SGLang (int4 + DSpark), vLLM on Qwen3.8-27B; slice per-token timeline with `nsys`; read GDN and MTP structure | Baseline report + overhead breakdown |
| 1a (1 wk) | **Get it running first.** A ~1000-line PyTorch engine built for bs=1 from existing blocks: explicit state (preallocated contiguous KV, conv state, FP32 recurrent state, position counter), per-layer pure functions, chunked prefill, greedy loop; GDN via `fla`'s chunk kernel for prefill and the verified Triton step from `check_stack.py` for decode, SDPA attention. Quantization is the dumbest thing that fits: own int4 g128 RTN packing, dequant-then-matmul GEMV. Milestone: coherent greedy text on three prompts, an eager tok/s. Then the 4-bit GEMV shootout (torchao, Marlin, NVFP4, EXL3; packed GB/s in-graph at real shapes incl. `lm_head`) and swap the kernel in | A running, fully ours engine + first speed number (development number) |
| 1b (done) | Correctness and quality. **Engine-correctness gate: done 2026-09-08** (bf16 vs HF, 48/48 greedy tokens). **Quality table and quantization choice: done 2026-09-09** — the yardstick (KL to bf16 over 82k positions, PPL, top-1, a measured noise floor) for ours and four rivals, then our own GPTQ in the unchanged packing: KL 0.0546 -> **0.0232**, speed bit-identical. The bar is 0.0128 and the remaining gap is the codebook: `docs/quantization.md` has the whole thread and the open decision | Correctness baseline + quality table + the GPTQ checkpoint and its recipe in git; the quality row itself still owed |
| 2 (3–4 wk) | Own kernels, in payoff order: full-step CUDA graph first, then fused GDN single step, attention decode, fused sampling; GEMV last and only if measurement demands it. **Done 2026-09-08** (`docs/progress.md` steps 4–11): 102 tok/s = 82% of the wall short-context, 69 tok/s = 83% at 200k with FP8 KV, 256k usable; GEMV redesign deferred | Raw decode near the wall |
| 3a (done) | Speculation fused into the graph: MTP chain, acceptance-driven dynamic depth, sampling, long context, draft vocabulary; greedy speculative output identical to greedy raw output with shared kernels. **Done 2026-09-08/09** (`docs/progress.md` steps 12–22): 186 / 261 / 263 tok/s effective on prose / code / math, 195 prose at 200k. Trees and DSpark measured and dropped (step 23) | Effective-throughput headline, first version |
| 3b (done) | **DFlash2 as the draft**: z-lab's 1.92B block-diffusion draft, 7 drafts from one forward conditioned on the target's layer 5/19/33/47/61 hidden states, on our fused kernels, int4, ring cache, K=7 verify. **Done 2026-09-09** (steps 25–27): 217 / 355 / 350 tok/s on prose / code / math, 229 code at 200k; Chinese prose keeps the MTP chain (`--draft mtp`) | The effective-throughput headline: prose over the 200 floor, code and math in or above their bands |
| 3c (done) | **Marlin-class M-row int4 GEMM**: Marlin (Apache-2) ported to bf16, our asymmetric g128 format, fp32 reduction and a lock-free partial mode, as a torch extension for `sm_120` (`tokenrush/csrc/`); one kernel and one layout for M <= 16, the default backend. **Done 2026-09-09** (step 28): verify K=7 1.35x -> **1.13x**; 224 / 373 / 373 (DFlash2) and 213 / 306 / 286 (MTP chain) on prose / code / math, 211 / 238 at 200k; raw decode on the shared layout 97.6 (102 with `--backend triton`) | The verify step near free: +5% DFlash, +15% MTP chain |
| 3 (closed) | **Closed 2026-09-09** (step 29): `--draft auto` keeps both drafts resident and picks the MTP chain for CJK prompts; the sampled path is exact rejection sampling for deterministic drafts (measured 211 / 391 / 358 at T=0.7). Optional items in `docs/progress.md` step 29 | 224 / 373 / 373 greedy on prose / code / math; 179 / 310 / 251 Chinese; 211 / 238 at 200k |
| 4 (done) | **Final measurement, on one machine, in one sitting. Done 2026-09-09/10** (`docs/progress.md` step 32) on vast 36542: the wall re-anchored (1702 GB/s; 1701 kept), every rival re-run from the recipes in `docs/baselines.md` and the engine measured the same day; bytes from the headers. The matrix is at the top of `docs/baselines.md`; the Targets table above carries the results | 228 / 357 / 378 vs the best rival per family 136 / 144 / 207; raw 100.9 = 81% of the wall, 85.5% at 200k; 256k needle retrieved |
| README data (done) | **Re-measure for the README, on one machine, one sitting: 2026-09-12, vast 59052** — the short-context matrix, every rival, the ten-point raw sweep and the multi-position speculative sweep (six positions × 512 tokens on PG-19 and code, both drafts, llama.cpp + MTP and SGLang + DSpark on the same prompt files); `docs/progress.md` step 35, `results/2026-09-12-machine-59052/` | The numbers and curves the README is drawn from; figures and the README itself are still to be written |
| 5 (done) | **Serve it.** `python -m tokenrush.serve`: OpenAI Chat / Completions and the Anthropic Messages API over the resident engine, tool calling through the model's own format, the context kept between requests by state snapshots (a tool-result turn costs 0.07 s). **Done 2026-09-10** (`docs/progress.md` step 33, `docs/serving.md`); Claude Code verified on it end to end | The local model, opened every day — Claude Code included |

Step-by-step progress and the numbers each step produced: `docs/progress.md`.

## Scope

One model, one quantization, one card, bs=1, greedy/top-p, text path. Nothing else.

**Serving it is in scope since Phase 5** (`docs/serving.md`,
`python -m tokenrush.serve`): OpenAI Chat / Completions and the Anthropic
Messages API over the resident engine, one request at a time, with the
context kept between requests so a chat turn costs its own tokens — it is
how the author opens the model every day, Claude Code included. Not
serving *traffic*: no batching, no concurrency, no constrained decoding.

**Artifacts.** The engine's weights are published at
[`zyhector/Qwen3.8-27B-TokenRush-int4g128`](https://huggingface.co/zyhector/Qwen3.8-27B-TokenRush-int4g128)
(int4 g128 GPTQ + MSE, 4.25 bpw, 17.0 GB), the draft is
`z-lab/Qwen3.8-27B-DFlash2` (public, 3.9 GB), and **the Hub is the route to
both**: `python -m tokenrush.run --chat --prompt "..."` with no `--model`
downloads them into the Hub cache on first use (17 + 3.9 GB, one line of
notice; `--no-download` to refuse) and runs. `--model` and `--dflash-path`
take a repo id or a local directory; `bench/decode.py`, `bench/families.py`,
`bench/quality_gsm8k_engine.py` and `bench/needle.py` resolve theirs the same
way. Do not rebuild the checkpoint to run the engine: every quality number
in `docs/quantization.md` was measured on the published file and GPU floating
point is not bit-deterministic (`scripts/quantize/build.sh` exists for
changing the method, not for getting the weights). The rival stacks are
rebuilt from `docs/baselines.md`.

**The reported numbers are Phase 4's: vast machine 36542 (RTX 5090, driver
610 / CUDA 13.3, torch 2.14+cu130), 2026-09-09/10, the wall, every rival and
the engine on one day** (`docs/environment.md`, `docs/baselines.md`,
`results/2026-09-09-machine-36542/`). Phase 0 was measured on machine 94372
on 2026-09-04 and stands as the historical baseline; the two agree to a point
or two on every GPU-bound row, and the ollama and llama.cpp-external-draft
rows moved because those loops are host-CPU-bound (`docs/baselines.md`).
Development numbers in `docs/progress.md` steps 1–31 are from disposable
instances and are quoted against the 1701 GB/s wall; they are not the report.

**The README's data is the 2026-09-12 re-measurement on vast 59052** (same
card, a 500 W power cap, a Core Ultra 9 285K host; rival versions pinned to
Phase 4's): the whole matrix again on one machine in one sitting, plus the
**multi-position decode-vs-context sweep** that replaced the single-position
one of step 34 — PG-19 prose and torch code, six positions per context
length, 512 greedy tokens each, both drafts from one shared prefill, tok/s
as total tokens over total decode seconds, llama.cpp's MTP and SGLang's
DSpark on the same prompt files. Logs in `results/2026-09-12-machine-59052/`
(its README maps files to numbers), tables generated from them by
`scripts/matrix_table.py` (short context → `matrix.csv`) and
`scripts/sweep_table.py` (context sweep → `sweep.csv`), the protocol and
what it found in `docs/progress.md` step 35, the tables in
`docs/baselines.md` ("The 2026-09-12 re-measurement"), the machine in
`docs/environment.md`. Every GPU-bound row agrees with Phase 4 within 1.5%;
the host-bound rival rows (ollama, llama.cpp's drafts) are 4–6% lower on
this host. Regenerate the tables from the logs rather than retyping them.
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

Longer-form argument, precedents and sources: `docs/feasibility.md`. The
quantization thread end to end — the format, the yardstick, the rivals, GPTQ,
and what is left — is `docs/quantization.md`.
