# Feasibility — short version

**Verdict: feasible, and not a gamble — but the number worth writing down
only exists once Phase 2 and Phase 3 both land.** The win comes from a design
point general engines are unwilling to serve, not from luck. The physics, the
memory budget and the per-rival argument live in `CLAUDE.md`; this file holds
what that leaves out, and — since Phase 0 — the projection of where the engine
ends up and which comparison to headline.

## The thesis is already proven by precedent

"Trade generality for single-stream speed" is not an untested hypothesis:

- **gpt-fast** (PyTorch Labs): ~1000 lines, no serving features, `torch.compile` +
  int4 + speculative decoding, SOTA at bs=1.
- **ExLlamaV2/V3**: consumer-card single-user engines that have beaten general
  engines for years. The V3 author reports 4bpw on a 4090 as roughly memory-bound.
- **TensorRT-LLM**: consistently ahead of general serving engines at bs=1.

So no step of this project depends on an unproven principle. All risk sits in
engineering integration and execution.

The novelty is not any single technique — every one has public precedent. It is the
integration on a Blackwell consumer card, for a 2026 hybrid-architecture model, with
aggressive speculation tuning, plus a fair and credible comparison report. That is
the correct shape for a mini project.

## Known rival data points (as of Sept 2026 — re-verify before citing)

- llama.cpp on our own 5090, Qwen3.8-27B UD-Q4_K_M: **82.8 tok/s raw**,
  **130 with MTP** (+61% on prose, +100% on math). Raw decode reads 16.10
  GB/token, so that is **78.4% of the bandwidth wall**. Measured, not cited —
  see `docs/baselines.md`. The remaining 22% is the raw-decode headroom, and it
  is thinner than public mobile-5090 figures suggest: "+30% over llama.cpp"
  requires 108 tok/s, which at 4.0 bpw is 85% of the wall. llama.cpp's MTP is
  also far stronger on this host than public figures suggest, so the
  effective-throughput margin must come from deeper speculation, not from a
  weak rival.
- vLLM 0.28 on our own 5090, NVFP4: **80.0 tok/s raw = 88% of the wall** at
  bs=1 with torch.compile and full CUDA graphs, and **90% at 200k**. The strongest raw engine
  measured; on matched bytes it leaves ~8 points to the GEMV ceiling.
- SGLang 0.5.18 on our own 5090, NVFP4: **63.2 tok/s raw = 70% of the wall**,
  flat from empty context to 200k. With DSpark (RadixArk, 1.86B draft, gamma
  7): **104 / 137 / 205 tok/s** on prose / code / math, mean accepted length
  2.4 / 3.1 / 4.7. Measured — see `docs/baselines.md`. RadixArk's published
  bs=1 GSM8K figure (3.16x, accepted length 3.43) matches the math row.
- 24 GB Blackwell card at 262k context: 50 → 12.6 tok/s. Long context is real.

## Resources are in hand

The 5090 is rented at $0.591/hr (`docs/environment.md`), so a few hundred hours
is $120–240 total. Every dependency is open: Qwen3.8-27B ships its own MTP head,
RadixArk published the DSpark draft, GDN prefill can borrow
flash-linear-attention's Triton kernels (only the decode single-step needs to be
written), gpt-fast provides the skeleton, and FlashInfer/Marlin-class kernels
exist to compare against.

## How fast can this actually go

**Projection, not measurement.** The numbers below are extrapolated from
component benchmarks on this machine. They are not the output of a working
engine, and every one of them is contingent on the assumptions stated at the end.
Measured rival figures live in `docs/baselines.md`; do not cite these as results.

### The coefficient everything rests on

What fraction of the 1701 GB/s wall can a decode step hold? Measured by building
the real per-layer matmul shapes of Qwen3.8-27B, running them as bs=1 GEMVs
inside one CUDA graph (`scripts/env_check/check_gemv_sol.py`):

```
9 GDN + 3 attention layers, 63 GEMVs, 9.13 GB of weights
eager   : 5.58 ms   1638 GB/s   96.3% of wall
graphed : 5.56 ms   1643 GB/s   96.6% of wall
```

96.6% is the demonstrated ceiling for the dominant term. It is bf16 and
weights-only, so it excludes 4-bit dequantization, GDN state updates, attention,
and sampling. **88–95% is the realistic band; the table below uses 92%.**

### What that yields

Raw decode first. At 4.0 bpw the text path is 13.45 GB, the ceiling is
126.5 tok/s, and 92% of it is **116 tok/s** (90–95%: 114–120).

| Scenario | Token Rush | Rival | Gain |
|---|---|---|---|
| Raw decode, short context, 4.25 bpw | 110 tok/s | llama.cpp 82.8 | +32% |
| Raw decode, short context, 4.0 bpw | 116 tok/s | llama.cpp 82.8 / vLLM 80.0 | +40% / +45% |
| **Raw decode, matched 4.79 bpw** | **97 tok/s** | **llama.cpp 82.8** | **+17%** |
| **Raw decode, matched 18.79 GB (vLLM's bytes)** | **83 tok/s** | **vLLM 80.0** | **+5%** |
| Raw decode, matched 3.92 bpw | 119 tok/s | ExLlamaV3 77.0 | +54% |
| Decode at 200k context, 4.0 bpw | 78 tok/s | llama.cpp 44.3 / vLLM 61.0 | +77% / +28% |

The +40–45% raw rows are almost entirely bytes: we read 13.45 GB where vLLM's
NVFP4 checkpoint reads 18.79 (its `lm_head` is bf16) and llama.cpp reads
16.10. Held to the same bytes the engine itself is worth +5% over vLLM. That
is the fair-comparison row and it must be conceded up front.

**Speculation is where the number comes from.** The step-cost model, from
the component measurements on this machine:

- a raw step at 116 tok/s is 8.6 ms;
- a verify step for K ≤ 8 draft tokens reads the weights once, the same as a
  raw step; the extra K−1 positions cost only KV and recurrent-state work,
  which is bandwidth-trivial, so a verify step is taken as **1.1x a raw
  step** — 9.5 ms — when it is inside the same CUDA graph. The rivals pay
  1.44x (SGLang, 22.8 vs 15.8 ms) and 5.6x (vLLM, 70 vs 12.5 ms) because
  their draft-verify loop runs eager kernels outside the graph;
- the draft: the MTP head (0.21 GB at 4 bits) chained three deep costs ~1 ms
  in-graph; a 4-bit DSpark-class draft with a gamma-7 tree costs 4–6 ms;
- mean accepted length per step, including the bonus token, from the
  measurements in `baselines.md`: MTP chained two deep gets 2.24 / 2.36 /
  2.81 on prose / code / math (ExLlamaV3), DSpark at gamma 7 gets 2.39 /
  3.11 / 4.68 (SGLang). Three-deep MTP is taken as 2.4 / 2.6 / 3.2; a tree
  over a DSpark-class draft as 2.4–3.0 / 3.1–3.6 / 4.7–5.2.

| Effective tok/s | Token Rush | SGLang + DSpark | llama.cpp + MTP | ExLlamaV3 + MTP×2 | vLLM (best = raw) |
|---|---|---|---|---|---|
| prose | **200–240** | 104 (1.9–2.3x) | 130 (1.5–1.85x) | 127 (1.6–1.9x) | 80 (2.5–3x) |
| code | **230–280** | 137 (1.7–2.0x) | 128 (1.8–2.2x) | 142 (1.6–2.0x) | 80 (2.9–3.5x) |
| math | **300–380** | 205 (1.5–1.85x) | 162 (1.85–2.3x) | 160 (1.9–2.4x) | 80 (3.8–4.8x) |
| prose at 200k | **140–180** | does not fit in 32 GB | not measured | not measured | 61 (2.3–3x) |

Absolute optimistic ceiling — 4.0 bpw at 95% of the wall with mean accepted
length 3.5 — is roughly 385 tok/s effective. The realistic good outcome is
**114–120 tok/s raw and 200–240 tok/s effective on prose**, with code and
math above that.

### Which comparison to headline

Three candidates, in order of how well they survive a reviewer:

1. **% of the roofline.** 92% raw against vLLM's 88%, SGLang's 70%,
   llama.cpp's 78%, ExLlamaV3's 60%; and effective tok/s per GB/s of
   bandwidth. Bulletproof, invariant to quantization and to rivals
   improving, and not pretty.
2. **2x SGLang + DSpark on prose.** The project's stated opponent, the best
   speculative rival measured, and the comparison that is most durable:
   SGLang's verify cost is its scheduler's, not a bug. Report prose — it is
   the floor; code and math are higher, so nobody reproducing it lands below
   the claim.
3. **2.5–3x vLLM.** The strongest raw engine, whose best configuration on
   this model is *not speculating*, because every draft it has is slower than
   its raw decode. Large, true today, and perishable: that is an integration
   bug in vLLM's speculative path, not a structural limit, and if it is fixed
   before the writeup the margin shrinks toward 1.5x. Use it, date it, and do
   not build the story on it.

The headline is therefore **"2x the best speculative engine, 2.5–3x the
strongest raw engine, on one RTX 5090, with 256k context usable"**, backed by
the roofline fraction as the number that does not move. Short-context raw
decode is evidence of competence, not the selling point; long context is a
match with vLLM at the wall, plus the fact that a draft model and a 200k KV
cache fit together here and do not in SGLang.

### Is it worth doing

Yes, on three measured facts, none of them assumed: the GDN launch tax is
9.1 ms per token against a 10 ms budget, so a full-step graph alone moves the
raw number; every rival runs its draft-verify loop outside the graph and
pays 1.44x to 5.6x a raw step for it; and at bs=1 the 5090's tensor cores are
idle, so verification is free. Nobody has both a raw decode near the wall
and a speculative loop built for one stream. That gap is real.

The cost is that the number exists only after Phase 2 (full-step graph,
fused GDN step) *and* Phase 3 (speculation inside the graph) — about 6–7
weeks of core work with no standalone milestone in between. Phase 1 alone
yields a token-exact reference and a raw number that, held to matched bytes,
is +5% over vLLM: correct, publishable as a roofline fraction, and not a
resume line. If the time box only reaches the end of Phase 2, the honest
outcome is "92% of the wall against vLLM's 88%".

### What would falsify this

- **4-bit GEMV underperforming.** The 96.6% is bf16. No 4-bit dequant GEMV exists
  for `sm_120` yet, and NVFP4's real behaviour here is the single largest unknown
  on the path. If it holds only 85%, multiply every number above by 0.92.
- **Mean accepted length.** N is the entire lever on the speculation row and it is
  content-dependent — DSpark measures 2.4 on prose, 3.1 on code and 4.7 on
  math here, and llama.cpp's one-token MTP gets +60% / +60% / +100%. 260 tok/s
  assumes 2.5, which is the prose figure; code and math have more. Validate
  early in Phase 3 with the MTP head, which is a weaker draft than DSpark's
  1.86B model.
- **Quantization quality at 4.0 bpw.** The tok/s advantage over llama.cpp comes
  substantially from quantizing harder, so the quality gate in `CLAUDE.md`
  (mean KL to bf16 no worse than ExLlamaV3's 4.00 bpw; GSM8K within noise of
  Q4_K_M) is what makes the byte advantage count. If our quant fails it, the
  choice is a higher bpw and the short-context margin collapses toward the
  +17% row. Measure the rivals' quants with the same script first; the
  threshold is theirs, not a number picked in advance.
- **The verify step costing more than 1.1x a raw step.** The whole
  speculation table rests on it. The risk is concentrated in the 48 GDN
  layers: a K-token verify needs a K-token recurrent step inside the graph,
  `fla`'s fused kernel is unusable here, and the chunk kernel's efficiency
  at T=8 is unmeasured. At SGLang's 1.44x the prose row drops to about 190.
- **Rivals fixing their speculative paths.** vLLM's drafts being slower than
  its raw decode is a bug, not physics. The 2.5–3x row can halve before the
  writeup; the roofline fraction and the SGLang row cannot.

## Risks and honest boundaries

- Hybrid architecture is roughly **2x the engineering** of a standard transformer.
- **GDN decode has no mature single-stream reference to copy.** Correctness rests
  on differential testing against HF and against torch references for every
  kernel — the two-gate definition in `CLAUDE.md`, not token-exactness, which
  a quantized engine cannot meet and which never measured quantization anyway.
  Store recurrent state in FP32.
- **The bar moves.** RadixArk is actively optimizing this model; DSpark will get
  faster. Pin rival versions and date every benchmark.
- Take the free lunch first (launch overhead, fusion). Do not start by chasing the
  last 5% of bandwidth.

## Sources

- Qwen3.6-27B architecture (same 3:1 GDN hybrid as 3.8): https://67ailab.com/posts/qwen36-27b-deep-dive-architecture-efficiency/
- Qwen3.8-27B architecture and context: https://www.mindstudio.ai/blog/qwen3-8-27b-architecture-benchmarks
- Long-context measurements on a 24GB Blackwell card: https://piszczek.pl/blog/qwen38-27b-256k-50-tps-24gb-gpu
- RadixArk DSpark draft: https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark
- RTX 5090 llama.cpp benchmarks: https://www.hardware-corner.net/rtx-5090-llm-benchmarks/
- SGLang speculative decoding docs: https://docs.sglang.io/advanced_features/speculative_decoding.html
- ExLlamaV3: https://github.com/turboderp-org/exllamav3
- gpt-fast: https://github.com/pytorch-labs/gpt-fast
- 5090 rental price comparison: https://getdeploying.com/gpus/nvidia-rtx-5090
