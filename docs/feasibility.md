# Feasibility — short version

**Verdict: feasible, and not a gamble.** The win comes from a design point general
engines are unwilling to serve, not from luck. The physics, the memory budget and
the per-rival argument live in `CLAUDE.md`; this file holds what that leaves out.

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

| Scenario | Token Rush | llama.cpp | Gain |
|---|---|---|---|
| Raw decode, short context, 4.25 bpw | 110 tok/s | 82.8 | +32% |
| Raw decode, short context, 4.0 bpw | 116 tok/s | 82.8 | +40% |
| **Raw decode, matched 4.79 bpw** | **97 tok/s** | **82.8** | **+17%** |
| Decode at 200k context, 4.0 bpw | 78 tok/s | 44.3 | **+77%** (vLLM: 61.0, +28%) |
| Effective, with speculation (N≈2.5) | 260 tok/s | 130 | **2.0x** |

Absolute optimistic ceiling — 4.0 bpw at 95% of the wall with mean accepted
length 3.5 — is roughly 385 tok/s effective. The realistic good outcome is
**110–116 tok/s raw and 250–300 tok/s effective**.

### Which claims are safe, and which are not

**A 50% margin is not available at short-context raw decode.** Reaching it would
need 4.0 bpw *and* essentially 100% of the wall, above the 96.6% that a pure GEMV
with no other work achieves. That target is physically out of reach.

**The matched-bpw row is the weak point.** Held to llama.cpp's own 4.79 bpw, the
short-context engine win is +17%; held to vLLM's bytes, it is 92% of the wall
against vLLM's measured 88% — about +5%. A reviewer will ask for exactly that
comparison. Concede it early rather than be caught by it — and note it is a
fair-comparison artifact, not the project's claim.

**Long context is where 50%+ over llama.cpp lives; speculation is where 2x
lives.** The 200k figure asks only that we hold 92% of the wall at long
context, when llama.cpp holds 77% at short context and 60% at 200k. That is
not a miracle; it is declining to collapse — and vLLM already holds 90% there,
so against the best rival the long-context row is a match, not a margin. Speculation is nearly free at bs=1 because the tensor
cores are otherwise idle — but llama.cpp's own MTP already gets +60% on prose
here, so 2x over it needs a mean accepted length of about 2.3 at the projected
raw speed, and the acceptance numbers in `baselines.md` say what is realistic.

The headline should therefore be long context and effective throughput, with
short-context raw decode presented as evidence of competence rather than as the
selling point.

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
  substantially from quantizing harder. If 4.0 bpw is not acceptable in output
  quality, the short-context margin collapses toward the +17% row.

## Risks and honest boundaries

- Hybrid architecture is roughly **2x the engineering** of a standard transformer.
- **GDN decode has no mature single-stream reference to copy.** Correctness rests
  entirely on differential testing against HF. Store recurrent state in FP32.
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
