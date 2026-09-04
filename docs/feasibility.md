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

- llama.cpp on our own 5090, Qwen3.8-27B UD-Q4_K_M: **80.86 tok/s raw**,
  **91 with MTP** (+17%). Raw decode reads 16.10 GB/token, so that is **80.6% of
  the bandwidth wall**. Measured, not cited — see `docs/baselines.md`. The
  remaining 19% is the raw-decode headroom, and it is thinner than public
  mobile-5090 figures suggest: "+30% over llama.cpp" requires 105 tok/s, which
  at 4.0 bpw is 88% of the wall. The speculation gap is the wider opening.
- SGLang + DSpark (RadixArk, 1.86B draft): bs=1 GSM8K **3.16x**, mean accepted
  length 3.43. Falls to 2.48x at concurrency 8.
- 24 GB Blackwell card at 262k context: 50 → 12.6 tok/s. Long context is real.

## Resources are in hand

The 5090 is rented at $0.428/hr (`docs/environment.md`), so a few hundred hours
is $85–170 total. Every dependency is open: Qwen3.8-27B ships its own MTP head,
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

What fraction of the 1615 GB/s wall can a decode step hold? Measured by building
the real per-layer matmul shapes of Qwen3.8-27B, running them as bs=1 GEMVs
inside one CUDA graph:

```
9 GDN + 3 attention layers, 93 GEMVs, 8.37 GB of weights
eager   : 5.47 ms   1531 GB/s   94.8% of wall
graphed : 5.30 ms   1579 GB/s   97.7% of wall
```

97.7% is the demonstrated ceiling for the dominant term. It is bf16 and
weights-only, so it excludes 4-bit dequantization, GDN state updates, attention,
and sampling. **88–95% is the realistic band; the table below uses 92%.**

### What that yields

| Scenario | Token Rush | llama.cpp | Gain |
|---|---|---|---|
| Raw decode, short context, 4.25 bpw | 104 tok/s | 80.9 | +29% |
| Raw decode, short context, 4.0 bpw | 110 tok/s | 80.9 | +37% |
| **Raw decode, matched 4.79 bpw** | **92 tok/s** | **80.9** | **+14%** |
| Decode at 200k context | 71 tok/s | 40.6 | **+76%** |
| Effective, with speculation (N≈2.5) | 245 tok/s | 91 | **2.7x** |

Absolute optimistic ceiling — 4.0 bpw at 95% of the wall with mean accepted
length 3.5 — is roughly 375 tok/s effective. The realistic good outcome is
**105–110 tok/s raw and 240–290 tok/s effective**.

### Which claims are safe, and which are not

**A 50% margin is not available at short-context raw decode.** Reaching it would
need 4.0 bpw *and* essentially 100% of the wall, above the 97.7% that a pure GEMV
with no other work achieves. That target is physically out of reach.

**The matched-bpw row is the weak point.** Held to llama.cpp's own 4.79 bpw, the
short-context engine win is +14%. A reviewer will ask for exactly that comparison.
Concede it early rather than be caught by it — and note it is a fair-comparison
artifact, not the project's claim.

**Long context and speculation are where 50%+ lives.** The 200k figure asks only
that we hold 92% of the wall at long context, when llama.cpp fails to hold 77%
even at *short* context. That is not a miracle; it is declining to collapse.
Speculation is nearly free at bs=1 because the tensor cores are otherwise idle.

The headline should therefore be long context and effective throughput, with
short-context raw decode presented as evidence of competence rather than as the
selling point.

### What would falsify this

- **4-bit GEMV underperforming.** The 97.7% is bf16. No 4-bit dequant GEMV exists
  for `sm_120` yet, and NVFP4's real behaviour here is the single largest unknown
  on the path. If it holds only 85%, multiply every number above by 0.92.
- **Mean accepted length.** N is the entire lever on the speculation row and it is
  content-dependent — llama.cpp's MTP measured anywhere from +10% to +20% across
  prompts, and DSpark's 3.43 is on GSM8K. Code and long prose will accept less.
  Validate N early in Phase 3; 245 tok/s assumes 2.5.
- **Quantization quality at 4.0 bpw.** The tok/s advantage over llama.cpp comes
  substantially from quantizing harder. If 4.0 bpw is not acceptable in output
  quality, the short-context margin collapses toward the +14% row.

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
- llama.cpp MTP measurements on Qwen3.8-27B: https://github.com/sudoingX/qwen38-mtp
- Long-context measurements on a 24GB Blackwell card: https://piszczek.pl/blog/qwen38-27b-256k-50-tps-24gb-gpu
- RadixArk DSpark draft: https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark
- RTX 5090 llama.cpp benchmarks: https://www.hardware-corner.net/rtx-5090-llm-benchmarks/
- SGLang speculative decoding docs: https://docs.sglang.io/advanced_features/speculative_decoding.html
- ExLlamaV3: https://github.com/turboderp-org/exllamav3
- gpt-fast: https://github.com/pytorch-labs/gpt-fast
- 5090 rental price comparison: https://getdeploying.com/gpus/nvidia-rtx-5090
