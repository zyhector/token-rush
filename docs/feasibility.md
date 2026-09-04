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
