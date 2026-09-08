# Handoff — Phase 0 done and frozen; Phase 1 starts here

For the agent picking this up. Read `CLAUDE.md` first for what the project
is. This document covers only where things stand on 2026-09-08 and what to
do next.

## Where things stand

**Phase 0 is complete and its numbers are frozen.** Every rival in the
`CLAUDE.md` table has a first-party raw, long-context and speculative
measurement on vast machine **94372** (RTX 5090, driver 610, CUDA 13.3), all
in `docs/baselines.md` with the raw logs in
`results/2026-09-04-machine-94372/`. The roofline is anchored at
**1701 GB/s** (`docs/environment.md`) and the projection in
`docs/feasibility.md` is computed against it.

**Decision (2026-09-08): Phase 0 is not re-measured on development
instances.** The 94372 instance is gone; the project now runs on a fresh
instance (container 50295164, 150 GB disk, nothing installed yet). The
bandwidth wall, the launch-tax and GEMV numbers, and every rival's figures
stand as recorded until Phase 4, which re-runs `scripts/env_check/`, every
rival from the recipes in `baselines.md`, and the engine, on one machine on
one day, and produces the numbers the report cites. Until then every
engine speed number is a development number quoted against 1701 GB/s:
enough to decide what to build next, not something to publish. Do not spend
a day re-running rivals when a new instance is rented.

What Phase 0 found, in one paragraph: vLLM at bs=1 holds 88% of the wall
(90% at 200k) with torch.compile plus full-step CUDA graphs, so raw decode is
table stakes, not the headline; every rival's speculative path is weak in
its own way (vLLM's three drafts are all *slower* than its raw decode,
llama.cpp's external drafts do worse than its one-token MTP head, SGLang's
verify step costs 1.45x a raw step); llama.cpp's long-context collapse is
llama.cpp's alone. `fla`'s fused GDN decode kernel is miscompiled on
`sm_120` — the chunk kernel and the minimal Triton step in
`scripts/env_check/check_stack.py` are correct, so the Phase 1 reference
uses those and the Phase 2 fused step is ours as planned.

## The current instance

Container 50295164. Not yet inspected beyond: RTX 5090, driver 610.43.02,
CUDA 13.3 toolkit at `/usr/local/cuda`, `nsys` and `uv` present, **150 GB
disk with 150 GB free**, **60 GB of host RAM**, 32 cores visible,
`/workspace` is not a volume, `/venv/main` has no torch yet. Provision with
the recipe in `docs/environment.md` ("Software") before anything else.

Two things about this box to plan around:

- **60 GB of RAM, not 503.** The HF bf16 reference (55.6 GB) cannot be
  fully host-resident next to anything else. Load it memory-mapped from
  the safetensors on disk and stream layers to the GPU, or run the
  engine-correctness gate layer by layer (see Phase 1 below). Check the
  disk's read bandwidth before deciding.
- **150 GB of disk.** bf16 checkpoint 55.6 + our quant ~14 + venv ~8 leaves
  ~70 GB for rival quants used in the quality table (GGUF 16.5, EXL3 16,
  NVFP4 20) and for logits scratch. It fits if rival checkpoints are
  evaluated one at a time and deleted, not held together.

## What Phase 1 is, as the measurements now define it

From `CLAUDE.md` "The plan", row 1, with what Phase 0 added:

- **Reference implementation**, PyTorch, text path only, passing the
  engine-correctness gate in `CLAUDE.md` against HF `transformers` on
  greedy decode. GDN decode goes through `fla`'s `chunk_gated_delta_rule`
  at T=1 (correct, slow) or the minimal Triton step in `check_stack.py`
  (correct, fast, 30 lines — a fine starting point for the Phase 2
  kernel). **Not** `fused_recurrent_gated_delta_rule`; it returns NaN heads
  on this stack. Keep the differential test in the loop; store recurrent
  state in FP32.
- **Quantization choice at ≤4.25 bpw.** NVFP4 is Blackwell-native and the
  QUASAR checkpoint exists, but it leaves `lm_head` at bf16 (2.5 GB per
  step, the single biggest byte item after the body). ExLlamaV3's 4.00bpw
  quant reads 13.18 GB per step with a 6-bit head — that is the byte budget
  to match or beat. Decide between NVFP4 with a quantized head and int4
  groupwise; `docs/feasibility.md` has the tok/s each buys.
- **First speed number**: raw decode of the reference with the chosen
  quantization, eager, then under one CUDA graph. A development number
  (see above). The bar it has to clear to be interesting is vLLM's 88% of
  the wall, not llama.cpp's 78%.

## Traps

`docs/traps.md` is the list of things that cost time in Phase 0. The one
that cost the most, four times: `pkill -f` matching its own command line.
