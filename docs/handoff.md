# Handoff — Phase 0 is done; the disk is not

For the agent picking this up. Read `CLAUDE.md` first for what the project
is. This document covers only where things stand on 2026-09-04, what is
blocking, and what to do next.

## Where things stand

**Phase 0 is complete.** Every rival in the `CLAUDE.md` table has a
first-party raw, long-context and speculative measurement on this machine
(vast **94372**, RTX 5090, driver 610, CUDA 13.3), all in `docs/baselines.md`
with the raw logs in `results/2026-09-04-machine-94372/`. The roofline is
re-anchored at **1701 GB/s** (`docs/environment.md`), and the projection in
`docs/feasibility.md` is recomputed against it. Nothing here needs
re-measuring unless a rival version changes.

The findings that changed the plan, in one paragraph: vLLM at bs=1 holds 88%
of the wall (90% at 200k) with torch.compile plus full-step CUDA graphs, so
raw decode is table stakes, not the headline; every rival's speculative path
is weak in its own way (vLLM's three drafts are all *slower* than its raw
decode, llama.cpp's external drafts do worse than its one-token MTP head,
SGLang's verify step costs 1.45x a raw step); llama.cpp's long-context
collapse is llama.cpp's alone. `fla`'s fused GDN decode kernel is miscompiled
on `sm_120` (NaN heads, deterministic per call sequence) — the chunk kernel
and a minimal Triton step in `scripts/env_check/check_stack.py` are correct,
so the Phase 1 reference uses the chunk kernel and the Phase 2 fused step is
ours as planned.

**Git**: the commits since `4184678` ("Hand off to a CUDA 13 machine") are
**not pushed**. Push first; the instance does not persist.

## The blocker: 100 GB of disk, 2.8 GB free

`/workspace` is not a volume here and the overlay is 100 GB. What is on it:

| | GB | Needed for |
|---|---|---|
| `/workspace/models/Qwen3.8-27B-NVFP4` (QUASAR) | 20 | SGLang / vLLM re-runs only |
| `/workspace/models/Qwen3.8-27B-GGUF` (UD-Q4_K_M) | 16 | llama.cpp re-runs; the daily-driver chat setup |
| `/workspace/models/Qwen3.8-27B-exl3-4.0` | 16 | ExLlamaV3 re-runs only |
| `/workspace/models/Qwen3.8-27B-{DSpark,DSpark-GGUF,DFlash2,DFlash2-GGUF}` | 13 | draft models, re-runs only |
| `/workspace/venvs/sglang`, `/workspace/venvs/vllm` | 17 | rival re-runs only |
| `/usr/local/cuda-12.8` | 6.6 | nothing — the image's toolkit; 13.3 is installed and used. Untested whether vast's tooling links against it; check `ldd` on `/opt/instance-tools` binaries before removing |
| `/venv` (torch 2.14+cu130, fla, transformers) | 5.8 | **Phase 1** |
| `/usr/local/cuda-13.3`, `/opt/nvidia` (nsys), `/usr/local/lib/ollama` | 10 | Phase 1–2 (toolkit, profiler); ollama can go |
| `/workspace/rivals/llama.cpp` (source + build) | 0.9 | keep |

Everything under `/workspace/models` and `/workspace/venvs` re-downloads or
reinstalls in minutes from the recipes in `baselines.md` and
`environment.md`; deleting it loses nothing but time.

**Phase 1 needs more than can be freed.** The token-exact reference requires
the bf16 checkpoint `Qwen/Qwen3.8-27B` — **55.6 GB** — plus whatever we
quantize from it (13–15 GB at 4.0–4.5 bpw), plus room to work. That is ~75 GB
of new data on a disk that holds 100 GB with 64 GB of rival weights on it.
Even with every rival deleted, the box has ~65 GB free, which fits the bf16
checkpoint with no room for a second quantization or for re-running any
rival for the Phase 4 matrix.

Options, for the person whose decision it is:

1. **A new instance with a bigger disk.** vast disk is set at creation and
   cannot grow. 300 GB holds everything comfortably (bf16 + our quants + all
   rivals + all venvs). Cost: rebuilding this environment, ~1 hour of mostly
   waiting; the recipes exist and were followed once already today. Filter
   for a host with `NVreg_RestrictProfilingToAdminUsers=0` while at it —
   `ncu` has been blocked on both hosts so far (`check_env.sh` reports it).
2. **A vast volume** attached to a new instance (`/workspace` becomes
   persistent and survives recycle/destroy). Same rebuild cost, and the
   weights stop being at risk. Volumes are per-machine; check availability
   on the target host.
3. **Stay, delete the rivals, live with 65 GB.** Phase 1 fits; the Phase 4
   benchmark matrix would then be re-downloading rivals one at a time. Works,
   but the fair-comparison runs get slow and error-prone.
4. **Reduce what Phase 1 needs.** Drop the vision tower and MTP head shards
   from the bf16 download (`--include` the text-only safetensors, ~53 GB —
   the split is in `environment.md`), and quantize layer-by-layer without a
   second full copy. Saves a few GB, not the problem.

Whatever is chosen: **push the repo before touching the instance**, and if
moving, run `scripts/env_check/` on the new box first and update
`docs/environment.md` — the bandwidth wall is per card, and every "% of wall"
number depends on it.

## What Phase 1 is, as the measurements now define it

From `CLAUDE.md` "The plan", row 1, with what Phase 0 added:

- **Reference implementation**, PyTorch, text path only, token-exact against
  HF `transformers` 5.16 on greedy decode. GDN decode goes through `fla`'s
  `chunk_gated_delta_rule` at T=1 (correct, slow) or the minimal Triton step
  in `check_stack.py` (correct, fast, 30 lines — a fine starting point for
  the Phase 2 kernel). **Not** `fused_recurrent_gated_delta_rule`; it returns
  NaN heads on this stack. Keep the differential test in the loop; store
  recurrent state in FP32.
- **Quantization choice at ≤4.25 bpw.** NVFP4 is Blackwell-native and the
  QUASAR checkpoint exists, but it leaves `lm_head` at bf16 (2.5 GB per step,
  the single biggest byte item after the body). ExLlamaV3's 4.00bpw quant
  reads 13.18 GB per step with a 6-bit head — that is the byte budget to
  match or beat. Decide between NVFP4 with a quantized head and int4
  groupwise; `docs/feasibility.md` has the tok/s each buys.
- **First speed number**: raw decode of the reference with the chosen
  quantization, eager, then under one CUDA graph. The bar it has to clear to
  be interesting is vLLM's 88% of the wall, not llama.cpp's 78%.

## Instance state worth knowing

- Servers: none running. ollama is installed (`/usr/local/bin/ollama`) with
  no models; its runner holds VRAM for 10 minutes after a request
  (`docs/traps.md`).
- `/workspace/models/Qwen3.8-27B-DSpark-ct/` is a symlink copy of the DSpark
  draft with a patched `config.json` — the only way vLLM 0.28 loads it
  (`baselines.md`, vLLM section).
- `/workspace/venvs/exl3` was deleted to make room; the ExLlamaV3 recipe is in
  `baselines.md`.
- `docs/traps.md` is the list of things that cost time today. The one that
  cost the most, four times: `pkill -f` matching its own command line.
