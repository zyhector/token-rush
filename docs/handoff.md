# Handoff — 2026-09-09, end of the single-5090 session

Written at the session switch. The next session runs on a **two-GPU box**
(two RTX 5090s) to do Phase 1b; read this, then delete the file (handoffs
exist only at a switch; `docs/progress.md` is the record).

Read first: `CLAUDE.md`, `docs/progress.md` ("Where things stand" and
"Next"), `docs/quality_plan.md` (the job), `docs/environment.md` (what the
card and the stack do and do not do), `docs/traps.md`.

## State

Phases 0 (frozen), 1a, 2 and 3 are done; 1b is half done (engine-correctness
gate passed; quality table and calibrated quantization owed); 4 not started.
Last commit on `main` is the Phase 3 close (step 29). 76 tests pass.

Development numbers on the last instance (vast 50295164, one RTX 5090;
every one of them is re-measured in Phase 4):

| | |
|---|---|
| raw decode | 102 tok/s on the Triton GEMV (`--backend triton`), 97.6 on the default Marlin layout; 71.7 at 200k with fp8 KV |
| speculative greedy, prose / code / math | 224 / 373 / 373 (DFlash2), 213 / 306 / 286 (MTP chain) |
| Chinese essay / math / mixed | 179 / 310 / 251 (MTP chain; `--draft auto` picks it for CJK prompts) |
| sampled T=0.7 top-p 0.9 | 211 / 391 / 358 |
| at 200k, prose / code | 211 (MTP) / 238 (DFlash2) |
| context | 256k usable, ~30 GB peak with both drafts and fp8 KV |
| quantization | int4 g128 RTN, uncalibrated, KL 0.06 to bf16: **the quality row is not met** — that is the job |

## Rules that stand

- **Phase 0 is frozen** on machine 94372; do not re-measure rivals or the
  wall until Phase 4, which does all of it on one machine in one sitting.
- `docs/progress.md` is the primary record: one numbered step per piece of
  work with a long write-up and the numbers it produced; `docs/progress.csv`
  gets one row per step (for the plot). Every small step is logged.
- The default configuration is simply the fastest; do not over-consider cases.
- The user pushes to GitHub; local commits are fine.
- Correctness gates: bf16 path vs HF; spec == raw greedy with shared kernels;
  every kernel differential-tested; graph replay bit-exact.

## Setting up the two-GPU box

1. Stack as in `docs/environment.md` (torch 2.14+cu130, triton 3.8,
   transformers 5.16, fla 0.6 — block fla's fused GDN decode kernel, it is
   miscompiled on sm_120; `scripts/env_check/` re-anchors the wall if wanted,
   but Phase 0 numbers stay frozen). The Marlin extension
   (`tokenrush/csrc/`) needs `nvcc` (13.x) and `ninja`; it builds on first
   import (~10 s, cached in `tokenrush/csrc/build/`). If the box has no
   nvcc, `--backend triton` runs everything without it.
2. Models under `/workspace/models/`: `Qwen3.8-27B` (bf16 HF, public), the
   packed `Qwen3.8-27B-int4g128` (`python -m tokenrush.quantize`, see
   `docs/progress.md` step 1), `Qwen3.8-27B-DFlash2` (z-lab's draft, step 24
   records the source), and for Phase 1b the rivals' quants listed in
   `docs/quality_plan.md`. Corpora used by the benches (`/workspace/data/`:
   WikiText-103 prose, torch's Python sources as code, Chinese Wikipedia) are
   rebuilt as steps 17 and 21 describe.
3. Smoke: `python -m pytest tests -q`; `python -m tokenrush.run --model
   /workspace/models/Qwen3.8-27B-int4g128 --chat --prompt "..."`;
   `python bench/families.py --model ...` reproduces the family table.

## The job: Phase 1b (docs/quality_plan.md)

Part 1, the quality table: bf16 logits once (the model fits across two
cards without offload), then every quant through our own bf16 forward:
KL mean / p99, WikiText-2 perplexity delta, top-1 agreement, a GSM8K subset.
Part 2, a calibrated int4 (GPTQ or AWQ, g128 asymmetric) converted to our
packing — the format is unchanged, so the Marlin backend, the graphs and the
tests all carry over; re-run `bench/decode.py` and `bench/families.py`
afterwards to confirm the speed did not move. Write the table into
`docs/progress.md` (a new step) and the chosen row into the Targets table
of `CLAUDE.md`. Tell the user when the two cards are no longer needed:
Phase 4 runs on one.

## Then Phase 4

One machine, one day: `scripts/env_check/` for the wall, every rival from the
recipes in `docs/baselines.md`, the engine on the final checkpoint — raw
(`--backend triton` for the raw row; state which layout the row used),
speculative on the six families, sampled, 200k, 256k needle. Only those
numbers are cited in the writeup.

## Optional, not blocking

Phase 3 items left open (step 29): stochastic drafts with residual sampling;
the 64-row attention tile at long context; partial-mode Marlin on the wide
shapes; a Marlin-layout M=1 kernel to recover raw decode's 4%.
