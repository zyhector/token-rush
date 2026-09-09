# Handoff — 2026-09-09, into Phase 4

Written at the switch from the two-GPU box to a single-card machine. The next
session runs **Phase 4: the final measurement**. Read this, then delete the
file — handoffs exist only at a switch; `docs/progress.md` is the record.

Read first: `CLAUDE.md`, `docs/progress.md` ("Where things stand" and "Next"),
`docs/baselines.md` (the recipes for every rival, and the correction note at
the top), `docs/environment.md`, `docs/traps.md`. `docs/quantization.md` only
if the quantization question comes up again — it is closed.

## State

Phases 0 (frozen), 1a, 1b, 2 and 3 are **done**. Phase 4 is the only work
left. Last commits on `main` are the Phase 1b close and the byte-count
correction. 76 tests pass.

Development numbers on the last box (two RTX 5090s; every one of them is
re-measured in Phase 4):

| | |
|---|---|
| raw decode | 97.4 tok/s, 10.26 ms/step, 78.2% of the wall (13.65 GB/step, ceiling 124.6); 102 on `--backend triton` on the earlier instance; 71.7 at 200k with fp8 KV = 85.6% |
| speculative greedy, prose / code / math | 227 / 354 / 376 (DFlash2), 204 / 255 / 282 (MTP chain) |
| Chinese essay / math / mixed | 160 / 278 / 225 (MTP chain; `--draft auto` picks it for CJK) |
| sampled T=0.7 top-p 0.9 | 211 / 391 / 358 (measured on the previous checkpoint; re-measure) |
| at 200k, prose / code | 211 (MTP) / 238 (DFlash2) |
| context | 256k usable, ~30 GB peak with both drafts and fp8 KV |
| quantization | int4 g128 GPTQ + MSE, 4.25 bpw, KL 0.0232 to bf16, GSM8K 96.5% vs bf16's 96.0% |

## The checkpoint

**Published at
[`zyhector/Qwen3.8-27B-TokenRush-int4g128`](https://huggingface.co/zyhector/Qwen3.8-27B-TokenRush-int4g128),
17.04 GB.** Download it:

```
hf download zyhector/Qwen3.8-27B-TokenRush-int4g128 --local-dir /workspace/models/Qwen3.8-27B-int4g128-gptq-mse
```

**Prefer this over rebuilding.** `bash scripts/quantize/build.sh` reproduces
it in ~25 minutes on one card and the calibration ids are committed, so the
recipe is deterministic in intent — but GPU floating point is not
bit-deterministic, so a rebuild is a statistical sibling, not the same file,
and every quality number in `docs/quantization.md` was measured on the
published one.

Also needed: `z-lab/Qwen3.8-27B-DFlash2` (the draft, 3.9 GB) at
`/workspace/models/Qwen3.8-27B-DFlash2`. The MTP head ships inside our packed
checkpoint.

## Setting up the machine

One RTX 5090 is enough. **Ask for ~300 GB of disk**: Phase 4 needs our
checkpoint (17 GB), the DFlash draft (4 GB), four rival stacks in their own
venvs (~20 GB) and the rivals' weights (GGUF 16 + NVFP4 20 + EXL3 16 + the
DSpark and DFlash GGUF drafts ~7). It does **not** need the bf16 model or the
41 GB of reference logits — those were the two-GPU work and it is finished.

1. Stack as in `docs/environment.md` (torch 2.14+cu130, triton 3.8,
   transformers 5.17, fla 0.6 — block fla's fused GDN decode kernel, it is
   miscompiled on sm_120). The Marlin extension (`tokenrush/csrc/`) needs
   `nvcc` 13.x and `ninja` and builds on first import (~10 s); without nvcc,
   `--backend triton` runs everything.
2. Rival stacks and weights: every recipe is in `docs/baselines.md`
   ("Reproducing" under each engine). Disk fills fast — measure one rival,
   delete it, move on, and read `docs/traps.md` before serving any of them on
   32 GB (SGLang's mamba cache sizing, the prefill-graph memory, ollama
   holding VRAM after a run, `pkill` matching your own shell).
3. Smoke: `python -m pytest tests -q` (76 pass); `python -m tokenrush.run
   --model <ckpt> --chat --prompt "..."`; `python bench/families.py --model
   <ckpt>` reproduces the family table.

## The job: Phase 4

**One machine, one day, one sitting.** Everything cited in the writeup comes
from this run; nothing is mixed with a development number.

1. `scripts/env_check/` — re-anchor the wall (`check_bandwidth.py` is
   best-of, never mean; a cold run reads 30% low) and record the machine in a
   new `results/<date>-machine-<id>/`. Update `docs/environment.md` with this
   machine's measurements, keeping the Phase 0 section clearly labelled as
   the historical one.
2. Every rival from the recipes in `docs/baselines.md`: llama.cpp raw and
   +MTP, vLLM raw (its best config), SGLang + DSpark, ExLlamaV3, ollama as
   the floor. Same prompts, same contexts.
3. The engine on the final checkpoint: raw (state which backend and layout
   the row used — `--backend triton` is ~4% faster at M=1, Marlin serves the
   speculative step), speculative on the six families, sampled at T=0.7,
   200k, and the 256k needle.
4. **Byte counts from the headers, not from file sizes.** The embedding table
   is never streamed; counting it once cost this project a wrong "vLLM is at
   88% of the wall" in every doc until 2026-09-09. `bench/decode.py` does it
   right for us; the note at the top of `docs/baselines.md` and
   `bench/quality_sources.py::streamed_bpw` show how for the others.
5. Only then the benchmark matrix and the writeup.

## What the writeup has to say plainly

- **The quality row is half met.** GSM8K is within noise of bf16 (96.5% vs
  96.0% on 200 problems). The KL row is not: 0.0232 against ExLlamaV3's
  0.0128 at 4.25 bpw against its 4.10. The remaining gap is the uniform int4
  codebook, not the calibration, and closing it was judged not worth a week
  against a project whose claim is speed (`docs/quantization.md`, and step 30
  for the decision).
- **Raw decode is table stakes, and the fair row must be conceded up front.**
  On corrected byte counts our engine holds 78.2% of the wall against vLLM's
  76.1%: two points of engine efficiency. The raw tok/s margin comes from
  streaming 13.65 GB where vLLM streams 16.25.
- **The headline is speculation**: 227 / 354 / 376 effective against the best
  rival per family (llama.cpp+MTP 130, SGLang+DSpark 137 / 205).
- **Long context is a margin, not a caveat**: 85% of the wall at 200k against
  vLLM's 81%, SGLang's 66% and llama.cpp's 58%, while being 256k-usable.

## Queued for the next session: make the Hub the default route

**Asked for on 2026-09-09; do this before Phase 4's measurement runs, it is
half an hour.** `tokenrush/run.py` still treats a local rebuild as the way to
get weights. Its `--model` is `required=True` with the help text "packed
checkpoint (see tokenrush.quantize)", and when the path is not a packed
checkpoint it exits with:

```
{path} is not a packed checkpoint; run python -m tokenrush.quantize first
```

That sends the reader down the 25-minute rebuild for a file that downloads in
four, and `tokenrush.quantize` is the *uncalibrated* RTN packer, which is not
the checkpoint anyone should be running. What it should do instead:

- default `--model` to the published repo id
  (`zyhector/Qwen3.8-27B-TokenRush-int4g128`) and accept either a repo id or
  a local directory, resolving a repo id through `huggingface_hub`
  (`snapshot_download`, which caches, so a second run costs nothing);
- when a local path is missing or is not packed, name the download command in
  the error rather than the quantizer;
- decide whether to auto-download or only print the command — auto is
  friendlier, printing is more honest about a 17 GB transfer. My inclination
  is auto with a one-line notice of the size, and `--no-download` to opt out.

The same `--model` handling is worth giving `bench/decode.py`,
`bench/families.py` and `bench/quality_gsm8k_engine.py`, which are the three
that Phase 4 runs; the other benches can keep `required=True`. Note that
`bench/draft_vocab.py` still defaults to the old RTN path
(`/workspace/models/Qwen3.8-27B-int4g128`) and should be updated with them.

## Optional, not blocking

From Phase 3 (step 29): stochastic drafts with residual sampling; the 64-row
attention tile at long context; partial-mode Marlin on the wide shapes; a
Marlin-layout M=1 kernel to recover raw decode's 4%. From Phase 1b: the GDN
recurrent-state drift over a long prefill against HF, which needle retrieval
at 128k and 256k already covers functionally.
