# Phase 1b, second half: quantization quality — deferred, to be done on a two-GPU box

> **Done 2026-09-09** (`docs/progress.md` step 30): Part 1 measured as written here
> (`bench/quality_corpus.py`, `quality_logits.py`, `quality_sources.py`, `quality_hf.py`,
> `exl3_export.py`); Part 2 step 1 (GPTQ, `tokenrush/gptq.py`) done: KL 0.0546 -> 0.0265.
> The bar (0.0128) is not met; steps 2–4 below are the open decision. The
> numbers below this line are the plan as written before the measurement.

Written 2026-09-08, when Phase 1b's first half (the engine-correctness gate)
passed and the decision was taken to go into Phase 2 first. This is the
work that was skipped, written for the agent that picks it up on a machine
with **two RTX 5090s** (so the 55.6 GB bf16 model holds without offload).
Read `CLAUDE.md` and `docs/progress.md` first.

## Why it is owed

The engine runs on an int4 g128 **round-to-nearest** quantization with no
calibration (`tokenrush/quant.py`), chosen on day 2 to get the model running.
Measured on one 32-token prompt against HF bf16 (`bench/gate_engine.py`):
teacher-forced KL **0.06 nats** mean over 48 steps, top-1 agreement 37/48,
greedy diverges at step 4, residual error 19% by layer 63. A calibrated int4
is typically 0.01–0.03; ExLlamaV3's 4.00 bpw, the bar in `CLAUDE.md`, is
~0.01–0.02. So the current quant is 2–4x worse than it should be, and the
byte advantage the project's speed numbers rest on does not count until the
quality row is met.

Two things are owed, in this order: a **quality table** with a proper
yardstick, then a **better quantization** measured with it.

## Part 1 — the quality table (the yardstick)

Goal: KL divergence to bf16 (mean, p99), WikiText-2 perplexity delta, top-1
agreement, and one downstream task, for our quant(s) and the rivals' quants,
**through the same forward code**, so the table measures weights and not
engines.

### bf16 logits, once

Text: WikiText-2 test (~300k tokens; use 32–64k), plus ~4k tokens of code and
~4k of math prompts (the Phase 0 prompt families in `scripts/rivals/`), in
chunks of 4096. Full-vocab logits in bf16 are 0.5 MB/token: 64k tokens = 32 GB
on disk, fine; or store fp16 log-probs of the top-1024 plus the tail mass and
say so (an approximation of KL; full is better).

On two cards: load bf16 with `device_map="auto"` across both GPUs (about 28 GB
each, no CPU offload), or run our own `StreamingBF16Engine` with the layer
cache pinned across two devices. Either way `bench/hf_reference.py` is the
starting point; block `fla` as it does (HF's fused GDN decode is NaN on
sm_120, `docs/environment.md`). For prefill-only logits HF with fla blocked
is pure torch and slow but fine at 4096-token chunks.

### Every quant through our forward

For each candidate, dequantize its weights to bf16 and run **our** bf16 path
(`StreamingBF16Engine` with a tensor source that dequantizes on the fly, or a
bf16 `Engine` if the box has the VRAM):

| candidate | source | dequant |
|---|---|---|
| ours, int4 g128 RTN | `/workspace/models/Qwen3.8-27B-int4g128` | `tokenrush.quant.dequantize_int4` |
| ours, calibrated (Part 2) | to be produced | same |
| llama.cpp UD-Q4_K_M | `unsloth/Qwen3.8-27B-GGUF` | `gguf` python package (`gguf.dequantize`) — map GGUF names back to ours |
| ExLlamaV3 4.00bpw | `turboderp/Qwen3.8-27B-exl3` rev `4.00bpw` | exllamav3's own dequant (its venv, torch 2.11) — export bf16 safetensors from there |
| NVFP4 (QUASAR-QAT) | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` | fp4 codes x fp8 block scales x global scale, trivial |

Disk: 150 GB holds bf16 (56) + logits (32) + one rival checkpoint (16–20) at a
time; evaluate and delete, do not hold all rivals at once.

Metrics per candidate: KL(bf16 || quant) mean and p99 over all positions;
perplexity on WikiText-2 and the delta to bf16; top-1 agreement; GSM8K on a
200-problem subset with greedy decode (the "downstream task" in `CLAUDE.md`).
Write the table into `docs/progress.md` and the chosen row into the Targets
table in `CLAUDE.md`.

## Part 2 — a better quantization

Cheapest first; each step keeps the engine's packing so the Triton GEMV and
everything else are untouched:

1. **GPTQ or AWQ at int4 g128 asymmetric** with llm-compressor or GPTQModel,
   256–512 calibration sequences of 2048 tokens (general text plus some code).
   Convert the output to our packing (`tokenrush.weights.pack_checkpoint`
   takes bf16 tensors; add a path that takes pre-quantized codes + scales +
   zeros, which GPTQ formats provide directly; zero = mn + 8*scale in our
   convention, see `tokenrush/quant.py`). Expect KL ~0.02. On a 60 GB-RAM
   single-card box this may OOM when the tool stages the bf16 model on CPU;
   with two cards it will not.
   Watch for: the hybrid layers (`linear_attn.*`) must be included in the
   sequential targets; QUASAR did it with ModelOpt, so it is doable.
2. **`lm_head` at 6 or 8 bits** (ExLlamaV3 keeps a 6-bit head). +0.3 GB per
   step, ceiling 124.6 -> ~122 tok/s. Needs a 6/8-bit GEMV variant or bf16
   head (2.5 GB, ceiling ~107: too expensive). Decide by the table.
3. **NVFP4 QAT body from QUASAR** as a format change: 4.5 bpw (+6% bytes),
   needs an fp4 variant of the Triton GEMV. Only if 1+2 miss the bar.
4. **bpw**: 4.0 vs 4.25 moves the ceiling by ~7 tok/s; the table decides
   whether g128 asymmetric (4.25) or a symmetric/g256 (4.0–4.125) variant
   meets the bar.

## Also owed from Phase 1b

- Long-context functional check: needle retrieval at 32k, 128k, 256k, and
  GDN state drift over long prefills vs HF (compare final hidden states after
  a 16k prompt). Folded into Phase 2's attention/FP8-KV work; if not done
  there, do it here.
- The rival byte counts in `docs/baselines.md` likely include the bf16
  embedding table (never streamed): vLLM/SGLang 18.79 GB -> ~16.25, llama.cpp
  16.10 -> ~15.3. Verify from the safetensors/GGUF headers and recompute the
  "% of wall" rows before the writeup (vLLM 88% -> ~76%).

## What exists to build on

- `bench/hf_reference.py`: HF dump (per-layer residuals, prompt logits, greedy
  steps); extend to chunked full-logit dumps.
- `bench/gate_engine.py`: per-layer / logits / teacher-forced comparison and
  a `kl()` helper.
- `tokenrush/model.py::StreamingBF16Engine`, `tokenrush/weights.py::HFTensors`:
  bf16 forward from any name -> tensor source; a dequantizing source is the
  only new piece per rival.
- `tokenrush/quant.py`: the packing (`quantize_int4`, `dequantize_int4`) and
  the GEMV; `pack_checkpoint` in `weights.py`.
