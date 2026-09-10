# Quantization

The standing record of one thread of the project: how the 4-bit weights were
chosen, what each attempt was worth, and what is still open. `docs/progress.md`
tells the story in chronological steps; this file tells it by subject, because
the thread ran from day 2 to the end of Phase 1b and the reasons are worth
keeping in one place.

Reproducing anything here needs no two-GPU box and no state from a dead vast
instance: `scripts/quantize/build.sh` makes the checkpoint, `measure.sh`
measures it, and the calibration and evaluation token ids are in this repo
(`data/quality/*.npz`). The adopted checkpoint itself is published at
[`zyhector/Qwen3.8-27B-TokenRush-int4g128`](https://huggingface.co/zyhector/Qwen3.8-27B-TokenRush-int4g128)
— prefer downloading it over rebuilding, since GPU floating point is not
bit-deterministic and every quality number below was measured on that exact
file.

## Why it matters here, and how much

Single-stream decode is a pure bandwidth problem: `tok/s ≈ 1701 / bytes read
per token`, and the weights are almost all of those bytes. Bits per weight is
therefore the second-largest lever in the project after speculation, and it is
a lever with a price:

| bits/weight | weight bytes | ceiling, empty KV |
|---|---|---|
| 4.00 | 13.45 GB | 126 tok/s |
| 4.25 | 14.29 GB | 119 tok/s |
| 4.50 | 15.13 GB | 112 tok/s |
| 5.00 | 16.81 GB | 101 tok/s |

The 110–120 tok/s target in `CLAUDE.md` only exists at or below ~4.25 bpw. So
the project cannot buy its way out of a quality problem with bits: it has to
be good *at* 4.25.

The quality bar, also from `CLAUDE.md`: **mean KL divergence to bf16 no worse
than ExLlamaV3's 4.00 bpw**, the one rival quantization at our bits, plus GSM8K
within noise of llama.cpp's Q4_K_M. Without that row the byte advantage the
speed numbers rest on does not count — a faster engine that is measurably
dumber has not won anything.

## The format, which never changed

Group-wise asymmetric int4 along the input dimension: per group of 128 weights,
a bf16 scale and a bf16 minimum; each weight a 4-bit code; dequantization is
`w = code * scale + mn` in bf16. That is 4 + 32/128 = **4.25 bits per weight**.
Element `2j` sits in the low nibble of byte `j`, element `2j+1` in the high
nibble (`tokenrush/quant.py`). 401 matrices carry it: 48 GDN layers ×
(`in_proj_qkv`, `in_proj_z`, `out_proj`, gate/up/down), 16 attention layers ×
(q/k/v/o, gate/up/down), and `lm_head`. Everything else stays bf16 — the
embedding (one row read per token), the norms, the conv, the tiny
`in_proj_a/b`, and the MTP head.

**Nothing in this document changed that format**, and that is the point. Every
kernel (the Triton GEMV, the split-K variant, the M-row GEMM, the Marlin port
with its own weight layout), every CUDA graph, both drafts and all 76 tests
depend on the packing and not on how the codes were chosen. So a better
quantizer is a drop-in: build a new checkpoint, point the engine at it, and the
speed is bit-identical. That was worth designing for, and it paid off — see
"speed did not move" below.

## Chapter 1 — round-to-nearest, day 2

The first quantizer was the dumbest thing that fits (`quantize_int4`): per group
of 128, take the min and max of the weights, cut the range into 15 steps, round
each weight to the nearest step. No calibration, no data, three lines of torch.

This was deliberate, and it was right. Phase 1a's milestone was *coherent text
from our own engine*, and a quantizer is not on the critical path to that; the
project's own plan said "the dumbest thing that fits... Quality is Phase 1b's
problem". Round-to-nearest let the engine exist on day 2, and the format it
defined is the format the whole engine was then built on.

What it cost was recorded at the time as a single suspicious number: in the
engine-correctness gate (step 5), the int4 path's teacher-forced KL against HF
bf16 on one 32-token prompt was **0.06 nats**, where a calibrated 4-bit quant
is typically 0.01–0.03. That was enough to know the row was not met and not
enough to know by how much, which is what the yardstick below was for.

## Chapter 2 — building a yardstick before choosing anything

The mistake available here was to try quantizers and compare them on whatever
number came to hand. Instead, Phase 1b built the measurement first, and it was
designed so that it measures *weights* and not *engines*:

**One forward for everybody.** Each candidate is a tensor source
(`bench/quality_sources.py`) that presents its quantized matrices dequantized
to bf16 and everything else from the bf16 checkpoint; all of them then run
through our own eager prefill path — the one the engine-correctness gate
verified against HF transformers in step 5 — with each layer built from the
source on demand, since 55.6 GB of bf16 does not fit on a card. A GGUF's own
llama.cpp forward and an EXL3's own kernels are thus never in the comparison.

**The metric is KL, not accuracy.** For every position, KL(bf16 ‖ candidate)
over the full 248,320-entry vocabulary in fp32, plus top-1 agreement and the
next-token NLL for perplexity. 81,920 positions: WikiText-2 test (16 chunks of
4096), code (torch's `nn/modules`, 2 chunks), math (GSM8K train questions with
worked solutions, 2 chunks; the accuracy task uses the disjoint test split).
The reason for KL over a task score is resolution. A 200-problem GSM8K run has
a noise band of ±3.5 points and cannot separate a good 4-bit quant from a bad
one; KL over 82k positions separates them by a factor of six. Task scores are
kept as a sanity check, not as the metric.

**The floor of the ruler was measured.** HF transformers' own bf16 forward
against our bf16 reference on two chunks per corpus: KL 2.5e-4 to 6.2e-4 mean,
p99 3e-3 to 8e-3, top-1 0.988–0.996, perplexity within ±0.005. Every number
below is at least 15x above that floor, so the ranking is real and not an
artifact of two implementations of the same model disagreeing.

**It is single-card reproducible.** The bf16 reference logits (41 GB) come from
our own streamed forward, ~2 minutes on one card. Only two side-checks need the
two-GPU box: the noise floor above, and any GSM8K run of a *bf16* model (55.6 GB
does not fit one card, and CPU offload would take 40 hours).

## Chapter 3 — what the table said

81,920 positions; bits/weight over the 25.60B weights the engine streams per
token. Rivals were downloaded, evaluated, and deleted one at a time.

| candidate | bits/weight | KL mean | KL p99 | top-1 | WikiText-2 PPL (bf16 = 6.255) |
|---|---|---|---|---|---|
| llama.cpp UD-Q4_K_M (imatrix) | 4.80 | 0.0093 | 0.109 | 0.966 | 6.270 |
| **ExLlamaV3 4.00 bpw — the bar** | **4.10** | **0.0128** | 0.158 | 0.960 | 6.274 |
| NVFP4, QUASAR-QAT (bf16 head) | 5.07 | 0.0231 | 0.273 | 0.943 | 6.369 |
| RedHatAI INT4, AWQ+GPTQ (bf16 head) | 4.71 | 0.0458 | 0.597 | 0.924 | 6.460 |
| ours, RTN | 4.25 | 0.0546 | 0.549 | 0.905 | 6.495 |
| noise floor (HF bf16 vs ours) | 16 | 0.0005 | 0.005 | 0.992 | ±0.005 |

Three things fell out of it.

**Ours was 4.3x the bar**, at *more* bits than the bar (4.25 vs 4.10). The
step-5 estimate of "2–4x" was right in kind and slightly kind in degree.

**Calibration, not bits, is what separates the field.** The GGUF spends 0.55
more bits per weight than we do and is 5.9x better; ExLlamaV3 spends 0.15
*fewer* and is 4.3x better. Both are calibrated; we were not. No pair of rows in
the table is explained by bit count.

**Math is the hardest corpus for every quantization** — KL 2–3x the prose value
for all of them — and code has the lowest median with the heaviest tail (most
positions in code are near-deterministic; the rare uncertain ones are where the
error goes).

Reading each rival's format took some care, and two of the three had a trap:

- **GGUF**: llama.cpp's converter reorders the 48 GDN value heads (its head `j`
  is HF's head `3*(j % 16) + j // 16`; q and k heads keep their order). The
  first attempt gave relative errors of 1.0–1.4 on exactly the four GDN
  projections — a number that means "these are unrelated tensors", not
  "quantization is lossy". Recovering the permutation by matching rows against
  bf16 brought every tensor within its type's expected error. UD-Q4_K_M is also
  mixed per tensor (131 Q5_K, 117 IQ4_XS, 104 Q4_K, 29 Q6_K, 7 Q3_K, 7 IQ4_NL,
  4 IQ3_S, 106 Q8_0), so its 4.80 bpw is an average, and it quantizes the
  embedding (Q4_K) and the head (Q6_K) as well.
- **NVFP4** (compressed-tensors `nvfp4-pack-quantized`): e2m1 codes × e4m3
  block-16 scales ÷ an fp32 global scale, trivial to read; its `lm_head` is
  bf16 and its QAT left the embedding, norms, conv and A/dt bit-identical to
  the originals.
- **EXL3**: a trellis code with two Hadamard transforms and sign vectors, not
  readable by hand. Reconstructed with exllamav3's own `get_weight_tensor` in
  its own venv and exported as bf16 shards (`bench/exl3_export.py`), 56 s for
  the model. 4.005 bits on the body, a 6-bit head, 4.10 overall.

## Chapter 4 — buy or build: the Hub survey

Before writing a quantizer it was worth asking whether the work already
existed. A survey of the Hub found 1,993 repos matching Qwen3.8; 516 tagged as
quantizations of the 27B (211 MLX, 124 GGUF, 79 compressed-tensors, 45 FP8, 30
GPTQ, 23 AWQ, 20 EXL3, 12 AutoRound). No official int4 from Qwen, Intel or
unsloth; about thirty community int4 g128 checkpoints, all symmetric, all
keeping `lm_head` in bf16 (400 of our 401 matrices).

**Converting one is genuinely cheap**, which is why it was worth checking:
their codes are the same 4-bit integers, our `mn = -8 * scale` is exact in bf16
for a symmetric zero point of 8, and none of the g128 repos stores a column
permutation. `tokenrush/convert.py` re-packs a compressed-tensors checkpoint
into ours in a few minutes with no requantization; the reader for the format
lives in `bench/quality_sources.py`. Layouts differ by family (compressed-
tensors packs 8 nibbles per int32 along the input dim with an offset of 8;
GPTQ v1 packs along K and stores `zero - 1`; AutoAWQ interleaves columns
`[0,2,4,6,1,3,5,7]`; AMD Quark uses the AWQ ordering with an explicit zero
point) but every one of them is a re-pack, not a requantization.

So the deciding question was quality, and the strongest candidate by evidence
was `RedHatAI/Qwen3.8-27B-INT4`: AWQ smoothing plus GPTQ, 512 calibration
samples, three published bf16-relative evaluations at 99–101% recovery, 182k
downloads. Measured on this yardstick, with its own bf16 head: **KL 0.0458** —
the level of our uncalibrated RTN body (0.0469).

That result is worth stating carefully, because it is the most interesting
thing the survey produced. Its published evaluations are not wrong: gsm8k_platinum
96.77 vs 95.75, ifeval 91.93 vs 92.24, mmlu_pro 83.45 vs 84.46. A KL of 0.046
simply does not move task accuracies by more than their noise. Two measurements
of "quality" that disagree by this much are measuring different things, and for
an engine whose whole claim is fidelity at speed, the distribution is the thing
that matters.

Reading it right needed one care of its own: **AWQ smoothing folds per-channel
scales into the layer norms and the small projections**, so its "unquantized"
tensors are not the original bf16 values either — its `input_layernorm` differs
from HF's by a factor of 0.74–1.36 per channel (Qwen's norms are `1 + w`, which
is where the first attempt went wrong). Undoing the smoothing puts its
unquantized `in_proj_a` within 0.3% of the original and its quantized matrices
at 0.14–0.16 relative weight error — the GPTQ signature: weight error *above*
RTN's 0.10, output error below. Any conversion must therefore take every
non-quantized tensor from that checkpoint too, not from the base weights.

On layer 0's real input, its matrices are only a little better than
round-to-nearest. `bench/quality_layer_probe.py` measures relative output
error `‖(Ŵ − W)x‖/‖Wx‖` on 16,384 held-out WikiText-2 positions, restricted
to the two matrices whose input genuinely is `rmsnorm(embed[ids])` and so
needs no forward pass — everything deeper would need the residual stream
carried through each candidate's own layers, which is what the KL table
measures anyway:

| layer-0 output error, held-out text | `in_proj_qkv` | `in_proj_z` |
|---|---|---|
| RTN | 0.0316 | 0.0434 |
| RedHatAI (AWQ + GPTQ, smoothing undone) | 0.0237 | 0.0328 |
| our GPTQ | 0.0184 | 0.0254 |
| our GPTQ + MSE | **0.0176** | **0.0243** |

Note the weight-space errors move the other way — RTN 0.101, ours 0.118,
RedHatAI 0.159 — which is the GPTQ trade visible per matrix: worse weights,
better outputs.

Verdict: **build**. Not because the conversion was hard — it took an afternoon
and works — but because nothing on the Hub at our bits beats what a plain GPTQ
gets in twenty minutes.

## Chapter 5 — GPTQ, in our own packing

`tokenrush/gptq.py`, about 200 lines, no external quantization tool.

The idea (Frantar et al., 2022): round-to-nearest treats every weight
independently and ignores what the layer's input actually looks like, so it
spends the same precision on an input channel that is always large and one that
is always near zero. GPTQ minimizes the *output* error `‖WX − QX‖` instead:
collect the Hessian `H = XᵀX` of each linear's input over a calibration set,
quantize columns one at a time, and after each column push its rounding error
onto the not-yet-quantized columns through `H⁻¹` (the OBS update), so later
columns absorb what earlier ones got wrong. The codes come out on the same
uniform grid; they are just not always the nearest one.

Choices made, and why:

- **Sequential.** Layers are quantized in order and each layer's calibration
  input is recomputed through the already-quantized layers before it, so later
  layers compensate for earlier ones' error. This is what makes the run take 17
  minutes rather than 3: two forward passes of the calibration set per layer.
- **Group parameters as `quantize_int4` chooses them**, from the error-updated
  weights, rounded to bf16 *before* the codes are picked — so the codes are
  chosen against exactly the numbers the dequantizer will use.
- **`lm_head` last**, on the final-normed hidden states of the quantized body.
- **Calibration**: 256 sequences × 2048 tokens — 160 WikiText-103 *train*, 64
  from torch's Python sources *outside* `nn/modules`, 32 from GSM8K train rows
  2000 onward. Deliberately disjoint from the evaluation corpora, which use
  WikiText-2 *test*, `nn/modules`, and GSM8K train rows 0–45. The ids are
  committed (`data/quality/calib.npz`, sha256 `bb4e2c1068f2fe90`) because the
  code portion is read from the installed torch and cannot be rebuilt
  bit-for-bit under another version.
- **Damping** 1% of the mean Hessian diagonal, the standard value, for the
  Cholesky.

Two differential tests, both cheap and both worth having: with an **identity
Hessian** GPTQ must reduce to round-to-nearest, and it does, code for code
(this catches sign, ordering and grid bugs in one line); with a **correlated**
Hessian its output error must fall while its weight error rises, which is the
trade it exists to make — measured 0.100 → 0.070 output, 0.100 → 0.157 weight.

Result on the model: **KL 0.0546 → 0.0265**, a factor of 2.06.

## Chapter 6 — the refinements, and which one was kept

Each of these is a 20-minute run, so all of them were measured rather than
argued about.

| variant | bits/weight | KL mean | KL p99 | KL max | top-1 | wiki / code / math KL | PPL Δ wiki |
|---|---|---|---|---|---|---|---|
| RTN | 4.25 | 0.0546 | 0.549 | 18.7 | 0.905 | 0.0535 / 0.0337 / 0.0843 | +0.240 |
| GPTQ | 4.25 | 0.0265 | 0.320 | 18.3 | 0.937 | 0.0246 / 0.0205 / 0.0473 | +0.098 |
| GPTQ + act-order | 4.25 | 0.0254 | 0.324 | 18.6 | 0.940 | 0.0231 / 0.0199 / 0.0498 | +0.121 |
| **GPTQ + MSE range search** | **4.25** | **0.0232** | **0.282** | **6.1** | 0.942 | 0.0221 / 0.0185 / 0.0369 | +0.111 |
| GPTQ, group 64 | 4.50 | 0.0210 | 0.262 | 17.6 | 0.944 | 0.0196 / 0.0165 / 0.0364 | +0.069 |
| ExLlamaV3 (the bar) | 4.10 | 0.0128 | 0.158 | 18.2 | 0.960 | 0.0112 / 0.0105 / 0.0282 | +0.019 |

**Act-order** — quantize columns in order of decreasing Hessian diagonal, so
the most influential inputs are done while the most error budget remains, with
*static groups* so the packing keeps contiguous groups and stores no
permutation. Gains 4% of KL, loses on perplexity, and is not adopted: a real
change would have needed non-contiguous groups, which the format forbids.

**MSE range search** — for each row and group, instead of taking the group's
min and max, try 33 shrink factors from 1.0 down to 0.6 and keep the range that
minimizes `Σ |w − dequant(w)|^2.4`. Clipping a few outlier weights to buy
precision for the rest. **Adopted.** It gains 12% of KL at zero cost in bits or
speed, and its real effect is on the tail: worst-case KL falls from 18.3 to 6.1
and the math corpus from 0.047 to 0.037. It costs a little on WikiText
perplexity (+0.111 vs +0.098 against a bf16 6.255) — the two metrics disagree,
and the choice went to KL and top-1 because they look at the whole distribution
while perplexity looks only at the probability of the one correct token, and
because the target row is written in KL.

**Group 64** — 0.25 more bits buys 20% of KL. Not adopted: it costs ~7 tok/s of
ceiling, and the Marlin kernel that makes the speculative verify step nearly
free is a g128 layout, so a g64 checkpoint would fall back to the Triton
kernels and lose more than it gains.

**A higher-precision head** — bounded, not built. With `--keep-bf16 lm_head`
the GPTQ checkpoint measures 0.0240 against 0.0265, so an 8-bit head can be
worth at most 0.0025 of KL, for +0.15 bpw and a kernel that does not exist yet.
The same attribution on RTN: head-in-bf16 0.0469, body-in-bf16 0.0075, full
0.0546 — the two halves add, and the body is 86% of the RTN loss and 90% of the
GPTQ one.

**Speed did not move**, which is the whole reason the format was held fixed.
Raw decode on one card with nothing else running: the adopted checkpoint
97.4 tok/s at 10.26 ms/step, against the 97.6 recorded on the previous
instance before any of this. (RTN and plain GPTQ measured 95.7 each while
the other card was quantizing — a valid comparison with each other,
identical to the hundredth of a millisecond, and not with the 97.4.) On the
six prompt families with the DFlash2 draft: 227 / 354 / 376 / 124 / 294 /
219 tok/s for the MSE checkpoint against RTN's 218 / 368 / 376 / 117 / 291 /
218, at step times of 13.8–13.9 ms either way. The spread is draft
acceptance on different text, not step cost. All 76 tests pass unchanged.
(Phase 4, on the final machine, `docs/progress.md` step 32: raw 97.5 at
10.25 ms on the same layout, 100.9 on `--backend triton`; the six families
228 / 357 / 378 / 125 / 296 / 220; GSM8K through the engine **194/200 =
97.0%**, against 96.5% here and bf16's 96.0% — the same number inside the
±3.5-point band.)

## Where it stands

The default checkpoint is **int4 g128 GPTQ with the MSE range search**, 4.25
bits per weight, KL 0.0232 to bf16, top-1 0.942, WikiText-2 perplexity 6.365
against bf16's 6.255.

**The GSM8K half of the row is met.** 200 test problems, greedy, chat template
with thinking off, 1024 new tokens, the answer read from the last `\boxed{}`:
bf16 through HF transformers 192/200 = 96.0%, the adopted checkpoint through
the engine 193/200 = 96.5% (plain GPTQ, also 193/200). Paired per problem, bf16 is right where we are wrong on 1,
we are right where bf16 is wrong on 2, and both fail 6 — a net difference of
one problem, well inside a 200-problem run's ±3.5-point band. The row names
llama.cpp's Q4_K_M as the comparison; bf16 is the stronger anchor (that quant
is itself 0.0093 of KL away), so the clause holds a fortiori. This is also
the measurement that most needed the two-GPU box: 55.6 GB of bf16 does not
fit one card, and CPU offload would have taken 40 hours instead of 33
minutes.

**The bar is not met.** ExLlamaV3 is at 0.0128 and nothing in the table above
closes a factor of 1.8. That is now a well-understood gap rather than an open
question: calibration was worth 2.06x and is spent; the range search 1.14x and
is spent; act-order and a better head are worth a few percent between them;
0.25 more bits is worth 20% and costs more speed than it is worth. What is
left is the **codebook**. A uniform grid of 16 levels per group of 128 has a
floor, and both quantizations that beat us are off that grid: ExLlamaV3 uses a
trellis code (a non-uniform, sequence-dependent codebook) at 4.00 bits, and
llama.cpp's Q4_K_M mixes types per tensor with an importance matrix at 4.80.

So the open decision is a choice between two honest positions:

1. **Change the codebook.** A non-uniform grid is not a quantizer setting; it
   is a format and therefore a kernel — the Marlin port, the Triton GEMV, the
   M-row GEMM and their tests all assume `code * scale + mn`. Realistically a
   week, against a project whose remaining work is one measurement week.
2. **Keep uniform int4 and restate the row** with the measured number: at 4.25
   bpw we are at 0.0232 where the best 4-bit specialist is at 0.0128, and say
   so in the writeup rather than claiming parity.

Nothing about the engine's speed claims depends on which is chosen; the byte
count is the same either way. What depends on it is what the project can say
about fidelity.

## Reproducing

```bash
bash scripts/quantize/build.sh                        # bf16 -> the default checkpoint, ~25 min
bash scripts/quantize/measure.sh /workspace/models/Qwen3.8-27B-int4g128-gptq-mse
VARIANT=rtn bash scripts/quantize/build.sh            # the day-2 baseline, 40 s
```

The corpora are in the repo and are the ones every number above used:

| file | contents | sha256 (int32 ids) |
|---|---|---|
| `data/quality/calib.npz` | 256 × 2048, the GPTQ calibration set | `bb4e2c1068f2fe90` |
| `data/quality/chunks.npz` | 16+2+2 × 4096, the evaluation corpora | wiki `d237d726574bfd67`, code `fe83434d50b5477c`, math `7a4d33a248aefb72` |

The builders (`bench/quality_calib.py`, `bench/quality_corpus.py`) are kept for
the record of how the ids were chosen, but re-running them under a different
torch gives different code text and therefore different ids.

Tools, all of them used for the numbers above:

| script | what it does |
|---|---|
| `bench/quality_logits.py` | KL / perplexity / top-1 of one candidate against the bf16 reference; `--keep-bf16 <regex>` for attribution, `--head-from` to mix a rival's body with our head |
| `bench/quality_table.py` | regenerates the comparison tables in this file from `results/quality/*.json` |
| `bench/quality_layer_probe.py` | per-matrix output error on held-out text, and undoes AWQ smoothing |
| `bench/quality_hf.py` | the HF side: GSM8K, and the yardstick's noise floor (needs two cards for a bf16 model) |
| `bench/quality_gsm8k_engine.py` | GSM8K through our own engine, one card |
| `bench/exl3_export.py` | dequantizes an EXL3 checkpoint with its own kernels, in the exl3 venv |
| `tokenrush/gptq.py`, `tokenrush/convert.py` | the quantizer, and the re-packer for compressed-tensors int4 |
| `scripts/check_baselines.py` | verifies the roofline arithmetic in `docs/baselines.md` |

The published checkpoint's card is `docs/model_card.md` (the copy uploaded as
its `README.md`); the metadata that records how it was made travels with it as
`tokenrush.json`, and a copy is kept here as
`results/quality/checkpoint_gptq_mse_tokenrush.json`.

To measure a rival, point `bench/quality_logits.py` at a source spec:
`packed:<dir>` (ours), `gguf:<file>`, `nvfp4:<dir>`, `ct:<dir>`
(compressed-tensors int4), `bf16dir:<dir>` (an EXL3 export from
`bench/exl3_export.py`, which runs in the exl3 venv). Per-candidate results,
including every GSM8K answer, are in `results/quality/*.json`.

## Traps paid for in this thread

- **A rival's converter may permute heads.** A relative error near 1.0 on some
  tensors and 0.08 on the rest is a layout difference, not a quality
  difference. Match rows against the bf16 weights before believing anything.
- **A rival's *unquantized* tensors may also differ.** AWQ smoothing folds
  per-channel scales into layer norms and small projections; a checkpoint like
  that must be read whole, never overlaid on the base model's norms. Qwen's
  RMSNorm gain is `1 + w`, and undoing a smoothing with `w` instead of `1 + w`
  produces infinities that look like a broken reader.
- **Published task-accuracy recoveries can hide a 5x KL difference.** They are
  a floor ("the model still works"), not a ranking.
- **Cap the generation length by looking at the outputs.** A 512-token limit
  truncated a third of the GSM8K answers and read as 65% accuracy; the same
  checkpoint scores 96.5% at 1024.
- **Save the expensive intermediate before the cheap step that can fail.** 17
  minutes of GPTQ were lost to a full disk during the final `save_file`; the
  quantizer now writes its codes to `gptq_codes.pt` first and can re-pack from
  them (`--from-codes`).
- **`grep` in a pipeline to a log file buffers.** Without `--line-buffered` a
  long run looks hung for half an hour and then prints everything at once.
