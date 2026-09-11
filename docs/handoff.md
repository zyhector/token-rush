# Handoff — re-measure the speculative decode-vs-context sweep

For the agent picking this up. Read `CLAUDE.md` first for what the project
is. This document covers one thing: the README's decode-vs-context figure
(`docs/img/decode_vs_context.svg`, drawn by `scripts/plot_sweep.py` from
`results/2026-09-09-machine-36542/sweep10/sweep10.csv`) has speculative
curves that zig-zag, the zig-zag is a measurement artefact, and the sweep
needs re-running under a protocol that averages it out. Everything else is
done (`docs/progress.md`, "Where things stand").

## Why the speculative curves wobble

Speculative tok/s is the product of two quantities: accepted tokens per
verify step, and steps per second. Split the DFlash2 prose row of the sweep:

| context | ms/step | accepted/step | tok/s |
|---|---|---|---|
| 0 | 13.8 | 3.85 | 279 |
| 32k | 15.4 | 2.52 | 164 |
| 128k | 19.8 | 2.38 | 120 |
| 200k | 23.1 | 4.36 | 189 |
| 240k | 24.9 | 3.01 | 121 |

The step cost is monotone and smooth from 13.8 to 24.9 ms — that is the
engine's property. All of the wobble is in the acceptance column, which is
the text's property: `bench/spec_context.py` takes the first N tokens of
one WikiText-103 file as the prompt and generates **200 greedy tokens at
position N**, so every context point is a different spot in the corpus,
sampled over only 50–70 verify steps. The log shows what each spot was:
200k lands on "The club competed in League One, the third tier" (formulaic,
4.36 accepted), 96k on "iva and his consort Parvati) and Bhairava" (Hindu
names, 2.38), 240k on a `= = Professional career = =` section heading.

Evidence that it is the text and not the engine:

- Two unrelated drafters (DFlash2 and the MTP chain) dip and rise at the
  same contexts — both low from 32k to 128k, both high at 200k.
- The Phase 4 protocol run measured 200k at a second corpus position
  (`engine/spec_context_prose_shift_dflash.log`): acceptance 4.53 vs 4.36,
  step cost 23.1 ms at both. Moving the position moved only acceptance.
- The rivals wobble the same way on the same prompts: llama.cpp + MTP is
  faster at 16k than at 8k, ExLlamaV3 + MTP faster at 32k than at 16k.
- Greedy speculative output is verified token-identical to greedy raw
  output, so the acceptance counts are honest.

The raw rows are unaffected (raw decode does not depend on the text) and
need no re-measurement.

## How the field measures this

Nobody plots one 200-token continuation per context length. The long-context
speculative-decoding papers (MagicDec, TriForce, SpecPV) cut PG-19 books into
segments of each target length, continue each for 96–1024 tokens greedily,
and report mean acceptance length and mean tok/s over many segments per
length; MagicDec shows acceptance flat from 4k to 100k once averaged. LongSpec
picks LongBench tasks with long outputs (GovReport, QMSum, Multi-News, LCC,
RepoBench-P) because short outputs cannot measure speedup fairly. OWL built
LongSpecBench (4k–64k real conversations) specifically because acceptance vs
context is itself a property worth measuring, and it takes many samples to
see. Spec-Bench averages over hundreds of prompts and three runs. On the
engine side, `llama-bench` does not support speculation at all and its
maintainers note random-token contexts cannot measure acceptance; SGLang's
DFlash blog averages acceptance per request over whole datasets.

## The protocol to run

1. **Corpus.** Prose: **PG-19** (continuous books, the standard for this
   figure) instead of WikiText-103, whose article boundaries and `= =`
   headings are their own noise source. Code: keep the torch sources from
   `bench/long_text.py`. Add a PG-19 branch to `bench/long_text.py`; the
   file must be ≥ 260k tokens.
2. **Positions.** For each context length N in the ten-point grid, take
   **P = 5–8 fixed positions** p spread through the corpus (fixed offsets,
   recorded in the log), prompt = the N tokens ending at p, and generate
   **512 greedy tokens** at p. Same positions for every N and for every
   draft, so the drafters are compared on identical text. Add `--positions`
   and `--new 512` to `bench/spec_context.py`; the raw measurement can stay
   at one position (it does not depend on the text).
3. **Aggregation.** tok/s per context = total generated tokens / total
   decode seconds over the P positions (not the mean of per-position
   ratios). Record mean, min and max of accepted/step and of tok/s per
   context; `scripts/sweep_table.py` gains those columns.
4. **Rivals.** llama.cpp + MTP on the same prompt files at the same
   positions (its prompt files are cut with the HF tokenizer in
   `results/.../scripts/llama_stage.sh`). ExLlamaV3 + MTP runs on random
   tokens in its own harness and cannot follow this protocol; drop it from
   the figure and say so in the caption.
5. **Cost.** Prefill dominates: the ten contexts sum to ~944k prompt tokens
   per position, ~10 min at 1500 tok/s, plus ~2 min of decode. Eight
   positions × two drafts ≈ 3 h on a rented 5090; llama.cpp + MTP adds
   ~1.5 h. One night. The Phase 4 host recipe is in `docs/environment.md`
   and `docs/baselines.md`.

## What the result should look like

- **Accepted/step vs context: flat.** Both drafts accept about the same
  number of tokens at 240k as at 0 once averaged over positions (the sweep's
  ten single-position samples average ~3.0 for both drafts on prose, ~3.8
  MTP / ~4.9 DFlash2 on code). If a draft's acceptance genuinely drifts with
  context (DFlash2's context cache is a 2048-key window, OWL's finding for
  EAGLE3), this figure is where it shows, and it is a real finding, not noise.
- **tok/s vs context: a smooth, gently falling curve**, shaped by the step
  cost alone: prose ≈ 3.0 × 1000 / ms(ctx), i.e. DFlash2 from ~220 at 0 to
  ~120 at 240k, the MTP chain from ~240 to ~160; code ≈ 350 → 200 (DFlash2).
  The spread band across positions should be narrow — a few percent — and
  the raw curves underneath unchanged.
- **The crossover to settle.** In the single-position sweep the MTP chain is
  ahead of DFlash2 on prose at every context ≥ 16k (198 vs 189 at 200k, 160
  vs 121 at 240k), because its step grows less with context (12.6 → 18.5 ms
  against 13.8 → 24.9 ms) while acceptance is similar. `--draft auto`
  currently picks by script only (CJK → MTP chain, else DFlash2; step 29);
  if the averaged curves confirm the crossover, add a context-length term to
  `spec.pick_draft` and document the rule in `progress.md` and `CLAUDE.md`.
- **The figure.** `scripts/plot_sweep.py` then reads the new CSV unchanged
  for the mean lines; add a 10%-opacity same-hue band for the per-position
  range, drop the "hence the wobble" sentence from the caption, and re-render
  light and dark. Delete this file when that is done.
