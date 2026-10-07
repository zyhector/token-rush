# Token Rush on Apple Silicon

The same engine on a Mac, through MLX: `tokenrush/mlx/`. Same model, same
published int4 checkpoint, same drafts (the MTP head and DFlash2), same serving
layer. Measured on one machine, a MacBook Pro with an **M5 Pro and 64 GB**,
macOS 26.4, mlx 0.32.3, 2026-10-07. Development numbers from one laptop, not a
Phase 4 style sitting: the machine heats under sustained load and repeated runs
of the same configuration drift by 10-15% (see "Heat" below).

```
uv sync --extra mac-bench                     # mlx, CPU torch (the shared serving code imports it), mlx-lm for the benches
uv run python -m tokenrush.serve              # picks the MLX backend on macOS; 32k context by default
uv run --extra bench pytest tests/mac -q      # kernels, engine and session on a small random model
```

## The wall

`bench/mac/bandwidth.py`: a streaming reduction over a 4 GB buffer reads at
**291 GB/s**, 1/5.8 of the 5090's 1701. The text path streams 13.65 GB per token
(the embedding table excluded), so **the raw-decode ceiling is 21.3 tok/s**.

MLX's own int4 matmul at M = 1 already streams at 95-97% of that at every
layer shape, in-graph over more than 1 GB of distinct weights per shape. As on
the 5090, dense weight streaming is solved before we start.

## No conversion

Our packing (element 2j in the low nibble of byte j) read as little-endian
uint32 is MLX's affine 4-bit layout, and both dequantize as `q * scale + mn`.
The published checkpoint loads with a `view(uint32)` per matrix: no
requantization, the GPTQ codes and every quality number in
`docs/quantization.md` carried over unchanged (`tokenrush/mlx/weights.py`).

## Correctness

The reference is mlx-lm's Qwen3.5 implementation fed the same int4 bytes, built
in memory (`bench/mac/mlx_lm_ref.py`): a second implementation that shares no
kernel with ours (GDN, attention, norms, rope all differ).
`bench/mac/diff_mlx_lm.py`, three prompts, our engine decoding step by step and
mlx-lm scoring the sequence teacher-forced:

| prompt | top-1 agreement | KL mean | KL max |
|---|---|---|---|
| prose | 0.979 (one near-tie in 48) | 4.0e-4 | 2.6e-3 |
| code | 1.000 | 2.2e-4 | 2.6e-3 |
| math | 1.000 | 2.7e-4 | 3.8e-3 |

Two orders of magnitude below the quantization's own KL to bf16 (0.0232).
Greedy output matches mlx-lm's for the first 114 of 128 tokens on the decode
benchmark prompt. `tests/mac` checks every kernel against a plain-op reference,
chunked prefill against token-by-token decode, the speculative step against raw
greedy at every acceptance length, and a snapshot resume against a cold prefill
(each with a negative control that must fail).

## Raw decode

`bench/mac/decode.py`, 256 greedy tokens, same bytes for both, one run in one
process (`results/mac/decode.txt`, after the fusion below; a cooler machine earlier
in the day gave 18.5 pipelined, see "Heat"):

| | tok/s | % of the wall |
|---|---|---|
| ours, one eval per step | 17.3 | 80.9% |
| **ours, pipelined** (step t+1 built while t runs; the token never visits the host) | **18.4** | **86.2%** |
| mlx-lm 0.32 | 17.2 | 80.6% |

Removing int4 matmuls from the step leaves 7.7 ms of the 57 (everything else:
GDN, attention, norms, ~25 small kernels per layer). Compiling the position-free
elementwise groups (`model.FUSE`: the gated norm, SwiGLU, the attention gate, the
conv arithmetic, residual add + RMSNorm) with `mx.compile` takes pipelined decode
from 17.7 to 18.2 tok/s and the 8-row verify from 92 to 89 ms, interleaved A/B in
`bench/mac/fuse_ab.py`. Each compiled group is bit-identical to its uncompiled form
(`tests/mac`). The GDN gating and l2 norms are not compiled: compiled, they differ
by 3.8e-6 in fp32, the recurrence carries it forward, and on one of the three
correctness prompts the mean KL to mlx-lm went from 2.7e-4 to 6.3e-4 (max 3.8e-3 to
1.7e-2) on the same token sequence. Compiling them too gave 18.8 tok/s; the
0.6 tok/s is not worth that.

The same picture as the 5090: raw decode is close to the wall in every engine,
and the margin over the best rival is a few points. The headroom is speculation.

## The verify step is not free here

On the 5090 a K = 7 verify costs 1.13x a decode step (`docs/progress.md` step 28),
which is what made deep speculation pay. On this Mac, MLX's int4 matmul reads the
weights once per row for 2 <= M < 16 (`bench/mac/bandwidth.py`, `qmv_rows.py`):

| rows M | 2 | 4 | 6 | 8 | 16 |
|---|---|---|---|---|---|
| cost vs M = 1 (gate_up) | 1.04x | 1.36x | 2.62x | 3.39x | 1.72x |

M = 16 is cheaper than M = 8 because it switches to a tiled kernel on the M5
GPU's neural accelerators, but only for wide outputs: at N = 5120 (out_proj,
down_proj) M = 16 costs 5x, too few tiles and no split-K.

What we tried, all in `tokenrush/mlx/kernels.py`, measured in-graph:

- **scalar M-row GEMV** (`qmv_rows`): each weight dequantized once, applied to all
  rows. Ties MLX up to M = 4, ~3x at M = 8. Not a code problem: 8 FMA per weight
  at 291 GB/s is ~9 TFLOPS of fp32 scalar work.
- **simdgroup_float8x8** (`qmm8`), dequantizing straight into the lane's fragment
  registers: flat in M but ~1.7-2.0x even at M = 1. The same flops on the same ALUs.
- **Metal 4 `matmul2d` on the neural accelerators** (`qmm_nax`), W supplied
  transposed so each lane's four codes are half a word, bf16 dequant in registers,
  split-K for narrow outputs: **1.45-1.85x at M = 8** against MLX's 2.2-3.6x, but
  1.2-1.5x at M = 1 (the dequantization is the fixed cost).

Three more `matmul2d` variants were measured interleaved against `qmm_nax` and
removed: integer-only dequantization (the bf16 pattern `0x4300 | q` is 128 + q
exactly) with the affine part folded once per group (same speed, same error; the
error is the bf16 output rounding in all of them), a k permutation giving each lane
one 16-byte load per row per group (1.7x at M = 1, slower), and two 32-column tiles
per SIMD group (slower). The dequantization was not the bottleneck.

So the engine routes by row count: M <= 4 to MLX, 5..8 to `qmm_nax`, on M5 and
later GPUs only (`applegpu_g17` and up; earlier chips keep MLX's kernels, untested). Interleaved
measurement of the whole verify step (`bench/mac/step_profile.py`): M = 8 goes
from 215 ms to 113 ms; M <= 5 is unchanged. Closing the remaining gap at M = 4..8
means taking the dequantization off the critical path, the open item below.

## Speculative decoding

Same drafts as the CUDA engine, same acceptance rules (greedy: argmax; sampled:
one draw per position, accepted while it equals the draft). Each step's draft
chain and verify are one lazy MLX graph, evaluated once; the only value read
back is n accepted. The GDN layers use a **deferred commit**: a verify step
writes no recurrent state, the caller keeps the rows' (k, v, g, beta), and the
next call replays the accepted n + 1 of them before its own rows. The CUDA
engine instead writes K + 1 state snapshots per step (48 x 8 x 3.1 MB), cheap at
1.7 TB/s and not here.

`bench/mac/spec_bench.py`, Phase 0 prompts (`bench/mtp_accept.py`, thinking
off), 256 greedy tokens, two passes in one sitting (ranges are the two passes):

| | essay | code | math |
|---|---|---|---|
| raw | 16.4-17.2 | 16.7-17.4 | 16.7-17.3 |
| MTP chain, K = 3 | 28.4-33.2 (2.61/step) | 31.8-38.2 (2.98) | 35.1-42.8 (3.32) |
| **DFlash2** (K = 7) | 25.3-28.2 (3.08) | **39.9-42.2** (4.41) | **46.9-49.9** (5.22) |

The better draft per family is worth 1.7-1.9x over raw on essays (MTP), 2.4x on code
and 2.7-2.9x on math (DFlash2), against the CUDA engine's 2.3x / 3.5x / 3.7x
(228 / 357 / 378 over 100.9); the difference is the verify cost above. DFlash2's acceptance per step matches what
it does on the 5090, which is also evidence for the port. The MTP chain's best
depth is K = 3 here (K = 4: the M = 5 step jumps), not the CUDA chain's 3-4 with
dynamic depth; dynamic depth loses on this machine.

Greedy speculative output equals raw greedy until the first bf16 near-tie (45,
41, 140 tokens on the three prompts): the verify step's M-row matmul rounds
differently from the M = 1 one. The CUDA engine's `consistent` mode, which makes
them bit-identical, has no Mac counterpart yet.

## Serving

`python -m tokenrush.serve` on macOS loads `tokenrush/mlx/serve.py` and the MLX
`Session` (`tokenrush/mlx/session.py`, same interface as `tokenrush/session.py`),
so the OpenAI and Anthropic APIs, tool calls and the output parser are the CUDA
server's code. Snapshots keep references to the immutable recurrent states and
pending rows and copy only the conv rings. Checked end to end: an OpenAI chat
request for code ran at **61 tok/s** on DFlash2 (6.5 tokens per step), an
Anthropic request in Chinese on the MTP chain at 25 tok/s greedy (2.1 per step),
and a second turn reused the first turn's prompt.

## Against vLLM, SGLang and llama.cpp on the same Mac

Same machine, same day, same int4 bytes for every engine: the checkpoint written
as a standard mlx-lm directory by `bench/mac/export_mlx_lm.py` (renamed, not
requantized). Same client for every server, `bench/mac/rival_bench.py`: the Phase 0
prompts, greedy, 256 tokens, streamed, decode tok/s after the first token, best of
two. Logs and JSON in `results/mac/rivals/`.

| engine | essay | code | math | speculative decoding |
|---|---|---|---|---|
| **token-rush, `--draft auto`** (DFlash2 for these prompts) | **32.0** | **49.2** | **51.1** | DFlash2, K = 7 |
| token-rush, `--draft mtp` | 34.2 | 38.4 | 38.2 | MTP chain, K = 3 |
| token-rush, `--draft raw` | 15.2 | 15.7 | 15.9 | none |
| vllm-metal 0.31.0 (vLLM on MLX) | 16.9 | 17.0 | 16.9 | none available for this model |
| SGLang, MLX backend | did not run | | | |
| llama.cpp (b11429, b11474), unsloth UD-Q4_K_M GGUF | wrong output | | | |
| mlx-lm 0.32 (in process, `bench/mac/decode.py`) | 17.2 | | | none |

Raw decode is a tie: every MLX engine streams the same bytes at about the same
fraction of the wall. Run interleaved on a warm machine (vllm-metal, ours,
vllm-metal again; `*-ab*.json`), ours lands between the two vllm-metal runs on every
prompt (16.3 / 14.7 / 14.2 against 16.8-15.1 / 15.8-14.4 / 15.4-13.9). The server's
raw path syncs every step; the in-process pipelined loop gets 18.4. The
`--draft raw` row was run 35 minutes after vllm-metal's, on a warmer machine; the
interleaved runs are the comparison. The margin is speculation, and on this model neither
rival can use any:

- **vllm-metal** (installed with its `install.sh`, `vllm serve <dir>
  --limit-mm-per-prompt '{"image":0,"video":0}'`; the export needs the base repo's
  `preprocessor_config.json`, since vLLM treats the architecture as multimodal).
  `--speculative-config` with `mtp`: "not supported on Metal (supported: Gemma4 MTP,
  draft_model, ngram, dflash, dspark)". With `dflash` and the DFlash2 draft: "Model
  Runner V1 does not support: DFlash candidate-head drafts". With `ngram` (after
  `KMP_DUPLICATE_LIB_OK=TRUE` for a libomp clash): "Speculative decode verification
  is not supported for hybrid models on Metal yet". Any GDN hybrid is raw-only.
- **SGLang** (the documented Apple path: `pyproject_other.toml`, `[all_mps]`,
  `SGLANG_USE_MLX=1`, plus `SGLANG_BUILD_RUST_EXTS=none` without a Rust toolchain).
  v0.5.21 does not start on a Mac with any model, the documented Qwen3-0.6B
  included: `MlxModelRunnerStub` has no `kv_cache_configurator`. `model_runner.py`
  uses it from v0.5.16 on and the stub never gains it (read in the sources of each
  release; only v0.5.21 and v0.5.15 were run). v0.5.15 serves Qwen3-0.6B, but on this
  model it routes the GDN layers to the torch backend whose state pool the MLX path
  never builds (`'NoneType' object has no attribute 'conv'`, after
  `--mamba-scheduler-strategy no_buffer --disable-overlap-schedule`, without which
  it refuses with "extra_buffer needs CUDA/MUSA/NPU (FLA)").
- **llama.cpp** (Homebrew build 11429 and the b11474 release, `-ngl 999 -c 8192
  --jinja`, unsloth's `Qwen3.8-27B-UD-Q4_K_M.gguf`). It loads and runs, and it is
  the one rival with both drafts for this model (`--spec-type draft-mtp`,
  `draft-dflash` with z-lab's DFlash2 GGUF), but the text is wrong: "The capital of
  France is" continues " city of the the activity of the city of Paris. they
  remains the both are", the same on Metal and on the CPU, with tokenization
  identical to ours. Its speeds (16 raw, 10-11 with MTP, 8-10 with DFlash2, the drafts
  accepting 1.2-1.5 tokens a step against a broken target) are kept in
  `results/mac/rivals/llama.cpp-*.garbled.json` and not in the table. Another GGUF
  quantization was not tried.

## Long context

`bench/mac/ctx_step.py` sets the position directly (attention reads the same bytes
whatever the cache holds) and times one decode step and one 8-row verify step,
interleaved over positions. KV is bf16, 64 KB per token over the 16 attention layers.

| position | decode step | tok/s | ceiling | 8-row verify, before | after |
|---|---|---|---|---|---|
| 512 | 57.7 ms | 17.3 | 21.3 | 89 ms (1.58x) | 89 ms (1.54x) |
| 8192 | 59.0 ms | 16.9 | 20.5 | 104 ms (1.75x) | 94 ms (1.59x) |
| 32768 | 66.2 ms | 15.1 | 18.4 | 153 ms (2.31x) | 108 ms (1.63x) |
| 60000 | 73.9 ms | 13.5 | 16.6 | 209 ms (2.82x) | 127 ms (1.72x) |

Single-row decode holds 81-82% of its context-dependent ceiling at every length.
The verify step did not: MLX's multi-row attention kernel reads K and V once per
query head, so the 24 / 4 GQA heads read the cache six times (35 GB/s effective at
32k). Folding each KV head's six query heads into the row dimension, with an
explicit per-row causal mask, reads it once: 3.2x faster attention, same output to
1e-4, and the verify cost now grows from 1.54x to 1.72x across 60k instead of to
2.82x (`model.FOLD_ROWS`).

`bench/mac/long_context.py` puts bench/needle.py's needle in the middle of
pre-tokenized WikiText-2 / calibration text, asks for it, then asks for a summary
of the same text (reusing the prefix) and times 128 tokens of that answer:

| context | prefill | needle (raw / DFlash2) | raw | DFlash2 (tokens/step) |
|---|---|---|---|---|
| 8160 | 449 tok/s | found / found | 16.6 tok/s | **34.8** (3.76) |
| 32736 | 232 tok/s | found / found | 10.3 | **22.9** (3.94) |
| 59968 | 245 tok/s | found / found | 10.4 | **17.3** (3.12) |

This run is the heat-affected one: it prefills for minutes, and its raw numbers at
32k and 60k sit 30% under the step timings above. The acceptance per step is what
it measures. Prefill is compute-bound: MLX's int4 and bf16 matmuls both top out
at 27-30 TFLOPS at prefill shapes on this GPU (dequantize-then-dense is not
faster), and a 2048-token chunk spends 5.4 s in matmuls, 0.5 s in the sequential
GDN kernel and 0.1 s in everything else; at long context, attention over the cache
adds the rest.

## Heat

This is a laptop. Over a long benchmark session the same configuration got
slower: the verify step at M = 4 took 74 ms in one run and 115 ms in a later one,
while `pmset -g therm` reported no thermal warning. Compare configurations interleaved in one run,
as `step_profile.py` does, never across runs.

## Not done

- **mx.compile of the whole step.** The position-free parts are compiled (above);
  the rest needs positions on device (attention slices and rope offsets are Python
  ints). What it could still save is bounded by the 7.7 ms non-matmul share.
- **The verify kernel at M = 4..8.** `qmm_nax` costs 1.2-1.5x at M = 1 and the
  variants above did not move it; the bound is not the dequantization. Profiling
  the kernel itself (Xcode's Metal counters) is the next step there.
- **Context is 32k on a Mac by design.** That is the target here (`--max-len`
  default 32768, 2.1 GB of KV); 60k was measured once and works, longer is not a
  goal. Prefill of a full 32k prompt takes about 1.5 minutes, compute-bound.
- **The quality row on Mac.** KL against bf16 needs the bf16 checkpoint (55.6 GB);
  the engine-vs-mlx-lm KL above is the correctness evidence, and the quantization
  numbers carry over because the bytes are the same.
- **SGLang cannot serve Qwen3.8 on a Mac.** Open, to fix later. Its MLX backend
  fails at startup on any model from v0.5.16 on (`kv_cache_configurator`), and
  v0.5.15 has no Apple path for the GDN layers: the linear-attention backend
  expects the CUDA/FLA state pool (`'NoneType' object has no attribute 'conv'`,
  `gdn_backend.py:277`). Making it run means giving SGLang an MLX GDN layer with
  a recurrent-state pool, or fixing the stub in the newer versions; details in
  `results/mac/rivals/errors.md`.
- **Other Apple GPUs.** Everything was measured on one M5 Pro. M1-M4 run the same
  code with the `matmul2d` kernel switched off; their walls and verify costs are
  unmeasured.
