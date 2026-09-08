# 4-bit GEMV shootout

Which 4-bit weight-streaming kernel to put behind `tokenrush.quant.QLinear`.
Run on 2026-09-08 (Phase 1, step 3), development instance 50295164 (RTX
5090, CUDA 13.3); a development number, quoted against the 1701 GB/s wall of
`docs/environment.md`.

```bash
source /venv/main/bin/activate
python scripts/gemv_shootout/harness.py                 # torch-native + our Triton kernel
python scripts/gemv_shootout/harness.py --per-shape     # per matrix shape, L2-proof
/workspace/venvs/vllm/bin/python scripts/gemv_shootout/harness.py --only marlin-int4,nvfp4-cutlass
```

## Method

Every candidate is a (pack, run) pair. The harness builds the 12-layer set
(9 GDN + 3 attention, every projection a separate matrix) plus `lm_head`,
records every GEMV into one CUDA graph, and divides the packed bytes actually
read per replay by the replay time. A second run uses the **fused** layer
set (q|k|v, in_proj_qkv|z, gate|up concatenated), which is how the engine
packs them now. The whole-model column scales the 12-layer time to 64
layers, adds `lm_head` at its own bandwidth, and inverts. Correctness is the
relative error against `x @ W_dequant.T` using the candidate's own
dequantization, so a kernel cannot score by reading fewer bytes than it
claims.

## Results

| candidate | bpw | 12 layers | fused 12 layers | `lm_head` | model tok/s (fused) | rel err |
|---|---|---|---|---|---|---|
| bf16 cuBLAS (reference) | 16 | 96.2% | 97.5% | 91.8% | 32 | — |
| dequantize-then-matmul (the skeleton's first GEMV) | 4.25 | 2.6% | — | 2.2% | 3 | — |
| **triton-int4 (ours)** | 4.25 | 83.4% | **88.2%** | **98.8%** | **111** | 5e-3 |
| tinygemm-int4 (torch built-in) | 4.25 | 86.1% | 90.7% | 98.8% | 114 | 5e-3 |
| marlin-int4 (vLLM) | 4.125 | 81.8% | 87.8% | 97.4% | 114 | 2e-3 |
| nvfp4 cutlass W4A4 (vLLM) | 4.5 | 48.2% | 52.4% | 57.5% | 62 | 1e-1 (vs unquantized) |

Per shape, L2-proof (cycling through >400 MB of copies), the int4 kernels
are all in the same band: 26–35% on the 1024x5120 k/v projection (a 5–6 µs
per-launch floor on 2.8 MB), 74–81% on the two 5120-row matrices (`out_proj`,
`down_proj`: too few rows for a full wave of programs), 85–91% on the
10k–17k-row matrices, 94% on the fused 34816x5120 `gate_up`, 99% on
`lm_head`.

## What it decided

- **Ours (`triton-int4`) is the default.** Within 2.5 points of the fastest,
  it reads our own packing (so prefill dequantizes the same nibbles and runs
  cuBLAS), holds one copy of the weights, and is the kernel Phase 2 fuses
  further. Inside the real model it takes 9.1 ms of a decode step for
  14.2 GB: 1560 GB/s, 92% of the wall. Autotune space is pinned to the six
  configs the full sweep picked (`_GEMV_CONFIGS`); the full sweep costs 15 s
  on `lm_head` alone.
- **Fuse the projections.** q|k|v, in_proj_qkv|z and gate|up are one matrix
  each in the loader now: +4–5 points on every kernel, because a 2.8 MB
  GEMV costs 6 µs whatever the kernel.
- **tinygemm kept as a reference backend.** Its packed layout is opaque, so
  the backend holds only that layout and serves prefill from it, 6x slower
  than dequant+GEMM at T >= 512. In eager mode it decodes faster than the
  Triton default (46 vs 36 tok/s) because an aten launch costs ~25 µs less
  Python than a Triton launch and the eager step is CPU-bound; under a
  CUDA graph only GPU time counts and the two are within 5%.
- **NVFP4's cutlass GEMM is not a GEMV.** At M=1 it holds half the wall.
  vLLM's own bs=1 NVFP4 serving path was not isolated here (it selects a
  kernel through its linear-backend abstraction), so this row says only that
  the obvious kernel is the wrong tool for bs=1; NVFP4 as a *format* is a
  Phase 1b quality question, and its 4.5 bpw already costs 6% of the byte
  budget against int4 g128.
- **Marlin** matches the others and needs vLLM; nothing to gain from it here.
- **Not measured**: ExLlamaV3's trellis kernel (needs its quantizer's output
  and its own venv; its Phase 0 deficit was engine overhead, not the kernel).

## Traps

- **Per-shape numbers lie unless you defeat L2.** A 47 MB matrix replayed
  alone in a graph reads at 2500 GB/s: it lives in the 96 MB L2. `--per-shape`
  cycles through >400 MB of copies. The 12-layer set (2.4 GB) needs no such
  care.
- **tinygemm's nibble order is the reverse of the obvious one** (the high
  nibble is the even element) and it dequantizes as `(q - 8) * scale + zero`.
  Probed, not documented; the harness's correctness column catches it.
