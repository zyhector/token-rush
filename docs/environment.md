# Environment

The machine Token Rush is being built on. Measured 2026-09-04.

Re-run `scripts/env_check/check_env.sh` and `scripts/env_check/check_bandwidth.py`
on any new instance and update this file — every "% of wall" claim in the project
is relative to the bandwidth recorded here.

## Instance

Rented from vast.ai. Benchmarks must cite this machine.

| | |
|---|---|
| Instance / container id | 49818067 |
| **Machine id** | **25132** (host 18) |
| Location | Alberta, CA |
| Image | `vastai/base-image:cuda-12.8.1-auto` |
| Price | $0.428/hr |
| Host reliability | 0.9985 |
| Motherboard | ROME2D32GM-2T |
| Public IP | 198.53.64.194 |

At $0.428/hr the whole project — quoted at a few hundred hours in
`feasibility.md` — lands around $85–170.

## GPU — RTX 5090

| | |
|---|---|
| Architecture | Blackwell consumer, **SM120** (`compute_cap 12.0`) |
| SMs | 170 |
| VRAM | 32607 MiB (31.4 GiB usable) |
| Memory clock | 14001 MHz (GDDR7, 512-bit) |
| SM clock (max) | 3090 MHz |
| Power limit | 575 W (default = max; min 400 W) |
| PCIe | Gen4 x16 (~26.2 GB/s host transfer) |
| VBIOS | 98.02.2E.40.AF |
| UUID | `GPU-a98bb22d-3ce8-d451-1390-62496971682c` |

Idle and unshared — 1 GPU, no other tenants, no thermal or power throttling
observed (`HW Slowdown: Not Active`).

SM120 is the consumer Blackwell lineage: `mma.sync` tensor cores plus FP4, but
no Hopper `wgmma` and no datacenter-Blackwell `tcgen05`. Kernels must target
`sm_120` specifically.

### Measured bandwidth

The number the whole project is anchored to.

| | Measured | % of 1792 GB/s spec |
|---|---|---|
| **Read-only (reduction)** | **1605–1610 GB/s** | **~89.6%** |
| Copy (read + write) | 1519–1520 GB/s | ~84.8% |

Weight streaming at bs=1 is a pure read, so **1605 GB/s is the wall**, not 1792.
Reproducible to within 0.3% across runs. (vast's own advertised figure for this
machine is a more conservative 1451 GB/s.)

## CPU / memory / storage

| | |
|---|---|
| CPU | 2x AMD EPYC 7B13 64-core (256 threads visible) |
| **Effective cores allocated** | **32** — the 256 shown are the host's |
| L3 | 512 MiB, 8 NUMA nodes |
| RAM | 503 GiB visible |
| `/` (overlay) | 100 GB — **wiped on recycle/destroy** |
| `/workspace` | **200 GB persistent volume** (`workspace_is_volume: true`) |
| Disk bandwidth | ~5.3 GB/s |
| Network | ~2.9 Gbps down / 3.3 Gbps up |

Only `/workspace` survives a recycle. Weights, quantized checkpoints and
benchmark results belong there; nothing irreplaceable goes on `/`.

Model download measured at **131 MB/s** from HuggingFace — the full 55.6 GB
bf16 repo pulls in about 7 minutes, so re-downloading after a wipe is cheap.

## Software

Driver **575.51.03** (supports up to CUDA 12.9), CUDA toolkit **12.8** with
`nvcc` confirmed emitting working `sm_120` cubins.

The base image ships **no Python ML stack**. Installed for this project:

| | |
|---|---|
| Python | 3.12.14 (`/venv/main`) |
| torch | 2.11.0+cu128 |
| triton | 3.6.0 |
| transformers | 5.16.1 |
| flash-linear-attention | 0.5.2 |
| numpy / safetensors / einops | 2.5.2 / 0.8.0 / 0.8.2 |

CUDA 12.8+ builds are mandatory: an older wheel (e.g. `cu124`) installs cleanly
and then fails at the first GPU op with *no kernel image is available*.

Verified working on this card: `sm_120` Triton codegen, CUDA graph
capture/replay, and `fla`'s `fused_recurrent_gated_delta_rule` at the bs=1
decode shape with FP32 recurrent state.

## Profiling — one real gap

| Tool | State |
|---|---|
| **nsys** (timeline, kernel trace) | **Works** — needs 2025.1.3+ |
| **ncu** (per-kernel HW counters) | **Blocked** — `ERR_NVGPUCTRPERM` |

The container has no `CAP_SYS_ADMIN` / `CAP_PERFMON` (`CapEff a80405fb`), and
`NVreg_RestrictProfilingToAdminUsers` is a host kernel-module parameter. **This
cannot be fixed from inside the container.**

What still works: full kernel timelines, so the Phase 0 per-token timeline slice
and all wall-clock work are unaffected. Note that the `nsys` in the CUDA 12.8
apt repo is 2024.6.2, which predates Blackwell and records an *empty* trace on
`sm_120` instead of erroring — install `nsight-systems-2025.1.3`.

What is lost: per-kernel DRAM throughput, achieved occupancy, memory-level
parallelism, warp stall reasons — the instruments for tuning the last few
percent toward the roofline in Phase 2.

Workarounds, in order of preference:

1. Ask the host to set `NVreg_RestrictProfilingToAdminUsers=0`, or filter for a
   machine that already has it. One-line host change.
2. Derive effective bandwidth from wall-clock timing and known byte counts.
   Exact for GEMV, and sufficient for the headline "% of roofline" metric, which
   is computed this way anyway.

Not a blocker. Option 2 carries Phases 0–2; only the final occupancy tuning
suffers, and `feasibility.md` already says not to start by chasing the last 5%.

## Model availability

`Qwen/Qwen3.8-27B` is **public and ungated**. Config verified against the
architecture assumed in `CLAUDE.md` — all of it matches: 64 layers as 48
`linear_attention` + 16 `full_attention` (`full_attention_interval: 4`), hidden
5120, FFN 17408, attention 24 Q / 4 KV heads at head dim 256, GDN 48 V / 16 QK
heads at head dim 128, conv kernel 4, vocab 248320.

Exact parameter split, read from the safetensors headers:

| Component | Params |
|---|---|
| Text body | 25.625 B |
| `lm_head` | 1.271 B |
| **Text path (what we serve)** | **26.896 B** |
| MTP head (`mtp.*`) | 0.425 B |
| Vision tower | 0.461 B |
| Total repo | 27.781 B (55.6 GB bf16) |

**The MTP head ships with the weights** — 15 tensors under the `mtp.` prefix,
one full-attention layer plus an `fc` projection. Phase 3's primary route is
available without training anything.

The vision tower is cleanly separable by prefix and is discarded.

Rival and comparison artifacts all exist on the Hub: GGUF (`unsloth`), NVFP4
(`unsloth`, `QUASAR-QAT`), EXL3 (`turboderp`, for the ExLlamaV3 comparison),
official FP8 (`Qwen`), and the DSpark draft (`RadixArk`).

## One-shot findings

Measured once to validate the plan, recorded here rather than kept as scripts.
Both are `bench/` material when they need re-measuring against the real engine.

**A naive Triton GEMV already saturates GDDR7.** A textbook bf16 GEMV, no
tuning, reached 1569 GB/s — 97.8% of the 1605 GB/s wall. The dense
weight-streaming path needs no hand-written kernel.

**The launch-dispatch tax is large.** A 48-layer Gated DeltaNet chain, eager vs.
captured into a single CUDA graph:

| | ms/token |
|---|---|
| Eager (48 separate launches) | 4.39 |
| One CUDA graph | 0.15 |
| **Tax removed** | **4.23** |

That is ~42% of the entire 10 ms budget for 100 tok/s, spent on launch overhead
alone. Graphed, the chain sits within 1.6x of its 0.094 ms state-bandwidth floor
(151 MB of FP32 state).

Caveat: the benchmark replays one layer's tensors 48 times, so weights and state
stay cache-warm. The launch-tax figure is what it measures reliably; the graphed
absolute is a lower bound, not a prediction for the real model.

Together these two say the Phase 2 win is concentrated in graph capture and GDN
fusion rather than in GEMV tuning.

Incidental: cuBLAS dispatches `cutlass_80_tensorop_*` — Ampere-lineage kernels —
for bf16 matmul on this card.
