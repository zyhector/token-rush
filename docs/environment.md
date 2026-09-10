# Environment

**The Phase 4 benchmark machine: vast 36542 (RTX 5090), 2026-09-09.** Every
number in the final report — the wall, every rival, the engine — was measured
on it on that day (`results/2026-09-09-machine-36542/`). Its read bandwidth
measured **1702 GB/s**, within 0.1% of the 1701 GB/s recorded on the Phase 0
machine (94372, 2026-09-04), so the project keeps quoting **1701 GB/s as the
wall**: the difference is below run-to-run noise and moves no "% of wall"
figure by as much as a tenth of a point.

The Phase 0 machine is kept below as the historical section: its
measurements were the baselines the engine was built against, and `docs/
baselines.md` says which machine each rival number comes from.

## Instance (Phase 4 — the reported one)

| | |
|---|---|
| Instance / container id | 50438878 |
| **Machine id** | **36542** (host 110266) |
| Location | Idaho, US |
| Image | `vastai/base-image:cuda-13.3.1-auto` |
| Price | $1.047/hr |
| Host reliability | 0.998 |
| Motherboard | ProArt X870E-CREATOR WIFI |
| Public IP | 135.131.174.76 |

### GPU — RTX 5090

| | |
|---|---|
| Architecture | Blackwell consumer, **SM120** (`compute_cap 12.0`), 170 SMs |
| VRAM | 32607 MiB (31.4 GiB usable) |
| Memory clock | 14001 MHz (GDDR7, 512-bit) |
| SM clock (max) | 3090 MHz |
| Power limit | 575 W |
| PCIe | Gen5 x16 |
| VBIOS | 98.02.2E.40.AF |
| UUID | `GPU-1b4a7aa5-8835-76e7-d42f-88898869366c` |

Idle and unshared, no throttling reported (`HW Slowdown: Not Active`).

### Measured bandwidth (`check_bandwidth.py`)

| | Measured | % of 1792 GB/s spec |
|---|---|---|
| **Read-only, Triton streaming kernel** | **1702 GB/s** | **95.0%** |
| Read-only, `torch.sum` | 1687 GB/s | 94.2% |
| Copy (read + write) | 1530 GB/s | 85.4% |

Same card, same wall: 1702 against 94372's 1701. **1701 GB/s stays the
project's wall** (see the top of this file).

### CPU / memory / storage

| | |
|---|---|
| CPU | AMD Ryzen 9 9950X 16-core, Zen 5 (32 threads), 5.76 GHz boost |
| **Effective cores allocated** | **16** |
| L3 | 64 MiB, 1 NUMA node |
| RAM | 186 GiB visible |
| `/` (overlay) | 300 GB — **wiped on recycle/destroy** |
| `/workspace` | **not a volume** (`workspace_is_volume: false`) |
| Disk bandwidth | ~11 GB/s |
| Network | ~5.0 Gbps down / 3.3 Gbps up (the 17 GB checkpoint downloads in ~4 min) |

### Software

Driver **610.43.02** (CUDA 13.3 max), CUDA toolkit 13.3 from the image,
`nvcc` 13.3.73, gcc 13.3, cmake 3.28.

| | |
|---|---|
| Python | 3.12.14 (`/venv/main`) |
| torch | 2.14.0+cu130 |
| triton | 3.8.0 |
| transformers | 5.17.0 |
| flash-linear-attention | 0.6.0 (git main `cfaac24`) |
| numpy / safetensors / einops / huggingface-hub | 2.5.3 / 0.8.0 / 0.8.2 / 1.30.0 |
| Marlin extension (`tokenrush/csrc/`) | builds on first import with nvcc 13.3 + ninja 1.13.2 |

Rival stacks, in their own venvs, all current PyPI/GitHub on 2026-09-09:
llama.cpp `434ddbbc0` (built with CUDA 13.3, `sm_120`), SGLang 0.5.19
(torch 2.13.0+cu130), vLLM 0.29.0 (torch 2.13.0+cu130), ExLlamaV3 1.4.6
(`cu132.torch2.11.0` wheel), ollama 0.33.3. Versions in
`results/2026-09-09-machine-36542/install/`.

### What is verified on this card (`check_stack.py`, `check_gemv_sol.py`)

Same picture as the Phase 0 machine, line for line:

| | |
|---|---|
| `sm_120` Triton codegen, CUDA graph capture / replay | work |
| `fla` `chunk_gated_delta_rule` at the decode shape, FP32 state | correct (2.1e-4 / 4.6e-3 over 6 calls) |
| `fla` `fused_recurrent_gated_delta_rule` | **still miscompiled** — NaN heads over 6 calls (fla 0.6.0 `cfaac24`, triton 3.8.0) |
| our minimal Triton step | correct (9.7e-5 / 1.2e-7) |
| cuBLAS bf16 GEMVs at the real layer shapes, one CUDA graph | **1641 GB/s = 96.4% of the wall** (`gemvx` kernels) |
| `nsys` 2026.2.1 | works |
| `ncu` | **blocked**, `ERR_NVGPUCTRPERM` (third host in a row) |

**One difference that matters for the rivals, not for us.** The 48-layer
eager GDN chain costs **4.43 ms/token here against 10.50 on the Phase 0
EPYC** (graphed: 1.48 vs 1.45 — identical). The Zen 5 desktop core issues
kernel launches 2.4x faster than the EPYC 9B14 server core. Our engine
replays one graph per step and does not see it; engines with host-side work
between small kernels do — which is why llama.cpp's external drafts (DSpark,
DFlash2) measure higher here than on 94372 while its `llama-bench` decode is
identical to 0.05 tok/s. `docs/baselines.md` names the machine on every row.

---

# Phase 0 machine — 94372 (historical)

The machine Phase 0 was measured on, 2026-09-04. Its numbers were the
baselines the engine was built against; they are superseded for the report
by the Phase 4 section above. Kept unedited.

## Instance

Rented from vast.ai.

| | |
|---|---|
| Instance / container id | 49832910 |
| **Machine id** | **94372** (host 71705) |
| Location | North Carolina, US |
| Image | `vastai/base-image:cuda-12.8.1-auto` |
| Price | $0.591/hr |
| Host reliability | 0.987 |
| Motherboard | S8056GME |
| Public IP | 108.255.76.60 |

At $0.591/hr the whole project — quoted at a few hundred hours in
`feasibility.md` — lands around $120–240.

## GPU — RTX 5090

| | |
|---|---|
| Architecture | Blackwell consumer, **SM120** (`compute_cap 12.0`) |
| SMs | 170 |
| VRAM | 32607 MiB (31.4 GiB usable) |
| Memory clock | 14001 MHz (GDDR7, 512-bit) |
| SM clock (max) | 3120 MHz |
| Power limit | 600 W |
| PCIe | Gen5 x16 (vast quotes 27 GB/s host transfer) |
| VBIOS | 98.02.2E.80.4A |
| UUID | `GPU-3a108a21-07b6-4d16-67f2-23a26c597e8b` |

Idle and unshared — 1 GPU, no other tenants, no thermal or power throttling
observed (`HW Slowdown: Not Active`). Bit-exact repeatability checked: 200
rounds of large matmul / reduction / copy and a host round trip, zero
mismatches.

SM120 is the consumer Blackwell lineage: `mma.sync` tensor cores plus FP4, but
no Hopper `wgmma` and no datacenter-Blackwell `tcgen05`. Kernels must target
`sm_120` specifically.

### Measured bandwidth

The number the whole project is anchored to.

| | Measured | % of 1792 GB/s spec |
|---|---|---|
| **Read-only, Triton streaming kernel** | **1701 GB/s** | **94.9%** |
| Read-only, `torch.sum` | 1686 GB/s | 94.1% |
| Copy (read + write) | 1530 GB/s | 85.4% |

Weight streaming at bs=1 is a pure read, so **1701 GB/s is the wall**, not
1792. Reproducible to within 0.5% across runs. (vast's own advertised figure
for this machine is a more conservative 1456 GB/s.)

`check_bandwidth.py` streams a 2 GB buffer — more than 20x the 96 MB L2, so no
re-read can hit cache — and reports the fastest of several timed blocks rather
than the mean. Two things it guards against, both of which read low rather
than high: the card idles at 405 MHz memory clock and boosts to 14001 MHz, so a
block overlapping the ramp reads low by up to 30%; and a buffer small enough to
finish in a few hundred microseconds is dominated by fixed launch and
final-reduce cost (a 512 MB buffer reads 4% low for that reason alone).

## CPU / memory / storage

| | |
|---|---|
| CPU | AMD EPYC 9B14 96-core, Zen 4 (192 threads visible) |
| **Effective cores allocated** | **24** — the 192 shown are the host's |
| L3 | 384 MiB, 4 NUMA nodes |
| RAM | 503 GiB visible |
| `/` (overlay) | 100 GB — **wiped on recycle/destroy** |
| `/workspace` | **not a volume** (`workspace_is_volume: false`) |
| Disk bandwidth | ~7 GB/s |
| Network | ~3.0 Gbps down / 1.5 Gbps up |

**Nothing on this instance survives a recycle or destroy.** The repo is on
GitHub; weights re-download from the Hub in minutes (37 GB in about three);
everything else is rebuilt by the recipes in this file and `baselines.md`.

## Software

Driver **610.43.02** (supports up to CUDA 13.3). The image ships CUDA 12.8;
CUDA toolkit **13.3** is installed alongside it from the NVIDIA apt repo
(`cuda-toolkit-13-3`) and `/usr/local/cuda` points at it. `nvcc` 13.3 emits
working `sm_120` cubins; llama.cpp is built with it.

The base image ships **no Python ML stack**. Installed for this project:

| | |
|---|---|
| Python | 3.12.14 (`/venv/main`) |
| torch | 2.14.0+cu130 (bundles cuda-bindings 13.3) |
| triton | 3.8.0 |
| transformers | 5.16.1 |
| flash-linear-attention | 0.6.0 (git main `3468279`) |
| numpy / safetensors / einops | 2.5.2 / 0.8.0 / 0.8.2 |

```bash
source /venv/main/bin/activate
uv pip install torch --torch-backend=cu130
uv pip install numpy transformers einops safetensors huggingface-hub accelerate pytest   # accelerate: HF CPU-offload for the reference dump
uv pip install --no-deps "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention"
```

CUDA 12.8+ builds are mandatory: an older wheel (e.g. `cu124`) installs cleanly
and then fails at the first GPU op with *no kernel image is available*.

Rival stacks live in their own venvs: `/workspace/venvs/sglang` (SGLang 0.5.18,
torch 2.13.0+cu130), `/workspace/venvs/vllm` (vLLM 0.28.0, torch 2.13.0+cu130)
and `/workspace/venvs/exl3` (ExLlamaV3 1.4.6, torch 2.11.0+cu130). All install
cleanly on this driver; SGLang and vLLM need `cuda-bindings 13`, which is why
the project moved to a CUDA 13 host. Together with the four checkpoints they
fill the 100 GB disk to within a few GB — `uv cache clean` and `apt-get clean`
are the first things to run when it fills.

### What is verified on this card (`check_stack.py`)

| | |
|---|---|
| `sm_120` Triton codegen | works |
| CUDA graph capture / replay | works |
| `fla` `chunk_gated_delta_rule` at the bs=1 decode shape, FP32 state | **correct** — max error 2e-4 on outputs, 5e-3 on state, over repeated calls |
| `fla` `fused_recurrent_gated_delta_rule` at the same shape | **miscompiled** — see below |

**`fused_recurrent_gated_delta_rule` cannot be used on this stack.** It runs
without error and returns the right shapes, but whole heads of its output and
final state come back NaN. The set of bad heads is deterministic for a given
sequence of calls and changes from call to call on identical inputs; it
survives every launch-config and source variant tried (`num_stages`,
`num_warps`, scalar vs. block loads, static loop, no int64 indexing, `tl.exp`
vs. fla's `exp`), Triton 3.6 / 3.7 / 3.8, ptxas 12.8 / 12.9, and fla 0.5.2 /
main; `compute-sanitizer memcheck` reports nothing; adding a debug store inside
the loop makes the bug vanish. A plain-torch reference and the chunk kernel
agree with each other, and a minimal single-kernel Triton step written for
the same recurrence (in `check_stack.py`, same grid, same per-program tile)
is correct to 1e-7 on the state over the same call sequence. So Triton on
`sm_120` is fine; something specific to fla's kernel source trips the
compiler. Not root-caused further — nothing about it is fixable from the
project side, and the project's own fused step does not depend on it.

Consequences: the Phase 1 reference uses `chunk_gated_delta_rule` (slow at T=1
but correct), and the Phase 2 fused GDN step — already the plan — is written
by us. The differential test in `check_stack.py` is the guard; **a kernel that
"runs" is not a kernel that works**, and this one was recorded as working
before it was tested against a reference.

### PTX that does not execute here

`createpolicy.fractional.L2::evict_first` followed by `cp.async.cg.shared.global.L2::cache_hint`
(stock Marlin's weight-streaming copy) raises `cudaErrorIllegalInstruction`
(715) on this card, driver 610 / CUDA 13.3, even when compiled for
`sm_120` natively. A plain `cp.async.cg.shared.global` in its place runs and
costs nothing measurable (`docs/progress.md` step 28). Anything vendored
from datacenter-tuned kernels that carries L2 eviction hints needs the same
edit.

## Profiling — one real gap

| Tool | State |
|---|---|
| **nsys** 2026.1.3 (timeline, kernel trace) | **Works** |
| **ncu** (per-kernel HW counters) | **Blocked** — `ERR_NVGPUCTRPERM` |

The container has no `CAP_SYS_ADMIN` / `CAP_PERFMON` (`CapEff a80405fb`), and
`NVreg_RestrictProfilingToAdminUsers` is a host kernel-module parameter. **This
cannot be fixed from inside the container**, and this is the second host in a
row to block it — expect it to be the norm on vast.

What still works: full kernel timelines on `sm_120`, so the Phase 0 per-token
timeline slice and all wall-clock work are unaffected.

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

`Qwen/Qwen3.8-27B` is **public and ungated**. Its `config.json` gives 64 layers
as 48 `linear_attention` + 16 `full_attention` (`full_attention_interval: 4`),
hidden 5120, FFN 17408, attention 24 Q / 4 KV heads at head dim 256, GDN 48 V /
16 QK heads at head dim 128, conv kernel 4, vocab 248320.

Parameter split, read from the safetensors headers:

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
(`QUASAR-QAT`; `RadixArk`'s is a mixed FP8/FP4 config and `unsloth`'s is FP8
despite the name), EXL3 (`turboderp`, for the ExLlamaV3 comparison), official
FP8 (`Qwen`), and the DSpark draft (`RadixArk`, 1.86B, bf16, 3.7 GB). Local
copies live under `/workspace/models/`.

## What this machine implies for the plan

Three measurements that set where Phase 2 effort belongs
(`check_gemv_sol.py`, `check_stack.py`).

**cuBLAS feeds the card at bs=1.** 63 bf16 GEMVs at the real per-layer shapes
of Qwen3.8-27B (9 GDN + 3 attention layers, 9.13 GB of weights), replayed from
one CUDA graph, stream at **1643 GB/s — 96.6% of the wall**; eager, 1638 GB/s.
The dense weight-streaming path needs no hand-written kernel. With torch
2.14+cu130, cuBLAS dispatches a dedicated `gemvx` kernel for these shapes, not
the Ampere-lineage `cutlass_80_tensorop_*` GEMM it used on CUDA 12.8 — the
headroom argument in `CLAUDE.md` no longer cites that fallback.

**The launch-dispatch tax is large.** A 48-layer Gated DeltaNet decode chain —
conv1d step, gating, delta-rule state update, gated RMSNorm, output gate,
projections excluded — eager vs. captured into a single CUDA graph:

| | ms/token |
|---|---|
| Eager | 10.55 |
| One CUDA graph | 1.44 |
| **Tax removed** | **9.11** |

That is ~91% of the entire 10 ms budget for 100 tok/s, spent on launch
overhead alone. The state update is a minimal single-kernel Triton step
written in `check_stack.py` (the fused fla kernel being unusable here, and the
minimal kernel passing the same differential test); the rest of the layer is
about two dozen small torch ops, which is what a kernel-per-op engine launches.
The graphed figure is a lower bound: the benchmark replays one layer's tensors
48 times, so its 3 MB of state stays cache-warm, and those two dozen ops per
layer are what GDN fusion in Phase 2 collapses further.

Together these say the Phase 2 win is concentrated in graph capture and GDN
fusion rather than in GEMV tuning.
