# env_check

"Is this box usable?" — run on the machine whose numbers get reported. It
was run once on the Phase 0 machine (94372, `docs/environment.md`) and is run
again in Phase 4 on the final benchmark machine. It is **not** run on every
development instance in between; those measurements are not re-taken until
there is an engine to measure alongside them.

These are deliberately **not** in `bench/`. `bench/` measures the engine — tok/s,
acceptance rate, time-to-first-token — against the real model. Nothing here
measures Token Rush; it only establishes that the machine can host it.

| Script | Answers |
|---|---|
| `check_env.sh` | What machine is this, and do the profilers work? |
| `check_bandwidth.py` | What is this card's real read bandwidth, and what decode ceiling does that imply? |
| `check_stack.py` | Do Triton, CUDA graphs and `fla`'s GDN step actually work here — against a reference — and what does the GDN launch tax cost? |
| `check_gemv_sol.py` | How close to the wall do cuBLAS GEMVs get at the real layer shapes, and which kernels does it dispatch? |

```bash
source /venv/main/bin/activate
scripts/env_check/check_env.sh
scripts/env_check/check_bandwidth.py
scripts/env_check/check_stack.py
scripts/env_check/check_gemv_sol.py <read GB/s from check_bandwidth>
```

Record the output in `docs/environment.md`.

## What each one is for

**`check_env.sh`** records the vast machine/host id — benchmarks must name the
machine they ran on — and checks `nsys` and `ncu`. Treat the `ncu` line as a
rental accept/reject criterion: GPU counter access is a per-host kernel-module
setting, so an otherwise identical 5090 may or may not allow kernel-level
profiling.

**`check_bandwidth.py`** anchors the roofline. Every speed target in `CLAUDE.md`
is a percentage of measured read bandwidth, not of the 1792 GB/s spec sheet, and
the measured figure varies by card. It reports the best of `torch.sum` and a
Triton streaming-read kernel over a 2 GB buffer. Run it before quoting any "% of
wall" number from a new instance.

**`check_stack.py`** is the differential test. It compares `fla`'s fused and
chunk GDN kernels, and a minimal Triton step of its own, against a plain FP32
reference over repeated calls — repeated, because the failure it exists to
catch (whole heads of NaN from a miscompiled kernel) depends on the call
sequence and can pass once. Then it times a 48-layer GDN decode chain eager vs.
graphed, which is the launch-tax number in `CLAUDE.md`.

**`check_gemv_sol.py`** is the speed-of-light for weight streaming: the real
per-layer shapes of Qwen3.8-27B as bs=1 bf16 GEMVs in one CUDA graph, in GB/s
against the wall, plus the cuBLAS kernel names it dispatched (the difference
between a native GEMV and an Ampere-lineage tensor-op GEMM is visible here).

## Provisioning a fresh instance

The vast base image ships no Python ML stack and no `nsys`. Use current
versions; nothing here is pinned to an old toolkit.

```bash
apt-get install -y nsight-systems-2026.1.3 cuda-toolkit-13-3   # toolkit ≤ driver_max_cuda
source /venv/main/bin/activate
uv pip install torch --torch-backend=cu130
uv pip install numpy transformers einops safetensors huggingface-hub
uv pip install --no-deps "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention"
```

An older torch build (`cu124`) installs cleanly and then fails at the first GPU
op with *no kernel image is available*. `check_env.sh` uses the newest `nsys`
installed; `ncu` is already present.

## Traps

- **Never name a probe script after a stdlib module** (`nt.py`, `os.py`). The
  script directory goes on `sys.path` first, the interpreter dies before CUDA
  init, and `nsys` faithfully records a trace containing no GPU work. An empty
  trace looks identical to a profiler permission failure — confirm the target
  program actually ran before blaming the profiler.
- **A kernel that runs is not a kernel that works.** Shapes and dtypes coming
  back right says nothing; compare against a reference, more than once.
- **Bandwidth needs a big buffer and best-of timing.** A 512 MB read is short
  enough that launch and reduce overhead cost 4%; the memory clock ramps from
  405 MHz to 14001 MHz under load and a cold block reads 30% low.
