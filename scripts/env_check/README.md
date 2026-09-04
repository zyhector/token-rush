# env_check

Throwaway "is this box usable?" checks. Run once when a new GPU instance is
rented, then forget about them.

These are deliberately **not** in `bench/`. `bench/` is for measuring the engine
— tok/s, acceptance rate, time-to-first-token — against the real model. Nothing
here measures Token Rush; it only establishes that the machine can host it.

| Script | Answers |
|---|---|
| `check_env.sh` | What machine is this, and do the profilers work? |
| `check_bandwidth.py` | What is this card's real read bandwidth, and what decode ceiling does that imply? |

```bash
source /venv/main/bin/activate
scripts/env_check/check_env.sh
scripts/env_check/check_bandwidth.py
```

## Why these two survive

**`check_env.sh`** records the vast machine/host id (benchmarks must name the
machine they ran on) and checks `nsys` and `ncu`. The `ncu` result is a rental
accept/reject criterion: GPU counter access is a per-host kernel-module setting,
so an otherwise identical 5090 may or may not allow kernel-level profiling.

**`check_bandwidth.py`** re-anchors the roofline. Every speed target in
`CLAUDE.md` is a percentage of measured read bandwidth, not of the 1792 GB/s
spec sheet — and the measured figure varies by card. Re-run it before quoting
any "% of wall" number from a new instance.

## Deliberately not kept

Two one-shot probes were used to validate the plan and then deleted, having
answered their question:

- A naive Triton GEMV, to confirm `sm_120` codegen works and to see how close
  an unoptimised kernel lands to the bandwidth ceiling (it landed at ~98%).
- A 48-layer Gated DeltaNet chain run eager vs. captured in one CUDA graph, to
  size the launch-dispatch tax (~4.2 ms/token).

Both numbers are recorded in `docs/environment.md`. When they need re-measuring
it will be against the real model inside the engine, which makes them `bench/`
material, not environment checks.

## Setup notes

The base image ships no Python ML stack. What this project needed:

```bash
uv pip install torch --torch-backend=cu128     # Blackwell needs CUDA >= 12.8
uv pip install numpy transformers flash-linear-attention
apt-get install -y nsight-systems-2025.1.3     # repo default is pre-Blackwell
```

Two traps worth remembering:

- The `nsys` in the CUDA 12.8 apt repo is 2024.6.2, which predates Blackwell and
  records an **empty trace** on `sm_120` rather than failing loudly.
- Never name a probe script after a stdlib module (`nt.py`, `os.py`). The script
  directory goes on `sys.path` first, the interpreter dies before CUDA init, and
  the result looks exactly like a profiler permission failure.
