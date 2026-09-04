# env_check

"Is this box usable?" — run once on a newly rented GPU instance, then get on
with the project.

These are deliberately **not** in `bench/`. `bench/` measures the engine — tok/s,
acceptance rate, time-to-first-token — against the real model. Nothing here
measures Token Rush; it only establishes that the machine can host it.

| Script | Answers |
|---|---|
| `check_env.sh` | What machine is this, and do the profilers work? |
| `check_bandwidth.py` | What is this card's real read bandwidth, and what decode ceiling does that imply? |

```bash
source /venv/main/bin/activate
scripts/env_check/check_env.sh
scripts/env_check/check_bandwidth.py
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
the measured figure varies by card. Run it before quoting any "% of wall" number
from a new instance.

## Provisioning a fresh instance

The vast base image ships no Python ML stack:

```bash
uv pip install torch --torch-backend=cu128     # Blackwell requires CUDA >= 12.8
uv pip install numpy transformers flash-linear-attention
```

An older torch build (`cu124`) installs cleanly and then fails at the first GPU
op with *no kernel image is available*.

`nsys` comes from the CUDA apt repo (`cuda-nsight-systems-12-8`); `check_env.sh`
uses the newest version installed. `ncu` is already present.

## One trap

Never name a probe script after a stdlib module (`nt.py`, `os.py`). The script
directory goes on `sys.path` first, the interpreter dies before CUDA init, and
`nsys` faithfully records a trace containing no GPU work. An empty trace looks
identical to a profiler permission failure — confirm the target program actually
ran before blaming the profiler.
