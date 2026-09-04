# Handoff — moving to a CUDA 13 machine

For the agent picking this up on the new instance. Everything below was
established on vast machine **25132** (RTX 5090, driver 575.51.03, CUDA 12.8),
which is being abandoned.

Read `CLAUDE.md` first for what the project is. This document covers only the
move: why, what to verify, what to re-measure, and what not to touch.

## Why we are moving

SGLang + DSpark is the project's stated primary rival, and **it cannot be run on
the old machine at all**:

- DSpark support exists only in SGLang **0.5.16+**.
- Every SGLang from **0.5.11** onward requires `cuda-python>=13.0`, which pulls
  `cuda-bindings>=13.0`; torch cu128 requires `cuda-bindings<13`. Irreconcilable.
- The last CUDA-12-compatible SGLang is **0.5.10** — eight minor versions stale
  and with no DSpark.
- The old host's driver (575.51.03) caps at CUDA 12.9. The driver is
  host-injected and must never be replaced from inside the container, and CUDA
  forward-compatibility is datacenter-only, so a GeForce card cannot run cu130
  there. A different host was the only option.

Benchmarking a stale rival would have contradicted the project's own rule —
`feasibility.md`: *"The bar moves. Pin rival versions and date every benchmark."*

Secondary motivation, **reasoned but not measured**: CUDA 12.8 was the first
toolkit to support `sm_120` at all, and we observed cuBLAS dispatching
Ampere-lineage `cutlass_80_tensorop_*` kernels for bf16 matmul on this card.
NVFP4 is Blackwell-native and is the likely Phase 1 quantization choice; its
kernel ecosystem matured after 12.8. CUDA 13 is expected to be better ground for
Phase 2, but nobody has demonstrated that yet — see "Open questions".

## Step 0 — qualify the machine before investing in it

Do this before rebuilding anything. It takes about five minutes and can save an
hour of wasted setup.

```bash
git clone git@github.com:zyhector/token-rush.git /workspace/token-rush
cd /workspace/token-rush
source /venv/main/bin/activate
uv pip install torch --torch-backend=cu130   # or cu128 if the driver is older
uv pip install numpy
scripts/env_check/check_env.sh
scripts/env_check/check_bandwidth.py
```

Three things must hold:

| Check | Required | Why |
|---|---|---|
| `driver_version` | **≥ 580.x** (`cuda_max_good ≥ 13.0`) | otherwise the whole move was pointless |
| `compute_cap` | 12.0, RTX 5090 | the project targets this card only |
| Read bandwidth | ~1600 GB/s | if materially lower, the card is being shared or throttled — reject |

**Also check the `NCU` line.** On the old host it read
`BLOCKED — ERR_NVGPUCTRPERM`: GPU performance counters are gated by a host
kernel-module setting (`NVreg_RestrictProfilingToAdminUsers`) that cannot be
changed from inside an unprivileged container. It is a **per-host** setting, so a
different machine may allow it.

If it says counters are readable, that is a significant win — it restores
per-kernel DRAM throughput, achieved occupancy and memory-level parallelism,
which is exactly the instrumentation Phase 2 kernel tuning wants. If it is still
blocked, it is not a blocker (wall-clock timing against known byte counts carries
Phases 0–2), but it is worth trying another host before settling in.

Record the result either way and update `docs/environment.md`.

## Step 1 — rebuild the stack

`/workspace` is a **host-local volume**. It does not follow you to a new machine.
Everything below must be rebuilt. Budget roughly 45 minutes, mostly waiting.

### Our own stack

```bash
source /venv/main/bin/activate
uv pip install torch --torch-backend=cu130
uv pip install numpy transformers flash-linear-attention
```

**Re-validate these three before trusting anything downstream** — they were
confirmed on CUDA 12.8 and are now unverified:

1. Triton generates working `sm_120` code.
2. CUDA graph capture and replay work.
3. `fla`'s `fused_recurrent_gated_delta_rule` runs at the bs=1 decode shape
   (B=1, T=1, 16 QK heads, 48 V heads, head_dim 128) with FP32 recurrent state.

`fla` is Triton-based so it will probably be fine, but "probably" is not
"confirmed", and GDN decode is the part of this project with no mature reference
implementation to fall back on.

### llama.cpp and weights

Full recipe is in `docs/baselines.md` under "Reproducing". Summary:

```bash
git clone https://github.com/ggml-org/llama.cpp /workspace/rivals/llama.cpp
cd /workspace/rivals/llama.cpp && git checkout 6703d78
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON
cmake --build build --config Release -j 32      # ~25 min

hf download unsloth/Qwen3.8-27B-GGUF --include "Qwen3.8-27B-UD-Q4_K_M.gguf" \
   --local-dir /workspace/models/Qwen3.8-27B-GGUF                    # 16.5 GB
hf download QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 \
   --local-dir /workspace/models/Qwen3.8-27B-NVFP4                   # 20.6 GB
```

Pin `6703d78` for the first pass so the numbers are comparable to the table
below. Once they reconcile, re-pin to current upstream and re-measure — the
project benchmarks current rivals, not convenient ones.

`CMAKE_CUDA_ARCHITECTURES=120` is not optional; a default build produces no
`sm_120` kernels.

## Step 2 — what to re-measure, and what not to

### Must re-measure (machine- or toolkit-dependent)

| What | Old value | Where it is recorded |
|---|---|---|
| Read bandwidth — **the roofline anchor** | 1615 GB/s | `environment.md`, `CLAUDE.md` |
| Machine identity, price, CPU, storage | machine 25132 | `environment.md` |
| `ncu` counter availability | BLOCKED | `environment.md` |
| llama.cpp raw decode (`llama-bench` tg128) | 80.86 tok/s | `baselines.md` |
| llama.cpp with `--spec-type draft-mtp` | ~91 tok/s | `baselines.md` |
| llama.cpp decode vs context (0 / 22k / 90k / 200k) | 77.0 / 70.5 / 54.1 / 40.6 | `baselines.md` |
| Speed-of-light coefficient (93 GEMVs, one CUDA graph) | 97.7% of wall | `feasibility.md` |
| GDN launch tax (48 layers, eager vs graphed) | 4.39 → 0.15 ms | `environment.md`, `CLAUDE.md` |

**Every "% of the wall" figure in the project is a fraction of the measured read
bandwidth.** If the new machine reads a different number, all of them shift.
Re-anchor first, then recompute, then update every document that quotes one —
`grep -rn "1615" CLAUDE.md docs/` finds them.

### Do NOT re-measure (properties of the model files, not the machine)

These were derived from `config.json` and safetensors/GGUF headers and are
already recorded. Re-deriving them wastes time and risks introducing drift:

- Architecture: 64 layers as 48 `linear_attention` + 16 `full_attention`,
  `full_attention_interval: 4`, hidden 5120, FFN 17408, attention 24 Q / 4 KV
  heads at head_dim 256, GDN 48 V / 16 QK heads at head_dim 128, conv kernel 4,
  vocab 248320, native context 262144.
- Parameter split: text body 25.625B, `lm_head` 1.271B, **text path 26.896B**,
  MTP head 0.425B, vision tower 0.461B, repo total 27.781B.
- The MTP head ships with the weights (`mtp.*`, 15 tensors) and survives GGUF
  conversion as `blk.64.nextn.*` with `nextn_predict_layers: 1`. Phase 3 trains
  nothing.
- GGUF `UD-Q4_K_M` byte accounting: file 16.464 GB, MTP head 0.351 GB,
  **text path read per decode step 16.102 GB = 4.79 bpw**.
- NVFP4 (QUASAR) byte accounting: text body 16.245 GB + `lm_head` 2.543 GB =
  **18.788 GB**. `lm_head` is left at BF16, so this checkpoint reads *more* per
  token than llama.cpp's Q4_K_M despite being "4-bit".
- Upstream llama.cpp supports this model natively (`LLM_ARCH_QWEN35`, MTP graph
  in `src/models/qwen35.cpp`). No fork needed; the `sudoingX/qwen38-mtp` fork
  referenced in older notes is superseded.

## Step 3 — the work that motivated the move

### SGLang + DSpark (the primary rival, never yet measured)

```bash
uv venv /workspace/venvs/sglang --python 3.12
VIRTUAL_ENV=/workspace/venvs/sglang uv pip install "sglang[all]"   # 0.5.18+
```

Use the NVFP4 checkpoint (`QUASAR-QAT`, genuine `nvfp4-pack-quantized`, 4-bit,
group 16, FP8 scales). **Do not use `unsloth/Qwen3.8-27B-NVFP4`** — despite the
name its config is `float-quantized` at 8 bits, not NVFP4.

Measure, at bs=1:

1. Raw decode, short context.
2. **Decode vs context length — this is the most valuable single measurement.**
   llama.cpp holds 76.7% of the wall at short context and collapses to 49.6% at
   200k. If SGLang collapses the same way, headroom argument #5 in `CLAUDE.md`
   stops being "llama.cpp is weak here" and becomes "the whole field is weak
   here", which is a much stronger and better-defended claim. If SGLang holds up,
   that argument needs revising downward — report it honestly either way.
3. With DSpark speculation, and record the **mean accepted length**. That number
   is the entire lever on the project's effective-throughput projection, which
   currently assumes N≈2.5 and is the least-supported assumption in
   `feasibility.md`.

Compute % of roofline as `measured ÷ (bandwidth ÷ bytes-read-per-token)`, using
18.788 GB for NVFP4 weights plus the KV actually re-read each step (32 KB per
token of context at FP8, over the 16 attention layers). **Never compare raw tok/s
across engines running different quantizations** — llama.cpp at 4.79 bpw and
NVFP4 at an effective 5.6 bpw read different amounts, and tok/s alone will
mislead by ~15%.

### vLLM at bs=1

Lower value — a general-engine tax reference. Do it after SGLang.

## Reference numbers from machine 25132

For detecting regressions and for judging whether CUDA 13 actually helps.

| | |
|---|---|
| Read bandwidth (best-of, not mean) | 1615 GB/s, 90% of the 1792 spec |
| Copy (read+write) | 1519 GB/s |
| Naive Triton bf16 GEMV | 1569 GB/s, 97% of wall |
| 93 realistic GEMVs in one CUDA graph | 1579 GB/s, **97.7% of wall** |
| Same, eager | 1531 GB/s, 94.8% |
| 48 GDN layers, eager | 4.39 ms/token |
| 48 GDN layers, one CUDA graph | 0.15 ms/token |
| llama.cpp raw (`llama-bench` tg128) | 80.86 ± 0.22 tok/s = 80.6% of wall |
| llama.cpp + MTP | ~91 tok/s (+17%) |
| llama.cpp at 200k | 40.6 tok/s = 49.6% of wall |
| VRAM, llama.cpp at 256k with q8_0 KV | 29.4 GB of 32.6 GB |

## Traps already paid for

Each of these cost real time on the old machine. Do not rediscover them.

- **Never name a scratch script after a stdlib module** (`nt.py`, `os.py`). The
  script directory goes on `sys.path` first, the interpreter dies before CUDA
  init, and `nsys` faithfully records a trace containing no GPU work. An empty
  trace looks exactly like a profiler permission failure. This cost an incorrect
  conclusion that `nsys` did not support Blackwell — it does; both 2024.6.2 and
  2025.1.3 trace `sm_120` correctly.
- **Test profiler output positively.** Checking that `nsys stats` output lacks
  the string `SKIPPED` reports success when the report file does not exist at all
  (stats then errors with a different message). Count kernel rows instead.
- **`pkill -f <pattern>` matches your own shell** when the pattern appears in the
  command line you are currently running. It kills the calling shell. Use
  `pkill -x llama-server`, or `pgrep` + `kill` in a separate step.
- **`nohup cmd &` returns the launcher's exit code, not the command's.** A build
  that is 22% done reports "completed, exit 0". Wait on an actual artifact.
- **Measure bandwidth as best-of, never mean.** The card idles at 810 MHz memory
  clock and boosts to 14001 MHz. A run starting cold read 1142 GB/s against a
  true 1615 — a 29% error. `check_bandwidth.py` already does best-of; keep it
  that way.
- **Port 8080 is taken by Jupyter** on the vast base image. Use 8090.
- **`llama-cli` no longer accepts `-no-cnv`**; use `--single-turn`.
- **Do not install or upgrade the NVIDIA driver from apt.** It is host-injected
  and must match the host kernel module. If a package wants a newer CUDA major
  than the driver supports, the answer is a different machine, not a new driver.

## Local chat setup (optional, not part of the project)

The old machine also ran the model as a daily driver through Claude Code. This is
convenience tooling, deliberately kept outside the repo; rebuild it only if
wanted. It lived in `/workspace/rivals/` with its own README.

llama.cpp serves the Anthropic Messages API natively at `/v1/messages`, so Claude
Code needs no proxy — point `ANTHROPIC_BASE_URL` at it. Two things are required:

1. **Patch the chat template.** Claude Code appends a system message at the *end*
   of the messages array (its system-reminder mechanism), and Qwen's stock
   template hard-fails on any system message not at position 0. Extract the
   template from the GGUF, replace the
   `raise_exception('System message must be at the beginning.')` branch with one
   that renders the message as a user turn, and pass `--chat-template-file`.
2. **Match the context windows.** Claude Code does not recognise this model and
   assumes a 200k window, which will overrun a server started with less. Set
   `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to the server's `--ctx-size`.

At 256k with `q8_0` KV this used 29.4 GB of 32.6 GB. f16 KV does not fit.

## Open questions to settle on the new machine

1. **Does CUDA 13 actually help on `sm_120`?** Re-run the speed-of-light
   experiment (93 realistic GEMVs in one CUDA graph — 97.7% on CUDA 12.8) and
   check whether cuBLAS still dispatches Ampere-lineage `cutlass_80_tensorop_*`
   kernels for bf16 matmul. If it now dispatches native Blackwell kernels, that
   both justifies the move and weakens headroom argument #4, which cites the
   Ampere fallback as evidence. Report it honestly either way.
2. **Are GPU performance counters readable here?** See Step 0.
3. **Does SGLang degrade at long context like llama.cpp does?** See Step 3.
4. **What is DSpark's real mean accepted length on this model?** The 220–280
   tok/s effective projection assumes N≈2.5 and is the weakest link in
   `feasibility.md`.

## Documents to update once measurements land

- `docs/environment.md` — new machine identity, bandwidth, profiler state.
- `docs/baselines.md` — re-measured llama.cpp figures, new SGLang section.
- `docs/feasibility.md` — the projection is anchored to 1615 GB/s and 97.7%;
  both may move.
- `CLAUDE.md` — physics section, roofline table, and the rival table.

Write documents in the present tense, describing the current state only. The
project's convention is that no document narrates what a number used to be.
