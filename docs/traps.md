# Traps already paid for

Each of these cost real time. Do not rediscover them.

## Measuring

- **Measure bandwidth as best-of, never mean, over a big buffer.** The card
  idles at 405 MHz memory clock and boosts to 14001 MHz; a run starting cold
  reads ~30% low. A 512 MB buffer reads 4% low from fixed launch and reduce
  overhead alone. `check_bandwidth.py` does both right; keep it that way.
- **A kernel that runs is not a kernel that works.** `fla`'s fused GDN decode
  step returns the right shapes and dtypes and NaNs whole heads
  (`environment.md`). Differential-test every kernel against a reference, over
  several calls on the same inputs — the failure depends on the call sequence
  and passes once.
- **A kernel benchmark on one matrix reads from L2, not DRAM.** Anything
  under ~96 MB replayed alone in a CUDA graph reports 2500 GB/s on this card.
  Cycle through >400 MB of copies, or benchmark a set that does not fit
  (`scripts/gemv_shootout/`).
- **Bytes per token must use the KV format's real size.** `q8_0` is 34.8
  KB/token for this model's 16 attention layers, FP8 is 32 KB. A wrong KV byte
  count moves a "% of wall" figure at 200k by ten points.
- **`nohup cmd &` returns the launcher's exit code, not the command's.** A build
  that is 22% done reports "completed, exit 0". Wait on an actual artifact.
- **`sglang.launch_server` re-execs itself**, so the PID of the process you
  started exits while the server lives on. Poll `/health`, not the PID.
- **Test profiler output positively.** Checking that `nsys stats` output lacks
  `SKIPPED` reports success when the report file does not exist at all. Count
  kernel rows instead.
- **Never name a scratch script after a stdlib module** (`nt.py`, `os.py`). The
  interpreter dies before CUDA init and `nsys` records an empty trace that
  looks exactly like a permission failure. `nsys` traces `sm_120` fine.
- **`grep` in a pipeline into a log file buffers.** Without `--line-buffered`
  a long run shows nothing for half an hour and then prints everything at
  once, which is indistinguishable from a hang.
- **`pkill -f <pattern>` matches your own shell** when the pattern appears in
  the command line you are running — including a `kill-then-relaunch` one-liner
  whose relaunch half contains the very string being killed. Anchor the pattern
  to the executable (`pkill -f "^/workspace/venvs/vllm/bin/vllm"`), use
  `pkill -x <name>`, kill by PID, or kill in a separate step. The `[x]`
  bracket trick does not help when the same word also appears elsewhere in
  the command line (a shell function named after the thing being killed).
  Paid for seven times: the fifth was a Phase 4 wrapper script *named*
  `sglang_ctx.sh`, killed by the previous stage's `pkill -9 -f sglang`; the
  sixth, `pkill -f tokenrush.serve` typed into a shell whose own command line
  contained it. Kill servers by PID from a `ps | grep "[t]okenrush"` listing
  that excludes `bash`, and wait for `nvidia-smi` to show the memory gone
  before starting the next GPU process — a killed 30 GB process takes
  seconds to release the card, and the next process OOMs meanwhile.
- **ollama keeps its model resident for `keep_alive` (10 min by default)**
  after the last request, holding ~20 GB of VRAM; the next engine's startup
  fails with "free memory less than desired utilization". `ollama stop
  <model>` or kill the runner before launching anything else.

## Quantization and rival checkpoints

(The full thread is `docs/quantization.md`; these are the ones that cost time.)

- **A rival converter may permute heads.** llama.cpp's GGUF reorders the 48
  GDN value heads (its head `j` is HF's `3*(j % 16) + j // 16`). A relative
  error near 1.0 on *some* tensors and 0.08 on the rest is a layout
  difference, not a quality difference — match rows against the bf16 weights
  before believing any number computed from a foreign checkpoint.
- **A rival's *unquantized* tensors may differ too.** AWQ smoothing folds
  per-channel scales into the layer norms and small projections, so such a
  checkpoint must be read whole and never overlaid on the base model's norms.
  Qwen's RMSNorm gain is `1 + w`; undoing a smoothing with `w` gives
  infinities that look like a broken reader.
- **Published task-accuracy "99–101% recovery" can sit on a 5x worse KL.**
  Task scores are a floor, not a ranking; a 200-problem GSM8K has a ±3.5 point
  noise band and cannot rank 4-bit quantizations.
- **Cap generation length by looking at the outputs.** A 512-token limit
  truncated a third of the GSM8K answers and read as 65% accuracy where the
  same checkpoint scores 96.5% at 1024.
- **Save the expensive intermediate before the cheap step that can fail.**
  17 minutes of GPTQ died in `save_file` on a full disk; the quantizer now
  writes its codes first and can re-pack with `--from-codes`.
- **Differential-test a quantizer against its own degenerate case.** With an
  identity Hessian, GPTQ must reproduce round-to-nearest code for code; that
  one line catches sign, ordering and grid bugs.

## Building the engine

- **The eager prefill path costs ~1.8 s per call whatever its length.** A
  chunk longer than the fused row count (8) goes through the
  dequantize-then-GEMM backend, which unpacks all 16 GB of int4 to bf16
  every call. Fine amortized over 4096 tokens (that *is* the 1500 tok/s
  prefill number); fatal for a 20-token chat turn. The fused M-row path
  costs 1.4 ms/token at 8 rows; `Session` uses it for deltas up to 1024
  tokens. The first call of each row count M autotunes for 1–4 s — the
  server warms all of them at startup.
- **`forward_hidden` on a chunk of <= 8 tokens left the recurrent state in
  the wrong slot** (slot 0 instead of T-1, which `forward()` got right):
  a prompt whose last prefill chunk was 1–8 tokens (4097–4104 tokens, say)
  continued from the state after the chunk's *first* token. Never hit by a
  benchmark; found by the prefix-reuse tests, where short deltas are the
  common case. Fixed in `model.py`; `tests/test_session.py` guards it.
- **The graphed steps advance the drafts' cursors on the device only.**
  `DFlashDraft.ctx` (host) is stale after `spec_step_dflash` replays; the
  cache actually ends at `pos - n - 1`. Anything eager that touches the
  draft cache after graphed steps must set the host mirror first
  (`session._finish_spec`).

- **Coherent text is not a correctness signal.** Dropping three quarters of
  the last MLP layer's output (a `[-1:]` slice on split-K partials) left the
  chat output fluent and moved the teacher-forced KL against HF from 0.06 to
  0.20. Run `bench/gate_engine.py` after every change to the decode path;
  it takes 20 seconds. Random-weight tests with small init do not see
  bugs in small-magnitude contributions.

## Serving rivals on 32 GB

- **SGLang's hybrid state cache is sized in units of 5 slots per request.**
  `--max-mamba-cache-size` below 5 gives `max_num_reqs=0` and the server
  refuses to start; the default sizes it for dozens of requests and eats
  3.4 GB. Use 6 for bs=1.
- **SGLang captures prefill CUDA graphs up to `--chunked-prefill-size`** and
  the 32k default runs out of memory next to a 256k KV pool. Pass
  `--disable-prefill-cuda-graph`; decode graphs (the ones bs=1 cares about)
  still capture.
- **A 16k prefill chunk needs ~400 MB of scratch for the GDN chunk kernel.**
  With the KV pool taking everything `--mem-fraction-static` allows, that
  allocation fails on the first long prompt. Use `--chunked-prefill-size 4096`
  and bound the pool with `--max-total-tokens`.
- **Port 8080 is taken by Jupyter** on the vast base image. Use 8090 or 30000.
- **`llama-cli` no longer accepts `-no-cnv`**; use `--single-turn`. It prints
  `[ Prompt: N t/s | Generation: N t/s ]` at exit instead of `eval time` lines.

## The machine

- **Do not install or upgrade the NVIDIA driver from apt.** It is host-injected
  and must match the host kernel module. If a package wants a newer CUDA major
  than the driver supports, the answer is a different machine, not a new driver.
- **`ncu` is blocked on every vast host seen so far** (`ERR_NVGPUCTRPERM`).
  Plan on wall-clock timing against known byte counts.
- **Nothing on this instance survives a recycle.** `/workspace` is not a volume
  here (`workspace_is_volume: false`). Rebuild recipes live in
  `environment.md` and `baselines.md`; check `vast-capabilities` before
  assuming otherwise on a new machine.
- **MTP speedups are host-dependent.** The same llama.cpp binary and weights
  give +17% on one host and +60% on this one; the speculative loop has
  host-side work between tiny kernels. Name the machine with every number.

## Local chat setup (optional, not part of the project)

llama.cpp serves the Anthropic Messages API natively at `/v1/messages`, so
Claude Code needs no proxy — point `ANTHROPIC_BASE_URL` at it. Two things are
required:

1. **Patch the chat template.** Claude Code appends a system message at the
   *end* of the messages array, and Qwen's stock template hard-fails on any
   system message not at position 0. Extract the template from the GGUF,
   replace the `raise_exception('System message must be at the beginning.')`
   branch with one that renders the message as a user turn, and pass
   `--chat-template-file`.
2. **Match the context windows.** Claude Code assumes a 200k window. Set
   `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to the server's `--ctx-size`.

At 256k with `q8_0` KV this needs ~29.4 GB of 32.6 GB. f16 KV does not fit.

## A captured graph holds tensors by address

Anything a CUDA graph was captured over must stay allocated and must stay
*that* tensor. Re-assigning an engine attribute after capture (a second
`attach_*` allocating its own `drafts` buffer or its own copy of the draft
head) leaves the graph reading the old allocation: silently stale data if the
old tensor is still referenced somewhere, garbage and a device-side assert if
it was freed and reused (step 29: the 128k-row draft head replaced under the
DFlash graph produced out-of-range token ids). Allocate shared buffers once,
at engine construction; make `attach_*` idempotent for what it shares.
