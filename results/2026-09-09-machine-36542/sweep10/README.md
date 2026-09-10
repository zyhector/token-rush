# Ten-point decode-vs-context sweep — 2026-09-10, vast 36542 (RTX 5090)

The data behind the context-length plots. Same machine and stacks as the
Phase 4 run one directory up, measured 02:46–06:00 the following night; the
protocol run's four points (0 / 22k / 90k / 200k) stand as the reported
tables, this is the dense version for drawing. `sweep10.csv` is the whole
thing (116 rows; `scripts/sweep_table.py` regenerates it from these logs).

Contexts: 0, 8k, 16k, 32k, 64k, 96k, 128k, 160k, 200k, 240k tokens already
in the cache; then 128 (llama-bench) / 256 (servers) / 200 (ours, real
text) / 30 timed steps (ours, raw) tokens of decode.

| file | what | points |
|---|---|---|
| `engine_raw_{triton,marlin}.log` | our raw decode, random-token prefill, fp8 KV, both GEMV backends (`bench/decode.py`) | 10 + 10 |
| `engine_spec_{prose,code}_{mtp,dflash}.log` | our speculative decode on real text (WikiText-103 / torch sources, `bench/spec_context.py`); tok/s, accepted/step, ms/step | 4 × 10 |
| `llama_depth.log` | llama.cpp `llama-bench -d`, q8_0 KV | 10 |
| `llama_mtp_context.log` | llama.cpp `--spec-type draft-mtp` on prose prompt files of each length (`/workspace/data/prompts/prose_N.txt`, cut with the HF tokenizer) | 10 |
| `vllm_context.log` | vLLM at `--max-model-len 245000 --gpu-memory-utilization 0.95` | 10 |
| `sglang_context.log` | SGLang with `--max-total-tokens 245000` | 10 |
| `exl3_context.log` | ExLlamaV3, FP16 cache | 9 — **240k does not fit** (15.7 GB of cache next to 13.2 GB of weights) |
| `exl3_mtp_context.log` | ExLlamaV3 chained MTP (2 draft tokens), random-token context | 7 — **160k and up do not fit** with the draft resident |
| `*_server_*.log` | the servers' launch lines and logs | |

Reading the speculative rows: the step cost (ms/step) is the engine's
property and grows smoothly with context; the accepted tokens per step is
the text's property at that position (a 200-token continuation of one
spot in one corpus) and bounces between 2.4 and 4.4, so tok/s zig-zags.
Plot ms/step, or tok/s with that caveat. llama.cpp's MTP row is on the
same prose (prompt files), ExLlamaV3's on random tokens (its harness).

Skipped, with the reason: ollama has no raw mode and no long-context
harness; SGLang + DSpark only fits a 30k window on this card; vLLM's
speculative paths are slower than its raw decode at every length measured
in Phase 4, so they were not swept.
