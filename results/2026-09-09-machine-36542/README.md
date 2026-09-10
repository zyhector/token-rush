# Raw evidence — 2026-09-09/10, vast machine 36542 (RTX 5090): the Phase 4 run

The unedited logs behind every number in `docs/baselines.md`,
`docs/environment.md` and the Phase 4 entry of `docs/progress.md` (step 32).
One machine, one sitting: the wall, every rival and the engine, in the order
listed. Nothing here is a result on its own; the tables in the docs are the
results, and this directory is what they were read from. The stage scripts
that drove each rival are in `scripts/` (they ran from a scratch directory;
copied here verbatim, including the `pkill` that cost one re-run).

| Directory / file | Backs |
|---|---|
| `env_check/` | `environment.md`: machine identity, profilers (nsys works, ncu blocked), the 1702 GB/s wall, stack validation (fla's fused GDN step still miscompiled), GEMV speed-of-light (96.4%), GDN launch tax (4.4 ms eager on this host) |
| `install/` | package versions as installed: `sglang[all]` 0.5.19, vllm 0.29.0, exllamav3 1.4.6, ollama 0.33.3, the llama.cpp cmake log (`434ddbbc0`, CUDA 13.3, `sm_120`) |
| `llama.cpp/llama_bench_tg128.log` | the tg128 row |
| `llama.cpp/llama_cli_{raw,mtp,mtp4,dspark,dflash}_{essay,code,math}_r{1,2,3}.log` | every `llama-cli` run, three per cell; `console.txt` is the one-line-per-run summary |
| `llama.cpp/llama_depth.log` | decode vs. context (`llama-bench -d`) |
| `exllamav3/exl3_{raw,context,mtp1,mtp2}.log` | ExLlamaV3 raw, context sweep, chained MTP with 1 and 2 draft tokens |
| `vllm/results_{raw,context,mtp1,dspark,dflash}.log`, `vllm/server_*.log` | vLLM raw, context sweep at 220k, MTP, DSpark, DFlash2; the server logs carry the launch args and the `SpecDecoding metrics` lines (acceptance lengths) |
| `vllm/dspark_ct_config.json` | the patched draft `config.json` vLLM needs for DSpark (`Qwen3DSparkModel` + a no-op compressed-tensors quant config) |
| `sglang/results_raw.log`, `results_context.log`, `server_raw.log`, `server_ctx.log` | SGLang raw and the context sweep; `server_raw.log` is the 16k-chunk server that OOM'd at 22k (the trap in `docs/traps.md`), `server_ctx.log` the 4096-chunk one the sweep ran on |
| `sglang/results_spec.log`, `server_spec.log` | SGLang + DSpark, with `spec_verify_ct` per request |
| `ollama/results.log`, `ollama/serve.log` | ollama's runner command line, per-request `print_timing` (eval ms/token, draft acceptance) — the evidence for the host-bound finding |
| `engine/decode_{marlin,triton}.log`, `decode_200k_{marlin,triton}.log` | raw decode, both GEMV backends, short and at 200k fp8 KV (`bench/decode.py`) |
| `engine/families_{greedy,sampled}.log` | the six-family speculative table, greedy and T=0.7 top-p 0.9, both drafts (`bench/families.py`) |
| `engine/spec_context_{prose,code}_{mtp,dflash}.log` | speculative and raw decode vs. context on real text (`bench/spec_context.py`); `*_shift_*` are the 200k attribution probe on the same corpora from 1.5 MB in |
| `engine/needle.log` | needle retrieval at 128k and 256k (`bench/needle.py`) |
| `engine/gsm8k.log`, `gsm8k_engine.json` | GSM8K, 200 problems, through the engine (`bench/quality_gsm8k_engine.py`) |

Timing method for the server engines is in the `scripts/rivals/*_bench.py`
docstrings: streamed requests, tokens between first and last chunk. The
long-text corpora are rebuilt by `bench/long_text.py`.
