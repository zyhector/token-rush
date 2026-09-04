# Raw evidence — 2026-09-04, vast machine 94372 (RTX 5090)

The unedited logs behind every number in `docs/baselines.md` and
`docs/environment.md`. Nothing here is a result on its own; the tables in the
docs are the results, and this directory is what they were read from.
Server logs include the exact launch arguments (`server_args=` /
`non-default args` lines) and the crash logs record the memory tuning that
the reproduce sections in `baselines.md` skip past.

| Directory / file | Backs |
|---|---|
| `env_check/` | `environment.md`: machine identity, profilers, bandwidth wall, stack validation (fla miscompile), GEMV speed-of-light, GDN launch tax |
| `llama.cpp/llama_cmake.log` | build configuration (CUDA 13.3, `sm_120`) |
| `llama.cpp/llama_bench_tg128.log`, `llama_cli_*.log` | `baselines.md` llama.cpp results table: raw, MTP, MTP n-max 4, DSpark and DFlash2 drafts, three prompts each |
| `llama.cpp/llama_depth.log` | llama.cpp decode vs. context (`llama-bench -d`) |
| `llama.cpp/llama_old_bench.log` | commit `6703d78` on this machine, the "MTP gain is not a code change" claim |
| `llama.cpp/llama_dspark.log`, `console/chain-llama-dspark-exl3-raw.txt`, `console/chain-vllm-dflash-dspark-llama-dflash-ollama.txt`, `console/chain-llama-mtp4.txt` | the draft-model runs as they printed |
| `sglang/results_raw_context.log`, `console/sglang-raw-context-streamed.txt` | SGLang raw and decode vs. context (streamed timing — the earlier `results_context.log` used the subtraction method and is superseded) |
| `sglang/results_spec.log`, `sglang/server_spec.log`, `console/sglang-dspark.txt` | SGLang + DSpark, with `spec_verify_ct` per request and the server's own accept-length log |
| `sglang/server_raw_done.log` | the raw/context server, launch args and decode log lines |
| `sglang/server_raw_crash*.log`, `server_raw_oom22k.log`, `server_spec_fail1.log` | the memory-tuning failures behind the flags in the reproduce section |
| `vllm/results_raw.log`, `vllm/server_raw.log` | vLLM raw decode |
| `vllm/results_context.log`, `vllm/server_ctx.log`, `console/vllm-context-220k.txt` | vLLM decode vs. context at `--max-model-len 220000` (`server_262k_fail.log` is why not 262144) |
| `vllm/results_mtp1.log`, `vllm/server_mtp1.log`, `console/vllm-mtp1.txt` | vLLM with its MTP head |
| `vllm/results_dspark3.log`, `vllm/server_dspark3.log` | vLLM + DSpark (`server_dspark.log`, `server_dspark2.log` are the two failed attempts: draft quant config, wrong draft class) |
| `vllm/results_dflash2.log`, `vllm/server_dflash2.log` | vLLM + DFlash2 (`server_dflash.log`: the KV-budget failure at 32k) |
| `exllamav3/exl3_results.log`, `exl3_results2.log`, `console/exl3-context-sweep.txt` | ExLlamaV3 raw and decode vs. context |
| `exllamav3/exl3_mtp.log`, `console/exl3-mtp.txt` | ExLlamaV3 chained MTP, 1 and 2 draft tokens |
| `ollama/serve.log`, `ollama/Modelfile` | ollama's runner command line (the `llama-server` invocation with `--spec-type draft-mtp --spec-draft-n-max 4`), offload state, MTP statistics; the bench numbers are in `console/chain-vllm-dflash-dspark-llama-dflash-ollama.txt` |
| `install/` | package versions as installed: `sglang[all]`, `vllm`, ollama, and the apt install of nsys 2026.1.3 + CUDA 13.3 |

Timing method for the server engines is in the `scripts/rivals/*_bench.py`
docstrings: streamed requests, tokens between first and last chunk. The
`sglang/results_context.log` file predates that fix and shows the
prefix-cache artifact (112 tok/s at 22k, above the physical ceiling) that
forced it.
