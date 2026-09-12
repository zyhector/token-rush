# Raw evidence — 2026-09-12, vast machine 59052 (RTX 5090): the README's run

The unedited logs behind every number in the top-level `README.md`, the
figures in `docs/img/`, the machine section of `docs/environment.md` and
step 35 of `docs/progress.md`. One machine, one sitting, in the order listed:
the machine checks, the engine, then every rival from the same recipes as
Phase 4 (`docs/baselines.md`, versions pinned to that day's), then the
engine's multi-position speculative sweep. Nothing here is a result on its
own; the tables in the docs are the results, and this directory is what they
were read from. The stage scripts that drove each part are in `scripts/`
(copied verbatim; `run_all.sh` is the queue that ran them).

Why this run exists: the first decode-vs-context sweep (`../2026-09-09-machine-36542/sweep10/`)
measured speculative decoding at one corpus position per context length, and
acceptance at a single position is a property of that text, so the curves
zig-zagged. This run measures **six positions per context length, 512 greedy
tokens each, on PG-19**, for both of our drafts and for llama.cpp's MTP on the
same prompt files, and aggregates tok/s as total tokens over total seconds
(`bench/spec_context.py`, `bench/cut_prompts.py`, `scripts/sweep_table.py`).

| Directory / file | Backs |
|---|---|
| `env_check/` | `environment.md`: machine identity (Core Ultra 9 285K, 62 GB, driver 595.84, a 500 W power cap), profilers (nsys works, ncu blocked), the 1703 GB/s wall, stack validation (fla's fused GDN step still miscompiled), GEMV speed-of-light (96.4%), GDN launch tax (5.22 ms eager, 1.51 graphed) |
| `install/` | package versions as installed: torch 2.14.0+cu130 / triton 3.8.0 / fla 0.6.0 in `/venv/main`; `sglang[all]` 0.5.19, vllm 0.29.0, exllamav3 1.4.6, ollama 0.33.3; the llama.cpp cmake log (`434ddbbc0`, CUDA 13.3, `sm_120`); the model downloads; the corpora build |
| `engine/decode_{marlin,triton}.log`, `decode_200k_{marlin,triton}.log` | raw decode, both GEMV backends, short and at 200k fp8 KV (`bench/decode.py`) |
| `engine/families_greedy.log` | the six-family speculative table, greedy, both drafts (`bench/families.py`) |
| `engine/needle.log` | needle retrieval at 128k and 256k (`bench/needle.py`) |
| `sweep/engine_raw_{triton,marlin}.log` | our raw decode vs. context, ten points, random-token prefill, fp8 KV (`bench/decode.py`) |
| `sweep/engine_spec_{pg19,code}_both.log` | **the multi-position sweep**: per (context, position) lines with raw, DFlash2 and MTP-chain tok/s, accepted/step, ms/step and whether the speculative output equalled raw greedy; `summary` lines with the aggregate and the min–max over positions (`bench/spec_context.py --positions 6 --new 512 --draft both`) |
| `sweep/engine_spec_code_160k_gdb.log` | the one unit that crashed five resumed attempts in a row at startup, run under `gdb` (no crash; its two lines were appended to the code log) |
| `sweep/engine_spec_*_fill.strace`, `sweep/engine_raw_marlin_96k.strace` | strace signal traces of the runs that caught the sweep's silent deaths: host-side SIGSEGV in the Marlin-backend prefill (`docs/traps.md`). The sweep logs carry `===== continued …` markers where a resumed run appended; `scripts/engine_spec_resume.sh` is the resumable driver, `bench/spec_context.py --resume-log` folds the earlier units into the summaries |
| `sweep.csv`, `matrix.csv` | all of the above and the rivals' sweeps in one table (`scripts/sweep_table.py`); the short-context matrix (`scripts/matrix_table.py`) |
| `llama.cpp/llama_bench_tg128.log` | the tg128 row |
| `llama.cpp/llama_cli_{raw,mtp,mtp4,dspark,dflash}_{essay,code,math}_r{1,2,3}.log` | every `llama-cli` run, three per cell |
| `llama.cpp/llama_depth.log` | decode vs. context, ten points (`llama-bench -d`, q8_0 KV) |
| `llama.cpp/llama_mtp_pg19.log`, `server_mtp_pg19.log` | llama.cpp's built-in MTP (`--spec-draft-n-max` default 3) on the 60 PG-19 prompt files through `llama-server` `/completion` (raw completion, 512 tokens; `scripts/rivals/llama_server_bench.py`; re-run by `scripts/llama_mtp_stage.sh` on port 8091 after the first attempt hit Jupyter on 8080) |
| `vllm/results_{raw,context,mtp1,dspark,dflash}.log`, `vllm/server_*.log` | vLLM raw, the ten-point context sweep at 245k, MTP, DSpark, DFlash2 |
| `sglang/results_{raw,context}.log`, `results_spec.log`, `results_spec_pg19.log`, `server_*.log` | SGLang raw and the ten-point sweep; SGLang + DSpark on the short prompts and on the PG-19 prompt files that fit its 30k window (contexts 0 / 8k / 16k) |
| `exllamav3/exl3_{raw,context,mtp1,mtp2,mtp_context}.log` | ExLlamaV3 raw, context (240k does not fit), chained MTP with 1 and 2 draft tokens, MTP ×2 vs. random-token context |
| `ollama/results.log`, `ollama/serve.log` | ollama's default (a 4-token MTP chain), per-request timings |
| `console_*.txt` | the console of each stage as it ran; `console_run_all.txt` is the first queue (`scripts/run_all.sh`), `console_run_engine*.txt` and `console_engine_spec_resume.txt` the follow-ups that re-ran what the silent deaths cut short |

The long-text corpora are rebuilt by `bench/long_text.py` (PG-19 from the
`emozilla/pg19` mirror, skipping the Bible and Shakespeare; WikiText-103 for
the needle; torch's sources for code), the prompt files by
`bench/cut_prompts.py` with the same positions the engine used (printed at the
top of `sweep/engine_spec_pg19_both.log`).
