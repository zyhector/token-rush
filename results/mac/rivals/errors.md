# Rival runs on the M5 Pro, 2026-10-07 (docs/mac.md, 'Against vLLM and SGLang')

## vllm-metal 0.31.0
version 0.31.0
raw: see vllm-metal.json

--speculative-config mtp:
NotImplementedError: Speculative method 'mtp' is not supported on Metal (supported: Gemma4 MTP, draft_model, ngram, dflash, dspark).
--speculative-config dflash (z-lab/Qwen3.8-27B-DFlash2):
Value error, Model Runner V1 does not support: DFlash candidate-head drafts [type=value_error, input_value=ArgsKwargs((), {'model_co... 'shutdown_timeout': 0}), input_type=ArgsKwargs]
--speculative-config ngram (KMP_DUPLICATE_LIB_OK=TRUE):
NotImplementedError: Speculative decode verification is not supported for hybrid models on Metal yet.

## SGLang, MLX backend (SGLANG_USE_MLX=1)
v0.5.21 (run), any model incl. mlx-community/Qwen3-0.6B-4bit; v0.5.16-v0.5.20 have the same reference in the source (not run):
AttributeError: 'MlxModelRunnerStub' object has no attribute 'kv_cache_configurator'
v0.5.15, Qwen3.8-27B export, default: AssertionError: extra_buffer needs CUDA/MUSA/NPU (FLA).
v0.5.15, --mamba-scheduler-strategy no_buffer --disable-overlap-schedule:
AttributeError: 'NoneType' object has no attribute 'conv'
  at sglang/srt/layers/attention/linear/gdn_backend.py:277 (__init__)

## llama.cpp (brew build 11429 and release b11474), unsloth/Qwen3.8-27B-GGUF UD-Q4_K_M
Runs, but the output is wrong, so its speeds are not comparable and are left out of
docs/mac.md. Same text on Metal (`-ngl 999`, with and without `-fa off`) and on the CPU
(`-ngl 0`); tokenization matches ours ([760, 6511, 314, 9338, 369] for the prompt below).
/completion "The capital of France is", greedy:
  ' city of the the activity of the city of Paris. they remains the both are'
chat, "Write a Python function that checks whether a number is prime.", thinking off:
  '的模型语言（在其.名称是反的.'
Speeds of the garbled runs (llama.cpp-*.garbled.json, best of two, essay/code/math):
raw 16.0/16.2/15.9, draft-mtp 11.2/10.1/10.0 (mean accepted length 1.38-1.49),
draft-dflash with the Q4_K_M DFlash2 GGUF 9.7/7.8/9.4 (1.24-1.49): the drafts cannot
predict a broken target. Not tried: another quantization of the GGUF (no disk room).
