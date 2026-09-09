---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - int4
  - gptq
  - token-rush
---

# Qwen3.8-27B-TokenRush-int4g128

4-bit weights for [Token Rush](https://github.com/zyhector/token-rush), a
single-stream inference engine for Qwen3.8-27B on one RTX 5090.

**This checkpoint only loads in that engine.** The packing is its own — group-wise
asymmetric int4 with a bf16 scale and a bf16 minimum per group of 128 along the
input dimension — so `transformers`, vLLM, SGLang and llama.cpp cannot read it.

| | |
|---|---|
| Format | int4, group 128, asymmetric: **4.25 bits per weight** |
| Method | GPTQ with a per-row MSE range search; 256 x 2048 calibration tokens |
| Quantized | the 401 text-path matrices, `lm_head` included |
| Left bf16 | embeddings, norms, conv, the small GDN projections, the MTP head |
| Size | 17.0 GB |
| KL divergence to bf16 | **0.0232** mean over 81,920 positions |
| GSM8K | **96.5%** against bf16's 96.0% (200 problems, greedy) |

Text path only; the vision tower is not included.

Details, the measurements behind those numbers, and the build recipe are in the
[repository](https://github.com/zyhector/token-rush).

Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
Apache 2.0.
