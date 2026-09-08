#!/usr/bin/env python3
"""Dump the HF transformers reference for the engine-correctness gate.

    python bench/hf_reference.py --src /workspace/models/Qwen3.8-27B --out /workspace/ref/hf_ref.pt

Runs Qwen3_5ForCausalLM in bf16, CPU-offloaded, with `fla` blocked so the
GDN path is HF's pure-torch implementation (fla's fused decode kernel is
miscompiled on sm_120, docs/environment.md). Saves, for one chat-formatted
prompt: the residual stream after the embedding and after every layer, the
prompt logits, and `steps` greedy tokens with the full logits at each step.
"""
import argparse
import sys
import time

sys.modules["fla"] = None              # `import fla` now raises: HF falls back to torch
sys.modules["causal_conv1d"] = None

import torch  # noqa: E402
from transformers import AutoTokenizer, Qwen3_5ForCausalLM  # noqa: E402

PROMPT = "Explain in three sentences why the sky is blue, then name two things that are also blue."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--gpu-mem", default="26GiB")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.src)
    text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(text)
    print(f"prompt: {len(ids)} tokens")

    t0 = time.time()
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        a.src, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: a.gpu_mem, "cpu": "56GiB"}, output_loading_info=True)
    model.eval()
    print(f"loaded in {time.time() - t0:.0f}s; missing {len(info['missing_keys'])}, "
          f"unexpected {len(info['unexpected_keys'])}, mismatched {len(info.get('mismatched_keys', []))}")
    if info["missing_keys"]:
        print("  missing e.g.", info["missing_keys"][:5])
    gdn = model.model.layers[0].linear_attn
    import transformers.models.qwen3_5.modeling_qwen3_5 as m
    print("GDN chunk impl:", m.torch_chunk_gated_delta_rule.__module__, "| recurrent impl:",
          m.torch_recurrent_gated_delta_rule.__module__, "| attn:", model.config._attn_implementation)

    dev = model.device
    x = torch.tensor([ids], device=dev)
    with torch.no_grad():
        t0 = time.time()
        out = model(input_ids=x, output_hidden_states=True, use_cache=True)
        print(f"prefill {time.time() - t0:.1f}s; {len(out.hidden_states)} hidden states of {tuple(out.hidden_states[0].shape)}")
        hidden = torch.stack([h[0].to("cpu", torch.bfloat16) for h in out.hidden_states])   # [L+1, T, hidden]
        prompt_logits = out.logits[0].to("cpu", torch.bfloat16)                               # [T, vocab]
        pkv = out.past_key_values
        nxt = out.logits[0, -1].argmax()
        gen, gen_logits = [], []
        t0 = time.time()
        for i in range(a.steps):
            gen.append(int(nxt))
            out = model(input_ids=nxt.view(1, 1), past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            gen_logits.append(out.logits[0, -1].to("cpu", torch.bfloat16))
            nxt = out.logits[0, -1].argmax()
            if i % 8 == 7:
                print(f"  step {i + 1}: {(time.time() - t0) / (i + 1):.1f} s/step  {tok.decode(gen)!r}")
    print("generated:", repr(tok.decode(gen)))
    torch.save({"prompt_text": text, "input_ids": torch.tensor(ids), "hidden": hidden,
                "prompt_logits": prompt_logits, "gen_tokens": torch.tensor(gen),
                "gen_logits": torch.stack(gen_logits), "transformers": __import__("transformers").__version__},
               a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
