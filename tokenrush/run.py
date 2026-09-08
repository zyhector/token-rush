"""python -m tokenrush.run --model <packed dir> --prompt "..." [--chat] [--max-new N]"""
import argparse
import json

import torch
from transformers import AutoTokenizer

from .generate import generate
from .model import Engine
from .weights import is_packed, load_packed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="packed checkpoint (see tokenrush.quantize)")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--chat", action="store_true", help="wrap the prompt in the chat template, thinking off")
    ap.add_argument("--think", action="store_true", help="with --chat, leave thinking on")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=32768, help="preallocated context")
    ap.add_argument("--chunk", type=int, default=4096, help="prefill chunk")
    ap.add_argument("--backend", default="triton", choices=("triton", "tinygemm", "dequant"), help="int4 GEMV")
    a = ap.parse_args()
    if not is_packed(a.model):
        raise SystemExit(f"{a.model} is not a packed checkpoint; run python -m tokenrush.quantize first")

    cfg, w, _ = load_packed(a.model, backend=a.backend)
    tok = AutoTokenizer.from_pretrained(a.model)
    engine = Engine(cfg, w, max_len=a.max_len)
    print(f"weights {w.nbytes / 1e9:.2f} GB, state {engine.state.nbytes / 1e9:.2f} GB, "
          f"cuda allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    if a.chat:
        text = tok.apply_chat_template([{"role": "user", "content": a.prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=a.think)
    else:
        text = a.prompt
    ids = tok.encode(text)
    stop = set(cfg.eos_ids) | {tok.eos_token_id}
    print(f"--- prompt ({len(ids)} tokens) ---\n{text}\n--- output ---")
    _, st = generate(engine, tok, ids, a.max_new, stop, chunk=a.chunk)
    print("---")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()}))


if __name__ == "__main__":
    main()
