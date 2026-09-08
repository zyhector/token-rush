"""python -m tokenrush.run --model <packed dir> --prompt "..." [--chat] [--max-new N]"""
import argparse
import json
import time

import torch
from transformers import AutoTokenizer

from .generate import generate
from .model import Engine
from .mtp import MTPHead, build_mtp
from .spec import generate_spec_graph
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
    ap.add_argument("--eager", action="store_true", help="decode eagerly instead of replaying CUDA graphs")
    ap.add_argument("--no-spec", action="store_true", help="raw decode instead of speculative (MTP chain)")
    ap.add_argument("--spec-depth", default="3:4", help="Kmin:Kmax adaptive draft depth")
    ap.add_argument("--draft-vocab", default="64k", choices=("64k", "full"),
                    help="the draft chain's lm_head: the 65536 most frequent token ids (default) or the full vocabulary")
    ap.add_argument("--kv", default="bf16", choices=("bf16", "fp8"), help="KV cache dtype")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    if not is_packed(a.model):
        raise SystemExit(f"{a.model} is not a packed checkpoint; run python -m tokenrush.quantize first")

    spec = not a.no_spec and not a.eager
    kmin, kmax = (int(v) for v in a.spec_depth.split(":"))
    cfg, w, mtp_t = load_packed(a.model, backend=a.backend, with_mtp=spec)
    tok = AutoTokenizer.from_pretrained(a.model)
    engine = Engine(cfg, w, max_len=a.max_len, kv_dtype=torch.float8_e4m3fn if a.kv == "fp8" else torch.bfloat16,
                    max_spec=kmax if spec else 0)
    mtp = None
    if not a.eager:
        t0 = time.perf_counter()
        engine.capture()
        msg = f"captured the decode graph"
        if spec:
            mtp = MTPHead(cfg, build_mtp(cfg, mtp_t, "cuda", int4=True), w.embed, w.lm_head, a.max_len,
                          kv_dtype=engine.state.kv_dtype)
            dv = None
            if a.draft_vocab == "64k":
                import os
                dv = torch.load(os.path.join(os.path.dirname(__file__), "draft_vocab_64k.pt")).long()
            engine.attach_mtp(mtp, draft_vocab=dv)
            for k in range(kmin, kmax + 1):
                engine.capture_spec(k)
            msg += f" and speculative graphs for depths {kmin}..{kmax}"
        print(f"{msg} in {time.perf_counter() - t0:.1f}s")
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
    if spec:
        _, st = generate_spec_graph(engine, mtp, tok, ids, a.max_new, stop, chunk=a.chunk, dynamic=(kmin, kmax),
                                    temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, seed=a.seed)
    else:
        _, st = generate(engine, tok, ids, a.max_new, stop, chunk=a.chunk, graphed=not a.eager,
                         temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, seed=a.seed)
    print("---")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()}, default=str))


if __name__ == "__main__":
    main()
