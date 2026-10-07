"""Single-stream decode speed of any OpenAI-compatible server, measured the same way for
every engine: the Phase 0 prompts (bench/mtp_accept.py, thinking off), greedy, streamed,
decode tok/s = tokens after the first / time after the first. Used for token-rush's own
server, vllm-metal and SGLang's MLX backend on one Mac (docs/mac.md).

    uv run python bench/mac/rival_bench.py --url http://127.0.0.1:8000 --name token-rush
"""
import argparse
import ast
import json
import os
import time
import urllib.request


def phase0_prompts():
    src = open(os.path.join(os.path.dirname(__file__), "..", "mtp_accept.py")).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PROMPTS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise LookupError("PROMPTS not found")


def run(url, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    chunks, usage = 0, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and (d["choices"][0].get("delta") or {}).get("content"):
                now = time.perf_counter()
                first = first or now
                last = now
                chunks += 1
    n = (usage or {}).get("completion_tokens", chunks)
    return {"ttft_s": first - t0, "tokens": n, "decode_tok_s": (n - 1) / (last - first) if last > first else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default=None, help="the served model name; default: the first /v1/models entry")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--json")
    a = ap.parse_args()
    model = a.model or json.load(urllib.request.urlopen(a.url + "/v1/models"))["data"][0]["id"]
    prompts = phase0_prompts()
    run(a.url, model, "Say hello.", 16)                       # warm up
    rows = []
    for fam, p in prompts.items():
        best = max((run(a.url, model, p, a.tokens) for _ in range(a.reps)), key=lambda r: r["decode_tok_s"])
        rows.append({"engine": a.name, "family": fam, **best})
        print(f"{a.name:<14} {fam:<6} {best['decode_tok_s']:6.1f} tok/s  ({best['tokens']} tokens, TTFT {best['ttft_s']:.2f} s)",
              flush=True)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
