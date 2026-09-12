#!/usr/bin/env python3
"""Raw-completion decode tok/s against a running llama-server, from its own timings.

llama-cli wraps every prompt in the chat template (the model then *answers* the
text, with thinking), so the multi-position protocol of bench/spec_context.py
— continue the book at a position — runs through llama-server's /completion
instead: the prompt file is the raw prompt, `n_predict` fixed, greedy, no
prompt caching. tok/s is the server's `timings.predicted_n / predicted_ms`
(decode only); with `--spec-type` the timings also carry draft_n and
draft_n_accepted, printed as accepted tokens per verify step.

  llama_server_bench.py files a.txt b.txt ... [--n 512] [--url http://127.0.0.1:8080]
"""
import argparse
import json
import urllib.request


def completion(url, prompt, n):
    payload = {"prompt": prompt, "n_predict": n, "temperature": 0, "cache_prompt": False, "stream": False,
               "return_tokens": False}
    req = urllib.request.Request(f"{url}/completion", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["files"])
    p.add_argument("files", nargs="+")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    a = p.parse_args()
    for path in a.files:
        out = completion(a.url, open(path).read(), a.n)
        t = out["timings"]
        tps = t["predicted_n"] / (t["predicted_ms"] / 1e3)
        extra = ""
        if t.get("draft_n"):
            # each verify step decodes one target token plus the accepted drafts
            steps = t["predicted_n"] - t["draft_n_accepted"]
            extra = f"   {t['predicted_n'] / max(steps, 1):.2f} per step (drafted {t['draft_n']}, accepted {t['draft_n_accepted']})"
        print(f"file {path}: {tps:7.1f} tok/s   (prompt {t['prompt_n']} tok at {t['prompt_per_second']:.0f} t/s, "
              f"{t['predicted_n']} new){extra}", flush=True)


if __name__ == "__main__":
    main()
