#!/usr/bin/env python3
"""The short-context matrix of a measurement run, read from the raw logs: for the
engine and every rival, raw decode tok/s (and % of the 1701 GB/s wall on that
engine's bytes per step), raw at 200k, and the speculative essay / code / math
row for each configuration that was run. Prints markdown and writes
<dir>/matrix.csv.

    python scripts/matrix_table.py results/2026-09-12-machine-59052
    python scripts/matrix_table.py results/2026-09-09-machine-36542   # the Phase 4 run, same layout

Rival rows are the median of the three llama-cli repeats, the best-of-3 the
server benches print, ollama's eval_count / eval_duration; the engine's are
bench/decode.py and bench/families.py. Byte counts as in docs/baselines.md
(checkpoint headers, embedding excluded; KV per token at 200k: fp8 32768 B,
fp8+scales 33280, q8_0 34816, fp16 65536).
"""
import csv
import glob
import os
import re
import statistics
import sys

W = 1701.0
KV = {"fp8": 32768, "fp8+scales": 33280, "q8_0": 34816, "fp16": 65536}
BYTES = {"Token Rush": (13.65, "fp8+scales"), "llama.cpp": (15.387, "q8_0"), "ollama": (15.83, "q8_0"),
         "vLLM": (16.245, "fp8"), "SGLang": (16.245, "fp8"), "ExLlamaV3": (13.184, "fp16")}
FAM = ("essay", "code", "math")


def read(p):
    return open(p).read() if os.path.exists(p) else ""


def pct(engine, tps, ctx=0):
    gb, kv = BYTES[engine]
    return tps * (gb + ctx * KV[kv] / 1e9) / W * 100


def three(text, pat=r"^(essay|code|math)\s+([\d.]+) tok/s"):
    d = {m.group(1): float(m.group(2)) for m in re.finditer(pat, text, re.M)}
    return [d.get(f) for f in FAM]


def fmt3(v):
    return " / ".join("—" if x is None else f"{x:.1f}" for x in v) if v and any(x is not None for x in v) else "—"


def main():
    d = sys.argv[1].rstrip("/")
    rows = []   # engine, config, raw_short, raw_200k, spec essay/code/math, note

    def add(engine, config, raw=None, raw200=None, spec=None, note=""):
        rows.append({"engine": engine, "config": config, "raw_short": raw, "pct_short": pct(engine, raw) if raw else None,
                     "raw_200k": raw200, "pct_200k": pct(engine, raw200, 200000) if raw200 else None,
                     "spec_essay": spec[0] if spec else None, "spec_code": spec[1] if spec else None,
                     "spec_math": spec[2] if spec else None, "note": note})

    # engine
    e = f"{d}/engine"
    dec = lambda f: (lambda m: float(m.group(1)) if m else None)(re.search(r"= ([\d.]+) tok/s", read(f"{e}/{f}")))
    fam = read(f"{e}/families_greedy.log")
    for dr, name in (("dflash", "DFlash2 in-graph, K=7"), ("mtp", "MTP chain in-graph, depth 3:4")):
        m = re.search(rf"^{dr}\s+\|\s+(\d+) [\d.]+ [\d.]+ \|\s+(\d+) [\d.]+ [\d.]+ \|\s+(\d+) [\d.]+ [\d.]+", fam, re.M)
        spec = [float(m.group(i)) for i in (1, 2, 3)] if m else None
        add("Token Rush", f"spec, {name}", dec("decode_marlin.log"), dec("decode_200k_marlin.log"), spec, "raw on the Marlin layout the verify step shares")
    add("Token Rush", "raw, --backend triton", dec("decode_triton.log"), dec("decode_200k_triton.log"), None, "the fast raw kernel")
    # llama.cpp
    l = f"{d}/llama.cpp"
    m = re.search(r"tg128\s+\|\s+([\d.]+) ±", read(f"{l}/llama_bench_tg128.log"))
    tg = float(m.group(1)) if m else None
    m = re.search(r"tg128 @ d200000\s+\|\s+([\d.]+) ±", read(f"{l}/llama_depth.log"))
    tg200 = float(m.group(1)) if m else None

    def cli(name):
        out = []
        for f in FAM:
            v = []
            for p in sorted(glob.glob(f"{l}/llama_cli_{name}_{f}_r*.log")):
                mm = re.findall(r"Generation: ([\d.]+) t/s", read(p))
                if mm:
                    v.append(float(mm[-1]))
            out.append(statistics.median(v) if v else None)
        return out
    add("llama.cpp", "raw (llama-bench tg128; llama-cli median of 3)", tg, tg200, None, f"llama-cli raw {fmt3(cli('raw'))}")
    for name, cfg in (("mtp", "spec, built-in MTP (default n-max 3)"), ("mtp4", "spec, built-in MTP n-max 4"), ("dspark", "spec, DSpark draft (bf16)"), ("dflash", "spec, DFlash2 draft (bf16)")):
        v = cli(name)
        if any(v):
            add("llama.cpp", cfg, None, None, v, "median of 3, includes thinking")
    # ollama
    v = three(read(f"{d}/ollama/results.log"))
    if any(v):
        add("ollama", "default: MTP chain n-max 4 (no raw mode)", None, None, v)
    # vLLM
    vd = f"{d}/vllm"
    m = re.search(r"context\s+200000:\s+([\d.]+) tok/s", read(f"{vd}/results_context.log"))
    raw = three(read(f"{vd}/results_raw.log"))
    add("vLLM", "raw", raw[0] if raw else None, float(m.group(1)) if m else None, None, f"raw essay / code / math {fmt3(raw)}")
    for f, cfg in (("results_mtp1.log", "spec, MTP x1"), ("results_dspark.log", "spec, DSpark"), ("results_dflash.log", "spec, DFlash2")):
        v = three(read(f"{vd}/{f}"))
        if any(v):
            add("vLLM", cfg, None, None, v)
    # SGLang
    sd = f"{d}/sglang"
    m = re.search(r"context\s+200000:\s+([\d.]+) tok/s", read(f"{sd}/results_context.log"))
    raw = three(read(f"{sd}/results_raw.log"))
    add("SGLang", "raw", raw[0] if raw else None, float(m.group(1)) if m else None, None, f"raw essay / code / math {fmt3(raw)}")
    spec = three(read(f"{sd}/results_spec.log"))
    acc = re.findall(r"mean accepted length ([\d.]+)", read(f"{sd}/results_spec.log"))
    if any(spec):
        add("SGLang", "spec, DSpark block 7", None, None, spec, "mean accepted length " + " / ".join(acc))
    # ExLlamaV3
    xd = f"{d}/exllamav3"
    m = re.search(r"context\s+200000:\s+([\d.]+) tok/s", read(f"{xd}/exl3_context.log"))
    raw = three(read(f"{xd}/exl3_raw.log"))
    add("ExLlamaV3", "raw", raw[0] if raw else None, float(m.group(1)) if m else None, None, f"raw essay / code / math {fmt3(raw)}")
    for f, cfg in (("exl3_mtp1.log", "spec, MTP x1"), ("exl3_mtp2.log", "spec, MTP x2")):
        v = three(read(f"{xd}/{f}"))
        acc = re.findall(r"-> ([\d.]+) per step", read(f"{xd}/{f}"))
        if any(v):
            add("ExLlamaV3", cfg, None, None, v, "per step " + " / ".join(acc))

    with open(f"{d}/matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("| engine | config | raw, short | % wall | raw at 200k | % wall | spec essay / code / math | note |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        f = lambda x, s="": "—" if x is None else f"{x:.1f}{s}"
        print(f"| {r['engine']} | {r['config']} | {f(r['raw_short'])} | {f(r['pct_short'], '%')} | {f(r['raw_200k'])} | {f(r['pct_200k'], '%')} | "
              f"{fmt3([r['spec_essay'], r['spec_code'], r['spec_math']])} | {r['note']} |")
    print(f"\n{len(rows)} rows -> {d}/matrix.csv")


if __name__ == "__main__":
    main()
