#!/usr/bin/env python3
"""Collect the ten-point decode-vs-context sweep (results/<run>/sweep10/*.log)
into one CSV and a markdown table: engine, config, context, tok/s, bytes per
step, ceiling, % of the 1701 GB/s wall, ms/step and accepted tokens per step
where speculative. Byte counts are the ones docs/baselines.md derives from the
checkpoint headers; KV bytes per token are the cache format's exact size.

    python scripts/sweep_table.py results/2026-09-09-machine-36542/sweep10
"""
import csv
import os
import re
import sys

W = 1701.0
KV = {"fp8": 32768, "fp8+scales": 33280, "q8_0": 34816, "fp16": 65536}
ENGINES = {  # engine, config -> (weights GB, kv format)
    ("Token Rush", "raw, triton GEMV"): (13.65, "fp8+scales"),
    ("Token Rush", "raw, Marlin layout"): (13.65, "fp8+scales"),
    ("Token Rush", "spec, DFlash2, prose"): (13.65, "fp8+scales"),
    ("Token Rush", "spec, MTP chain, prose"): (13.65, "fp8+scales"),
    ("Token Rush", "spec, DFlash2, code"): (13.65, "fp8+scales"),
    ("Token Rush", "spec, MTP chain, code"): (13.65, "fp8+scales"),
    ("llama.cpp", "raw"): (15.387, "q8_0"),
    ("llama.cpp", "spec, MTP x1, prose"): (15.387, "q8_0"),
    ("vLLM", "raw"): (16.245, "fp8"),
    ("SGLang", "raw"): (16.245, "fp8"),
    ("ExLlamaV3", "raw"): (13.184, "fp16"),
    ("ExLlamaV3", "spec, MTP x2"): (13.184, "fp16"),
}


def rows_of(d):
    out = []
    ctx = re.compile(r"context\s+(\d+):\s+([\d.]+) tok/s")

    def add(engine, config, c, tps, ms=None, acc=None):
        gb, kv = ENGINES[(engine, config)]
        b = gb + c * KV[kv] / 1e9
        out.append({"engine": engine, "config": config, "context": c, "tok_s": round(tps, 1), "bytes_per_step_gb": round(b, 3),
                    "ceiling_tok_s": round(W / b, 1), "pct_of_wall": round(tps * b / W * 100, 1),
                    "ms_per_step": ms, "accepted_per_step": acc})

    for b, cfg in (("triton", "raw, triton GEMV"), ("marlin", "raw, Marlin layout")):
        for m in re.finditer(r"context (\d+): ([\d.]+) ms/step = ([\d.]+) tok/s", open(f"{d}/engine_raw_{b}.log").read()):
            add("Token Rush", cfg, int(m.group(1)) - 40, float(m.group(3)), float(m.group(2)))
    for t in ("prose", "code"):
        for dr, name in (("dflash", "DFlash2"), ("mtp", "MTP chain")):
            for m in re.finditer(r"context\s+(\d+): raw\s+\S+ tok/s \| spec\s+([\d.]+) tok/s, ([\d.]+) tokens/step, ([\d.]+) ms/step",
                                 open(f"{d}/engine_spec_{t}_{dr}.log").read()):
                c = int(m.group(1))
                add("Token Rush", f"spec, {name}, {t}", 0 if c == 64 else c, float(m.group(2)), float(m.group(4)), float(m.group(3)))
    for m in re.finditer(r"tg128(?: @ d(\d+))?\s+\|\s+([\d.]+) ±", open(f"{d}/llama_depth.log").read()):
        add("llama.cpp", "raw", int(m.group(1) or 0), float(m.group(2)))
    for m in re.finditer(r"context (\d+): \[ Prompt: [\d.]+ t/s \| Generation: ([\d.]+) t/s", open(f"{d}/llama_mtp_context.log").read()):
        add("llama.cpp", "spec, MTP x1, prose", int(m.group(1)), float(m.group(2)))
    for f, e, cfg in (("vllm_context.log", "vLLM", "raw"), ("sglang_context.log", "SGLang", "raw"), ("exl3_context.log", "ExLlamaV3", "raw")):
        for m in ctx.finditer(open(f"{d}/{f}").read()):
            add(e, cfg, int(m.group(1)), float(m.group(2)))
    for m in re.finditer(r"context\s+(\d+):\s+([\d.]+) tok/s\s+([\d.]+) per step", open(f"{d}/exl3_mtp_context.log").read()):
        add("ExLlamaV3", "spec, MTP x2", int(m.group(1)), float(m.group(2)), None, float(m.group(3)))
    return out


def main():
    d = sys.argv[1]
    rows = rows_of(d)
    with open(os.path.join(d, "sweep10.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    ctxs = sorted({r["context"] for r in rows})
    print("| engine / config | " + " | ".join(f"{c // 1000}k" if c else "0" for c in ctxs) + " |")
    print("|---|" + "---|" * len(ctxs))
    for key in ENGINES:
        rr = {r["context"]: r for r in rows if (r["engine"], r["config"]) == key}
        if not rr:
            continue
        cells = [(f"{rr[c]['tok_s']:.0f} ({rr[c]['pct_of_wall']:.0f}%)" if c in rr else "—") for c in ctxs]
        print(f"| {key[0]}, {key[1]} | " + " | ".join(cells) + " |")
    print(f"\n{len(rows)} rows -> {d}/sweep10.csv")


if __name__ == "__main__":
    main()
