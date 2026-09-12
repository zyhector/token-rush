#!/usr/bin/env python3
"""Collect a decode-vs-context sweep into one CSV and a markdown table: engine,
config, context, tok/s, bytes per step, ceiling, % of the 1701 GB/s wall, ms/step
and accepted tokens per step where speculative. Byte counts are the ones
docs/baselines.md derives from the checkpoint headers; KV bytes per token are
the cache format's exact size.

Two layouts:

    python scripts/sweep_table.py results/2026-09-09-machine-36542/sweep10
        the first, single-position sweep: every log in one directory

    python scripts/sweep_table.py results/2026-09-12-machine-59052
        the multi-position protocol (docs/progress.md step 34): the engine's
        sweep/engine_spec_{pg19,code}_both.log `summary` lines carry the
        position-aggregated tok/s and acceptance with their min/max over
        positions; the rivals' per-file lines (llama.cpp/llama_mtp_pg19.log,
        sglang/results_spec_pg19.log) are aggregated here the same way — total
        tokens over total seconds per context. Extra CSV columns: tok_s_min,
        tok_s_max, accepted_min, accepted_max, positions.

Writes <dir>/sweep.csv (sweep10.csv for the old layout).
"""
import csv
import os
import re
import sys
from collections import defaultdict

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
    ("llama.cpp", "spec, MTP x1, prose"): (15.387, "q8_0"),          # the first sweep's label (llama-cli, default draft depth)
    ("llama.cpp", "spec, MTP n-max 3, prose"): (15.387, "q8_0"),      # the multi-position sweep: llama-server, --spec-draft-n-max default 3
    ("vLLM", "raw"): (16.245, "fp8"),
    ("SGLang", "raw"): (16.245, "fp8"),
    ("SGLang", "spec, DSpark, prose"): (16.245, "fp8"),
    ("ExLlamaV3", "raw"): (13.184, "fp16"),
    ("ExLlamaV3", "spec, MTP x2"): (13.184, "fp16"),
}
FIELDS = ["engine", "config", "context", "tok_s", "bytes_per_step_gb", "ceiling_tok_s", "pct_of_wall", "ms_per_step",
          "accepted_per_step", "tok_s_min", "tok_s_max", "accepted_min", "accepted_max", "positions"]


def make_add(out):
    def add(engine, config, c, tps, ms=None, acc=None, **spread):
        gb, kv = ENGINES[(engine, config)]
        b = gb + c * KV[kv] / 1e9
        row = {"engine": engine, "config": config, "context": c, "tok_s": round(tps, 1), "bytes_per_step_gb": round(b, 3),
               "ceiling_tok_s": round(W / b, 1), "pct_of_wall": round(tps * b / W * 100, 1),
               "ms_per_step": ms, "accepted_per_step": acc}
        row.update(spread)
        out.append(row)
    return add


def read(path):
    return open(path).read() if os.path.exists(path) else ""


def rows_single(d):
    """The first sweep's layout (results/2026-09-09-machine-36542/sweep10)."""
    out = []
    add = make_add(out)
    ctx = re.compile(r"context\s+(\d+):\s+([\d.]+) tok/s")
    for b, cfg in (("triton", "raw, triton GEMV"), ("marlin", "raw, Marlin layout")):
        for m in re.finditer(r"context (\d+): ([\d.]+) ms/step = ([\d.]+) tok/s", read(f"{d}/engine_raw_{b}.log")):
            add("Token Rush", cfg, int(m.group(1)) - 40, float(m.group(3)), float(m.group(2)))
    for t in ("prose", "code"):
        for dr, name in (("dflash", "DFlash2"), ("mtp", "MTP chain")):
            for m in re.finditer(r"context\s+(\d+): raw\s+\S+ tok/s \| spec\s+([\d.]+) tok/s, ([\d.]+) tokens/step, ([\d.]+) ms/step",
                                 read(f"{d}/engine_spec_{t}_{dr}.log")):
                c = int(m.group(1))
                add("Token Rush", f"spec, {name}, {t}", 0 if c == 64 else c, float(m.group(2)), float(m.group(4)), float(m.group(3)))
    for m in re.finditer(r"tg128(?: @ d(\d+))?\s+\|\s+([\d.]+) ±", read(f"{d}/llama_depth.log")):
        add("llama.cpp", "raw", int(m.group(1) or 0), float(m.group(2)))
    for m in re.finditer(r"context (\d+): \[ Prompt: [\d.]+ t/s \| Generation: ([\d.]+) t/s", read(f"{d}/llama_mtp_context.log")):
        add("llama.cpp", "spec, MTP x1, prose", int(m.group(1)), float(m.group(2)))
    for f, e, cfg in (("vllm_context.log", "vLLM", "raw"), ("sglang_context.log", "SGLang", "raw"), ("exl3_context.log", "ExLlamaV3", "raw")):
        for m in ctx.finditer(read(f"{d}/{f}")):
            add(e, cfg, int(m.group(1)), float(m.group(2)))
    for m in re.finditer(r"context\s+(\d+):\s+([\d.]+) tok/s\s+([\d.]+) per step", read(f"{d}/exl3_mtp_context.log")):
        add("ExLlamaV3", "spec, MTP x2", int(m.group(1)), float(m.group(2)), None, float(m.group(3)))
    return out


def per_file(text, new):
    """Aggregate `file .../N_<ctx>_p_<pos>.txt: X tok/s ... [A per step]` lines per context:
    total tokens / total seconds (every file generates `new` tokens), min/max over positions,
    accepted per step as total tokens over total verify steps."""
    by = defaultdict(list)
    for line in text.splitlines():
        m = re.match(r"file \S*/N_(\d+)_p_(\d+)\.txt:\s+([\d.]+) tok/s", line)
        if not m:
            continue
        a = re.search(r"([\d.]+) per step", line)
        by[int(m.group(1))].append((float(m.group(3)), float(a.group(1)) if a else None))
    res = {}
    for c, v in sorted(by.items()):
        tps = [t for t, _ in v]
        secs = sum(new / t for t in tps)
        r = {"tok_s": new * len(v) / secs, "tok_s_min": min(tps), "tok_s_max": max(tps), "positions": len(v)}
        accs = [a for _, a in v if a is not None]
        if accs:
            steps = sum(new / a for a in accs)
            r.update({"acc": round(new * len(accs) / steps, 2), "acc_min": min(accs), "acc_max": max(accs)})
        res[c] = r
    return res


def rows_multi(root):
    """The multi-position protocol's layout (results/2026-09-12-machine-59052)."""
    out = []
    add = make_add(out)
    sw = f"{root}/sweep"
    for b, cfg in (("triton", "raw, triton GEMV"), ("marlin", "raw, Marlin layout")):
        for m in re.finditer(r"context (\d+): ([\d.]+) ms/step = ([\d.]+) tok/s", read(f"{sw}/engine_raw_{b}.log")):
            add("Token Rush", cfg, int(m.group(1)) - 40, float(m.group(3)), float(m.group(2)))
    num = r"([\d.]+)"
    spec = re.compile(rf"(dflash|mtp)\s+{num} tok/s \[{num}-{num}\], {num} tokens/step \[{num}-{num}\], {num} ms/step")
    for t, label in (("pg19", "prose"), ("code", "code")):
        text = read(f"{sw}/engine_spec_{t}_both.log")
        summaries = {}                       # a resumed run reprints a folded-in context's summary: keep the last
        for line in text.splitlines():
            m = re.match(r"summary context\s+(\d+):(.*)\| (\d+) positions x (\d+) tokens", line)
            if m:
                summaries[int(m.group(1))] = m
        for c, m in sorted(summaries.items()):
            c = 0 if c == 64 else c
            P = int(m.group(3))
            for s in spec.finditer(m.group(2)):
                name = "DFlash2" if s.group(1) == "dflash" else "MTP chain"
                add("Token Rush", f"spec, {name}, {label}", c, float(s.group(2)), float(s.group(8)), float(s.group(5)),
                    tok_s_min=float(s.group(3)), tok_s_max=float(s.group(4)), accepted_min=float(s.group(6)),
                    accepted_max=float(s.group(7)), positions=P)
    for m in re.finditer(r"tg128(?: @ d(\d+))?\s+\|\s+([\d.]+) ±", read(f"{root}/llama.cpp/llama_depth.log")):
        add("llama.cpp", "raw", int(m.group(1) or 0), float(m.group(2)))
    for c, r in per_file(read(f"{root}/llama.cpp/llama_mtp_pg19.log"), 512).items():
        add("llama.cpp", "spec, MTP n-max 3, prose", c, r["tok_s"], None, r.get("acc"), tok_s_min=r["tok_s_min"], tok_s_max=r["tok_s_max"],
            accepted_min=r.get("acc_min"), accepted_max=r.get("acc_max"), positions=r["positions"])
    for c, r in per_file(read(f"{root}/sglang/results_spec_pg19.log"), 512).items():
        add("SGLang", "spec, DSpark, prose", c, r["tok_s"], None, r.get("acc"), tok_s_min=r["tok_s_min"], tok_s_max=r["tok_s_max"],
            accepted_min=r.get("acc_min"), accepted_max=r.get("acc_max"), positions=r["positions"])
    ctx = re.compile(r"context\s+(\d+):\s+([\d.]+) tok/s")
    for f, e in (("vllm/results_context.log", "vLLM"), ("sglang/results_context.log", "SGLang"), ("exllamav3/exl3_context.log", "ExLlamaV3")):
        for m in ctx.finditer(read(f"{root}/{f}")):
            add(e, "raw", int(m.group(1)), float(m.group(2)))
    for m in re.finditer(r"context\s+(\d+):\s+([\d.]+) tok/s\s+([\d.]+) per step", read(f"{root}/exllamav3/exl3_mtp_context.log")):
        add("ExLlamaV3", "spec, MTP x2", int(m.group(1)), float(m.group(2)), None, float(m.group(3)))
    return out


def main():
    d = sys.argv[1].rstrip("/")
    multi = os.path.isdir(os.path.join(d, "sweep"))
    rows = rows_multi(d) if multi else rows_single(d)
    name = "sweep.csv" if multi else "sweep10.csv"
    with open(os.path.join(d, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS if multi else FIELDS[:9])   # the old CSV keeps its columns
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
    if multi:
        # the speculative rows with their spread over positions: tok/s [min–max], accepted/step [min–max], ms/step
        print("\n| engine / config | " + " | ".join(f"{c // 1000}k" if c else "0" for c in ctxs) + " |")
        print("|---|" + "---|" * len(ctxs))
        for key in ENGINES:
            rr = {r["context"]: r for r in rows if (r["engine"], r["config"]) == key and r.get("positions")}
            if not rr:
                continue
            cells = []
            for c in ctxs:
                if c not in rr:
                    cells.append("—"); continue
                r = rr[c]
                cell = f"**{r['tok_s']:.0f}** [{r['tok_s_min']:.0f}–{r['tok_s_max']:.0f}]"
                if r.get("accepted_per_step"):
                    cell += f", {r['accepted_per_step']:.2f}/step [{r['accepted_min']:.2f}–{r['accepted_max']:.2f}]"
                if r.get("ms_per_step"):
                    cell += f", {r['ms_per_step']:.1f} ms"
                cells.append(cell)
            print(f"| {key[0]}, {key[1]} | " + " | ".join(cells) + " |")
    print(f"\n{len(rows)} rows -> {d}/{name}")


if __name__ == "__main__":
    main()
