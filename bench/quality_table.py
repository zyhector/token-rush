#!/usr/bin/env python3
"""Print the quantization-quality comparison table from results/quality/*.json.

    python bench/quality_table.py                       # everything, ordered by pooled KL
    python bench/quality_table.py --md                  # markdown, for docs/quantization.md

The tables in `docs/quantization.md` and `docs/progress.md` step 30 came from
this; regenerating them after adding a candidate should reproduce them.
"""
import argparse
import glob
import json
import os

LABELS = {                                   # file stem -> the label used in the docs
    "bf16_ref": "bf16 (reference)",
    "int4_rtn": "ours, RTN",
    "int4_gptq": "ours, GPTQ",
    "int4_gptq_ao": "ours, GPTQ + act-order",
    "int4_gptq_mse": "ours, GPTQ + MSE (adopted)",
    "int4_gptq_g64": "ours, GPTQ group 64",
    "int4_rtn_bf16head": "ours, RTN, head in bf16",
    "int4_rtn_bf16body": "ours, RTN, body in bf16",
    "int4_gptq_bf16head": "ours, GPTQ, head in bf16",
    "int4_gptq_bf16body": "ours, GPTQ, body in bf16",
    "exl3_4.00bpw": "ExLlamaV3 4.00 bpw (the bar)",
    "gguf_ud_q4_k_m": "llama.cpp UD-Q4_K_M",
    "nvfp4_quasar": "NVFP4 QUASAR-QAT",
    "redhat_int4_bf16head": "RedHatAI INT4 (bf16 head)",
}
COLS = ["candidate", "bpw", "KL mean", "KL p99", "KL max", "top-1", "PPL wiki", "KL wiki/code/math", "GSM8K"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/quality")
    ap.add_argument("--md", action="store_true", help="markdown table")
    ap.add_argument("--attribution", action="store_true", help="include the head/body attribution rows")
    a = ap.parse_args()

    GSM_ALIAS = {"bf16_ref": "bf16"}
    gsm = {}
    for fn in glob.glob(os.path.join(a.dir, "gsm8k_*.json")):
        d = json.load(open(fn))
        stem = os.path.basename(fn)[len("gsm8k_"):-len(".json")].replace("_engine", "")
        gsm[stem] = f"{d['accuracy'] * 100:.1f}%"
    rows = []
    for fn in sorted(glob.glob(os.path.join(a.dir, "*.json"))):
        stem = os.path.basename(fn)[:-len(".json")]
        if stem.startswith("gsm8k_") or stem.startswith("noise_floor"):
            continue
        d = json.load(open(fn))
        if "pooled" not in d:
            continue
        if not a.attribution and ("bf16head" in stem or "bf16body" in stem) and not stem.startswith("redhat"):
            continue
        p, c = d["pooled"], d["per_corpus"]
        rows.append((p["kl_mean"], [
            LABELS.get(stem, stem), f"{d['streamed_bpw']:.2f}",
            "—" if p["kl_mean"] == 0 else f"{p['kl_mean']:.4f}",
            "—" if p["kl_mean"] == 0 else f"{p['kl_p99']:.3f}",
            "—" if p["kl_mean"] == 0 else f"{p['kl_max']:.1f}",
            f"{p['top1_agreement']:.3f}", f"{c['wikitext2']['ppl']:.3f}",
            " / ".join(f"{c[k]['kl_mean']:.4f}" for k in ("wikitext2", "code", "math")),
            gsm.get(GSM_ALIAS.get(stem, stem), "—"),
        ]))
    nf = os.path.join(a.dir, "noise_floor_hf_vs_ours.json")
    if os.path.exists(nf):
        n = json.load(open(nf))["per_corpus"]
        kl = sum(v["kl_mean"] for v in n.values()) / len(n)
        rows.append((1e9, ["noise floor (HF bf16 vs ours)", "16", f"{kl:.4f}",
                           f"{max(v['kl_p99'] for v in n.values()):.3f}", "—",
                           f"{sum(v['top1_agreement'] for v in n.values()) / len(n):.3f}", "—", "—", "—"]))
    rows.sort(key=lambda r: r[0])
    table = [COLS] + [r[1] for r in rows]
    w = [max(len(r[i]) for r in table) for i in range(len(COLS))]
    sep = "|" + "|".join("-" * (x + 2) for x in w) + "|" if not a.md else "|" + "|".join("---" for _ in w) + "|"
    for i, r in enumerate(table):
        print("| " + " | ".join(c.ljust(w[j]) for j, c in enumerate(r)) + " |")
        if i == 0:
            print(sep)


if __name__ == "__main__":
    main()
