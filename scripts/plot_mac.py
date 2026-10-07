#!/usr/bin/env python3
"""Single-stream decode speed on an M5 Pro, every engine through the same client
(bench/mac/rival_bench.py), from results/mac/rivals/*.json: one bar per prompt
family and engine configuration, with the raw bandwidth wall marked. Writes
docs/img/mac_rivals in svg + png, light and dark.

    python scripts/plot_mac.py [light|dark]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/img/mac_rivals"
RESULTS = "results/mac/rivals"
WALL = 21.3  # 291 GB/s over the 13.65 GB a raw step reads
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", wall="#b9b8b1",
                  essay="#2a78d6", code="#eb6834", math="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835", wall="#4a4a47",
                 essay="#3987e5", code="#d95926", math="#199e70"),
}
ENGINES = [  # json file, label; top of the chart first
    ("token-rush-auto", "token-rush, DFlash2 draft"),
    ("token-rush-mtp", "token-rush, MTP chain"),
    ("token-rush-raw", "token-rush, raw"),
    ("vllm-metal", "vllm-metal, raw (no draft runs)"),
]
FAMILIES = ["essay", "code", "math"]


def load():
    rows = []
    for name, label in ENGINES:
        path = os.path.join(RESULTS, name + ".json")
        if os.path.exists(path):
            r = {d["family"]: d["decode_tok_s"] for d in json.load(open(path))}
            rows.append((label, name.startswith("token-rush"), [r.get(f, 0.0) for f in FAMILIES]))
    return rows


def draw(theme):
    t = THEMES[theme]
    rows = load()
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(rows) + 1.3), facecolor=t["surface"])
    ax.set_facecolor(t["surface"])
    h = 0.26
    for i, (label, ours, vals) in enumerate(rows):
        y = len(rows) - 1 - i
        for j, (fam, v) in enumerate(zip(FAMILIES, vals)):
            yy = y + (1 - j) * h
            ax.barh(yy, v, height=h * 0.9, color=t[fam], alpha=1.0 if ours else 0.55,
                    label=fam if i == 0 else None)
            ax.text(v + 0.4, yy, f"{v:.1f}", va="center", fontsize=8, color=t["ink2"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9.5, color=t["ink"])
    for lab, (_, ours, _) in zip(ax.get_yticklabels(), rows[::-1]):
        lab.set_fontweight("bold" if ours else "normal")
    ax.axvline(WALL, color=t["wall"], ls="--", lw=1)
    ax.text(WALL + 0.3, len(rows) - 0.45, f"raw wall {WALL} tok/s", fontsize=8, color=t["muted"])
    ax.set_xlabel("decode tok/s, single stream, greedy", color=t["ink2"], fontsize=9)
    ax.set_xlim(0, max(max(r[2]) for r in rows) * 1.12)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.grid(axis="x", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["ink2"], length=0)
    ax.set_title("Qwen3.8-27B on a MacBook Pro M5 Pro (64 GB)", loc="left", color=t["ink"], fontsize=12, pad=12)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.tight_layout()
    suffix = "" if theme == "light" else "_dark"
    for ext in ("svg", "png"):
        fig.savefig(f"{OUT}{suffix}.{ext}", facecolor=t["surface"], dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("theme", nargs="?", choices=list(THEMES))
    a = ap.parse_args()
    for theme in [a.theme] if a.theme else THEMES:
        draw(theme)


if __name__ == "__main__":
    main()
