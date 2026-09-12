#!/usr/bin/env python3
"""Raw decode speed by development step, from docs/progress.csv: the Phase 2 climb
from the first coherent tokens to the bandwidth wall, and where the later phases
left it. Writes docs/img/progress_steps in svg + png, light and dark.

    python scripts/plot_progress.py [light|dark]

Development numbers from disposable instances (docs/progress.md), all against the
1701 GB/s wall; not the report.
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "docs/progress.csv"
OUT = "docs/img/progress_steps"
WALL = 124.6   # tok/s at the engine's 13.65 GB per step, empty context
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", wall="#b9b8b1", line="#2a78d6"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835", wall="#4a4a47", line="#3987e5"),
}
# the steps worth a label: step -> (label, x, y, ha, va), placed by hand off the step line
LABELS = {
    2: ("first coherent tokens\n(dequant-then-matmul)", 3.2, 6, "left", "bottom"),
    3: ("own int4 GEMV", 3.1, 33, "left", "top"),
    4: ("whole-step CUDA graph", 4.15, 66, "left", "top"),
    6: ("fused GDN step", 6.15, 81, "left", "top"),
    7: ("fused attention decode", 7.15, 97, "left", "top"),
    8: ("split-K GEMV", 8.15, 106, "left", "bottom"),
    22: ("flash-decoding, 3 stages", 22, 105, "center", "bottom"),
    28: ("Marlin-class GEMM\n(shared layout: raw −3%)", 28.15, 94, "left", "top"),
    32: ("Phase 4 measurement", 32, 105, "center", "bottom"),
}


def draw(theme):
    t = THEMES[theme]
    rows = [r for r in csv.DictReader(open(CSV)) if r["decode_tok_s"]]
    xs = [int(r["step"]) for r in rows]
    ys = [float(r["decode_tok_s"]) for r in rows]
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.5, "svg.fonttype": "none",
        "text.color": t["ink2"], "axes.labelcolor": t["ink2"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "axes.edgecolor": t["axis"],
    })
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=100, facecolor=t["surface"])
    fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.20)
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    ax.axhline(WALL, color=t["wall"], lw=1.2, zorder=1)
    ax.text(xs[-1] + 0.3, WALL + 2, "bandwidth wall, 124.6 tok/s", ha="right", va="bottom", fontsize=8.8, color=t["muted"])
    ax.step(xs, ys, where="post", color=t["line"], lw=2.2, zorder=3)
    ax.scatter(xs, ys, s=18, color=t["line"], edgecolor=t["surface"], linewidth=1.2, zorder=4)
    for step, (label, x, y, ha, va) in LABELS.items():
        if step in xs:
            ax.text(x, y, label, ha=ha, va=va, fontsize=8.6, color=t["ink2"], linespacing=1.25)
    ax.set_xlim(1, xs[-1] + 1)
    ax.set_ylim(0, 140)
    ax.set_xlabel("development step (docs/progress.md)", color=t["muted"], labelpad=8)
    ax.set_ylabel("tok/s", color=t["muted"], rotation=0, ha="right", va="bottom", labelpad=0)
    ax.yaxis.set_label_coords(0.0, 1.02)
    fig.text(0.06, 0.95, "Raw decode speed by development step", fontsize=15, fontweight="bold", color=t["ink"], va="top")
    fig.text(0.06, 0.885, "Qwen3.8-27B int4 on one RTX 5090, batch size 1, short context. Development numbers from disposable instances, against the same 1701 GB/s wall.",
             fontsize=10.5, color=t["ink2"], va="top")
    fig.text(0.06, 0.025, "Steps 12–27 and 29–34 changed the speculative decoder, quality or the serving layer, not raw decode.\n"
             "The Marlin layout (step 28) costs raw decode 3% and is what serves the verify step; --backend triton keeps the faster raw kernel.",
             fontsize=8.6, color=t["muted"], va="bottom", linespacing=1.4)
    for ext in ("svg", "png"):
        fig.savefig(f"{OUT}{'' if theme == 'light' else '_dark'}.{ext}", facecolor=t["surface"], dpi=200 if ext == "png" else 100)
    plt.close(fig)


if __name__ == "__main__":
    for theme in sys.argv[1:] or ("light", "dark"):
        draw(theme)
