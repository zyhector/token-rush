#!/usr/bin/env python3
"""Decode speed step by step through the build, from docs/progress.csv: raw greedy
decode and the effective tok/s with speculative decoding on the essay / code /
math prompts (the default draft of each step), as a staircase — a number holds
until the step that changed it. Writes docs/img/progress in svg + png, light and dark.

    python scripts/plot_progress.py [light|dark]
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/img/progress"
WALL = 124.6  # 1701 GB/s over the 13.65 GB a raw step reads at short context
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", wall="#b9b8b1",
                  raw="#898781", essay="#2a78d6", code="#eb6834", math="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835", wall="#4a4a47",
                 raw="#898781", essay="#3987e5", code="#d95926", math="#199e70"),
}
SERIES = [  # csv column, label, colour key
    ("spec_math_tok_s", "speculative, math", "math"),
    ("spec_code_tok_s", "speculative, code", "code"),
    ("spec_essay_tok_s", "speculative, essay", "essay"),
    ("decode_tok_s", "raw decode", "raw"),
]
MILESTONES = [  # step, its title in docs/progress.md
    (3, "own int4 GEMV"),
    (4, "whole-step CUDA graph"),
    (6, "fused GDN step"),
    (7, "fused attention decode"),
    (13, "verify step in one graph"),
    (14, "MTP draft chain in the graph"),
    (20, "truncated draft vocabulary"),
    (25, "DFlash2 draft in the graph"),
    (28, "Marlin-class int4 GEMM"),
]


def load():
    rows = list(csv.DictReader(open("docs/progress.csv")))
    return [(int(r["step"]), r) for r in rows]


def spread(ys, min_gap):
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    shift = (sum(out) - sum(ys)) / len(ys)
    return [y - shift for y in out]


def draw(theme):
    t = THEMES[theme]
    rows = load()
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.5, "svg.fonttype": "none",
        "text.color": t["ink2"], "axes.labelcolor": t["ink2"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "axes.edgecolor": t["axis"],
    })
    fig, ax = plt.subplots(figsize=(12.5, 5.6), dpi=100, facecolor=t["surface"])
    fig.subplots_adjust(left=0.05, right=0.84, top=0.74, bottom=0.14)
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)

    last = max(s for s, _ in rows)
    ax.axhline(WALL, color=t["wall"], lw=1.2, zorder=1)
    ax.text(last + 0.3, WALL + 4, "bandwidth wall, raw decode", ha="right", va="bottom", fontsize=8.8, color=t["muted"])

    ends = []
    for col, label, ck in SERIES:
        xs, ys = [], []
        for s, r in rows:
            if r[col]:
                xs.append(s); ys.append(float(r[col]))
        c = t[ck]
        # staircase: the value holds until the next step that changed it
        ax.plot(xs + [last], ys + [ys[-1]], color=c, lw=2.2 if ck != "raw" else 1.8, drawstyle="steps-post",
                ls="-" if ck != "raw" else (0, (1.2, 2.2)), solid_capstyle="round", dash_capstyle="round", zorder=4)
        ax.scatter(xs, ys, s=18, color=c, edgecolor=t["surface"], linewidth=1.2, zorder=5)
        ends.append((last, ys[-1], label, c))
    ax.set_xlim(1.5, last + 0.5)
    ax.set_xticks([2, 5, 10, 15, 20, 25, 30, 35])
    ax.set_xlabel("step in docs/progress.md", color=t["muted"], labelpad=8)
    ax.set_ylim(0, 420)
    ax.set_yticks([0, 100, 200, 300, 400])
    ax.set_ylabel("tok/s", color=t["muted"], rotation=0, ha="right", va="bottom", labelpad=0)
    ax.yaxis.set_label_coords(0.0, 1.01)

    # end labels
    gap = 420 * 0.055
    for (x, y, label, c), yl in zip(ends, spread([e[1] for e in ends], gap)):
        ax.annotate(label, xy=(x, y), xytext=(last + 1.3, yl), textcoords="data", va="center", ha="left", fontsize=9.6,
                    color=t["ink2"], annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.9, alpha=0.7, shrinkA=0, shrinkB=4,
                                    connectionstyle="arc,angleA=180,angleB=0,armA=0,armB=10,rad=0"))

    # milestones: a hairline at the step, the label above the plot
    for i, (s, label) in enumerate(MILESTONES):
        ax.axvline(s, color=t["grid"], lw=0.8, zorder=0)
        ax.annotate(label, xy=(s, 1.0), xycoords=("data", "axes fraction"), xytext=(2, 4 + 13 * (i % 3)),
                    textcoords="offset points", ha="left", va="bottom", fontsize=8.6, color=t["muted"], annotation_clip=False)

    fig.text(0.05, 0.955, "Decode speed through the build", fontsize=15, fontweight="bold", color=t["ink"], va="top")
    fig.text(0.05, 0.897, "Qwen3.8-27B on one RTX 5090, batch size 1, greedy, short context. One number per development step.",
             fontsize=10.5, color=t["ink2"], va="top")
    for ext in ("svg", "png"):
        fig.savefig(f"{OUT}{'' if theme == 'light' else '_dark'}.{ext}", facecolor=t["surface"], dpi=200 if ext == "png" else 100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("themes", nargs="*", default=["light", "dark"])
    for theme in ap.parse_args().themes:
        draw(theme)


if __name__ == "__main__":
    main()
