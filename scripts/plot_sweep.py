#!/usr/bin/env python3
"""Decode speed vs. context length, ours and the rivals, one panel, from the
ten-point sweep CSV (scripts/sweep_table.py writes it). Colour = engine, solid =
speculative decoding on prose, dashed = raw. Writes docs/img/decode_vs_context
in svg + png, light and dark.

    uv run --with matplotlib python scripts/plot_sweep.py [light|dark]
"""
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "results/2026-09-09-machine-36542/sweep10/sweep10.csv"
OUT = "docs/img/decode_vs_context"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", wall="#b9b8b1",
                  color={"Token Rush": "#2a78d6", "llama.cpp": "#eb6834", "vLLM": "#1baf7a", "SGLang": "#eda100", "ExLlamaV3": "#e87ba4"}),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835", wall="#4a4a47",
                 color={"Token Rush": "#3987e5", "llama.cpp": "#d95926", "vLLM": "#199e70", "SGLang": "#c98500", "ExLlamaV3": "#d55181"}),
}
RAW, SPEC = (0, (4, 2.2)), "-"
SERIES = [  # (engine, config), end label, linestyle
    (("Token Rush", "spec, DFlash2, prose"), "Token Rush · DFlash2", SPEC),
    (("llama.cpp", "spec, MTP x1, prose"), "llama.cpp · MTP", SPEC),
    (("ExLlamaV3", "spec, MTP x2"), "ExLlamaV3 · MTP ×2", SPEC),
    (("Token Rush", "raw, triton GEMV"), "Token Rush", RAW),
    (("llama.cpp", "raw"), "llama.cpp", RAW),
    (("vLLM", "raw"), "vLLM", RAW),
    (("SGLang", "raw"), "SGLang", RAW),
    (("ExLlamaV3", "raw"), "ExLlamaV3", RAW),
]


def load():
    series = defaultdict(list)
    for r in csv.DictReader(open(CSV)):
        series[(r["engine"], r["config"])].append((int(r["context"]), float(r["tok_s"]), float(r["ceiling_tok_s"])))
    for k in series:
        series[k].sort()
    return series


def spread(ys, min_gap):
    """Push label y positions apart (keeping order) so none overlap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    shift = (sum(out) - sum(ys)) / len(ys)
    return [y - shift for y in out]


def draw(theme):
    t = THEMES[theme]
    series = load()
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.5, "svg.fonttype": "none",
        "text.color": t["ink2"], "axes.labelcolor": t["ink2"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "axes.edgecolor": t["axis"],
    })
    fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=100, facecolor=t["surface"])
    fig.subplots_adjust(left=0.06, right=0.80, top=0.80, bottom=0.21)

    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    ax.set_xlim(-4000, 246000)
    ax.set_xticks([0, 50000, 100000, 150000, 200000])
    ax.set_xticklabels(["0", "50k", "100k", "150k", "200k"])
    ax.set_xlabel("tokens already in context", color=t["muted"], labelpad=8)
    ax.margins(y=0.06)

    # the 1701 GB/s wall at Token Rush's bytes/step: what raw decode cannot exceed
    pts = series[("Token Rush", "raw, triton GEMV")]
    ax.plot([c for c, _, _ in pts], [w for _, _, w in pts], color=t["wall"], lw=1.2, zorder=1)

    ends = []
    for key, label, ls in SERIES:
        pts = series[key]
        xs = [c for c, _, _ in pts]; ys = [v for _, v, _ in pts]
        ours = key[0] == "Token Rush"
        col = t["color"][key[0]]
        ax.plot(xs, ys, color=col, lw=2.4 if ours else 1.7, ls=ls, solid_capstyle="round", solid_joinstyle="round",
                dash_capstyle="round", zorder=4 if ours else 3)
        ax.scatter(xs, ys, s=22 if ours else 15, color=col, edgecolor=t["surface"], linewidth=1.2, zorder=5 if ours else 4)
        ends.append((xs[-1], ys[-1], label, col, ours))

    ax.set_ylim(bottom=0)
    ymin, ymax = ax.get_ylim()
    gap = (ymax - ymin) * 0.05
    xr = ax.get_xlim()[1]
    ly = spread([e[1] for e in ends], gap)
    for (x, y, label, col, ours), yl in zip(ends, ly):
        ax.annotate(label, xy=(x, y), xytext=(xr + 5000, yl), textcoords="data", va="center", ha="left",
                    fontsize=10.5, color=t["ink"] if ours else t["ink2"], fontweight="bold" if ours else "normal",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.9, alpha=0.7, shrinkA=0, shrinkB=4,
                                    connectionstyle="arc,angleA=180,angleB=0,armA=0,armB=14,rad=0"))
    wx = [c for c, _, _ in pts]; wy = [w for _, _, w in pts]
    xm = 176000
    ym = wy[-2] + (wy[-1] - wy[-2]) * (xm - wx[-2]) / (wx[-1] - wx[-2])
    (x0, y0), (x1, y1) = ax.transData.transform([(wx[-3], wy[-3]), (wx[-1], wy[-1])])
    import math
    ax.text(xm, ym + gap * 0.6, "bandwidth wall", rotation=math.degrees(math.atan2(y1 - y0, x1 - x0)),
            rotation_mode="anchor", va="bottom", ha="center", fontsize=8.8, color=t["muted"])

    # line-style key, top right where the plot is empty
    kx, ky = 0.985, 0.96
    for i, (ls, text) in enumerate(((SPEC, "speculative decoding, prose"), (RAW, "raw decode"))):
        y = ky - i * 0.062
        ax.plot([kx - 0.075, kx - 0.03], [y, y], transform=ax.transAxes, color=t["muted"], lw=1.8, ls=ls,
                dash_capstyle="round", clip_on=False)
        ax.text(kx - 0.083, y, text, transform=ax.transAxes, ha="right", va="center", fontsize=10, color=t["ink2"])

    ax.set_ylabel("tok/s", color=t["muted"], rotation=0, ha="right", va="bottom", labelpad=0)
    ax.yaxis.set_label_coords(0.0, 1.01)

    fig.text(0.06, 0.955, "Single-stream decode speed vs. context length", fontsize=15, fontweight="bold", color=t["ink"], va="top")
    fig.text(0.06, 0.895, "Qwen3.8-27B on one RTX 5090, batch size 1, greedy. Every engine measured on the same card the same night.",
             fontsize=10.5, color=t["ink2"], va="top")
    fig.text(0.06, 0.025,
             "Solid: speculative decoding on prose (WikiText-103). Speculative tok/s = accepted tokens per step × steps per second; acceptance depends on the\n"
             "text at each position, hence the wobble, while the step cost grows smoothly. ExLlamaV3's MTP row is on random-token context and stops where it no longer fits.\n"
             "Token Rush raw: Triton GEMV backend, fp8 KV cache. Gray line: the 1701 GB/s bandwidth wall at Token Rush's bytes per step.",
             fontsize=8.6, color=t["muted"], va="bottom", linespacing=1.4)

    for ext in ("svg", "png"):
        fig.savefig(f"{OUT}{'' if theme == 'light' else '_dark'}.{ext}", facecolor=t["surface"], dpi=200 if ext == "png" else 100)
    plt.close(fig)


if __name__ == "__main__":
    for theme in sys.argv[1:] or ("light", "dark"):
        draw(theme)
