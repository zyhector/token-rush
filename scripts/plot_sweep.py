#!/usr/bin/env python3
"""Decode speed vs. context length, ours and the rivals, two panels, from a sweep
CSV (scripts/sweep_table.py writes it). Left: raw decode as a fraction of the
bandwidth wall at each engine's own bytes per step (the engine property).
Right: effective tok/s with speculative decoding on prose; the position-to-
position spread of the multi-position protocol is a light band of the same hue.
Writes docs/img/decode_vs_context in svg + png, light and dark.

    python scripts/plot_sweep.py [--csv results/2026-09-12-machine-59052/sweep.csv] [light|dark]
"""
import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/img/decode_vs_context"
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", wall="#b9b8b1",
                  color={"Token Rush": "#2a78d6", "llama.cpp": "#eb6834", "vLLM": "#1baf7a", "SGLang": "#eda100", "ExLlamaV3": "#e87ba4"}),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835", wall="#4a4a47",
                 color={"Token Rush": "#3987e5", "llama.cpp": "#d95926", "vLLM": "#199e70", "SGLang": "#c98500", "ExLlamaV3": "#d55181"}),
}
RAW = [  # (engine, config), label
    (("Token Rush", "raw, triton GEMV"), "Token Rush"),
    (("vLLM", "raw"), "vLLM"),
    (("ExLlamaV3", "raw"), "ExLlamaV3"),
    (("SGLang", "raw"), "SGLang"),
    (("llama.cpp", "raw"), "llama.cpp"),
]
SPEC = [  # (engine, config), label, linestyle
    (("Token Rush", "spec, DFlash2, prose"), "Token Rush · DFlash2", "-"),
    (("Token Rush", "spec, MTP chain, prose"), "Token Rush · MTP chain", (0, (4, 2.2))),
    (("llama.cpp", "spec, MTP x1, prose"), "llama.cpp · MTP", "-"),
    (("SGLang", "spec, DSpark, prose"), "SGLang · DSpark", "-"),
    (("Token Rush", "raw, triton GEMV"), "Token Rush · raw", (0, (1.2, 2.2))),
]


def load(path):
    series = defaultdict(list)
    for r in csv.DictReader(open(path)):
        lo = float(r["tok_s_min"]) if r.get("tok_s_min") else None
        hi = float(r["tok_s_max"]) if r.get("tok_s_max") else None
        series[(r["engine"], r["config"])].append((int(r["context"]), float(r["tok_s"]), float(r["ceiling_tok_s"]), float(r["pct_of_wall"]), lo, hi))
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


def style_axis(ax, t):
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


def end_labels(ax, ends, t, gap_frac=0.055):
    ymin, ymax = ax.get_ylim()
    gap = (ymax - ymin) * gap_frac
    xr = ax.get_xlim()[1]
    ly = spread([e[1] for e in ends], gap)
    for (x, y, label, col, ours), yl in zip(ends, ly):
        ax.annotate(label, xy=(x, y), xytext=(xr + 5000, yl), textcoords="data", va="center", ha="left",
                    fontsize=9.6, color=t["ink"] if ours else t["ink2"], fontweight="bold" if ours else "normal",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.9, alpha=0.7, shrinkA=0, shrinkB=4,
                                    connectionstyle="arc,angleA=180,angleB=0,armA=0,armB=12,rad=0"))


def draw(theme, path, subtitle, caption):
    t = THEMES[theme]
    series = load(path)
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.5, "svg.fonttype": "none",
        "text.color": t["ink2"], "axes.labelcolor": t["ink2"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "axes.edgecolor": t["axis"],
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2), dpi=100, facecolor=t["surface"])
    fig.subplots_adjust(left=0.05, right=0.845, top=0.80, bottom=0.235, wspace=0.40)

    # left: raw decode, % of the wall
    style_axis(ax1, t)
    ax1.axhline(100, color=t["wall"], lw=1.2, zorder=1)
    ax1.text(240000, 101, "bandwidth wall", ha="right", va="bottom", fontsize=8.8, color=t["muted"])
    ends = []
    for key, label in RAW:
        pts = series.get(key)
        if not pts:
            continue
        xs = [p[0] for p in pts]; ys = [p[3] for p in pts]
        ours = key[0] == "Token Rush"; col = t["color"][key[0]]
        ax1.plot(xs, ys, color=col, lw=2.4 if ours else 1.7, solid_capstyle="round", zorder=4 if ours else 3)
        ax1.scatter(xs, ys, s=22 if ours else 15, color=col, edgecolor=t["surface"], linewidth=1.2, zorder=5 if ours else 4)
        ends.append((xs[-1], ys[-1], label, col, ours))
    ax1.set_ylim(0, 108)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_yticklabels(["0", "20", "40", "60", "80", "100%"])
    end_labels(ax1, ends, t)
    ax1.set_title("Raw decode, fraction of the bandwidth wall", loc="left", fontsize=11.5, color=t["ink"], pad=12)

    # right: effective tok/s with speculation, prose
    style_axis(ax2, t)
    pts = series[("Token Rush", "raw, triton GEMV")]
    ax2.plot([p[0] for p in pts], [p[2] for p in pts], color=t["wall"], lw=1.2, zorder=1)
    ends = []
    for key, label, ls in SPEC:
        pts = series.get(key)
        if not pts:
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ours = key[0] == "Token Rush"; col = t["color"][key[0]]
        if all(p[4] is not None for p in pts):
            ax2.fill_between(xs, [p[4] for p in pts], [p[5] for p in pts], color=col, alpha=0.10, lw=0, zorder=2)
        ax2.plot(xs, ys, color=col, lw=2.4 if ours and "raw" not in key[1] else 1.7, ls=ls, solid_capstyle="round", dash_capstyle="round",
                 zorder=4 if ours else 3)
        ax2.scatter(xs, ys, s=22 if ours else 15, color=col, edgecolor=t["surface"], linewidth=1.2, zorder=5 if ours else 4)
        ends.append((xs[-1], ys[-1], label, col, ours))
    ax2.set_ylim(bottom=0)
    ax2.margins(y=0.06)
    end_labels(ax2, ends, t)
    wx = [p[0] for p in series[("Token Rush", "raw, triton GEMV")]]; wy = [p[2] for p in series[("Token Rush", "raw, triton GEMV")]]
    ax2.text(120000, wy[wx.index(128000)] + 4 if 128000 in wx else wy[len(wy) // 2], "bandwidth wall (raw)", ha="center", va="bottom",
             fontsize=8.8, color=t["muted"])
    ax2.set_title("Effective tok/s with speculative decoding, prose", loc="left", fontsize=11.5, color=t["ink"], pad=12)
    ax2.set_ylabel("tok/s", color=t["muted"], rotation=0, ha="right", va="top", labelpad=0)
    ax2.yaxis.set_label_coords(-0.005, 0.995)

    fig.text(0.05, 0.955, "Single-stream decode speed vs. context length", fontsize=15, fontweight="bold", color=t["ink"], va="top")
    fig.text(0.05, 0.897, subtitle, fontsize=10.5, color=t["ink2"], va="top")
    fig.text(0.05, 0.025, caption, fontsize=8.6, color=t["muted"], va="bottom", linespacing=1.4)
    for ext in ("svg", "png"):
        fig.savefig(f"{OUT}{'' if theme == 'light' else '_dark'}.{ext}", facecolor=t["surface"], dpi=200 if ext == "png" else 100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("themes", nargs="*", default=["light", "dark"])
    ap.add_argument("--csv", default="results/2026-09-12-machine-59052/sweep.csv")
    ap.add_argument("--subtitle", default="Qwen3.8-27B on one RTX 5090, batch size 1, greedy. Every engine measured on the same card on the same day (vast 59052, 2026-09-12).")
    ap.add_argument("--caption", default=(
        "Left: tok/s over each engine's own ceiling (1701 GB/s divided by its weights plus KV bytes per step at that context). Right: PG-19 prose, six positions per context,\n"
        "512 greedy tokens each, tok/s = total tokens / total decode seconds; the band is the min–max over positions. llama.cpp + MTP on the same prompt files through llama-server;\n"
        "SGLang + DSpark on the ones that fit its 30k window. vLLM's speculative paths are slower than its raw decode (table) and ExLlamaV3's MTP runs only on random-token\n"
        "context, so neither is drawn. Token Rush raw: Triton GEMV, fp8 KV. Gray: the bandwidth wall at Token Rush's bytes per step."))
    a = ap.parse_args()
    for theme in a.themes:
        draw(theme, a.csv, a.subtitle, a.caption)


if __name__ == "__main__":
    main()
