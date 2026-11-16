"""Plot the results of experiment 01: novel-view PSNR against each capture factor.

Levels are drawn evenly spaced, in increasing order.

Usage: python experiments/01_capture_rig/plot.py [--results results/01_capture_rig]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PANELS = [
    ("cameras", "Number of cameras", None),
    ("sync", "Sync lag (frames of 40 ms)", 0),
    ("calibration", "Calibration error (pixels)", 0),
    ("exposure", "Exposure mismatch (fraction)", 0),
    ("depth_noise", "Depth noise (voxels)", 0),
]
INK, MUTED, SERIES = "#0b0b0b", "#52514e", "#2a78d6"


def series(results, kind, zero_level):
    """Levels and per-sequence PSNR lists for one factor; `zero_level` adds the unperturbed rig."""
    per_seq = results["per_sequence"]
    points = []
    if zero_level is not None:
        points.append((zero_level, [m["psnr"] for m in per_seq["cameras=4"].values()]))
    for key, seqs in per_seq.items():
        name, level = key.split("=")
        if name == kind:
            points.append((float(level), [m["psnr"] for m in seqs.values()]))
    return sorted(points)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(here, "..", "..", "results", "01_capture_rig"))
    args = ap.parse_args()
    with open(os.path.join(args.results, "results.json")) as fh:
        results = json.load(fh)

    fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 3.4), sharey=True)
    for ax, (kind, label, zero_level) in zip(axes, PANELS):
        points = series(results, kind, zero_level)
        labels = [f"{p[0]:g}" for p in points]
        levels = list(range(len(points)))
        means = [sum(p[1]) / len(p[1]) for p in points]
        ax.fill_between(levels, [min(p[1]) for p in points], [max(p[1]) for p in points],
                        color=SERIES, alpha=0.15, linewidth=0)
        ax.plot(levels, means, color=SERIES, linewidth=2, marker="o", markersize=6)
        for x, y in zip(levels, means):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center",
                        fontsize=8, color=INK)
        ax.set_xlabel(label, color=MUTED, fontsize=9)
        ax.set_xticks(levels, labels)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c2")
    axes[0].set_ylabel("PSNR at held-out views (dB)", color=MUTED, fontsize=9)
    fig.suptitle("Novel-view quality against each capture factor "
                 "(line: mean of 12 sequences, band: lowest to highest sequence)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(args.results, "capture_rig.png"), dpi=150, facecolor="white")


if __name__ == "__main__":
    main()
