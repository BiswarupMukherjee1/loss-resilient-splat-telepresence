"""Example images for experiment 01: one held-out view of three sequences, reference and reconstructions.

Usage: python experiments/01_capture_rig/figure.py [--out results/01_capture_rig]
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lossysplat import pipeline  # noqa: E402
from lossysplat.cameras import RigConfig, capture_rig, eval_rig  # noqa: E402
from lossysplat.data import Sequence  # noqa: E402
from lossysplat.metrics import psnr  # noqa: E402
from lossysplat.zbuffer import composite, render_points  # noqa: E402

ROWS = [("Gymnast", 120), ("FlowerDance", 60), ("BlueBackpack", 120)]
COLUMNS = [("2 cameras", 2, 0), ("3 cameras", 3, 0), ("4 cameras", 4, 0), ("6 cameras", 6, 0),
           ("4 cameras, two lag 80 ms", 4, 2)]
VIEW = 0  # index into the evaluation rig: on the ring, 40 degrees from the first camera


def reconstruct(seq, frame, n_cams, lag, cfg, dev):
    """Gaussians from `n_cams` cameras; with `lag`, cameras 1 and 3 deliver a frame `lag` frames later."""
    xyz, rgb = seq.frame(frame, dev)
    cams = [c.to(dev) for c in capture_rig(seq.center_xz(), cfg, n_cams)]
    rgbs, depths, alphas = pipeline.capture(xyz, rgb, cams)
    if lag:
        late = pipeline.capture(*seq.frame(frame + lag, dev), [cams[1], cams[3]])
        for dst, src in ((rgbs, late[0]), (depths, late[1]), (alphas, late[2])):
            dst[1], dst[3] = src[0], src[1]
    return pipeline.reconstruct(rgbs, depths, alphas, cams)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(here, "..", "..", "results", "01_capture_rig"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RigConfig()
    fig, axes = plt.subplots(len(ROWS), len(COLUMNS) + 1, figsize=(1.9 * (len(COLUMNS) + 1), 2.6 * len(ROWS)))
    for r, (name, frame) in enumerate(ROWS):
        seq = Sequence(name)
        cam = eval_rig(seq.center_xz(), cfg)[VIEW].to(dev)
        ref_rgb, _, ref_alpha = render_points(*seq.frame(frame, dev), cam)
        ref = composite(ref_rgb, ref_alpha)
        axes[r, 0].imshow(ref.cpu().numpy(), interpolation="nearest")
        axes[r, 0].set_ylabel(name, fontsize=9, color="#52514e")
        for c, (title, n_cams, lag) in enumerate(COLUMNS, start=1):
            img = pipeline.render(reconstruct(seq, frame, n_cams, lag, cfg, dev), cam)[0].clamp(0, 1)
            axes[r, c].imshow(img.cpu().numpy(), interpolation="nearest")
            axes[r, c].set_xlabel(f"{psnr(img, ref):.1f} dB", fontsize=8, color="#0b0b0b")
            if r == 0:
                axes[r, c].set_title(title, fontsize=8, color="#0b0b0b")
        if r == 0:
            axes[r, 0].set_title("Reference (full point cloud)", fontsize=8, color="#0b0b0b")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "examples.png"), dpi=170, facecolor="white")


if __name__ == "__main__":
    main()
