"""Experiment 01: how good can a splat reconstruction from a ring of RGB-D cameras be, and what breaks it?

No learning and no transmission here. Every sequence is captured by a virtual rig, lifted to pixel-aligned
Gaussians and scored at held-out viewpoints against renders of the full point cloud. One factor is changed
at a time around the reference rig (4 cameras, perfect sensors):

    cameras      number of cameras on the ring
    sync         two of the four cameras lag by some frames (25 fps, so 1 frame = 40 ms)
    calibration  random extrinsic error per camera, given as reprojection error in pixels
    exposure     alternating per-camera gain error
    depth_noise  Gaussian noise on the depth maps, in voxels

Usage: python experiments/01_capture_rig/run.py [--step 10] [--out results/01_capture_rig]
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lossysplat import pipeline  # noqa: E402
from lossysplat.cameras import RigConfig, capture_rig, eval_rig, perturb_camera  # noqa: E402
from lossysplat.data import Sequence, store_dir  # noqa: E402
from lossysplat.data_split import ALL  # noqa: E402
from lossysplat.metrics import Lpips  # noqa: E402

CONDITIONS = (
    [("cameras", k) for k in (2, 3, 4, 6, 8)]
    + [("sync", d) for d in (1, 2, 4)]
    + [("calibration", e) for e in (0.25, 0.5, 1.0, 2.0, 4.0)]
    + [("exposure", e) for e in (0.05, 0.1, 0.2)]
    + [("depth_noise", s) for s in (0.5, 1.0, 2.0, 4.0)]
)


def run_condition(kind, level, seq, frame, cfg, center, eval_cams, lpips, gen, dev):
    xyz, rgb = seq.frame(frame, dev)
    n_cams = level if kind == "cameras" else 4
    cams = [c.to(dev) for c in capture_rig(center, cfg, n_cams)]
    rgbs, depths, alphas = pipeline.capture(xyz, rgb, cams)
    lift_cams = cams
    if kind == "sync":
        late_xyz, late_rgb = seq.frame(frame + level, dev)
        late = pipeline.capture(late_xyz, late_rgb, [cams[1], cams[3]])
        for dst, src in ((rgbs, late[0]), (depths, late[1]), (alphas, late[2])):
            dst[1], dst[3] = src[0], src[1]
    elif kind == "calibration":
        rot = math.degrees(math.atan(level / float(cams[0].K[0, 0])))
        lift_cams = [perturb_camera(c, rot_deg=rot, generator=gen) for c in cams]
    elif kind == "exposure":
        gains = torch.tensor([1 + level, 1 - level, 1 + level, 1 - level], device=dev)
        rgbs = (rgbs * gains[:, None, None, None]).clamp(0, 1)
    elif kind == "depth_noise":
        noise = torch.randn(depths.shape, generator=gen).to(dev) * level
        depths = torch.where(alphas > 0, depths + noise, depths)
    gaussians = pipeline.reconstruct(rgbs, depths, alphas, lift_cams)
    return pipeline.score(gaussians, xyz, rgb, eval_cams, lpips=lpips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=10, help="use every step-th frame of each sequence")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                                  "results", "01_capture_rig"))
    ap.add_argument("--seqs", nargs="*", default=ALL)
    args = ap.parse_args()
    os.environ.setdefault("TORCH_HOME", os.path.join(store_dir(), "cache", "torch"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RigConfig()
    lpips = Lpips(dev)
    gen = torch.Generator().manual_seed(0)
    results = {}
    t0 = time.time()
    for name in args.seqs:
        seq = Sequence(name)
        center = seq.center_xz()
        eval_cams = [c.to(dev) for c in eval_rig(center, cfg)]
        frames = [f for f in range(0, len(seq), args.step) if f + 4 < len(seq)]
        for kind, level in CONDITIONS:
            rows = []
            for f in frames:
                rows += run_condition(kind, level, seq, f, cfg, center, eval_cams, lpips, gen, dev)
            results.setdefault(f"{kind}={level}", {})[name] = pipeline.mean_scores(rows)
        print(f"{name}: done after {time.time() - t0:.0f} s", flush=True)
    summary = {cond: pipeline.mean_scores(list(per_seq.values())) for cond, per_seq in results.items()}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump({"frames_step": args.step, "sequences": args.seqs, "rig": vars(cfg),
                   "mean_over_sequences": summary, "per_sequence": results}, fh, indent=1)
    for cond, m in summary.items():
        print(f"{cond:18s} psnr {m['psnr']:.2f}  psnr_fg {m['psnr_fg']:.2f}  ssim {m['ssim']:.4f}  "
              f"lpips {m['lpips']:.4f}  hole {100 * m['hole']:.2f}%")


if __name__ == "__main__":
    main()
