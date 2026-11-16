"""The vectorised rasteriser must agree with a direct per-pixel loop, and the full chain must
reconstruct a simple object."""
import math

import torch

from lossysplat import pipeline
from lossysplat.cameras import RigConfig, capture_rig, eval_rig
from lossysplat.splat import render_gaussians


def _reference_render(means, covs, colors, opacities, cam, lowpass, background=1.0):
    """Front-to-back alpha compositing written as plain loops."""
    xc = means @ cam.R.T + cam.t
    order = torch.argsort(xc[:, 2])
    fx, fy = float(cam.K[0, 0]), float(cam.K[1, 1])
    img = torch.zeros(cam.height, cam.width, 3, dtype=torch.float64)
    trans = torch.ones(cam.height, cam.width, dtype=torch.float64)
    for i in order.tolist():
        x, y, z = [float(c) for c in xc[i]]
        J = torch.tensor([[fx / z, 0.0, -fx * x / z**2], [0.0, fy / z, -fy * y / z**2]])
        M = J @ cam.R
        c2 = M @ covs[i] @ M.T
        det0 = float(torch.linalg.det(c2))
        c2 = c2 + lowpass * torch.eye(2)
        det = float(torch.linalg.det(c2))
        inv = torch.linalg.inv(c2)
        u, v = fx * x / z + float(cam.K[0, 2]), fy * y / z + float(cam.K[1, 2])
        for py in range(cam.height):
            for px in range(cam.width):
                d = torch.tensor([px + 0.5 - u, py + 0.5 - v])
                a = float(opacities[i]) * math.sqrt(det0 / det) * math.exp(-0.5 * float(d @ inv @ d))
                a = min(a, 0.99)
                if a < 1.0 / 255.0 or abs(px - math.floor(u)) > 4 or abs(py - math.floor(v)) > 4:
                    continue
                img[py, px] += trans[py, px] * a * colors[i].double()
                trans[py, px] *= 1.0 - a
    return img + trans[..., None] * background, 1.0 - trans


def test_rasteriser_matches_per_pixel_loop():
    g = torch.Generator().manual_seed(0)
    cfg = RigConfig(height=16, width=12, half_extent=16.0)
    cam = capture_rig((0.0, 0.0), cfg)[0]
    n = 30
    means = torch.stack([torch.rand(n, generator=g) * 16 - 8, cfg.center_y + torch.rand(n, generator=g) * 20 - 10,
                         torch.rand(n, generator=g) * 40 - 20], dim=1)
    A = torch.randn(n, 3, 3, generator=g) * 0.8
    covs = A @ A.transpose(1, 2) + 0.05 * torch.eye(3)
    colors = torch.rand(n, 3, generator=g)
    opacities = 0.2 + 0.75 * torch.rand(n, generator=g)
    rgb, alpha, _ = render_gaussians(means, covs, colors, opacities, cam, lowpass=0.1)
    ref_rgb, ref_alpha = _reference_render(means, covs, colors, opacities, cam, lowpass=0.1)
    assert float(alpha.max()) > 0.5
    assert torch.allclose(rgb.double(), ref_rgb, atol=2e-3)
    assert torch.allclose(alpha.double(), ref_alpha, atol=2e-3)


def test_chain_reconstructs_a_sphere():
    """Four ring cameras see the poles of a sphere only at a grazing angle, so a few percent of the
    surface stays uncovered; the rest must be reconstructed closely."""
    cfg = RigConfig()
    grid = torch.arange(-70, 71, dtype=torch.float32)
    x, y, z = torch.meshgrid(grid, grid, grid, indexing="ij")
    r = torch.sqrt(x**2 + y**2 + z**2)
    shell = (r <= 70) & (r > 68)
    xyz = torch.stack([x[shell] + 250, y[shell] + cfg.center_y, z[shell] + 250], dim=1)
    rgb = 0.5 + 0.4 * torch.sin(xyz / 20.0)
    cams = capture_rig((250.0, 250.0), cfg, n_cams=4)
    gaussians = pipeline.reconstruct(*pipeline.capture(xyz, rgb, cams), cams)
    scores = pipeline.mean_scores(pipeline.score(gaussians, xyz, rgb, eval_rig((250.0, 250.0), cfg)[:4]))
    assert scores["psnr"] > 30.0
    assert scores["hole"] < 0.08
