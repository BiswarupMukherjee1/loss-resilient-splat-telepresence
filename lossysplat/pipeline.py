"""The reference chain used by every experiment: capture views -> Gaussians -> held-out views -> scores."""
import torch

from .lift import lift_views
from .metrics import foreground_region, psnr, ssim
from .splat import render_gaussians
from .zbuffer import composite, render_points


def capture(xyz, rgb, cams):
    """Render the point cloud in every camera. Returns rgb (K, H, W, 3), depth (K, H, W), alpha (K, H, W)."""
    views = [render_points(xyz, rgb, cam) for cam in cams]
    return tuple(torch.stack([v[i] for v in views]) for i in range(3))


def reconstruct(rgbs, depths, alphas, cams, **lift_args):
    """Pixel-aligned Gaussians from all capture views."""
    return lift_views(rgbs, depths, alphas, cams, **lift_args)


def render(gaussians, cam, **render_args):
    """Image over a white background, alpha and depth of the Gaussians seen from `cam`."""
    return render_gaussians(gaussians.means, gaussians.covs, gaussians.colors, gaussians.opacities, cam,
                            **render_args)


def score(gaussians, xyz, rgb, eval_cams, lpips=None, **render_args):
    """Compare renders of the Gaussians with renders of the full point cloud at the evaluation cameras.

    Returns one dict per camera: psnr, psnr_fg (person region only), ssim, hole (fraction of the
    person's pixels left uncovered) and lpips when an `Lpips` instance is given.
    """
    out = []
    for cam in eval_cams:
        ref_rgb, _, ref_alpha = render_points(xyz, rgb, cam)
        ref = composite(ref_rgb, ref_alpha)
        img, alpha, _ = render(gaussians, cam, **render_args)
        img = img.clamp(0, 1)
        person = ref_alpha > 0.5
        row = {
            "psnr": psnr(img, ref),
            "psnr_fg": psnr(img, ref, foreground_region(ref_alpha, alpha)),
            "ssim": ssim(img, ref),
            "hole": float(((alpha < 0.5) & person).sum() / person.sum().clamp(min=1)),
        }
        if lpips is not None:
            row["lpips"] = lpips(img, ref)
        out.append(row)
    return out


def mean_scores(rows):
    """Average a list of score dicts key by key."""
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}
