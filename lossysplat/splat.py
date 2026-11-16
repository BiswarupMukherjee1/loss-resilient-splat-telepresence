"""A compact Gaussian splat rasteriser in PyTorch.

Every Gaussian is projected to a 2D Gaussian (EWA approximation, as in 3D Gaussian Splatting), evaluated
on a fixed window of pixels around its centre, and the resulting fragments are alpha-composited front to
back in exact per-pixel depth order. There is no tiling and no compiled extension; it is meant for scenes
of up to a few hundred thousand small Gaussians, which is the regime of pixel-aligned splats.
"""
import torch


def render_gaussians(means, covs, colors, opacities, cam, window=4, background=1.0, lowpass=0.1,
                     sharpness=1.0, extra=None):
    """Render 3D Gaussians.

    means (N, 3), covs (N, 3, 3) world-space covariances, colors (N, 3), opacities (N,).
    `window`: half-size in pixels of the footprint evaluated per Gaussian.
    `lowpass`: variance in pixels^2 added to the projected covariance (anti-aliasing); the opacity is
    scaled down so that the splat keeps its integral.
    `sharpness`: the kernel is opacity * exp(-0.5 * q**sharpness) with q the squared Mahalanobis
    distance. 1 is a Gaussian; larger values give a flatter top and a faster fall-off.
    `extra`: optional (N, C) per-Gaussian features composited like colour (no background added).

    Returns rgb (H, W, 3), alpha (H, W), depth (H, W) (alpha-weighted mean depth, 0 where empty), and
    the composited `extra` (H, W, C) when given.
    """
    H, W = cam.height, cam.width
    dev = means.device
    R, t, K = cam.R, cam.t, cam.K
    xc = means @ R.T + t
    z = xc[:, 2]
    keep = z > 1e-3
    order = torch.argsort(z[keep])
    idx = torch.nonzero(keep, as_tuple=False)[:, 0][order]
    xc, z = xc[idx], z[idx]
    colors, opacities = colors[idx], opacities[idx]
    if extra is not None:
        extra = extra[idx]

    fx, fy = K[0, 0], K[1, 1]
    J = torch.zeros(len(idx), 2, 3, device=dev, dtype=means.dtype)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * xc[:, 0] / z**2
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * xc[:, 1] / z**2
    M = J @ R
    cov2 = M @ covs[idx] @ M.transpose(1, 2)
    b = cov2[:, 0, 1]
    det0 = (cov2[:, 0, 0] * cov2[:, 1, 1] - b * b).clamp(min=1e-12)
    a = cov2[:, 0, 0] + lowpass
    c = cov2[:, 1, 1] + lowpass
    det = (a * c - b * b).clamp(min=1e-12)
    opacities = opacities * torch.sqrt(det0 / det)
    ia, ib, ic = c / det, -b / det, a / det

    u = fx * xc[:, 0] / z + K[0, 2]
    v = fy * xc[:, 1] / z + K[1, 2]
    offs = torch.arange(-window, window + 1, device=dev)
    oy, ox = torch.meshgrid(offs, offs, indexing="ij")
    px = torch.floor(u).long()[:, None] + ox.reshape(1, -1)
    py = torch.floor(v).long()[:, None] + oy.reshape(1, -1)
    dx = px.to(means.dtype) + 0.5 - u[:, None]
    dy = py.to(means.dtype) + 0.5 - v[:, None]
    q = (ia[:, None] * dx * dx + 2.0 * ib[:, None] * dx * dy + ic[:, None] * dy * dy).clamp(min=0)
    if sharpness != 1.0:
        q = q**sharpness
    alpha = (opacities[:, None] * torch.exp(-0.5 * q)).clamp(max=0.99)
    valid = (alpha >= 1.0 / 255.0) & (px >= 0) & (px < W) & (py >= 0) & (py < H)

    # Fragments are generated in depth order; a stable sort by pixel keeps that order inside each pixel.
    gid = torch.arange(len(idx), device=dev)[:, None].expand_as(px)[valid]
    pix = (py * W + px)[valid]
    alpha = alpha[valid]
    pix, perm = torch.sort(pix, stable=True)
    gid, alpha = gid[perm], alpha[perm]

    # Transmittance in front of each fragment: exclusive cumulative product within its pixel.
    log_t = torch.log1p(-alpha.double())
    excl = torch.cumsum(log_t, dim=0) - log_t
    n_frag = pix.shape[0]
    if n_frag > 0:
        is_start = torch.ones(n_frag, dtype=torch.bool, device=dev)
        is_start[1:] = pix[1:] != pix[:-1]
        pos = torch.arange(n_frag, device=dev)
        start = torch.cummax(torch.where(is_start, pos, torch.zeros_like(pos)), dim=0).values
        trans = torch.exp(excl - excl[start]).to(means.dtype)
    else:
        trans = alpha
    weight = alpha * trans

    acc = torch.zeros(H * W, device=dev, dtype=means.dtype).index_add_(0, pix, weight)
    rgb = torch.zeros(H * W, 3, device=dev, dtype=means.dtype).index_add_(0, pix, weight[:, None] * colors[gid])
    dep = torch.zeros(H * W, device=dev, dtype=means.dtype).index_add_(0, pix, weight * z[gid])
    rgb = rgb + (1.0 - acc)[:, None] * background
    dep = torch.where(acc > 1e-6, dep / acc.clamp(min=1e-6), torch.zeros_like(dep))
    out = (rgb.reshape(H, W, 3), acc.reshape(H, W), dep.reshape(H, W))
    if extra is not None:
        feat = torch.zeros(H * W, extra.shape[1], device=dev, dtype=means.dtype)
        feat.index_add_(0, pix, weight[:, None] * extra[gid])
        out = out + (feat.reshape(H, W, -1),)
    return out
