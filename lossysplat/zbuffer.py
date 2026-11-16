"""Point-cloud renderer used as the virtual RGB-D sensor and for ground-truth images."""
import math

import torch
import torch.nn.functional as F

from .cameras import Camera


def _render_hard(xyz, rgb, cam, point_size, tol):
    """Z-buffer render without anti-aliasing. Each voxel covers the pixel centres inside a square of side
    `point_size` voxels (at least one pixel) around its projection. A pixel takes the nearest depth and the
    mean colour and depth of the points within `tol` world units of it (the front surface).

    Returns colour (H, W, 3), depth (H, W), hit (H, W) bool.
    """
    H, W = cam.height, cam.width
    uv, z = cam.project(xyz)
    radius = (0.5 * point_size * cam.K[0, 0] / z.clamp(min=1e-3)).clamp(min=0.5)
    reach = int(math.ceil(float(radius.max()) - 0.5)) if len(z) else 0
    offs = torch.arange(-reach, reach + 1, device=xyz.device)
    oy, ox = [o.reshape(1, -1) for o in torch.meshgrid(offs, offs, indexing="ij")]
    px = torch.floor(uv[:, 0]).long()[:, None] + ox
    py = torch.floor(uv[:, 1]).long()[:, None] + oy
    inside = ((px + 0.5 - uv[:, 0:1]).abs() <= radius[:, None]) & ((py + 0.5 - uv[:, 1:2]).abs() <= radius[:, None])
    ok = inside & (z[:, None] > 1e-3) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    src = torch.arange(len(z), device=xyz.device)[:, None].expand_as(px)[ok]
    pix = (py * W + px)[ok]
    zf = z[src]
    zmin = torch.full((H * W,), float("inf"), device=xyz.device, dtype=z.dtype)
    zmin = zmin.scatter_reduce(0, pix, zf, reduce="amin")
    front = (zf <= zmin[pix] + tol).to(z.dtype)
    count = torch.zeros(H * W, device=xyz.device, dtype=z.dtype).index_add_(0, pix, front)
    color = torch.zeros(H * W, 3, device=xyz.device, dtype=z.dtype).index_add_(0, pix, rgb[src] * front[:, None])
    depth = torch.zeros(H * W, device=xyz.device, dtype=z.dtype).index_add_(0, pix, zf * front)
    hit = count > 0
    color = color / count.clamp(min=1)[:, None]
    depth = depth / count.clamp(min=1)
    return color.reshape(H, W, 3), depth.reshape(H, W), hit.reshape(H, W)


def render_points(xyz, rgb, cam, supersample=3, point_size=1.2, tol=2.5):
    """Render a voxelised coloured point cloud as a camera with a perfect matte would see it.

    The cloud is rendered at `supersample` times the resolution and averaged down, so silhouette pixels
    get a fractional alpha.

    Returns
        rgb   (H, W, 3) foreground colour (not mixed with any background; 0 where alpha is 0)
        depth (H, W)    depth of the front surface inside the pixel; 0 where alpha is 0
        alpha (H, W)    fraction of the pixel covered
    """
    s = supersample
    K = cam.K.clone()
    K[:2] = K[:2] * s
    hi = Camera(K, cam.w2c, cam.height * s, cam.width * s)
    color, depth, hit = _render_hard(xyz, rgb, hi, point_size, tol)
    hitf = hit.to(color.dtype)
    alpha = F.avg_pool2d(hitf[None, None], s)[0, 0]
    col = F.avg_pool2d((color * hitf[..., None]).permute(2, 0, 1)[None], s)[0].permute(1, 2, 0)
    col = col / alpha.clamp(min=1e-6)[..., None]
    # Front surface inside the pixel: sub-pixels within `tol` of the nearest one.
    big = torch.where(hit, depth, torch.full_like(depth, float("inf")))
    near = -F.max_pool2d(-big[None, None], s)[0, 0]
    near_up = near.repeat_interleave(s, 0).repeat_interleave(s, 1)
    front = (hit & (depth <= near_up + tol)).to(color.dtype)
    dsum = F.avg_pool2d((depth * front)[None, None], s)[0, 0]
    dcnt = F.avg_pool2d(front[None, None], s)[0, 0]
    dep = torch.where(dcnt > 0, dsum / dcnt.clamp(min=1e-6), torch.zeros_like(dsum))
    return col * (alpha > 0)[..., None], dep, alpha


def composite(rgb, alpha, background=1.0):
    """Foreground colour and alpha over a constant background."""
    return rgb * alpha[..., None] + background * (1.0 - alpha[..., None])
