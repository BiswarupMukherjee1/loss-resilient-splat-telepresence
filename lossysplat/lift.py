"""Turn RGB-D views into pixel-aligned Gaussians.

Each foreground pixel becomes one Gaussian at its unprojected 3D point. Its covariance is built from the
3D offsets to the neighbouring pixels, so a splat is a thin disc lying on the local surface and stretches
where the surface is seen at a slant.
"""
from dataclasses import dataclass

import torch


@dataclass
class Gaussians:
    means: torch.Tensor      # (N, 3)
    covs: torch.Tensor       # (N, 3, 3)
    colors: torch.Tensor     # (N, 3)
    opacities: torch.Tensor  # (N,)
    view: torch.Tensor       # (N,) index of the source view
    pixel: torch.Tensor      # (N,) flat pixel index in the source view

    def __len__(self):
        return self.means.shape[0]

    def select(self, mask):
        return Gaussians(self.means[mask], self.covs[mask], self.colors[mask], self.opacities[mask],
                         self.view[mask], self.pixel[mask])

    @staticmethod
    def cat(parts):
        return Gaussians(*[torch.cat([getattr(p, f) for p in parts]) for f in
                           ("means", "covs", "colors", "opacities", "view", "pixel")])


def _tangent(points, valid, depth, axis, footprint, max_jump):
    """Central (or one-sided) 3D difference along an image axis, ignoring depth discontinuities."""
    fwd_p, bwd_p = torch.roll(points, -1, dims=axis), torch.roll(points, 1, dims=axis)
    fwd_d, bwd_d = torch.roll(depth, -1, dims=axis), torch.roll(depth, 1, dims=axis)
    fwd_ok = torch.roll(valid, -1, dims=axis) & ((fwd_d - depth).abs() < max_jump * footprint)
    bwd_ok = torch.roll(valid, 1, dims=axis) & ((bwd_d - depth).abs() < max_jump * footprint)
    n = points.shape[axis]
    edge = torch.arange(n, device=points.device)
    shape = [1, 1]
    shape[axis] = n
    fwd_ok = fwd_ok & (edge < n - 1).reshape(shape)
    bwd_ok = bwd_ok & (edge > 0).reshape(shape)
    both = (fwd_ok & bwd_ok)[..., None]
    tangent = torch.where(both, (fwd_p - bwd_p) / 2.0,
                          torch.where(fwd_ok[..., None], fwd_p - points,
                                      torch.where(bwd_ok[..., None], points - bwd_p, torch.zeros_like(points))))
    return tangent, fwd_ok | bwd_ok


def lift_view(rgb, depth, alpha, cam, view_index=0, sigma=0.5, opacity=0.95, thickness=0.05,
              max_stretch=1.5, max_jump=6.0, alpha_threshold=0.1, grazing=(0.3, 0.7)):
    """RGB (H, W, 3), depth (H, W), alpha (H, W) of one camera -> Gaussians.

    `rgb` is the foreground colour. A pixel's Gaussian has opacity `opacity` times the pixel's alpha.
    `sigma`: standard deviation of a splat in units of the pixel spacing on the surface.
    `max_jump`: neighbours further apart in depth than this many pixel footprints are treated as a
    different surface. `max_stretch`: cap on the tangent length, in pixel footprints.
    `grazing` = (lo, hi): with c the cosine between the surface normal and the viewing ray, the opacity is
    multiplied by a ramp that is 0 for c <= lo and 1 for c >= hi, so surfaces seen at a slant, which a
    camera samples poorly, give way to other cameras' splats. (0, 0) disables it.
    """
    valid = alpha > alpha_threshold
    points = cam.unproject(depth)
    footprint = depth / cam.K[0, 0]
    right, down = cam.R[0], cam.R[1]
    tangents = []
    for axis, fallback in ((1, right), (0, down)):
        tan, ok = _tangent(points, valid, depth, axis, footprint, max_jump)
        tan = torch.where(ok[..., None], tan, footprint[..., None] * fallback)
        length = tan.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cap = (max_stretch * footprint)[..., None]
        tangents.append(tan * torch.clamp(cap / length, max=1.0))
    tu, tv = tangents
    normal = torch.nn.functional.normalize(torch.linalg.cross(tu, tv, dim=-1), dim=-1)
    cov = sigma**2 * (tu[..., :, None] * tu[..., None, :] + tv[..., :, None] * tv[..., None, :])
    cov = cov + ((thickness * footprint) ** 2)[..., None, None] * (normal[..., :, None] * normal[..., None, :])
    weight = opacity * alpha.clamp(0, 1)
    if grazing[1] > 0:
        ray = torch.nn.functional.normalize(points - cam.position, dim=-1)
        cos = (normal * ray).sum(dim=-1).abs()
        weight = weight * ((cos - grazing[0]) / max(grazing[1] - grazing[0], 1e-6)).clamp(0, 1)
        valid = valid & (weight > 1e-3)
    flat = valid.reshape(-1)
    pixel = torch.nonzero(flat, as_tuple=False)[:, 0]
    return Gaussians(
        points.reshape(-1, 3)[flat], cov.reshape(-1, 3, 3)[flat], rgb.reshape(-1, 3)[flat],
        weight.reshape(-1)[flat],
        torch.full((pixel.shape[0],), view_index, device=rgb.device, dtype=torch.long), pixel,
    )


def lift_views(rgbs, depths, alphas, cams, **kwargs):
    """Lift several views and merge them into one set of Gaussians."""
    return Gaussians.cat([lift_view(rgbs[k], depths[k], alphas[k], cams[k], view_index=k, **kwargs)
                          for k in range(len(cams))])
