"""Pinhole cameras and the capture / evaluation rigs.

Convention: camera x points right, y down, z forward. Pixel (i, j) covers [j, j + 1) x [i, i + 1) and
its centre is at (j + 0.5, i + 0.5). World coordinates are voxel indices of the dataset, with +y up.
"""
import math
from dataclasses import dataclass

import torch


@dataclass
class Camera:
    K: torch.Tensor      # (3, 3) intrinsics
    w2c: torch.Tensor    # (4, 4) world-to-camera
    height: int
    width: int

    def to(self, device):
        return Camera(self.K.to(device), self.w2c.to(device), self.height, self.width)

    @property
    def R(self):
        return self.w2c[:3, :3]

    @property
    def t(self):
        return self.w2c[:3, 3]

    @property
    def position(self):
        return -self.R.T @ self.t

    def world_to_cam(self, xyz):
        return xyz @ self.R.T + self.t

    def cam_to_world(self, xyz_cam):
        return (xyz_cam - self.t) @ self.R

    def project(self, xyz):
        """World points (N, 3) -> pixel coordinates (N, 2) and camera depth (N,)."""
        xc = self.world_to_cam(xyz)
        z = xc[:, 2]
        u = self.K[0, 0] * xc[:, 0] / z + self.K[0, 2]
        v = self.K[1, 1] * xc[:, 1] / z + self.K[1, 2]
        return torch.stack([u, v], dim=1), z

    def unproject(self, depth):
        """Depth map (H, W) -> world points (H, W, 3) at the pixel centres."""
        dev = depth.device
        v, u = torch.meshgrid(
            torch.arange(self.height, device=dev, dtype=depth.dtype) + 0.5,
            torch.arange(self.width, device=dev, dtype=depth.dtype) + 0.5,
            indexing="ij",
        )
        x = (u - self.K[0, 2]) / self.K[0, 0] * depth
        y = (v - self.K[1, 2]) / self.K[1, 1] * depth
        return self.cam_to_world(torch.stack([x, y, depth], dim=-1).reshape(-1, 3)).reshape(
            self.height, self.width, 3)


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """World-to-camera matrix of a camera at `eye` looking at `target`."""
    eye = torch.as_tensor(eye, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    up = torch.as_tensor(up, dtype=torch.float32)
    forward = torch.nn.functional.normalize(target - eye, dim=0)
    right = torch.nn.functional.normalize(torch.linalg.cross(forward, up), dim=0)
    down = torch.linalg.cross(forward, right)
    R = torch.stack([right, down, forward], dim=0)
    w2c = torch.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = -R @ eye
    return w2c


def orbit_camera(center, radius, azimuth_deg, elevation_deg, half_extent, height, width):
    """Camera on a sphere around `center`; `half_extent` world units are visible above and below it."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    cx, cy, cz = [float(c) for c in center]
    eye = (cx + radius * math.cos(el) * math.sin(az), cy + radius * math.sin(el),
           cz + radius * math.cos(el) * math.cos(az))
    focal = radius / half_extent * (height / 2.0)
    K = torch.tensor([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])
    return Camera(K, look_at(eye, (cx, cy, cz)), height, width)


@dataclass
class RigConfig:
    """Geometry shared by the capture rig and the evaluation cameras (lengths in voxels).

    The defaults frame a standing person of the 9-bit UVG-VPC sequences at 2 voxels per pixel.
    """
    radius: float = 1600.0
    half_extent: float = 256.0
    height: int = 256
    width: int = 192
    center_y: float = 247.0
    depth_range: float = 256.0

    @property
    def near(self):
        return self.radius - self.depth_range

    @property
    def far(self):
        return self.radius + self.depth_range


def capture_rig(center_xz, cfg, n_cams=4, azimuth_offset_deg=0.0):
    """`n_cams` cameras evenly spaced on a horizontal ring at mid-body height."""
    center = (center_xz[0], cfg.center_y, center_xz[1])
    return [
        orbit_camera(center, cfg.radius, azimuth_offset_deg + 360.0 * k / n_cams, 0.0,
                     cfg.half_extent, cfg.height, cfg.width)
        for k in range(n_cams)
    ]


# (azimuth, elevation) in degrees. None of them coincides with a camera of a 2, 3, 4, 6 or 8 camera ring.
EVAL_VIEWS = [(40.0, 0.0), (130.0, 0.0), (215.0, 0.0), (305.0, 0.0), (80.0, 20.0), (260.0, 20.0)]


def eval_rig(center_xz, cfg, azimuth_offset_deg=0.0):
    """Held-out viewpoints: four on the capture ring between cameras and two raised by 20 degrees."""
    center = (center_xz[0], cfg.center_y, center_xz[1])
    return [
        orbit_camera(center, cfg.radius, azimuth_offset_deg + az, el, cfg.half_extent, cfg.height, cfg.width)
        for az, el in EVAL_VIEWS
    ]


def perturb_camera(cam, rot_deg=0.0, shift=0.0, generator=None):
    """Camera with a random pose error: a rotation of `rot_deg` degrees about a random axis and a shift
    of `shift` world units in a random direction. Models an imperfect extrinsic calibration."""
    axis = torch.nn.functional.normalize(torch.randn(3, generator=generator), dim=0)
    ang = math.radians(rot_deg)
    kx = torch.tensor([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    dR = torch.eye(3) + math.sin(ang) * kx + (1 - math.cos(ang)) * (kx @ kx)
    direction = torch.nn.functional.normalize(torch.randn(3, generator=generator), dim=0)
    w2c = cam.w2c.clone().cpu()
    R, pos = w2c[:3, :3], cam.position.cpu()
    R_new = R @ dR
    pos_new = pos + shift * direction
    w2c[:3, :3] = R_new
    w2c[:3, 3] = -R_new @ pos_new
    return Camera(cam.K, w2c.to(cam.w2c.device), cam.height, cam.width)
