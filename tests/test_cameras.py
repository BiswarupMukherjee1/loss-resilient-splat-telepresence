"""Projection and unprojection must be inverse, and the rig must look at its centre."""
import torch

from lossysplat.cameras import RigConfig, capture_rig, eval_rig, perturb_camera


def test_unproject_then_project_returns_pixel_centres():
    cam = eval_rig((250.0, 260.0), RigConfig())[4]
    depth = 1400.0 + 300.0 * torch.rand(cam.height, cam.width, generator=torch.Generator().manual_seed(0))
    uv, z = cam.project(cam.unproject(depth).reshape(-1, 3))
    v, u = torch.meshgrid(torch.arange(cam.height) + 0.5, torch.arange(cam.width) + 0.5, indexing="ij")
    assert torch.allclose(uv[:, 0], u.reshape(-1), atol=1e-2)
    assert torch.allclose(uv[:, 1], v.reshape(-1), atol=1e-2)
    assert torch.allclose(z, depth.reshape(-1), rtol=1e-5)


def test_rig_cameras_look_at_the_centre():
    cfg = RigConfig()
    centre = torch.tensor([[250.0, cfg.center_y, 260.0]])
    for cam in capture_rig((250.0, 260.0), cfg, n_cams=6):
        uv, z = cam.project(centre)
        assert torch.allclose(uv[0], torch.tensor([cfg.width / 2, cfg.height / 2]), atol=1e-2)
        assert abs(float(z) - cfg.radius) < 1e-2


def test_perturbed_camera_keeps_its_position_and_a_rotation_matrix():
    cam = capture_rig((250.0, 260.0), RigConfig())[1]
    moved = perturb_camera(cam, rot_deg=0.5, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(moved.position, cam.position, atol=1e-2)
    assert torch.allclose(moved.R @ moved.R.T, torch.eye(3), atol=1e-5)
    assert not torch.allclose(moved.R, cam.R, atol=1e-4)
