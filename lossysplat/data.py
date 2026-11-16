"""Access to packed point-cloud sequences (see scripts/fetch_uvg_vpc.py) and default locations."""
import os

import numpy as np
import torch


def store_dir():
    """Root of large files (data, runs). Set LOSSYSPLAT_STORE to move it."""
    default = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "store")
    return os.path.abspath(os.environ.get("LOSSYSPLAT_STORE", default))


def _compact(v):
    """Keep every third bit of v (inverse of the Morton bit spreading)."""
    v = v & 0x09249249
    v = (v | (v >> 2)) & 0x030C30C3
    v = (v | (v >> 4)) & 0x0300F00F
    v = (v | (v >> 8)) & 0x030000FF
    v = (v | (v >> 16)) & 0x3FF
    return v


def decode_frame(geometry, color):
    """Packed frame -> xyz uint16 (N, 3), rgb uint8 (N, 3). See scripts/fetch_uvg_vpc.py for the layout."""
    delta = np.ascontiguousarray(geometry.T).view("<u4")[:, 0]
    code = np.cumsum(delta, dtype=np.uint64).astype(np.uint32)
    xyz = np.stack([_compact(code), _compact(code >> 1), _compact(code >> 2)], axis=1).astype(np.uint16)
    chans = np.cumsum(color, axis=1, dtype=np.uint8)
    green = chans[0]
    rgb = np.stack([chans[1] + green, green, chans[2] + green], axis=1).astype(np.uint8)
    return xyz, rgb


class Sequence:
    """All frames of one voxelised point-cloud sequence, kept packed in memory and decoded per frame."""

    def __init__(self, name, root=None):
        root = root or os.path.join(store_dir(), "data", "uvg_vpc")
        with np.load(os.path.join(root, f"{name}.npz")) as f:
            self.geometry, self.color, self.offsets = f["geometry"], f["color"], f["offsets"]
        self.name = name
        self._bounds = None

    def __len__(self):
        return len(self.offsets) - 1

    def frame_numpy(self, index):
        a, b = self.offsets[index], self.offsets[index + 1]
        return decode_frame(self.geometry[:, a:b], self.color[:, a:b])

    def frame(self, index, device="cpu"):
        """Points (N, 3) float32 in voxel units and colours (N, 3) in [0, 1]."""
        xyz, rgb = self.frame_numpy(index)
        return (torch.from_numpy(xyz.astype(np.float32)).to(device),
                torch.from_numpy(rgb).to(device).float() / 255.0)

    def bounds(self, step=10):
        """Bounding box (lo, hi) over every `step`-th frame."""
        if self._bounds is None:
            lo, hi = np.full(3, 65535), np.zeros(3)
            for i in range(0, len(self), step):
                xyz, _ = self.frame_numpy(i)
                lo, hi = np.minimum(lo, xyz.min(axis=0)), np.maximum(hi, xyz.max(axis=0))
            self._bounds = (lo.astype(np.float32), hi.astype(np.float32))
        return self._bounds

    def center_xz(self):
        """Centre of the sequence's horizontal bounding box, used as the rig centre."""
        lo, hi = self.bounds()
        return float((lo[0] + hi[0]) / 2), float((lo[2] + hi[2]) / 2)
