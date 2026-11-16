"""The packed dataset layout must decode to exactly the points that were written."""
import importlib.util
import os

import numpy as np

from lossysplat.data import Sequence, decode_frame

_spec = importlib.util.spec_from_file_location(
    "fetch_uvg_vpc", os.path.join(os.path.dirname(__file__), "..", "scripts", "fetch_uvg_vpc.py"))
fetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch)


def _random_frame(rng, n):
    xyz = np.unique(rng.integers(0, 512, size=(n, 3)), axis=0)
    rgb = rng.integers(0, 256, size=(len(xyz), 3)).astype(np.uint8)
    return xyz, rgb


def _as_set(xyz, rgb):
    return {tuple(row) for row in np.concatenate([xyz, rgb], axis=1).astype(np.int64)}


def test_frame_round_trip():
    xyz, rgb = _random_frame(np.random.default_rng(0), 5000)
    back_xyz, back_rgb = decode_frame(*fetch.encode_frame(xyz, rgb))
    assert _as_set(back_xyz, back_rgb) == _as_set(xyz, rgb)


def test_sequence_file_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    frames = [_random_frame(rng, n) for n in (300, 1, 2000)]
    fetch.write_packed(str(tmp_path / "Toy.npz"), frames)
    seq = Sequence("Toy", root=str(tmp_path))
    assert len(seq) == 3
    for i, (xyz, rgb) in enumerate(frames):
        assert _as_set(*seq.frame_numpy(i)) == _as_set(xyz, rgb)
