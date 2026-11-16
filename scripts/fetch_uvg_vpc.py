"""Download UVG-VPC sequences (9-bit voxelised point clouds) and pack each one into a single .npz file.

The archives are streamed through tar into a temporary directory, parsed, and removed, so only the
packed file stays on disk. Each .npz holds all frames of one sequence, losslessly, at about 2 bytes per
point. Inside a frame the points are sorted along a Morton (Z-order) curve and stored as differences:
    geometry  uint8 (4, P)   little-endian bytes of the Morton-code differences (restart at each frame)
    color     uint8 (3, P)   differences modulo 256 of (green, red - green, blue - green)
    offsets   int64 (F + 1,) frame f owns columns offsets[f]:offsets[f + 1]
`decode_frame` below (and lossysplat.data) turns a frame back into coordinates and colours.

The dataset is released by Tampere University for non-commercial research under its own agreement
(https://ultravideo.fi/UVG-VPC/). It is not redistributed here.

Usage: python scripts/fetch_uvg_vpc.py --out DATA_DIR [--tmp TMP_DIR] [--seqs Gymnast BlueSpin ...]
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np

BASE_URL = "https://ultravideo.fi/UVG-VPC"
SEQUENCES = [
    "BlueBackpack", "BlueSpin", "BlueSquat", "CasualSpin", "CasualSquat", "ElegantDance",
    "ElegantWave", "FlowerDance", "FlowerWave", "Gymnast", "HelloGoodbye", "ReadyForWinter",
]
PLY_TYPES = {
    "double": "<f8", "float64": "<f8", "float": "<f4", "float32": "<f4",
    "uchar": "u1", "uint8": "u1", "int": "<i4", "int32": "<i4", "uint": "<u4",
    "short": "<i2", "ushort": "<u2", "char": "i1",
}


def read_ply(path):
    """Return (xyz float64 (N, 3), rgb uint8 (N, 3)) from a binary little-endian PLY file."""
    with open(path, "rb") as fh:
        fields, count, fmt = [], None, None
        while True:
            line = fh.readline().decode("ascii", "replace").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("element") and count is not None:
                raise ValueError(f"{path}: unexpected extra element '{line}'")
            elif line.startswith("property"):
                _, typ, name = line.split()
                fields.append((name, PLY_TYPES[typ]))
            elif line == "end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: unsupported PLY format {fmt}")
        dtype = np.dtype(fields)
        data = np.frombuffer(fh.read(count * dtype.itemsize), dtype=dtype)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1)
    return xyz, rgb


def _spread(v):
    """Insert two zero bits between the 10 low bits of v."""
    v = v & 0x3FF
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v


def _compact(v):
    """Inverse of _spread."""
    v = v & 0x09249249
    v = (v | (v >> 2)) & 0x030C30C3
    v = (v | (v >> 4)) & 0x0300F00F
    v = (v | (v >> 8)) & 0x030000FF
    v = (v | (v >> 16)) & 0x3FF
    return v


def encode_frame(xyz, rgb):
    """Voxel coordinates (N, 3) < 1024 and colours (N, 3) uint8 -> geometry (4, N), color (3, N)."""
    xyz = xyz.astype(np.uint32)
    code = _spread(xyz[:, 0]) | (_spread(xyz[:, 1]) << 1) | (_spread(xyz[:, 2]) << 2)
    order = np.argsort(code, kind="stable")
    code = code[order]
    delta = np.diff(code, prepend=np.uint32(0)).astype("<u4")
    geometry = delta.view(np.uint8).reshape(-1, 4).T.copy()
    c = rgb[order].astype(np.int16)
    chans = np.stack([c[:, 1], c[:, 0] - c[:, 1], c[:, 2] - c[:, 1]])
    color = np.diff(chans, axis=1, prepend=np.zeros((3, 1), np.int16)).astype(np.uint8)
    return geometry, color


def decode_frame(geometry, color):
    """Inverse of encode_frame: returns xyz uint16 (N, 3) and rgb uint8 (N, 3), in Morton order."""
    delta = np.ascontiguousarray(geometry.T).view("<u4")[:, 0]
    code = np.cumsum(delta, dtype=np.uint64).astype(np.uint32)
    xyz = np.stack([_compact(code), _compact(code >> 1), _compact(code >> 2)], axis=1).astype(np.uint16)
    chans = np.cumsum(color, axis=1, dtype=np.uint8)
    green = chans[0]
    rgb = np.stack([chans[1] + green, green, chans[2] + green], axis=1).astype(np.uint8)
    return xyz, rgb


def write_packed(out_path, frames):
    """`frames`: iterable of (xyz, rgb) per frame. Writes the packed file and checks it decodes back."""
    geo, col, offsets = [], [], [0]
    for xyz, rgb in frames:
        if xyz.min() < 0 or xyz.max() > 1023:
            raise ValueError("coordinates outside the 10-bit voxel grid")
        g, c = encode_frame(xyz, rgb)
        back_xyz, back_rgb = decode_frame(g, c)
        order = np.lexsort((np.arange(len(xyz)), _morton(xyz)))
        if not (np.array_equal(back_xyz, xyz[order].astype(np.uint16))
                and np.array_equal(back_rgb, rgb[order].astype(np.uint8))):
            raise RuntimeError("packed frame does not decode to the input")
        geo.append(g)
        col.append(c)
        offsets.append(offsets[-1] + len(xyz))
    tmp_path = out_path + ".tmp.npz"
    np.savez_compressed(tmp_path, geometry=np.concatenate(geo, axis=1), color=np.concatenate(col, axis=1),
                        offsets=np.asarray(offsets, dtype=np.int64))
    os.replace(tmp_path, out_path)
    return len(offsets) - 1, offsets[-1]


def _morton(xyz):
    xyz = xyz.astype(np.uint32)
    return _spread(xyz[:, 0]) | (_spread(xyz[:, 1]) << 1) | (_spread(xyz[:, 2]) << 2)


def _ply_frames(files):
    for f in files:
        xyz, rgb = read_ply(f)
        if xyz.min() < 0 or not np.allclose(xyz, np.round(xyz)):
            raise ValueError(f"{f}: coordinates are not voxel indices")
        yield xyz.astype(np.int64), rgb.astype(np.uint8)


def pack_sequence(ply_dir, out_path):
    files = sorted(glob.glob(os.path.join(ply_dir, "**", "*.ply"), recursive=True))
    if not files:
        raise RuntimeError(f"no PLY files under {ply_dir}")
    return write_packed(out_path, _ply_frames(files))


def fetch(seq, tmp_dir):
    """Stream one archive through tar into a fresh directory and return that directory."""
    dest = os.path.join(tmp_dir, f"{seq}_vox9")
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    url = f"{BASE_URL}/{seq}_vox9.tar.gz"
    curl = subprocess.Popen(["curl", "-sS", "--fail", "--retry", "3", url], stdout=subprocess.PIPE)
    tar = subprocess.Popen(["tar", "-xz", "-C", dest], stdin=curl.stdout)
    curl.stdout.close()
    tar_rc, curl_rc = tar.wait(), curl.wait()
    if curl_rc != 0 or tar_rc != 0:
        raise RuntimeError(f"{seq}: download failed (curl {curl_rc}, tar {tar_rc})")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=None, help="scratch directory for extraction (default: OUT/_tmp)")
    ap.add_argument("--seqs", nargs="*", default=SEQUENCES)
    ap.add_argument("--from-dir", default=None,
                    help="pack an already extracted sequence directory instead of downloading (one --seqs name)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tmp_dir = args.tmp or os.path.join(args.out, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    for seq in args.seqs:
        out_path = os.path.join(args.out, f"{seq}.npz")
        if os.path.exists(out_path):
            print(f"{seq}: already packed", flush=True)
            continue
        if args.from_dir:
            frames, points = pack_sequence(args.from_dir, out_path)
        else:
            dest = fetch(seq, tmp_dir)
            frames, points = pack_sequence(dest, out_path)
            shutil.rmtree(dest)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"{seq}: {frames} frames, {points} points, {size_mb:.0f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
