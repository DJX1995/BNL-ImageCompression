import argparse
import math
import random
import shutil
import sys
import os

import h5py
import hdf5plugin
import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2


def stitch_gray_grid_with_edges(
    patches: list[np.ndarray],
    coords,                          # (N,2) array-like of [y,x] top-lefts
    img_size: tuple[int, int] = (3269, 3110),       # original image size (H, W)
    to_uint16: bool = True,          # convert float [0,1] -> uint16
) -> np.ndarray:
    """
    Reconstruct a grayscale image from 256x256 non-overlapping patches placed on a
    regular 256 grid but with an offset due to cropping. Returns a (H,W) image where
    cropped margins are zero-filled.

    - patches are assumed float in [0,1] (dtype preserved until conversion).
    - coords are top-left (y,x) for each patch.
    - img_size is the original full image shape BEFORE cropping (H,W).
    """
    PATCH = 256
    H, W = img_size
    N = len(patches)
    if N == 0:
        return np.zeros((H, W), dtype=np.uint16 if to_uint16 else np.float32)

    # Stack & basic checks
    P = np.stack(patches, axis=0)  # (N,256,256)
    if P.shape[1:] != (PATCH, PATCH):
        raise ValueError("All patches must be 256x256.")

    coords = np.asarray(coords)
    if coords.shape[-1] != 2 or coords.shape[0] != N:
        raise ValueError("coords must have shape (N,2) with [y,x] per patch.")
    ys = coords[:, 0].astype(np.int64)
    xs = coords[:, 1].astype(np.int64)

    # Normalize coords so min -> 0 (tight grid origin)
    off_h = int(ys.min())
    off_w = int(xs.min())
    ys0 = ys - off_h
    xs0 = xs - off_w

    # Validate 256-grid alignment
    if not ((ys0 % PATCH == 0).all() and (xs0 % PATCH == 0).all()):
        raise ValueError("Coords are not aligned to the 256 grid after normalization.")

    # Compute grid size for tight canvas
    ny = int(ys0.max() // PATCH) + 1
    nx = int(xs0.max() // PATCH) + 1
    if ny * nx != N:
        raise ValueError(f"Missing/extra patches: expected {ny*nx}, got {N}.")

    # Sort patches into raster order (top->bottom, left->right)
    order = np.lexsort((xs0, ys0))
    P = P[order]

    # Build tight canvas via a zero-copy block view
    tight = np.zeros((ny * PATCH, nx * PATCH), dtype=P.dtype)
    # strides in elements for (ny, nx, PATCH, PATCH) view; convert to bytes internally
    st_elems = (PATCH * nx * PATCH, PATCH, nx * PATCH, 1)
    block = as_strided(
        tight,
        shape=(ny, nx, PATCH, PATCH),
        strides=tuple(s * tight.itemsize for s in st_elems),
    )
    block[...] = P.reshape(ny, nx, PATCH, PATCH)

    # Paste tight canvas into full image at (off_h, off_w), zero-filled elsewhere
    y2 = off_h + tight.shape[0]
    x2 = off_w + tight.shape[1]
    if y2 > H or x2 > W:
        raise ValueError(
            f"Tight region (end at {(y2,x2)}) exceeds img_size {(H,W)}. "
            "Check coords or img_size."
        )

    out = np.zeros((H, W), dtype=tight.dtype)
    out[off_h:y2, off_w:x2] = tight

    # Optional conversion to uint16 with desired scaling
    if to_uint16:
        out = out.astype(np.uint16)
    return out


import os, h5py

def fix_master_for_adxv(master_in: str,
                        recon_data_h5: str,
                        master_out: str,
                        internal_path="/entry/data/data",
                        keep_numbered_link=True):
    """
    Create a clean master that ADXV likes:
      - copies /entry metadata from master_in,
      - rebuilds /entry/data with a single 'data' ExternalLink -> recon_data_h5:/entry/data/data,
      - sets NX_class/ signal attrs properly, optionally also adds data_000001 -> same target.
    """
    # compute relative path from master_out to recon_data_h5 (so the link resolves)
    out_dir = os.path.dirname(os.path.abspath(master_out))
    rel = os.path.relpath(os.path.abspath(recon_data_h5), start=out_dir)

    # make sure destination dir exists and no stale file blocks writing
    os.makedirs(out_dir or ".", exist_ok=True)
    try:
        os.remove(master_out)
    except FileNotFoundError:
        pass

    with h5py.File(master_in, "r") as fin, h5py.File(master_out, "w") as fout:
        # copy root attrs
        for an, av in fin.attrs.items():
            fout.attrs[an] = av

        # copy /entry (whole group) from fin to fout
        fin.copy("/entry", fout)  # <-- correct way

        # rebuild /entry/data
        entry_out = fout["/entry"]
        if "data" in entry_out:
            del entry_out["data"]
        data_out = entry_out.create_group("data")

        # NeXus attrs (must be bytes)
        entry_out.attrs["default"] = b"data"
        data_out.attrs["NX_class"] = b"NXdata"
        data_out.attrs["signal"]   = b"data"

        # main link name 'data' -> recon
        data_out["data"] = h5py.ExternalLink(rel, internal_path)

        # optional legacy name
        if keep_numbered_link:
            data_out["data_000001"] = h5py.ExternalLink(rel, internal_path)

        # some viewers get confused by stray 'axes' attrs
        if "axes" in data_out.attrs:
            del data_out.attrs["axes"]

    # quick sanity check: follow the link
    with h5py.File(master_out, "r") as fm:
        lnk = fm.get("/entry/data/data", getlink=True)
        assert isinstance(lnk, h5py.ExternalLink)
        target_file = os.path.join(os.path.dirname(os.path.abspath(master_out)), lnk.filename)
        with h5py.File(target_file, "r") as fd:
            _ = fd[lnk.path]  # raises if wrong

    return master_out




if __name__ == "__main__":
    test_master_h5   = "../data/CBASS/split2/test_master.h5"
    # recon_data_h5    = "../experiments/data_recon/full_cbass_lambda15.0_recon_image_full.h5"  # has /entry/data/data
    # recon_master_out = "../experiments/data_recon/full_cbass_lambda15.0_recon_image_full_master.h5"
    recon_data_h5    = "../experiments/deeplearning/subset_cbass_image_recon.h5"  # has /entry/data/data
    recon_master_out = "../experiments/deeplearning/subset_cbass_image_recon_master.h5"
    out = fix_master_for_adxv(test_master_h5, recon_data_h5, recon_master_out)
    print("Wrote:", out)