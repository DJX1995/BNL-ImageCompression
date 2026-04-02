import tqdm
import argparse
import math
import random
import sys
import os
import time
import h5py
import hdf5plugin
import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2
import matplotlib.pyplot as plt
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from generate_h5file import stitch_gray_grid_with_edges

def visual_recon():
    indexs = range(30,50)
    suffixs = []
    lmbda = 10.

    load_path = f'../data/CBASS/test_patch.h5'
    # New output file to store full images instead of patches
    save_path_full = f'../data/CBASS/test_image_stitch.h5'
    with h5py.File(load_path, "r") as fin, h5py.File(save_path_full, "w") as fout:
        # Copy file-level attributes
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

        # Create groups and copy everything except entry/data/patches
        entry_in = fin['entry']
        entry_out = fout.create_group('entry')
        for k, v in entry_in.attrs.items():
            entry_out.attrs[k] = v

        # Copy non-data children of entry as-is (links copy without loading data into RAM)
        for name, obj in entry_in.items():
            if name == 'data':
                continue
            entry_in.copy(name, entry_out, name=name)

        data_in = entry_in['data']
        data_out = entry_out.create_group('data')
        for k, v in data_in.attrs.items():
            data_out.attrs[k] = v
        for name, obj in data_in.items():
            if name == 'patches':
                continue
            data_in.copy(name, data_out, name=name)

        # We'll append stitched images one-by-one to this resizable dataset
        images_dset = None
        write_idx = 0

        num_patches = data_in['patches'].shape[0]
        full_img_patches = []
        full_img_postions = []
        
        for index in range(num_patches):
            x = data_in['patches'][index]       # shape (N, H, W)
            img_indices = data_in['img_indices'][index]
            patch_indices = data_in['patch_indices'][index]
            patch_positions = data_in['patch_positions'][index]
            suffixs.append([img_indices, patch_indices])
            full_img_patches.append(x)
            full_img_postions.append(patch_positions)
            
            # Stitch and write out a full image when a complete set of patches is accumulated
            if (index + 1) % 144 == 0:
                all_patches = np.stack(full_img_patches, axis=0)
                all_positions = np.stack(full_img_postions, axis=0)
                img_full = stitch_gray_grid_256(all_patches, all_positions)  # (H, W)

                # Lazily create resizable images dataset after first full image
                if images_dset is None:
                    H, W = img_full.shape
                    images_dset = data_out.create_dataset(
                        'images',
                        shape=(0, H, W),
                        maxshape=(None, H, W),
                        dtype=img_full.dtype,
                        chunks=(1, H, W),
                        compression='gzip',
                        compression_opts=4,
                        shuffle=True,
                    )

                images_dset.resize(write_idx + 1, axis=0)
                images_dset[write_idx] = img_full
                write_idx += 1

                full_img_patches = []
                full_img_postions = []
            # save_path = f'../experiments/data_recon/images/original_{img_indices}_{patch_indices}.png'
            # visualize_patch(x, save_path)

    load_path = f'../experiments/data_recon/lambda{lmbda}_recon.h5'
    with h5py.File(load_path, "r") as f:
        for i, index in enumerate(indexs):
            x = f["recon"][index]          # shape (N, H, W)
            # print(x.shape, x.dtype)
            img_indices, patch_indices = suffixs[i]
            save_path = f'../experiments/data_recon/images/compressed_lambda{lmbda}_{img_indices}_{patch_indices}.png'
            visualize_patch(x.astype(np.float32), save_path)



def visualize_patch(image_patch, save_path='img.png'):
    if np.issubdtype(image_patch.dtype, np.uint16):
        image_patch = image_patch.view(np.int16)
        image_patch[image_patch > 255] = 255  # artifacts caused by detector?
        image_patch[image_patch < 0] = 0  # grids are -1, artifacts are -2 values?
        image_patch = image_patch.astype(np.float32)
        image_patch = image_patch / 255.0
    h, w = image_patch.shape
    full_image = np.zeros((h, w, 3))
    full_image_norm = image_patch
    # full_image_norm = cv2.normalize(image_patch, dst=None, alpha=0, beta=4, norm_type=cv2.NORM_MINMAX)
    full_image_norm[full_image_norm > 1] = 1
    full_image[:, :, 0] += full_image_norm
    full_image[:, :, 1] += full_image_norm
    full_image[:, :, 2] += full_image_norm

    full_image[full_image > 1] = 1
    full_image = full_image * 255
    cv2.imwrite(save_path, full_image)


def patch_recon_and_save_to_image(test_patch_h5=f'../data/CBASS/full/test_patch.h5', 
                                  recon_patch_h5=f'../experiments/data_recon/full_cbass_hyperprior_lambda10.0_recon_softclip.h5',
                                  out_h5 = f'../experiments/data_recon/full_cbass_lambda10.0_recon_image_full.h5'):
    K = 144

    # read coords (y,x) for all patches
    with h5py.File(test_patch_h5, "r") as fp:
        if "coords" in fp:
            coords_all = fp["coords"][:, 1:3].astype(np.int64)     # (N, 2)
        else:
            coords_all = fp["entry/data/patch_positions"][:].astype(np.int64)

    with h5py.File(recon_patch_h5, "r") as fr:
        recon = fr["recon"]                                        # (N, pH, pW)
        N = recon.shape[0]
        assert N % K == 0, f"N_patches={N} not divisible by K={K}"
        n_images = N // K

        # ---------- create output file & dataset ----------
        os.makedirs(os.path.dirname(os.path.abspath(out_h5)) or ".", exist_ok=True)
        with h5py.File(out_h5, "w") as fo:
            entry = fo.require_group("entry")
            data  = entry.require_group("data")
            dimg  = None  # lazy-created after stitching the first image

            for i in tqdm.tqdm(range(n_images), desc=f"Stitch -> {os.path.basename(out_h5)}"):
                s, e = i * K, (i + 1) * K
                patches_block  = recon[s:e, ...].view(np.int16)                     # (K, pH, pW)
                np.maximum(patches_block, 0, out=patches_block)
                coords_block   = coords_all[s:e, :]                # (K, 2)

                img = stitch_gray_grid_with_edges(patches_block, coords_block)  # (H, W)

                # create dataset after first stitched image
                if dimg is None:
                    H, W = img.shape
                    dimg = data.create_dataset(
                        "data",
                        shape=(n_images, H, W),
                        dtype=img.dtype,
                        chunks=(1, H, W),
                        **hdf5plugin.Bitshuffle(nelems=0, cname="lz4")
                    )

                dimg[i, ...] = img


def stitch_gray_grid_256(
    patches: list[np.ndarray],
    coords: list[tuple[int, int]],
) -> np.ndarray:
    """
    Reconstruct a grayscale float image (H, W) from 256x256 non-overlapping patches
    that form a complete centered grid (uniform stride=256, no gaps/overlaps).

    Assumes patch values are in [0,1]. Preserves dtype (e.g., float32).
    """
    PATCH = 256  # local constant
    image_shape = PATCH * 12

    H, W = image_shape, image_shape
    N = len(patches)
    if N == 0:
        return np.zeros((H, W), dtype=patches.dtype)

    # Stack and basic checks
    P = np.stack(patches, axis=0)           # (N, 256, 256)
    assert P.shape[1:] == (PATCH, PATCH), "All patches must be 256x256."

    # Normalize coords so min -> 0
    ys = coords[:, 0]
    xs = coords[:, 1]
    off_h = int(ys.min())
    off_w = int(xs.min())
    ys0 = ys - off_h
    xs0 = xs - off_w

    # Ensure they lie on a 256 grid
    if not ((ys0 % PATCH == 0).all() and (xs0 % PATCH == 0).all()):
        raise ValueError("Coords are not aligned to the 256 grid after normalization.")

    # Grid size (tight image dims will be ny*256, nx*256)
    ny = int(ys0.max() // PATCH) + 1
    nx = int(xs0.max() // PATCH) + 1
    if ny * nx != N:
        raise ValueError(f"Missing/extra patches: expected {ny*nx}, got {N}.")

    # Sort patches into raster order (top->bottom, left->right)
    order = np.lexsort((xs0, ys0))
    P = P[order]

    # Build the tight canvas with a block view (no loops)
    out = np.zeros((ny * PATCH, nx * PATCH), dtype=P.dtype)

    # strides in elements for (ny, nx, PATCH, PATCH) view; multiply by itemsize internally
    st_elems = (PATCH * nx * PATCH, PATCH, nx * PATCH, 1)
    block = as_strided(
        out,
        shape=(ny, nx, PATCH, PATCH),
        strides=tuple(s * out.itemsize for s in st_elems),
    )
    block[...] = P.reshape(ny, nx, PATCH, PATCH)

    if np.issubdtype(patches.dtype, np.uint16):
        out = out.view(np.int16)
        out[out > 255] = 255  # artifacts caused by detector?
        out[out < 0] = 0  # grids are -1, artifacts are -2 values?
        return out
    
    # Handle NaN and inf values before conversion
    out_clean = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    out_clean = np.clip(out_clean, 0.0, 1.0)  # Ensure values are in [0,1] range
    out_uint16 = np.round(out_clean * 255).astype(np.uint16)
    return out_uint16


def max_values_in_circles(image, centers, radius):
    """
    Get the maximum pixel value within a circular region of fixed radius around each center.

    Args:
        image (np.ndarray): 2D image (grayscale).
        centers (array-like): List or np.ndarray of (y, x) float coordinates.
        radius (int): Radius of circular region.

    Returns:
        np.ndarray: Max values within each circle, with 3*log(max_value)
    """
    max_vals = []

    for center in centers:
        cy, cx = int(round(center[0])), int(round(center[1]))

        y_min = int(max(cy - radius, 0))
        y_max = int(min(cy + radius + 1, image.shape[0]))
        x_min = int(max(cx - radius, 0))
        x_max = int(min(cx + radius + 1, image.shape[1]))

        patch = image[y_min:y_max, x_min:x_max]

        # Create meshgrid relative to center
        y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (y_grid - cy)**2 + (x_grid - cx)**2 <= radius**2

        masked_values = patch[mask]
        max_val = np.max(masked_values) if masked_values.size > 0 else 0.
        
        max_vals.append(3*np.log(max_val + 1e-8))

    return np.array(max_vals)


def plot_equal_width_hist(values, n_bins=30, save_dir=".", output_name="hist", upper_percentile=95., start_at_zero=True, handle_overflow='clip'):
    v = np.asarray(values).ravel()
    if v.size == 0:
        raise ValueError("Empty values.")

    vmin = 0.0 if start_at_zero else float(v.min())
    vmax_raw = float(v.max())

    # Cap vmax to chosen percentile
    vmax = float(np.percentile(v, upper_percentile))
    # Avoid degenerate case where percentile collapses
    if not np.isfinite(vmax):
        vmax = vmax_raw

    # ----- NEW: overflow handling -----
    overflow_mask = v > vmax
    overflow_count = int(overflow_mask.sum())
    if handle_overflow == "clip":
        v_hist = np.minimum(v, vmax)           # push overflow into last bin
    elif handle_overflow == "ignore":
        v_hist = v[~overflow_mask]             # drop overflow values
    else:
        raise ValueError("handle_overflow must be 'clip' or 'ignore'")

    # make n_bins equal-width edges in value space
    edges = np.linspace(vmin, vmax, n_bins + 1)
    # ensure the rightmost edge captures vmax exactly
    edges[-1] = np.nextafter(edges[-1], np.float64('inf'))

    # counts per equal-width bin
    counts, _ = np.histogram(v_hist, bins=edges)

    # ----- plotting (same visual width bars) -----
    n_bins = len(edges) - 1
    x_labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(n_bins)]
    x_positions = np.arange(n_bins)

    output_fpath = os.path.join(save_dir, f"{output_name}_spot_peak_hist.png")

    plt.figure(figsize=(max(12, 0.6 * n_bins), 6))
    plt.bar(x_positions, counts, edgecolor="black")  # constant-width bars

    # sparse tick labels to avoid clutter
    label_step = max(1, n_bins // 15)
    plt.xticks(
        ticks=x_positions[::label_step],
        labels=[x_labels[i] for i in range(0, n_bins, label_step)],
        rotation=45, ha='right'
    )

    plt.title("Histogram (equal-width bins)")
    plt.xlabel("Value range")
    plt.ylabel("Count")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    total_count = counts.sum()
    plt.text(0.98, 0.95, f"Total Count: {total_count:.1f}",
             ha='right', va='top', transform=plt.gca().transAxes, fontsize=10, color='blue')
    plt.text(0.98, 0.85, f"Total Bins: {n_bins}",
             ha='right', va='top', transform=plt.gca().transAxes, fontsize=10, color='blue')

    # ----- NEW: annotate overflow count if any -----
    if overflow_count > 0:
        how = "clipped" if handle_overflow == "clip" else "ignored"
        plt.text(0.98, 0.75, f"Overflow >P{upper_percentile:g}: {overflow_count} ({how})",
                 ha='right', va='top', transform=plt.gca().transAxes, fontsize=10, color='blue')

    plt.tight_layout()
    plt.savefig(output_fpath, dpi=300)
    plt.close()

    try:
        # Create an artifact
        artifact = wandb.Artifact(
            name="histograms",   # collection name
            type="image"          # you choose the type (e.g., "plot", "figure", "result")
        )
        
        # Add your file (relative path is fine, wandb will resolve it)
        artifact.add_file(output_fpath)

        # Log the artifact to the current run
        wandb.log_artifact(artifact)
        np.save(f"{output_name}_max_values.npy", np.array(values))
        artifact = wandb.Artifact("max-values", type="dataset")
        artifact.add_file(f"{output_name}_max_values.npy")
        wandb.log_artifact(artifact)
    except:
        pass
    return edges, counts, output_fpath



def compute_equal_count_radial_bins(values, n_bins=15):
    """
    values : array-like, shape (N,)
        Spot peak values (one per spot).
    n_bins : int
        Target number of quantile bins (actual may be smaller if many ties).
    Returns
    -------
    edges : np.ndarray, shape (B+1,)
        Bin edges (monotonic). B may be < n_bins if ties collapse edges.
    counts : np.ndarray, shape (B,)
        Count per bin.
    bin_idx : np.ndarray, shape (N,)
        Bin index per value in [0, B-1].
    spot_indices_per_bin : dict[int, list[int]]
        Indices of spots that fell into each bin.
    """
    x = np.asarray(values).ravel()
    if x.size == 0:
        raise ValueError("No spot peak values provided.")

    # cap bins to number of samples
    B = int(min(max(n_bins, 1), x.size))

    # compute quantile edges
    qs = np.linspace(0.0, 1.0, B + 1)
    edges = np.quantile(x, qs, method="linear")

    # force start at zero (for radial bins)
    edges[0] = 0.0

    # ensure strictly increasing edges (collapse duplicates)
    edges_unique = np.maximum.accumulate(edges)
    keep = np.r_[True, np.diff(edges_unique) > 0]
    edges = edges_unique[keep]

    if edges.size < 2:
        # degenerate: all values identical
        edges = np.array([0.0, x.max() if x.max() > 0 else 1.0])

    B = edges.size - 1

    # assign bins
    bin_idx = np.searchsorted(edges, x, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, B - 1)

    counts = np.bincount(bin_idx, minlength=B)

    spot_indices_per_bin = {i: [] for i in range(B)}
    for i, b in enumerate(bin_idx):
        spot_indices_per_bin[b].append(i)

    return edges, counts, bin_idx, spot_indices_per_bin


def false_positive_peak_value(fp_centers_list, image_org_list, radius=10, draw_hist=True, n_bins=15, save_path='./'):
    """
    fp_centers : list, each element in list in shape (N, 2), where N is the number of fp spots
        Spot peak values (one per spot).
    image_org : array-like, shape (N,)
        Spot peak values (one per spot).
    n_bins : int
        Target number of quantile bins (actual may be smaller if many ties).
    Returns
    -------
    edges : np.ndarray, shape (B+1,)
        Bin edges (monotonic). B may be < n_bins if ties collapse edges.
    counts : np.ndarray, shape (B,)
        Count per bin.
    bin_idx : np.ndarray, shape (N,)
        Bin index per value in [0, B-1].
    spot_indices_per_bin : dict[int, list[int]]
        Indices of spots that fell into each bin.
    """
    assert len(fp_centers_list) == len(image_org_list)
    max_values_list = []
    for fp_centers, image_org in zip(fp_centers_list, image_org_list):
        if len(fp_centers) == 0:
            continue
        max_values = max_values_in_circles(image_org, fp_centers, radius)
        max_values_list.extend(max_values.tolist())
    # max_values_list = np.array(max_values_list)

    # if draw_hist:
    #     output_fpath = os.path.join(save_path, 'fp_hist.png')
    #     edges, counts, bin_idx, spot_per_bin = compute_equal_count_radial_bins(max_values_list, n_bins=n_bins)

    #     plt.hist(max_values_list, bins=edges)
    #     plt.xlabel("Value")
    #     plt.ylabel("Count")
    #     plt.title("Equal-count radial bins (start=0)")
    #     plt.tight_layout()
    #     plt.savefig(output_fpath, dpi=300)
    #     plt.close()
    #     # Create an artifact
    #     artifact = wandb.Artifact(
    #         name="histograms",   # collection name
    #         type="image"          # you choose the type (e.g., "plot", "figure", "result")
    #     )

    #     # Add your file (relative path is fine, wandb will resolve it)
    #     artifact.add_file(output_fpath)

    #     # Log the artifact to the current run
    #     wandb.log_artifact(artifact)
    return max_values_list



def plot_metric(x_labels, eval_metrics):
    # Plot and annotate with predicted spot count, and show GT spot number as a reference
    fig, axs = plt.subplots(1, 1, figsize=(12, 9), sharex=True)

    metric_labels = ['Precision', 'Recall', 'F1-score']
    gt_spotnum = 1026  # Ground truth spot number

    # Plot and annotate for eval_j
    for i in range(3):
        axs.plot(x_labels, eval_metrics[:, i], label=f'{metric_labels[i]}', marker='o')
    # for x, y, pred in zip(x_labels, eval_metrics[:, 2], eval_metrics[:, 3]):  # Annotate F1
    #     axs[0].annotate(f'{int(pred)}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)

    # # Plot and annotate for eval_h
    # for i in range(3):
    #     axs[1].plot(x_labels, eval_h[:, i], label=f'H-Compress: {metric_labels[i]}', marker='s')
    # for x, y, pred in zip(x_labels, eval_h[:, 2], eval_h[:, 3]):  # Annotate F1
    #     axs[1].annotate(f'{int(pred)}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)

    # # Add GT spot number text
    # axs[0].text(0.01, 0.25, f'GT Spot Number = {gt_spotnum}', transform=axs[0].transAxes,
    #             fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    # axs[1].text(0.01, 0.25, f'GT Spot Number = {gt_spotnum}', transform=axs[1].transAxes,
    #             fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    # Titles and labels
    axs.set_title('Spot Finder Results')
    # axs[1].set_title('H-Compress Spot Finder Results')
    axs.set_xlabel('Compression Ratio')

    # for ax in axs:
    axs.set_ylabel('Value')
    axs.legend()
    axs.grid(True)

    plt.tight_layout()
    # plt.show()
    plt.savefig("eval_metrics_plot.png", dpi=300)



def plot_fig(xvalues, yvalues, save_dir=".", output_name='lambda_x_ratio'):
    output_fpath = os.path.join(save_dir, f"{output_name}.png")
    x = np.array(xvalues)
    y = np.array(yvalues)

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, marker='o', linestyle='-')

    plt.xlabel("lambda")
    plt.ylabel("compression ratio")
    plt.tight_layout()
    plt.savefig(output_fpath, dpi=300)
    plt.close()



if __name__ == "__main__":

    values = np.load('fp_max_values.npy')
    plot_equal_width_hist(values, n_bins=20, output_name='fp_hist', upper_percentile=95., start_at_zero=False)
    values = np.load('fn_max_values.npy')
    plot_equal_width_hist(values, n_bins=20, output_name='fn_hist', upper_percentile=95., start_at_zero=False)

    # x_labels = np.array([110.6, 150.5, 182.1])
    # gt_spotnum = 1026
    # eval_metrics = np.array([[0.912, 0.980, 0.945],
    #                         [0.889, 0.983, 0.934],
    #                         [0.791, 0.983, 0.877]])
    # plot_metric(x_labels, eval_metrics)

    
    # save_dir="."
    # output_name='lambda_x_ratio'

    # xvalues = [10, 15, 20, 30]
    # yvalues = [160.1, 156.4, 135.6, 115.8]
    
    # output_fpath = os.path.join(save_dir, f"{output_name}.png")
    # x = np.array(xvalues)
    # y = np.array(yvalues)

    # plt.figure(figsize=(12, 6))
    # plt.plot(x, y, marker='o', linestyle='-')

    # plt.xlabel("lambda")
    # plt.ylabel("compression ratio")
    # plt.tight_layout()
    # plt.savefig(output_fpath, dpi=300)
    # plt.close()
    