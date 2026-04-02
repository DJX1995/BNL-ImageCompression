import os, h5py
import numpy as np
from tqdm import tqdm
# import hdf5plugin  # uncomment if your input uses bitshuffle/LZ4

def patchify_image_centered(image: np.ndarray, patch_size: int):
    """Return (patches, coords) where coords = [(y,x), ...] top-left for each patch."""
    H, W = image.shape
    rem_h, rem_w = H % patch_size, W % patch_size
    off_h, off_w = rem_h // 2, rem_w // 2
    usable_h = H - rem_h
    usable_w = W - rem_w

    patches, coords = [], []
    for y in range(off_h, off_h + usable_h, patch_size):
        for x in range(off_w, off_w + usable_w, patch_size):
            patches.append(image[y:y + patch_size, x:x + patch_size])
            coords.append((y, x))
    return patches, coords

def write_patches(split_data_h5: str, out_h5: str, patch_size: int = 256,
                  store_coords: bool = True, compression: str = "gzip", compression_opts: int = 4,
                  data_path: str = "/entry/data/data", images_per_batch: int = 32):
    """
    Reads split_data_h5:/entry/data/data (shape: [N, H, W]) and writes patches to out_h5:
      - out["patches"] : (N * K, patch_size, patch_size)  [K = patches per image]
      - out["coords"]  : (N * K, 3) int32  [img_idx, y, x]  (optional)
    """
    with h5py.File(split_data_h5, "r") as fsrc:
        dsrc = fsrc[data_path]  # N, H, W
        N, H, W = dsrc.shape
        dtype = dsrc.dtype

        # compute how many patches per image (constant given fixed size & centered grid)
        rem_h, rem_w = H % patch_size, W % patch_size
        usable_h = H - rem_h
        usable_w = W - rem_w
        if usable_h <= 0 or usable_w <= 0:
            raise ValueError(f"Patch size {patch_size} too large for image ({H},{W}).")
        patches_per_row = usable_w // patch_size
        patches_per_col = usable_h // patch_size
        K = patches_per_row * patches_per_col
        total_patches = N * K

        # prepare output
        os.makedirs(os.path.dirname(os.path.abspath(out_h5)) or ".", exist_ok=True)
        with h5py.File(out_h5, "w") as fdst:
            # patches dataset (chunk along first dim for streaming writes)
            chunk_p = min(64, max(1, K))  # small-ish chunk on first axis
            dpatch = fdst.create_dataset(
                "patches", shape=(total_patches, patch_size, patch_size), dtype=dtype,
                chunks=(chunk_p, patch_size, patch_size),
                compression=compression, compression_opts=compression_opts
            )
            if store_coords:
                dcoords = fdst.create_dataset(
                    "coords", shape=(total_patches, 3), dtype=np.int32,
                    chunks=(max(1024, K), 3), compression=compression, compression_opts=compression_opts
                )
            # minimal provenance
            fdst.attrs["source_file"] = os.path.basename(split_data_h5)
            fdst.attrs["data_path"] = data_path
            fdst.attrs["patch_size"] = patch_size
            fdst.attrs["image_shape_HW"] = np.array([H, W], dtype=np.int32)
            fdst.attrs["patches_per_image"] = np.int32(K)

            # write in small image batches
            write_pos = 0
            for start in tqdm(range(0, N, images_per_batch), desc=f"Patchifying {os.path.basename(split_data_h5)}"):
                stop = min(N, start + images_per_batch)
                # read a small batch of images
                batch = dsrc[start:stop, ...]  # shape: (B, H, W)
                # emit patches sequentially
                for b, img in enumerate(batch):
                    patches, coords = patchify_image_centered(img, patch_size)
                    # stack into array once (K, p, p)
                    pstack = np.stack(patches, axis=0)
                    dpatch[write_pos:write_pos + K, ...] = pstack
                    if store_coords:
                        # store [img_idx, y, x]
                        ci = np.empty((K, 3), dtype=np.int32)
                        ci[:, 0] = start + b
                        ci[:, 1:] = np.array(coords, dtype=np.int32)
                        dcoords[write_pos:write_pos + K, :] = ci
                    write_pos += K

    return out_h5

# ---------- convenience wrappers for your train/test files ----------

def make_train_test_patches(train_data_h5: str, test_data_h5: str, out_dir: str,
                            patch_size: int = 256, store_coords: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    train_patches_h5 = os.path.join(out_dir, "train_patch.h5")
    test_patches_h5  = os.path.join(out_dir, "test_patch.h5")
    write_patches(train_data_h5, train_patches_h5, patch_size=patch_size, store_coords=store_coords)
    write_patches(test_data_h5,  test_patches_h5,  patch_size=patch_size, store_coords=store_coords)
    print("Created:", train_patches_h5)
    print("Created:", test_patches_h5)
    return train_patches_h5, test_patches_h5


if __name__ == "__main__":
    root = '../data/CBASS/split2/'
    # paths returned from your split step
    train_data = os.path.join(root, 'train_data.h5') 
    test_data  = os.path.join(root, 'test_data.h5')
    out_dir    = root

    # make patches (datasets: "patches" and optional "coords")
    make_train_test_patches(train_data, test_data, out_dir, patch_size=256, store_coords=True)
