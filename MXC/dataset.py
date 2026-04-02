# ===== Patch utilities and dataset for HDF5 patches, HDF5 dataset for model  =====
import math
import h5py
import hdf5plugin
import os
from typing import List, Tuple, Optional, Iterator
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import json
from skimage import morphology, draw


class H5Dataset(Dataset):
    """Load a .h5 dataset. Either Training or testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - dataset_name/
            - train.h5
            - test.h5

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'test')
        size: size of the image
    """

    def __init__(self, root, transform=None, split="train",
                use_softclip=True,            # enable new normalization
                S=490.,  # if S provided, directly use S
                use_spots_region=True,
                config=None
                ):
        self.config = config
        self.root = root
        self.dataset_path = os.path.join(root, split + "_patch.h5")
        self.spotcenter_path = os.path.join(root, split + "_centers.json")
        if not os.path.exists(self.dataset_path):
            raise RuntimeError(f'Invalid directory "{self.dataset_path}"')
        self.use_spots_region = use_spots_region
        if self.use_spots_region:
            self.spot_radius = self.config.spot_radius
            self.spot_fg_radius = self.config.spot_fg_radius
            self.spot_type = self.config.spot_type
            if self.spot_type == 'diamond':
                self.footprint = morphology.diamond(self.spot_fg_radius)
            elif self.spot_type == 'disk':
                self.footprint = morphology.disk(self.spot_fg_radius)
            if not os.path.exists(self.spotcenter_path):
                raise RuntimeError(f'Invalid directory "{self.spotcenter_path}"')
            with open(self.spotcenter_path, 'r') as f:
                self.spot_centers =  json.load(f)
        
        self.split = split
        self.transform = transform
        self.rgb = False

        with h5py.File(self.dataset_path, "r") as f:  
            self.num_img = f['patches'].shape[0] 
            # self.num_img = f['entry']['data']['patches'].shape[0]  # f['entry']['data'] ['img_indices', 'patch_indices', 'patch_positions', 'patches']>
        self.sampleIDs = [i for i in range(self.num_img)]
        
        bg_threshold = 10
        if self.split == 'train':
            self.image_filtering(threshold=bg_threshold)
        
        self.use_softclip = use_softclip
        if self.use_softclip:
            if S is None:
                # --- Soft-clip config ---
                self.calibrate_softclip_roi_only(q_roi=0.995, alpha=0.95, radius=10, bg_max_value=bg_threshold, bg_eps=0.02)
            else:
                self.S = S     # will be set in pre_processing()
        
    
    def plot_histgram(
        self,
        bins: int = 512,
        logy: bool = True,
        save_path: str | None = None,
    ):
        """
        Stream a global histogram of raw pixel values without loading all data into RAM.

        Args:
            bins: number of histogram bins
            logy: plot y-axis in log scale
            show: call plt.show() at the end
            save_path: optional file path to save the figure
        Returns:
            (fig, ax, bin_edges, counts)
        """
        # -------- First pass: find global min/max over selected samples --------
        with h5py.File(self.dataset_path, "r") as f:
            dset = f["patches"]
            ids = self.sampleIDs

            # Initialize with first sample to avoid inf checks
            first = dset[ids[0]].view(np.int16)
            first[first<0] = 0
            gmin = int(first.min())
            gmax = int(first.max())

            for idx in ids[1:]:
                arr = dset[idx].view(np.int16)
                arr[arr<0] = 0
                vmin = int(arr.min())
                vmax = int(arr.max())
                if vmin < gmin: gmin = vmin
                if vmax > gmax: gmax = vmax

        # Handle degenerate case (all pixels identical)
        if gmin == gmax:
            gmin, gmax = gmin - 1, gmax + 1

        bin_edges = np.linspace(gmin, gmax, bins + 1, dtype=np.float64)
        counts = np.zeros(bins, dtype=np.int64)
        total_pixels = 0

        # -------- Second pass: accumulate histogram counts --------
        with h5py.File(self.dataset_path, "r") as f:
            dset = f["entry"]["data"]["patches"]
            for idx in self.sampleIDs:
                arr = dset[idx].view(np.int16)
                flat = arr.ravel()
                h, _ = np.histogram(flat, bins=bin_edges)
                counts += h
                total_pixels += flat.size

        # -------- Plot --------
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.step(bin_edges[:-1], counts, where="post")
        ax.set_xlabel("Pixel value (raw units)")
        ax.set_ylabel("Count")
        ax.set_title(f"Dataset Pixel Histogram (bins={bins}, pixels={total_pixels:,})")
        if logy:
            ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)

        return fig, ax, bin_edges, counts

    def get_spot_mask(self, img, centers, mask_type='binary'):
        if mask_type == 'binary':
            return self.get_spot_mask_binary(img, centers, footprint=self.footprint)
        else:
            return self.get_spot_mask_plateau_gaussian(img, centers, self.config.spot_radius, self.config.spot_gaus_sigma, Rcut=self.config.Rcut)

    def get_spot_mask_plateau_gaussian(self, patch, centers, R0=6.0, sigma=3.0, Rcut=30):
        mask = np.zeros_like(patch, dtype=np.float32)
        if centers is None or len(centers) == 0:
            return mask

        # Build an image with zeros at centers, ones elsewhere → EDT gives distance to nearest center
        img = np.ones_like(patch, dtype=np.uint8)
        c = np.asarray(centers, dtype=np.int32).copy()
        img[c[:, 0], c[:, 1]] = 0

        D = distance_transform_edt(img).astype(np.float32)  # distance to nearest center, shape (H, W)

        # Plateau
        inside = (D <= R0)
        mask[inside] = 1.0

        # Gaussian tail (truncated)
        mid = (D > R0) & (D < Rcut)
        x = D[mid] - R0
        mask[mid] = np.exp(-(x * x) / (2.0 * sigma * sigma))

        return mask
            
    def get_spot_mask_binary(self, patch, centers, footprint):
        """Load centers for a specific patch index from JSON"""     
        mask = np.zeros_like(patch)
        if len(centers):
            c = np.asarray(centers, dtype=np.int32)
            mask[c[:,0], c[:,1]] = 1
        mask = morphology.binary_dilation(mask, footprint).astype(np.uint8)
        return mask

    def _get_filter_cache_path(self) -> str:
        """Return path to cache JSON storing filtered sample IDs by threshold in root dir."""
        # Store per split in the provided root directory
        return os.path.join(self.root, f"{self.split}_filter_cache.json")

    def _load_cached_sample_ids(self, threshold: int) -> Optional[List[int]]:
        """Load cached sample IDs for a given threshold if present."""
        cache_path = self._get_filter_cache_path()
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            ids = data.get('threshold_to_ids', {}).get(str(threshold))
            if ids is None:
                return None
            return [int(x) for x in ids]
        except Exception:
            return None

    def _save_cached_sample_ids(self, threshold: int, ids: List[int]) -> None:
        """Persist filtered sample IDs keyed by threshold into root dir."""
        cache_path = self._get_filter_cache_path()
        try:
            data = {}
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    data = json.load(f)
            threshold_map = data.get('threshold_to_ids', {})
            threshold_map[str(threshold)] = ids  # stringify key for JSON
            data['threshold_to_ids'] = threshold_map
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass

    def image_filtering(self, threshold=15):
        # Try loading from cache first
        cached = self._load_cached_sample_ids(threshold)
        if cached is not None:
            self.sampleIDs = cached if len(cached) > 0 else [0]
            return

        with h5py.File(self.dataset_path, "r") as f:
            new_sampleIDs = []
            for index in self.sampleIDs:
                patch = f['patches'][index]
                patch = patch.view(np.int16)
                if patch.max() <= threshold:
                    continue
                new_sampleIDs.append(index)
            if len(new_sampleIDs) == 0:
                new_sampleIDs = [0]
            self.sampleIDs = new_sampleIDs
        self._save_cached_sample_ids(threshold, self.sampleIDs)
    
    def calibrate_softclip_roi_only(
        self,
        q_roi: float = 0.99,    # ROI percentile for H (e.g., 0.99 or 0.995)
        alpha: float = 0.90,    # target y(H) = alpha
        radius: int = 10,       # same dilation used in get_spot_mask
        # Optional background floor without bg percentile:
        bg_max_value: int | None = 10,  # treat any value <= this as "background of interest"
        bg_eps: float = 0.02,           # want y(bg_max_value) <= bg_eps
        verbose: bool = True,
    ):
        """
        ROI-only soft-clip calibration:
        S_roi = H * (1 - alpha) / alpha,  where H = ROI percentile (q_roi).
        Optional safety floor: S >= S_min = bg_max_value * (1 - bg_eps) / bg_eps
        to guarantee background stays dim up to bg_max_value.
        Sets self.S and self.alpha_eff.
        """

        # -------- Pass 1: global max after clamping negatives to 0 --------
        with h5py.File(self.dataset_path, "r") as f:
            dset = f["patches"]
            first = dset[self.sampleIDs[0]].view(np.int16).astype(np.int32)
            np.maximum(first, 0, out=first)
            gmax = int(first.max())
            for idx in self.sampleIDs[1:]:
                arr = dset[idx].view(np.int16).astype(np.int32)
                np.maximum(arr, 0, out=arr)
                vmax = int(arr.max());  gmax = max(gmax, vmax)
        gmax = max(gmax, 1)

        # -------- Pass 2: ROI histogram (streaming) --------
        roi_counts = np.zeros(gmax + 1, dtype=np.int64)
        n_roi = 0
        with h5py.File(self.dataset_path, "r") as f:
            dset = f["patches"]
            for idx in self.sampleIDs:
                patch = dset[idx].view(np.int16).astype(np.int32)
                np.maximum(patch, 0, out=patch)
                centers = self.spot_centers.get(str(idx), [])
                # mask = self.get_spot_mask(patch, centers, footprint=self.footprint).astype(bool)
                mask = self.get_spot_mask(patch, centers, mask_type='binary').astype(bool)
                if mask.any():
                    vals_roi = patch[mask]
                    roi_counts += np.bincount(vals_roi, minlength=gmax + 1)
                    n_roi += vals_roi.size

        # If no ROI pixels at all, fall back to a conservative H
        if n_roi == 0:
            H = int(0.9 * gmax)
        else:
            tot = int(roi_counts.sum())
            target = int(np.ceil(q_roi * tot))
            cdf = np.cumsum(roi_counts, dtype=np.int64)
            H = int(np.searchsorted(cdf, target, side="left"))

        # ROI-derived knee
        alpha = float(alpha) if alpha > 0 else 0.90
        S_roi = float(H) * (1.0 - alpha) / alpha if H > 0 else 1.0

        # Optional floor to keep background dim up to a fixed value (no bg percentile)
        if bg_max_value is not None and bg_max_value > 0 and bg_eps > 0:
            S_min = float(bg_max_value) * (1.0 - bg_eps) / bg_eps
            S = max(S_roi, S_min)
        else:
            S_min = 0.0
            S = S_roi

        alpha_eff = (float(H) / (float(H) + S)) if H > 0 else 0.0

        self.S = float(S)
        self.alpha_eff = float(alpha_eff)

        if verbose:
            print(
                f"[ROI-only SoftClip] gmax={gmax}  roi_pixels={n_roi:,}  "
                f"H(p{int(q_roi*100)})={H}  S_roi={S_roi:.3f}  "
                f"S_min(bg≤{bg_max_value}@≤{bg_eps:.3f})={S_min:.3f}  ->  S={S:.3f}  "
                f"alpha_eff={alpha_eff:.3f}"
            )
        # return {"S": self.S, "alpha_eff": self.alpha_eff, "H": int(H), "S_roi": float(S_roi), "S_min": float(S_min)}

    # ------------------------ Soft-clip mapping ------------------------
    def softclip_forward(self, x):      # y = x/(x+S)
        return x / (x + self.S)

    def softclip_inverse(self, y):      # x = S*y/(1-y)
        # y: np.ndarray, any dtype -> returns float32 (raw units)
        y = np.asarray(y, dtype=np.float32, copy=False)

        # clamp strictly below 1.0 in float32 space + clean NaN/Inf
        upper = np.nextafter(np.float32(1.0), np.float32(0.0))  # 0.99999994
        y[:] = np.nan_to_num(y, nan=0.0, posinf=upper, neginf=0.0)
        np.clip(y, 0.0, upper, out=y)

        S = np.float32(self.S)
        return S * (y / (np.float32(1.0) - y))  # exact inverse in fp32
        # np.clip(y, 0, 1.0, out=y)  # float32
        # return self.S * (y / (1.0 - y + 1e-7))

    def get_patch(self, index: int):
        """
        Split an image into non-overlapping centered patches.

        Args:
            index: patch index

        Returns:
            patches (list of np.ndarray): list of [patch_size, patch_size] arrays
        """
        with h5py.File(self.dataset_path, "r") as f:
            patch = f['patches'][index]
            coords = f['coords'][index]
            img_indices = coords[0]
            patch_positions = coords[1:]
            
        # Reinterpret as int16 to recover original values
        patch = patch.view(np.int16)
        np.clip(patch, 0, None, out=patch)
        if self.use_softclip:
            patch = patch.astype(np.float32)
            patch = self.softclip_forward(patch)
        else:
            patch[patch > 255] = 255  # artifacts caused by detector?
            patch = patch.astype(np.float32)
            patch = patch / 255.0
        return patch, img_indices, patch_positions

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        patch_idx = self.sampleIDs[index]
        img, img_indices, patch_positions = self.get_patch(patch_idx)  # PIL image patch
        if self.use_spots_region:
            centers = self.spot_centers[str(patch_idx)]
            mask = self.get_spot_mask(img, centers, mask_type=self.config.spot_mask_type)
            mask_fg = self.get_spot_mask_binary(img, centers, footprint=self.footprint)
        else:
            mask = np.zeros_like(img)
            mask_fg = np.zeros_like(img)
        if self.rgb:
            img = torch.from_numpy(img).unsqueeze(0).repeat(3,1,1)  # (3,H,W)
            mask = torch.from_numpy(mask).unsqueeze(0).repeat(3,1,1)  # (3,H,W)
        else:
            img = torch.from_numpy(img).unsqueeze(0)                # (1,H,W)
            mask = torch.from_numpy(mask).unsqueeze(0)                # (1,H,W)
            mask_fg = torch.from_numpy(mask_fg).unsqueeze(0)  
        meta = (patch_idx, img_indices, patch_positions)
        if self.transform:
            return self.transform(img), mask, mask_fg #, mask_small, mask_large
        return img, mask, mask_fg #, mask_small, mask_large

    def __len__(self):
        return len(self.sampleIDs)


def patchify_image_centered(image: np.ndarray, patch_size: int = 256) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Split a 2D image into non-overlapping centered patches.

    Args:
        image: 2D grayscale array [H, W]
        patch_size: size of square patch

    Returns:
        patches: list of (patch_size, patch_size) arrays
        coords:  list of (y, x) top-left coords per patch
    """
    H, W = image.shape
    rem_h, rem_w = H % patch_size, W % patch_size
    off_h, off_w = rem_h // 2, rem_w // 2

    usable_h = H - rem_h
    usable_w = W - rem_w

    patches: List[np.ndarray] = []
    coords: List[Tuple[int, int]] = []
    for y in range(off_h, off_h + usable_h, patch_size):
        for x in range(off_w, off_w + usable_w, patch_size):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            coords.append((y, x))
    return patches, coords


def _copy_metadata_except_data(src_entry: h5py.Group, dst_entry: h5py.Group) -> None:
    """Copy all groups/datasets and attributes from entry except the 'data' group."""
    # Copy attributes on entry
    for attr_name, attr_val in src_entry.attrs.items():
        dst_entry.attrs[attr_name] = attr_val
    # Copy groups/datasets except 'data'
    for key in src_entry.keys():
        if key == 'data':
            continue
        obj = src_entry[key]
        if isinstance(obj, h5py.Group):
            g = dst_entry.create_group(key)
            # Copy attrs
            for an, av in obj.attrs.items():
                g.attrs[an] = av
            # Shallow copy children (metadata typically small)
            for k2 in obj.keys():
                child = obj[k2]
                if isinstance(child, h5py.Group):
                    g.copy(child, k2)
                else:
                    ds = g.create_dataset(k2, data=child[()], dtype=child.dtype)
                    for an2, av2 in child.attrs.items():
                        ds.attrs[an2] = av2
        else:
            ds = dst_entry.create_dataset(key, data=obj[()], dtype=obj.dtype)
            for an, av in obj.attrs.items():
                ds.attrs[an] = av


def extract_patches_from_split(split_h5: str, split: str, patch_size: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""
	Extract patches from a split H5 file and return as arrays.
	
	Returns:
		patches: array of shape (num_patches, patch_size, patch_size) with dtype from source
		img_indices: array of shape (num_patches,) with source image index for each patch
		patch_indices: array of shape (num_patches,) with patch index within each image
		patch_positions: array of shape (num_patches, 2) with (y, x) top-left coordinates
	"""
	with h5py.File(split_h5, 'r') as src:
		dset_name = f'{split}_data'
		src_dset = src['entry']['data'][dset_name]
		num_images = src_dset.shape[0]
		dtype = src_dset.dtype

		print(f"[{split}] Extracting patches from {num_images} images")
		
		# First pass: count total patches
		total_patches = 0
		patches_per_image = []
		for img_idx in range(num_images):
			image = src_dset[img_idx]
			patches, _ = patchify_image_centered(image, patch_size)
			num_patches = len(patches)
			patches_per_image.append(num_patches)
			total_patches += num_patches
		
		print(f"[{split}] Total patches: {total_patches}")
		
		# Initialize arrays
		patches_array = np.zeros((total_patches, patch_size, patch_size), dtype=dtype)
		img_indices = np.zeros(total_patches, dtype=np.int32)
		patch_indices = np.zeros(total_patches, dtype=np.int32)
		patch_positions = np.zeros((total_patches, 2), dtype=np.int32)
		
		# Second pass: fill arrays
		patch_idx = 0
		for img_idx in tqdm(range(num_images), desc=f"{split}: extracting patches"):
			image = src_dset[img_idx]
			patches, coords = patchify_image_centered(image, patch_size)
			
			for local_patch_idx, (patch, (y, x)) in enumerate(zip(patches, coords)):
				patches_array[patch_idx] = patch
				img_indices[patch_idx] = img_idx
				patch_indices[patch_idx] = local_patch_idx
				patch_positions[patch_idx] = [y, x]
				patch_idx += 1
		
		return patches_array, img_indices, patch_indices, patch_positions


def create_patch_h5_from_split(split_h5: str, split: str, out_path: str, patch_size: int = 256,
								compression: Optional[str] = None, compression_level: int = 6) -> str:
	"""
	Create a patch file from a split H5 file, saving patches as arrays under entry/data.
	Memory-efficient: processes images one by one and writes patches chunk by chunk.
	Structure: entry/data contains 'patches', 'img_indices', 'patch_indices', 'patch_positions'
	"""
	with h5py.File(split_h5, 'r') as src, h5py.File(out_path, 'w') as dst:
		# Copy root 'entry' metadata except data
		entry_src = src['entry']
		entry_dst = dst.create_group('entry')
		_copy_metadata_except_data(entry_src, entry_dst)

		# Create data group for patches
		data_dst = entry_dst.create_group('data')

		dset_name = f'{split}_data'
		src_dset = src['entry']['data'][dset_name]
		num_images = src_dset.shape[0]
		dtype = src_dset.dtype

		# First pass: count total patches
		print(f"[{split}] Counting patches from {num_images} images...")
		total_patches = 0
		patches_per_image = []
		for img_idx in range(num_images):
			image = src_dset[img_idx]
			patches, _ = patchify_image_centered(image, patch_size)
			num_patches = len(patches)
			patches_per_image.append(num_patches)
			total_patches += num_patches
		
		print(f"[{split}] Total patches: {total_patches}")

		# Build optional compression kwargs
		kwargs = {}
		if compression is not None:
			kwargs['compression'] = compression
			if compression == 'gzip':
				kwargs['compression_opts'] = compression_level

		# Create datasets with proper shapes
		patches_ds = data_dst.create_dataset(
			'patches', 
			shape=(total_patches, patch_size, patch_size), 
			dtype=dtype,
			**kwargs
		)
		img_indices_ds = data_dst.create_dataset(
			'img_indices', 
			shape=(total_patches,), 
			dtype=np.int32,
			**kwargs
		)
		patch_indices_ds = data_dst.create_dataset(
			'patch_indices', 
			shape=(total_patches,), 
			dtype=np.int32,
			**kwargs
		)
		patch_positions_ds = data_dst.create_dataset(
			'patch_positions', 
			shape=(total_patches, 2), 
			dtype=np.int32,
			**kwargs
		)

		# Second pass: write patches chunk by chunk
		patch_idx = 0
		for img_idx in tqdm(range(num_images), desc=f"{split}: writing patches"):
			image = src_dset[img_idx]
			patches, coords = patchify_image_centered(image, patch_size)
			
			# Write patches for this image
			num_patches_this_image = len(patches)
			if num_patches_this_image > 0:
				# Write patches array
				patches_ds[patch_idx:patch_idx + num_patches_this_image] = patches
				
				# Write indices and positions
				img_indices_ds[patch_idx:patch_idx + num_patches_this_image] = img_idx
				patch_indices_ds[patch_idx:patch_idx + num_patches_this_image] = np.arange(num_patches_this_image)
				patch_positions_ds[patch_idx:patch_idx + num_patches_this_image] = coords
				
				patch_idx += num_patches_this_image
		
		# Add metadata attributes
		data_dst.attrs['num_patches'] = total_patches
		data_dst.attrs['patch_size'] = patch_size
		data_dst.attrs['source_split'] = split
		data_dst.attrs['source_file'] = split_h5
		
	return out_path


def extract_train_test_patches(train_h5: str, test_h5: str, patch_size: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""
	Extract patches from both train and test H5 files and return as arrays.
	
	Returns:
		train_patches, train_img_indices, train_patch_indices, train_patch_positions,
		test_patches, test_img_indices, test_patch_indices, test_patch_positions
	"""
	train_patches, train_img_indices, train_patch_indices, train_patch_positions = extract_patches_from_split(train_h5, 'train', patch_size)
	test_patches, test_img_indices, test_patch_indices, test_patch_positions = extract_patches_from_split(test_h5, 'test', patch_size)
	
	return (train_patches, train_img_indices, train_patch_indices, train_patch_positions,
			test_patches, test_img_indices, test_patch_indices, test_patch_positions)


def load_patch_data(patch_h5_path: str) -> dict:
	"""
	Load patch data from HDF5 file created by create_patch_h5_from_split.
	
	Returns:
		dict with keys: 'patches', 'img_indices', 'patch_indices', 'patch_positions'
	"""
	with h5py.File(patch_h5_path, 'r') as f:
		data = f['entry']['data']
		return {
			'patches': data['patches'][:],
			'img_indices': data['img_indices'][:],
			'patch_indices': data['patch_indices'][:],
			'patch_positions': data['patch_positions'][:]
		}


def create_train_test_patch_h5(train_h5: str, test_h5: str, out_dir: str,
								  patch_size: int = 256,
								  compression: Optional[str] = None, compression_level: int = 6) -> Tuple[str, str]:
	"""Create train_patch.h5 and test_patch.h5 from split H5 files."""
	os.makedirs(out_dir, exist_ok=True)
	train_out = os.path.join(out_dir, 'train_patch.h5')
	test_out = os.path.join(out_dir, 'test_patch.h5')
	create_patch_h5_from_split(train_h5, 'train', train_out, patch_size, compression, compression_level)
	create_patch_h5_from_split(test_h5, 'test', test_out, patch_size, compression, compression_level)
	return train_out, test_out



if __name__ == '__main__':
    # train_out, test_out = create_train_test_patch_h5(
    #     train_h5="../data/CBASS/train.h5",
    #     test_h5="../data/CBASS/test.h5",
    #     out_dir="../data/CBASS",
    #     patch_size=256,
    #     compression=None
    # )

    h5dset = H5Dataset(root="../data/CBASS/", split='train')
    print(h5dset.__getitem__(0).shape)