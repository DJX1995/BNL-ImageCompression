import os, json, math, random
import h5py
import numpy as np
from typing import List, Tuple, Dict, Optional
import hdf5plugin  # put this near the top of the file


# --- put near your imports ---
def _b(s: str) -> bytes:
    # Compatible with NumPy 1.x and 2.x; returns raw bytes for HDF5 attrs
    return s.encode("utf-8")
    

# ============================== Utilities ==============================

def _discover_blocks(master_path: str, data_group_path: str = "/entry/data"):
    """Return ordered list of data_* blocks and their specs (n, H, W, dtype, external link)."""
    out = []
    with h5py.File(master_path, "r") as f:
        g = f[data_group_path]
        keys = sorted(g.keys())
        for k in keys:
            lnk = g.get(k, getlink=True)
            if isinstance(lnk, h5py.ExternalLink):
                ext_file, ext_path = lnk.filename, lnk.path
                with h5py.File(os.path.join(os.path.dirname(master_path), ext_file), "r") as fe:
                    d = fe[ext_path]
                    n, H, W = d.shape
                    out.append(dict(key=k, n=n, H=H, W=W, dtype=str(d.dtype),
                                    external_file=ext_file, external_path=ext_path))
            else:
                d = g[k]
                n, H, W = d.shape
                out.append(dict(key=k, n=n, H=H, W=W, dtype=str(d.dtype),
                                external_file=None, external_path=None))
    return out

def _global_index_map(blocks):
    """Map (key, local_idx) -> global_idx; also return total frame count."""
    offs = {}
    acc = 0
    for b in blocks:
        offs[b["key"]] = acc
        acc += b["n"]
    return offs, acc

def _load_indexing_json(path: str) -> List[Tuple[str, int]]:
    with open(path, "r") as f:
        raw = json.load(f)
    return [(str(k), int(i)) for (k, i) in raw]

def _save_indexing_json(path: str, indices: List[Tuple[str, int]]):
    with open(path, "w") as f:
        json.dump(indices, f)

def _ensure_nexus_classes(fout: h5py.File):
    entry = fout.require_group("/entry")
    entry.attrs["NX_class"] = _b("NXentry")

    inst  = entry.require_group("instrument")
    inst.attrs["NX_class"] = _b("NXinstrument")

    samp  = entry.require_group("sample")
    samp.attrs["NX_class"] = _b("NXsample")

    data  = entry.require_group("data")
    data.attrs["NX_class"] = _b("NXdata")

    entry.attrs["default"] = _b("data")
    data.attrs["signal"]   = _b("data_000001")


def _copy_group_structure_and_scalars(src: h5py.Group, dst: h5py.Group, total_frames: int):
    """
    Copy groups recursively and copy datasets that are *not* per-image arrays
    (i.e., whose first dimension != total_frames). This preserves scalars and small arrays.
    """
    for name, obj in src.items():
        if isinstance(obj, h5py.Group):
            gdst = dst.create_group(name)
            for an, av in obj.attrs.items():
                gdst.attrs[an] = av
            _copy_group_structure_and_scalars(obj, gdst, total_frames)
        elif isinstance(obj, h5py.Dataset):
            shp = obj.shape
            is_per_image_like = (len(shp) >= 1 and shp[0] == total_frames)
            if not is_per_image_like:
                d = dst.create_dataset(name, data=obj[...], dtype=obj.dtype)
                for an, av in obj.attrs.items():
                    d.attrs[an] = av

def _scan_per_image_arrays(fin: h5py.File, total_frames: int,
                           roots=("/entry/sample", "/entry/instrument")):
    """
    Find datasets whose first dimension matches total_frames (per-image arrays).
    """
    cands = []
    for root in roots:
        if root not in fin:
            continue
        def visit(g: h5py.Group, base: str):
            for n, obj in g.items():
                p = f"{base}/{n}"
                if isinstance(obj, h5py.Group):
                    visit(obj, p)
                elif isinstance(obj, h5py.Dataset):
                    shp = obj.shape
                    if shp and len(shp) >= 1 and shp[0] == total_frames:
                        cands.append(p)
        visit(fin[root], root)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in cands:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


def _slice_and_write_per_image_arrays(fin: h5py.File, fout: h5py.File,
                                      indices: List[Tuple[str,int]],
                                      blocks: List[dict],
                                      per_image_paths: List[str],
                                      batch: int = 4096):
    """
    Slice per-image arrays (whose first dim == total frames) to match `indices`,
    streaming in chunks to avoid OOM. Two passes:
      (1) create destination datasets (shape = (len(indices), ...) + attrs)
      (2) fill values using sorted reads + scatter-back to original order
    """
    import numpy as np  # ensure np is available here
    # Map (block key, local idx) -> global idx
    offs, total = _global_index_map(blocks)

    # ---------- Pass 1: create destination datasets ----------
    for src_path in per_image_paths:
        if src_path not in fin:
            continue
        dsrc = fin[src_path]
        shp = dsrc.shape
        if not shp or shp[0] != total:
            # skip non-global or scalar datasets
            continue
        new_shp = (len(indices),) + shp[1:]
        grp = fout.require_group(os.path.dirname(src_path))
        if os.path.basename(src_path) in grp:
            del grp[os.path.basename(src_path)]  # overwrite if re-running
        dstd = grp.create_dataset(os.path.basename(src_path),
                                  shape=new_shp, dtype=dsrc.dtype, chunks=True)
        for an, av in dsrc.attrs.items():
            dstd.attrs[an] = av

    # ---------- Pass 2: fill in chunks with sorted reads ----------
    for src_path in per_image_paths:
        if src_path not in fin or src_path not in fout:
            continue
        dsrc = fin[src_path]
        dstd = fout[src_path]
        shp = dsrc.shape
        if not shp or shp[0] != total:
            continue

        pos = 0
        while pos < len(indices):
            end = min(len(indices), pos + batch)

            # Build global indices for this chunk
            glo = np.fromiter((offs[k] + i for (k, i) in indices[pos:end]),
                              dtype=np.int64, count=end - pos)

            # h5py fancy indexing requires increasing order for performance
            order = np.argsort(glo)
            glo_sorted = glo[order]

            # Read in sorted order, then scatter back to original
            vals_sorted = dsrc[glo_sorted, ...]
            vals = np.empty((end - pos,) + dsrc.shape[1:], dtype=dsrc.dtype)
            vals[order] = vals_sorted

            dstd[pos:end, ...] = vals
            pos = end


def create_split_master_and_data(
    master_in: str,
    output_dir: str,
    split_name: str,                  # "train" or "test"
    train_ratio: float = 0.8,
    random_state: int = 42,
    indexing_json: Optional[str] = None,  # if provided, use it (keeps split unchanged)
    data_group_path: str = "/entry/data",
    batch_frames: int = 32,
    compress: Optional[str] = "gzip",
    compress_opts: int = 4,
    shuffle: bool = True,
    split_mode: str = "per_block",    # NEW: "per_block" or "global_inorder"
):
    """
    Plan 2: build split_data.h5 (contiguous stack) + split_master.h5 (Eiger/NeXus),
    copying globals and slicing per-image arrays to the split order.

    split_mode:
      - "per_block": split each data_00000X independently (optionally shuffled)
      - "global_inorder": ignore shuffle; take the first floor(N*train_ratio) frames
         across all blocks in file order for 'train', remainder for 'test'.
    """
    os.makedirs(output_dir, exist_ok=True)
    split_master = os.path.join(output_dir, f"{split_name}_master.h5")
    split_data   = os.path.join(output_dir, f"{split_name}_data.h5")
    split_index  = os.path.join(output_dir, f"{split_name}_indexing.json") if not indexing_json else indexing_json

    # 1) Discover blocks and specs
    blocks = _discover_blocks(master_in, data_group_path=data_group_path)
    if not blocks:
        raise RuntimeError("No /entry/data/data_00000X datasets found.")
    H = blocks[0]["H"]; W = blocks[0]["W"]; dtype = np.dtype(blocks[0]["dtype"])
    for b in blocks:
        if (b["H"], b["W"]) != (H, W):
            raise RuntimeError("Inconsistent frame size across blocks.")
        if np.dtype(b["dtype"]) != dtype:
            raise RuntimeError("Inconsistent dtype across blocks.")

    offs, total_frames = _global_index_map(blocks)

    # 2) Build or load split indices
    if indexing_json and os.path.isfile(indexing_json):
        indices = _load_indexing_json(indexing_json)
    else:
        random.seed(random_state)

        if split_mode == "global_inorder":
            # ---- GLOBAL IN-ORDER CUT (no shuffle) ----
            n_train = int(math.floor(total_frames * train_ratio))
            taken = 0
            train_idx, test_idx = [], []
            for b in blocks:  # keep block order as discovered
                k, n = b["key"], b["n"]
                if taken >= n_train:
                    # everything from this block goes to test
                    test_idx.extend((k, i) for i in range(n))
                else:
                    need = n_train - taken
                    use = min(need, n)
                    # head of this block -> train
                    train_idx.extend((k, i) for i in range(use))
                    # tail (if any) -> test
                    if use < n:
                        test_idx.extend((k, i) for i in range(use, n))
                    taken += use
        else:
            # ---- PER-BLOCK SPLIT (optionally shuffled) ----
            train_idx, test_idx = [], []
            for b in blocks:
                idxs = list(range(b["n"]))
                if shuffle:
                    random.shuffle(idxs)
                cut = int(math.floor(b["n"] * train_ratio))
                train_idx += [(b["key"], i) for i in idxs[:cut]]
                test_idx  += [(b["key"], i) for i in idxs[cut:]]

        indices = train_idx if split_name == "train" else test_idx
        _save_indexing_json(split_index, indices)

    N = len(indices)
    if N == 0:
        raise RuntimeError(f"No frames selected for split '{split_name}'")

    # 3) Stream frames into split_data.h5:/entry/data/data
    with h5py.File(split_data, "w") as fd, h5py.File(master_in, "r") as fin:
        g = fd.require_group("entry"); data_grp = g.require_group("data")
        create_kwargs = dict(shape=(N, H, W), dtype=dtype, chunks=(1, H, W))
        if compress is None:
            dset = data_grp.create_dataset("data", **create_kwargs)
        else:
            dset = data_grp.create_dataset("data", **create_kwargs,
                                           compression=compress, compression_opts=compress_opts)

        # group indices by block for efficient access
        by_block: Dict[str, List[Tuple[int,int]]] = {}
        for j, (k, i) in enumerate(indices):
            by_block.setdefault(k, []).append((j, i))

        src_data = fin[data_group_path]
        for b in blocks:
            k = b["key"]
            pairs = by_block.get(k, [])
            if not pairs:
                continue

            lnk = src_data.get(k, getlink=True)
            if isinstance(lnk, h5py.ExternalLink):
                src_file = os.path.join(os.path.dirname(master_in), lnk.filename)
                src_path = lnk.path
                fsrc = h5py.File(src_file, "r"); dsrc = fsrc[src_path]; close_src = True
            else:
                dsrc = src_data[k]; close_src = False

            # stream in small batches
            for off in range(0, len(pairs), batch_frames):
                chunk = pairs[off:off+batch_frames]
                for j, i in chunk:
                    dset[j, ...] = dsrc[i, ...]
            if close_src: fsrc.close()

    # 4) Build split_master.h5: copy globals + slice per-image arrays + link data_000001
    with h5py.File(master_in, "r") as fin, h5py.File(split_master, "w") as fout:
        # Copy root attrs
        for an, av in fin.attrs.items():
            fout.attrs[an] = av

        # Copy /entry (structure & scalars), excluding per-image arrays
        entry_in  = fin["/entry"]
        entry_out = fout.create_group("entry")
        for an, av in entry_in.attrs.items():
            entry_out.attrs[an] = av

        for name in ["instrument", "sample"]:
            if name in entry_in:
                g_in  = entry_in[name]
                g_out = entry_out.create_group(name)
                for an, av in g_in.attrs.items():
                    g_out.attrs[an] = av
                _copy_group_structure_and_scalars(g_in, g_out, total_frames)

        # Create /entry/data and link to the new contiguous stack
        data_out = entry_out.create_group("data")
        _ensure_nexus_classes(fout)  # sets NX_class + entry.default + data.signal
        data_out["data_000001"] = h5py.ExternalLink(os.path.basename(split_data), "/entry/data/data")

        # Slice per-image arrays (global length == total_frames)
        per_img_paths = _scan_per_image_arrays(fin, total_frames, roots=("/entry/sample", "/entry/instrument"))
        _slice_and_write_per_image_arrays(fin, fout, indices, blocks, per_img_paths)

    return split_master, split_data, split_index


def flip_master_external_link(master_in: str, master_out: str, new_data_file: str,
                              target_link_name: str = "/entry/data/data_000001",
                              new_internal_path: str = "/entry/data/data"):
    with h5py.File(master_in, "r") as fin, h5py.File(master_out, "w") as fout:
        # copy root attrs
        for an, av in fin.attrs.items():
            fout.attrs[an] = av
        # copy all top-level objects
        for name in fin:
            fin.copy(name, fout, name=name)
        # overwrite the ExternalLink
        parent_path = os.path.dirname(target_link_name)
        link_name   = os.path.basename(target_link_name)
        parent = fout[parent_path]
        if link_name in parent:
            del parent[link_name]
        parent[link_name] = h5py.ExternalLink(os.path.basename(new_data_file), new_internal_path)



def validate_split(master_path: str, data_file: str):
    """
    Basic checks: ExternalLink resolves, shape/dtype match, per-image arrays match N.
    """
    with h5py.File(master_path, "r") as f:
        assert "/entry/data" in f, "No /entry/data in master."
        dgrp = f["/entry/data"]
        assert "data_000001" in dgrp, "No data_000001 link in master."
        lnk = dgrp.get("data_000001", getlink=True)
        assert isinstance(lnk, h5py.ExternalLink), "data_000001 is not an ExternalLink."
        assert os.path.basename(lnk.filename) == os.path.basename(data_file), "Link points to a different file."

    with h5py.File(data_file, "r") as fd:
        dd = fd["/entry/data/data"]
        N, H, W = dd.shape
        dtype = dd.dtype

    with h5py.File(master_path, "r") as f:
        # quick scan per-image arrays
        def count_bad():
            bad = []
            for root in ["/entry/sample", "/entry/instrument"]:
                if root not in f: continue
                def visit(g, base):
                    for n, obj in g.items():
                        p = f"{base}/{n}"
                        if isinstance(obj, h5py.Group):
                            visit(obj, p)
                        elif isinstance(obj, h5py.Dataset):
                            shp = obj.shape
                            if shp and shp[0] == N:
                                continue
                            # ignore non per-image arrays
                visit(f[root], root)
            return bad
        _ = count_bad()
    print(f"Validated: {master_path} links to {data_file}; shape={(N,H,W)} dtype={dtype}.")


def create_train_and_test(master_in, out_dir, train_ratio=0.8, random_state=42, batch_frames=50, shuffle=True, split_mode='per_block'):
    """
    Convenience wrapper to create both train and test splits in one call.
    Returns (train_master, train_data, test_master, test_data).
    """
    os.makedirs(out_dir, exist_ok=True)

    # First call creates train indices (saves JSON)
    train_master, train_data, train_index_json = create_split_master_and_data(
        master_in, out_dir,
        split_name="train",
        train_ratio=train_ratio,
        random_state=random_state,
        indexing_json=None,
        data_group_path="/entry/data",
        batch_frames=batch_frames,
        shuffle=shuffle,
        split_mode=split_mode
    )

    # Build complement indices for test
    import json, h5py
    all_pairs = []
    with h5py.File(master_in, "r") as f:
        g = f["/entry/data"]
        for k in sorted(g.keys()):
            link = g.get(k, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                ext_file, ext_path = link.filename, link.path
                with h5py.File(os.path.join(os.path.dirname(master_in), ext_file), "r") as fe:
                    n = fe[ext_path].shape[0]
            else:
                n = g[k].shape[0]
            all_pairs.extend([(k, i) for i in range(n)])
    used = set(tuple(x) for x in json.load(open(train_index_json)))
    test_pairs = [p for p in all_pairs if p not in used]
    test_index_json = os.path.join(out_dir, "test_indexing.json")
    with open(test_index_json, "w") as f: json.dump(test_pairs, f)

    # Now create test split
    test_master, test_data, _ = create_split_master_and_data(
        master_in, out_dir,
        split_name="test",
        indexing_json=test_index_json,
        data_group_path="/entry/data",
        batch_frames=batch_frames,
        shuffle=shuffle,
        split_mode=split_mode
    )

    return train_master, train_data, test_master, test_data


def convert_dataset(master_in, out_dir, random_state=42, batch_frames=50):
    """
    Convenience wrapper to create both train and test splits in one call.
    Returns (train_master, train_data, test_master, test_data).
    """
    os.makedirs(out_dir, exist_ok=True)

    # First call creates train indices (saves JSON)
    train_master, train_data, train_index_json = create_split_master_and_data(
        master_in, out_dir,
        split_name="train",
        train_ratio=1.0,
        random_state=random_state,
        indexing_json=None,
        data_group_path="/entry/data",
        batch_frames=batch_frames,
        shuffle=False
    )
    return train_master, train_data


if __name__ == "__main__":
    master_in = "../../BNL_data_compression/data/CBASS_Cap5_Endo6_23AA_2v_502/Endo6_23AA_2v_502_master.h5"
    out_dir   = "../data/CBASS/split2"
    # try not shuffle and use full dataset
    # convert_dataset(master_in, out_dir, random_state=42, batch_frames=50)
    train_master, train_data, test_master, test_data = create_train_and_test(
        master_in, out_dir, train_ratio=0.8, random_state=42, batch_frames=50, shuffle=False, split_mode='global_inorder'
    )

    validate_split(train_master, train_data)
    validate_split(test_master, test_data)
    