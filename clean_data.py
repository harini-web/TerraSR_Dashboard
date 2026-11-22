#!/usr/bin/env python3
"""
Stage 2 - Clean the raw pairs and create leakage-safe train / val / test splits.

Cleaning (Kramer et al., 2024):
  1. sanity checks   : finite values, no flat/empty or saturated (snow/cloud-like) LR patches
  2. spectral match  : per-channel histogram matching of the HR aerial patch to the S2 patch
  3. quality filter  : drop pairs whose bicubic-upsampled LR vs HR has SSIM < 0.45 or PSNR < 21 dB

Splitting (leakage-safe):
  * tile size is read from the data (--lr-size only overrides), nothing is hard-coded
  * tiles are grouped by GROUND position (UTM coordinates from manifest.csv), not by scene, so the same
    ground seen on a different date / in an overlapping scene lands in the same group
  * groups are split inside every region -> train / val / test (default 72/18/10 %), each split guaranteed non-empty
  * after splitting, any train/val tile whose footprint overlaps a test (or, for train, a val) tile is PURGED
    (this removes boundary leakage caused by overlapping tiles; see --gap-m for an extra safety margin)
  * 'test'        = held-out spatial blocks inside regions the model has seen
  * 'test_unseen' = (optional) whole regions never seen in training  ->  --holdout-regions a b | --holdout-auto N
    This is the split that measures generalisation to new geography.

Output: <out>/*.npz, splits.json, clean_report.csv, clean_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.exposure import match_histograms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/clean")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--min-ssim", type=float, default=0.45)
    p.add_argument("--min-psnr", type=float, default=21.0)
    p.add_argument("--no-hist-match", action="store_true")
    p.add_argument("--max-saturated", type=float, default=0.2, help="drop if >x of LR pixels are >=0.98 in all bands")
    p.add_argument("--min-std", type=float, default=0.01, help="drop flat LR patches")
    p.add_argument("--lr-size", type=int, default=None, help="LR tile size in px (default: read from the data)")
    p.add_argument("--pixel-m", type=float, default=10.0, help="LR ground sampling distance in metres")
    p.add_argument("--block", type=int, default=4, help="split-group side length, in tiles (block = block*tile*pixel-m metres)")
    p.add_argument("--gap-m", type=float, default=0.0, help="extra distance (m) required between tiles of different splits")
    p.add_argument("--split", type=float, nargs=3, default=[0.72, 0.18, 0.10], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--holdout-regions", nargs="+", default=[], help="regions reserved entirely for 'test_unseen'")
    p.add_argument("--holdout-auto", type=int, default=0, help="randomly reserve N regions for 'test_unseen'")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def bicubic_up(lr: np.ndarray, scale: int) -> np.ndarray:
    t = torch.from_numpy(lr.astype(np.float32))[None]
    return F.interpolate(t, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0, 1)[0].numpy()


def hist_match(hr: np.ndarray, lr: np.ndarray) -> np.ndarray:
    return np.stack([match_histograms(hr[i], lr[i]) for i in range(hr.shape[0])]).astype(np.float32)


def region_of(fname: str) -> str:
    return fname.split("__")[0] if "__" in fname else "unknown"


# ----------------------------------------------------------------------------- per-file worker
CFG = None


def _init(cfg):
    global CFG
    CFG = cfg
    torch.set_num_threads(1)


def clean_one(f):
    a = CFG
    try:
        d = np.load(Path(a["raw"]) / f)
        lr, hr = d["lr"].astype(np.float32), d["hr"].astype(np.float32)
    except Exception:
        return f, "unreadable", None, None
    if not (np.isfinite(lr).all() and np.isfinite(hr).all()):
        return f, "non_finite", None, None
    if lr.std() < a["min_std"] or hr.std() < a["min_std"]:
        return f, "flat", None, None
    if (lr >= 0.98).all(0).mean() > a["max_saturated"]:
        return f, "saturated", None, None
    if not a["no_hist_match"]:
        hr = np.clip(hist_match(hr, lr), 0, 1)
    up = bicubic_up(lr, a["scale"])
    psnr = float(peak_signal_noise_ratio(hr, up, data_range=1.0))
    ssim = float(structural_similarity(hr, up, data_range=1.0, channel_axis=0))
    if ssim < a["min_ssim"] or psnr < a["min_psnr"]:
        return f, "low_quality", psnr, ssim
    np.savez(Path(a["out"]) / f, lr=lr.astype(np.float16), hr=hr.astype(np.float16))
    return f, "kept", psnr, ssim


# ----------------------------------------------------------------------------- ground coordinates
def load_coords(raw: Path, files, pixel_m):
    """file -> (spatial_key, x_m, y_m). Uses manifest.csv (UTM coords of the tile corner) when available."""
    table = {}
    man = raw / "manifest.csv"
    if man.exists():
        with open(man, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    table[row["file"]] = (row["region"], float(row["x_left"]), float(row["y_top"]))
                except (KeyError, ValueError):
                    pass
    coords, missing = {}, 0
    for f in files:
        if f in table:
            coords[f] = table[f]
        else:                                           # fallback: scene-local pixel grid (cannot link different scenes)
            missing += 1
            scene, r, c = f[:-4].rsplit("_", 2)
            coords[f] = (f"{region_of(f)}|{scene}", int(c) * pixel_m, -int(r) * pixel_m)
    if missing:
        print(f"[warn] {missing:,} files have no coordinates in manifest.csv -> grouped per scene only; leakage between "
              f"different scenes covering the same ground cannot be excluded for them.")
    return coords


# ----------------------------------------------------------------------------- splitting
def split_region(files, coords, args, rng, block_m):
    groups = defaultdict(list)
    for f in files:
        _, x, y = coords[f]
        groups[(math.floor(x / block_m), math.floor(y / block_m))].append(f)
    keys = sorted(groups)
    rng.shuffle(keys)
    out = {"train": [], "val": [], "test": []}

    if len(keys) < 3:                                   # too few spatial groups: tile-level split (purged later)
        items = sorted(files); rng.shuffle(items); n = len(items)
        if n < 3:
            out["train"] = items
            return out
        n_test = min(max(1, round(n * args.split[2])), n - 2)
        n_val = min(max(1, round(n * args.split[1])), n - n_test - 1)
        out["test"], out["val"], out["train"] = items[:n_test], items[n_test:n_test + n_val], items[n_test + n_val:]
        return out

    total = len(files)
    t1, t2 = args.split[0] * total, (args.split[0] + args.split[1]) * total
    g, count = {"train": [], "val": [], "test": []}, 0
    for k in keys:
        name = "train" if count < t1 else "val" if count < t2 else "test"
        g[name].append(k); count += len(groups[k])
    for name in ("test", "val"):                        # guarantee every split owns >= 1 group
        if not g[name]:
            donor = max(g, key=lambda s: len(g[s]))
            g[name].append(g[donor].pop())
    for name, ks in g.items():
        for k in ks:
            out[name] += groups[k]
    return out


def make_splits(files, coords, args, block_m):
    rng = random.Random(args.seed)
    by_region = defaultdict(list)
    for f in files:
        by_region[region_of(f)].append(f)
    out = {"train": [], "val": [], "test": []}
    for reg in sorted(by_region):
        for k, v in split_region(by_region[reg], coords, args, rng, block_m).items():
            out[k] += v
    return out


def purge_overlaps(splits, coords, tile_m, gap_m):
    """Remove val/train tiles whose footprint (+gap) overlaps a tile of a higher-priority split (test > val > train)."""
    thr = tile_m + gap_m
    index, removed = defaultdict(list), Counter()
    cell = lambda v: math.floor(v / thr)
    for name in ("test", "val", "train"):
        keep = []
        for f in splits[name]:
            key, x, y = coords[f]
            cx, cy = cell(x), cell(y)
            hit = any(abs(x - x2) < thr and abs(y - y2) < thr
                      for dx in (-1, 0, 1) for dy in (-1, 0, 1) for (x2, y2) in index.get((key, cx + dx, cy + dy), ()))
            if hit:
                removed[name] += 1
            else:
                keep.append(f)
        for f in keep:                                  # index only after the split is done: same-split overlap is fine
            key, x, y = coords[f]
            index[(key, cell(x), cell(y))].append((x, y))
        splits[name] = keep
    return removed


def main():
    args = get_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in raw.glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz files in {raw}. Run download_data.py first.")
    raw_by_region = Counter(region_of(f) for f in files)
    print(f"RAW PAIRS FOUND: {len(files):,}  ({len(raw_by_region)} regions)")

    tile = args.lr_size or int(np.load(raw / files[0])["lr"].shape[-1])
    tile_m, block_m = tile * args.pixel_m, args.block * tile * args.pixel_m
    print(f"tile size {tile}px = {tile_m:.0f} m, split-group size {block_m:.0f} m")

    cfg = {k: getattr(args, k) for k in ["scale", "min_ssim", "min_psnr", "min_std", "max_saturated"]}
    cfg.update(raw=str(raw), out=str(out), no_hist_match=args.no_hist_match)
    kept, reasons = [], Counter()
    with open(out / "clean_report.csv", "w", newline="") as fh, mp.Pool(args.workers, _init, (cfg,)) as pool:
        w = csv.writer(fh); w.writerow(["file", "status", "psnr_bicubic", "ssim_bicubic"])
        it = pool.imap_unordered(clean_one, files, chunksize=16)
        for f, status, psnr, ssim in tqdm(it, total=len(files), desc="cleaning", unit="pair",
                                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed {elapsed} | remaining {remaining} | {rate_fmt}]"):
            w.writerow([f, status, "" if psnr is None else f"{psnr:.2f}", "" if ssim is None else f"{ssim:.3f}"])
            if status == "kept":
                kept.append(f)
            else:
                reasons[status] += 1
    kept.sort()
    if not kept:
        raise SystemExit("Everything was filtered out - relax --min-ssim/--min-psnr or check the download.")

    # ---- held-out (never seen) regions
    present = sorted({region_of(f) for f in kept})
    hold = set(args.holdout_regions)
    unknown = hold - set(present)
    if unknown:
        raise SystemExit(f"--holdout-regions not in data: {sorted(unknown)}. Available: {present}")
    if args.holdout_auto:
        pool_ = [r for r in present if r not in hold]
        if args.holdout_auto >= len(pool_):
            raise SystemExit("--holdout-auto must leave at least one region for training")
        hold |= set(random.Random(args.seed).sample(pool_, args.holdout_auto))
    seen = [f for f in kept if region_of(f) not in hold]
    unseen = [f for f in kept if region_of(f) in hold]
    if not seen:
        raise SystemExit("No training regions left after hold-out.")

    coords = load_coords(raw, kept, args.pixel_m)
    splits = make_splits(seen, coords, args, block_m)
    purged = purge_overlaps(splits, coords, tile_m, args.gap_m)
    if hold:
        splits["test_unseen"] = unseen
    for name in ("train", "val"):
        if not splits[name]:
            raise SystemExit(f"'{name}' split is empty (too little data). Download more pairs or lower --block.")
    if not splits["test"]:
        print("[warn] 'test' split is empty (too little data) - evaluate.py/train.py will skip it.")

    (out / "splits.json").write_text(json.dumps(splits, indent=1))
    kept_by_region = Counter(region_of(f) for f in kept)
    summary = {"raw_pairs": len(files), "clean_pairs": len(kept), "kept_pct": round(100 * len(kept) / len(files), 1),
               "dropped": dict(reasons), "tile_px": tile, "group_m": block_m,
               "splits": {k: len(v) for k, v in splits.items()}, "purged_for_overlap": dict(purged),
               "holdout_regions": sorted(hold),
               "per_region": {r: {"raw": raw_by_region[r], "clean": kept_by_region.get(r, 0)} for r in sorted(raw_by_region)}}
    (out / "clean_summary.json").write_text(json.dumps(summary, indent=1))

    print("\n" + "=" * 62)
    print(f"  RAW PAIRS BEFORE CLEANING : {len(files):,}")
    print(f"  CLEANED PAIRS AFTER       : {len(kept):,}  ({summary['kept_pct']} % kept)")
    print(f"  dropped (quality)         : {dict(reasons)}")
    print(f"  purged (split overlap)    : {dict(purged)}")
    print("  splits                    : " + " / ".join(f"{k} {len(v):,}" for k, v in splits.items()))
    if hold:
        print(f"  unseen-geography regions  : {sorted(hold)}")
    print("=" * 62)
    print(f"  {'region':28s} {'raw':>7s} {'clean':>7s}")
    for r in sorted(raw_by_region):
        print(f"  {r:28s} {raw_by_region[r]:7,d} {kept_by_region.get(r, 0):7,d}{'   (held out)' if r in hold else ''}")
    if len(kept) < 10000:
        print(f"\n[note] {len(kept):,} < 10,000 clean pairs. Download more (e.g. --target-raw {int(len(files) * 10500 / max(len(kept), 1)):,}).")


if __name__ == "__main__":
    main()
