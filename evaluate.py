#!/usr/bin/env python3
"""
Evaluate a trained model on the held-out split (or any folder of LR/HR .npz pairs).

  python evaluate.py --ckpt runs/exp1/best.pt --data data/clean --split test

Reports mean/median PSNR, SSIM (and LPIPS if `pip install lpips`) for bicubic vs the model, overall and per
region, and writes per_image_metrics.csv, summary.json and sample figures (random / best / worst) to
<ckpt folder>/eval_<split>/.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import build_model
from train import PairDataset, psnr_batch, ssim_batch


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    a = ck["args"]
    kw = {"feats": a["feats"], "n_blocks": a["blocks"]} if a["model"] == "resnet" else {}
    m = build_model(a["model"], a["scale"], **kw).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, a["scale"]


def stats(x):
    x = np.asarray(x)
    return {"mean": float(x.mean()), "median": float(np.median(x))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/clean")
    p.add_argument("--split", default="test", help="train | val | test | test_unseen | all (any key of splits.json)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--samples", type=int, default=6, help="patches per sample figure")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scale = load_model(a.ckpt, dev)
    root = Path(a.data)
    if a.split == "all":
        files = sorted(q.name for q in root.glob("*.npz"))
    else:
        splits = json.loads((root / "splits.json").read_text())
        if a.split not in splits:
            raise SystemExit(f"split '{a.split}' not in splits.json. Available: {sorted(splits)} (or 'all')")
        files = splits[a.split]
    if not files:
        raise SystemExit(f"split '{a.split}' is empty (dataset too small?). Try --split val / --split all, "
                         f"or build a larger dataset.")
    ds = PairDataset(root, files)
    dl = DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers)
    out = Path(a.out) if a.out else Path(a.ckpt).parent / f"eval_{a.split}"
    out.mkdir(parents=True, exist_ok=True)

    lp = None
    try:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    except Exception:
        print("(lpips not installed - skipping LPIPS; pip install lpips)")

    rows = defaultdict(list)           # metric -> list ; keys like psnr_model
    with torch.no_grad():
        for lr, hr in dl:
            lr, hr = lr.to(dev), hr.to(dev)
            outs = {"bicubic": F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0, 1),
                    "model": model(lr).clamp(0, 1)}
            for name, sr in outs.items():
                rows[f"psnr_{name}"] += psnr_batch(sr, hr).cpu().tolist()
                rows[f"ssim_{name}"] += ssim_batch(sr, hr).cpu().tolist()
                if lp is not None:
                    rows[f"lpips_{name}"] += lp(sr * 2 - 1, hr * 2 - 1).flatten().cpu().tolist()

    n = len(files)
    regions = [f.split("__")[0] if "__" in f else "unknown" for f in files]
    with open(out / "per_image_metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh); keys = list(rows)
        w.writerow(["file", "region"] + keys)
        for i, f in enumerate(files):
            w.writerow([f, regions[i]] + [f"{rows[k][i]:.4f}" for k in keys])

    metrics = ["psnr", "ssim"] + (["lpips"] if lp is not None else [])
    summary = {"split": a.split, "n_images": n, "overall": {}, "per_region": {}}
    print(f"\n{a.split} set: {n:,} image pairs   (PSNR/SSIM higher = better, LPIPS lower = better)")
    print(f"{'':10s}" + "".join(f"{m.upper():>18s}" for m in metrics))
    for name in ["bicubic", "model"]:
        summary["overall"][name] = {m: stats(rows[f"{m}_{name}"]) for m in metrics}
        print(f"{name:10s}" + "".join(
            f"{summary['overall'][name][m]['mean']:>9.3f}/{summary['overall'][name][m]['median']:<8.3f}" for m in metrics))
    print("(mean/median)")
    gain = np.array(rows["psnr_model"]) - np.array(rows["psnr_bicubic"])
    print(f"model beats bicubic PSNR on {100 * (gain > 0).mean():.1f} % of patches, mean gain {gain.mean():+.2f} dB")

    print(f"\n{'region':28s} {'n':>6s} {'PSNR bic':>9s} {'PSNR mod':>9s} {'gain':>7s} {'SSIM mod':>9s}")
    for reg in sorted(set(regions)):
        idx = [i for i, r in enumerate(regions) if r == reg]
        pb, pm = np.mean([rows["psnr_bicubic"][i] for i in idx]), np.mean([rows["psnr_model"][i] for i in idx])
        sm = np.mean([rows["ssim_model"][i] for i in idx])
        summary["per_region"][reg] = {"n": len(idx), "psnr_bicubic": pb, "psnr_model": pm, "ssim_model": sm}
        print(f"{reg:28s} {len(idx):6d} {pb:9.2f} {pm:9.2f} {pm - pb:+7.2f} {sm:9.3f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # sample figures: random / best / worst by PSNR gain over bicubic
    from train import save_samples
    order = np.argsort(gain)
    k = min(a.samples, n)
    for tag, idx in [("random", np.linspace(0, n - 1, k).astype(int)), ("best", order[::-1][:k]), ("worst", order[:k])]:
        sub = PairDataset(root, [files[i] for i in idx])
        save_samples(model, sub, dev, scale, out / f"samples_{tag}.png", n=k)
    print(f"\nsaved: {out}/summary.json, per_image_metrics.csv, samples_*.png")


if __name__ == "__main__":
    main()
