#!/usr/bin/env python3
"""
Stage 3 - Feed the cleaned dataset to the CNN, train, validate and test.

Training recipe (from the paper's pre-training stage): L1 loss, Adam, lr 2e-4, halve the lr when the
validation PSNR does not improve for 10 epochs, stop after 25 epochs without improvement.
The test set is evaluated against the bicubic baseline (PSNR / SSIM, mean and median; LPIPS if the
`lpips` package is installed).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import build_model


# ----------------------------------------------------------------------------- data
class PairDataset(Dataset):
    def __init__(self, root, files, augment=False):
        self.root, self.files, self.augment = Path(root), files, augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.root / self.files[i])
        lr, hr = d["lr"].astype(np.float32), d["hr"].astype(np.float32)
        if self.augment:                                   # same flip / rotation for LR and HR
            k = random.randint(0, 3)
            lr, hr = np.rot90(lr, k, (1, 2)), np.rot90(hr, k, (1, 2))
            if random.random() < 0.5:
                lr, hr = lr[:, :, ::-1], hr[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(lr)), torch.from_numpy(np.ascontiguousarray(hr))


# ----------------------------------------------------------------------------- metrics
def psnr_batch(x, y):
    mse = ((x - y) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


def _gauss_win(size=11, sigma=1.5, ch=3, device="cpu"):
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2)); g /= g.sum()
    return (g[:, None] @ g[None, :]).expand(ch, 1, size, size).contiguous()


def ssim_batch(x, y, c1=0.01 ** 2, c2=0.03 ** 2):
    w = _gauss_win(ch=x.shape[1], device=x.device)
    f = lambda t: F.conv2d(t, w, groups=x.shape[1])
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return s.flatten(1).mean(1)


@torch.no_grad()
def evaluate(model, loader, device, scale, lpips_fn=None, bicubic=False):
    if model is not None:
        model.eval()
    P, S, L = [], [], []
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        sr = F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False) if bicubic else model(lr)
        sr = sr.clamp(0, 1)
        P.append(psnr_batch(sr, hr).cpu()); S.append(ssim_batch(sr, hr).cpu())
        if lpips_fn is not None:
            L.append(lpips_fn(sr * 2 - 1, hr * 2 - 1).flatten().cpu())
    P, S = torch.cat(P).numpy(), torch.cat(S).numpy()
    res = {"psnr_mean": float(P.mean()), "psnr_median": float(np.median(P)),
           "ssim_mean": float(S.mean()), "ssim_median": float(np.median(S))}
    if L:
        L = torch.cat(L).numpy()
        res.update(lpips_mean=float(L.mean()), lpips_median=float(np.median(L)))
    return res


def save_samples(model, ds, device, scale, path, n=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    fig, ax = plt.subplots(len(idx), 4, figsize=(10, 2.6 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        lr, hr = ds[i]
        with torch.no_grad():
            sr = model(lr[None].to(device)).clamp(0, 1)[0].cpu()
        bic = F.interpolate(lr[None], scale_factor=scale, mode="bicubic", align_corners=False).clamp(0, 1)[0]
        for c, (img, t) in enumerate([(lr, "LR Sentinel-2"), (bic, "Bicubic"), (sr, "CNN SR"), (hr, "HR aerial")]):
            ax[r, c].imshow(img.permute(1, 2, 0).numpy()); ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(t)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------- main
def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/clean")
    p.add_argument("--out", default="runs/exp1")
    p.add_argument("--model", choices=["resnet", "srcnn"], default="resnet")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--feats", type=int, default=64)
    p.add_argument("--blocks", type=int, default=16)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-patience", type=int, default=10)
    p.add_argument("--stop-patience", type=int, default=25)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = get_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    splits = json.loads((Path(a.data) / "splits.json").read_text())
    ds = {k: PairDataset(a.data, v, augment=(k == "train")) for k, v in splits.items()}
    print({k: len(v) for k, v in ds.items()}, "device:", dev)
    if min(len(ds["train"]), len(ds["val"])) == 0:
        raise SystemExit("Empty train/val split - get more data (increase --max-items in download_data.py).")

    mk = lambda k, shuf: DataLoader(ds[k], batch_size=a.batch_size, shuffle=shuf, num_workers=a.workers,
                                    pin_memory=dev.type == "cuda", drop_last=shuf and len(ds[k]) > a.batch_size)
    tr, va = mk("train", True), mk("val", False)

    model = build_model(a.model, a.scale, **({"feats": a.feats, "n_blocks": a.blocks} if a.model == "resnet" else {})).to(dev)
    print(f"{a.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=a.lr_patience)
    use_amp = a.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, bad, t_start = -1.0, 0, time.time()
    fmt = lambda sec: f"{int(sec // 3600)}h{int(sec % 3600 // 60):02d}m"
    with open(out / "history.csv", "w", newline="") as fh:
        hw = csv.writer(fh); hw.writerow(["epoch", "train_l1", "val_psnr", "val_ssim", "lr"])
        for ep in range(1, a.epochs + 1):
            model.train(); run = 0.0
            for lr_img, hr_img in tqdm(tr, desc=f"epoch {ep}", leave=False):
                lr_img, hr_img = lr_img.to(dev), hr_img.to(dev)
                with torch.autocast(dev.type, enabled=use_amp):
                    loss = F.l1_loss(model(lr_img), hr_img)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                run += loss.item() * lr_img.size(0)
            run /= len(ds["train"])
            v = evaluate(model, va, dev, a.scale)
            sched.step(v["psnr_mean"])
            hw.writerow([ep, f"{run:.5f}", f"{v['psnr_mean']:.3f}", f"{v['ssim_mean']:.4f}", opt.param_groups[0]["lr"]]); fh.flush()
            el = time.time() - t_start
            eta = el / ep * (a.epochs - ep)               # upper bound: early stopping may end sooner
            print(f"ep {ep:3d}/{a.epochs} | L1 {run:.4f} | val PSNR {v['psnr_mean']:.2f} SSIM {v['ssim_mean']:.3f} "
                  f"| lr {opt.param_groups[0]['lr']:.1e} | elapsed {fmt(el)} | max remaining {fmt(eta)}")
            if v["psnr_mean"] > best:
                best, bad = v["psnr_mean"], 0
                torch.save({"model": model.state_dict(), "args": vars(a)}, out / "best.pt")
            else:
                bad += 1
                if bad >= a.stop_patience:
                    print("early stopping"); break

    # ---- final test evaluation with the best checkpoint
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev)["model"])
    lp = None
    try:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    except Exception:
        print("(lpips not available - skipping LPIPS)")
    results = {}
    for k in [k for k in ds if k.startswith("test")]:            # 'test' and, if present, 'test_unseen'
        if len(ds[k]) == 0:
            print(f"[warn] split '{k}' is empty - skipped")
            continue
        dl = mk(k, False)
        results[k] = {"n": len(ds[k]), "bicubic": evaluate(None, dl, dev, a.scale, lp, bicubic=True),
                      a.model: evaluate(model, dl, dev, a.scale, lp)}
    (out / "test_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    if len(ds["test"]):
        save_samples(model, ds["test"], dev, a.scale, out / "samples.png")
    print("saved to", out)
    print(f"next: python evaluate.py --ckpt {out}/best.pt --data {a.data}   |   python predict.py --ckpt {out}/best.pt --input your_image.tif")


if __name__ == "__main__":
    main()
