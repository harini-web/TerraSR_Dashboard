#!/usr/bin/env python3
"""
Super-resolve ANY RGB Sentinel-2 image (10 m -> 5 m for the default x2 model).

  GeoTIFF (keeps georeferencing, pixel size halves):
      python predict.py --ckpt runs/exp1/best.pt --input s2_rgb.tif --output s2_rgb_x2.tif --compare
  PNG / JPG:
      python predict.py --ckpt runs/exp1/best.pt --input patch.png --output patch_x2.png
  A folder of images:
      python predict.py --ckpt runs/exp1/best.pt --input some_folder/ --output out_folder/

Input value ranges (--mode): 
  s2     uint16 Sentinel-2 L2A digital numbers (reflectance*10000). Same stretch as training (refl / 0.3).
  uint8  8-bit RGB images (PNG, JPG, TCI products) - treated as already stretched to [0,1]
  float  float image already in [0,1]
  auto   (default) uint8 -> uint8, uint16 -> s2, float -> float
Band order must be R,G,B (use --bands 3 2 1 for a stack ordered B02,B03,B04).
For L2A baseline >= 04.00 (2022+) DNs carry a +1000 offset: pass --offset -0.1 (fetch_s2_rgb.py stores this in the
file tags and predict.py reads it automatically). Large images are processed in overlapping tiles.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate import load_model

GAIN = 1e-4


def to_unit(arr, mode, refl_max, offset):
    if mode == "uint8":
        return arr / 255.0
    if mode == "float":
        return arr
    return np.clip((arr * GAIN + offset) / refl_max, 0, 1)        # s2


def from_unit(x, mode, refl_max, offset):
    if mode == "uint8":
        return x * 255.0
    if mode == "float":
        return x
    return (x * refl_max - offset) / GAIN


@torch.no_grad()
def sr_tiled(model, x, scale, device, tile=128, overlap=16):
    """x: (3,H,W) tensor in [0,1] -> (3,H*scale,W*scale). Fully convolutional model, so tile + crop borders."""
    _, H, W = x.shape
    if H <= tile and W <= tile:
        return model(x[None].to(device)).clamp(0, 1)[0].cpu()
    stride = tile - 2 * overlap
    ph, pw = (-H) % stride, (-W) % stride
    # numpy 'reflect' keeps reflecting when the pad is larger than the image (torch's F.pad raises), so narrow
    # images (e.g. 10 x 5000 px) work too. 1-px-wide axes are edge-replicated automatically.
    xp = torch.from_numpy(np.pad(x.numpy(), ((0, 0), (overlap, overlap + ph), (overlap, overlap + pw)), mode="reflect"))
    out = torch.zeros(3, (H + ph) * scale, (W + pw) * scale)
    for y0 in range(0, H + ph, stride):
        for x0 in range(0, W + pw, stride):
            crop = xp[:, y0:y0 + tile, x0:x0 + tile][None].to(device)
            sr = model(crop).clamp(0, 1)[0].cpu()
            core = sr[:, overlap * scale:(overlap + stride) * scale, overlap * scale:(overlap + stride) * scale]
            out[:, y0 * scale:(y0 + stride) * scale, x0 * scale:(x0 + stride) * scale] = core
    return out[:, :H * scale, :W * scale]


def process(path, dst, model, scale, dev, a):
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        import rasterio
        from rasterio.transform import Affine
        with rasterio.open(path) as src:
            raw = src.read(list(a.bands))
            profile, tf, tags = src.profile, src.transform, src.tags()
            dtype = src.dtypes[0]
        mode = a.mode if a.mode != "auto" else ("uint8" if dtype == "uint8" else "s2" if dtype.startswith("uint") or dtype.startswith("int") else "float")
        offset = a.offset if a.offset is not None else float(tags.get("S2_REFLECTANCE_OFFSET", 0.0))
    else:
        from PIL import Image
        raw = np.asarray(Image.open(path).convert("RGB")).transpose(2, 0, 1)
        profile, tf, dtype = None, None, "uint8"
        mode, offset = (a.mode if a.mode != "auto" else "uint8"), (a.offset or 0.0)

    x = torch.from_numpy(to_unit(raw.astype(np.float32), mode, a.refl_max, offset).astype(np.float32))
    sr = sr_tiled(model, x, scale, dev, a.tile, a.overlap).numpy()
    print(f"{path.name}: {x.shape[2]}x{x.shape[1]} -> {sr.shape[2]}x{sr.shape[1]}  (mode={mode})")

    vals = from_unit(sr, mode, a.refl_max, offset)
    if np.issubdtype(np.dtype(dtype), np.integer):
        info = np.iinfo(dtype); vals = np.clip(np.rint(vals), info.min, info.max)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if profile is not None:
        profile.update(count=sr.shape[0], width=sr.shape[2], height=sr.shape[1], dtype=str(dtype),
                       transform=tf * Affine.scale(1 / scale), compress="deflate", tiled=False)
        profile.pop("blockxsize", None); profile.pop("blockysize", None)
        with rasterio.open(dst, "w", **profile) as o:
            o.write(vals.astype(dtype))
            o.update_tags(**tags)
    else:
        from PIL import Image
        Image.fromarray(np.clip(vals, 0, 255).astype(np.uint8).transpose(1, 2, 0)).save(dst)
    prev = (np.clip(sr, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
    from PIL import Image
    Image.fromarray(prev).save(dst.with_name(dst.stem + "_preview.png"))
    if a.compare:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h, w = x.shape[1:]; ch, cw = min(h, 96), min(w, 96); y0, x0 = (h - ch) // 2, (w - cw) // 2
        lr_c = x[:, y0:y0 + ch, x0:x0 + cw]
        bic = F.interpolate(lr_c[None], scale_factor=scale, mode="bicubic", align_corners=False).clamp(0, 1)[0]
        sr_c = torch.from_numpy(sr[:, y0 * scale:(y0 + ch) * scale, x0 * scale:(x0 + cw) * scale])
        fig, ax = plt.subplots(1, 3, figsize=(12, 4.3))
        for axi, im, t in zip(ax, [lr_c, bic, sr_c], ["Input (10 m)", "Bicubic x%d" % scale, "CNN SR x%d" % scale]):
            axi.imshow(im.permute(1, 2, 0).clamp(0, 1).numpy()); axi.set_title(t); axi.axis("off")
        plt.tight_layout(); plt.savefig(dst.with_name(dst.stem + "_compare.png"), dpi=140); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", required=True, help="image file or folder")
    p.add_argument("--output", default=None, help="output file or folder (default: <input>_x2)")
    p.add_argument("--mode", default="auto", choices=["auto", "s2", "uint8", "float"])
    p.add_argument("--bands", type=int, nargs=3, default=[1, 2, 3], help="1-based band indices for R G B (GeoTIFF)")
    p.add_argument("--offset", type=float, default=None, help="S2 reflectance offset (-0.1 for baseline >= 04.00)")
    p.add_argument("--refl-max", type=float, default=0.3, help="must match the value used at download time")
    p.add_argument("--tile", type=int, default=128)
    p.add_argument("--overlap", type=int, default=16)
    p.add_argument("--compare", action="store_true", help="also save a LR | bicubic | SR crop figure")
    a = p.parse_args()

    if a.tile <= 2 * a.overlap:
        p.error("--tile must be larger than 2 * --overlap")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scale = load_model(a.ckpt, dev)
    src = Path(a.input)
    if src.is_dir():
        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        files = sorted(f for f in src.iterdir() if f.suffix.lower() in exts)
        outdir = Path(a.output) if a.output else src.with_name(src.name + f"_x{scale}")
        for f in files:
            process(f, outdir / f"{f.stem}_x{scale}{f.suffix}", model, scale, dev, a)
    else:
        dst = Path(a.output) if a.output else src.with_name(f"{src.stem}_x{scale}{src.suffix}")
        process(src, dst, model, scale, dev, a)


if __name__ == "__main__":
    main()
