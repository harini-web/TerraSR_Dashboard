# Sentinel-2 ×2 super-resolution with a CNN (10 m → 5 m)

Re-implementation of the data recipe and CNN baseline of Kramer et al., *Enhancing Sentinel-2 Image Resolution* (2024),
scaled up to 10-15k+ pairs from 20 geographically diverse regions.

```
download_data.py -> clean_data.py -> train.py -> evaluate.py        (run_pipeline.py = all four)
                                              \-> predict.py  (+ fetch_s2_rgb.py to get inputs)
```

## Files (everything you need, all in one folder)
| file | stage | needed for |
|---|---|---|
| `requirements.txt` | setup | `python -m pip install -r requirements.txt` |
| `download_data.py` | 1. data collection | downloads paired Sentinel-2 / NAIP tiles |
| `clean_data.py` | 2. cleaning + splits | histogram matching, quality filter, leakage-safe train/val/test |
| `model.py` | 3. model | CNN definitions (imported by train/evaluate/predict) |
| `train.py` | 3. training | trains the CNN, writes `best.pt` |
| `evaluate.py` | 4. evaluation | metrics vs bicubic, per region (imports `train.py`, `model.py`) |
| `predict.py` | 5. prediction | super-resolves new images (imports `evaluate.py`, `model.py`) |
| `fetch_s2_rgb.py` | 5. prediction (optional) | gets a Sentinel-2 RGB GeoTIFF to feed `predict.py` (imports `download_data.py`) |
| `run_pipeline.py` | optional | runs download -> clean -> train -> evaluate in one command |
| `README.md` | docs | this file |

Keep all `.py` files in the same folder: they import each other.

## 1. Install
```bash
pip install -r requirements.txt          # add `pip install lpips` for the LPIPS metric
```

## 2. Build the dataset (10-15k+ pairs)
```bash
python download_data.py --list-regions                      # forest, cropland, desert, urban, wetland, coast ...
python download_data.py --out data/raw --target-raw 20000 --workers 6
python clean_data.py --raw data/raw --out data/clean
```
* Regions are visited round-robin, so every land-cover type is covered even if you stop early (Ctrl-C is safe; re-run to resume).
* The download shows a live bar with **elapsed / remaining time and pairs per second**, and at the end prints
  **TOTAL RAW PAIRS** plus a per-region table (`raw_summary.json`).
* Cleaning shows its own progress/ETA and prints **RAW PAIRS BEFORE**, **CLEANED PAIRS AFTER**, drop reasons, the
  train/val/test sizes and a raw-vs-clean table per region (`clean_summary.json`). Cleaning usually removes 20-40 %, so
  the default `--target-raw 20000` should leave ≥ 12k. If it prints a note that you have < 10k, raise `--target-raw`.
* Yield tips: `--stride 64` (default, ~1/3 tile overlap) gives ~120 tiles per NAIP scene; `--stride 96` = paper-style
  no overlap but needs 2x more scenes. Use `--min-vegetation 0.5` to focus on forest/crops, `--regions a b c` to pick regions.
**Download speed.** The first version took ~20 h for 20k pairs; it now reads each scene once instead of using many small
requests per tile (S2 and the aerial image are fetched once per scene, tiles are cut from memory, and cloudy/empty tiles are
rejected before any aerial pixels are downloaded). Further tips:
* `--workers 6` is the default. Planetary Computer **throttles aggressive clients** ("Request rate limited or blocked"), so
  stay at 4-8. The script shares one rate gate between all workers, retries throttled requests with exponential backoff
  (all workers pause together), caches Sentinel-2 searches, and stops by itself after 15 failed scenes in a row.
* Do a pilot first: `--target-raw 1500` and read the `remaining` time and pairs/s on the progress bar, then scale up.
* You do not need 20k raw pairs: cleaning keeps roughly 60-80 %, so `--target-raw 15000` gives about 10-12k clean pairs.
* Stop any time with Ctrl-C and resume later with the same command; use `--regions ...` to download in batches.
* Latency to the data centre matters: running from a cloud VM in the US/West Europe region is usually much faster than from
  a home connection.
* The target is exact: a download slot is reserved atomically before each tile is written, so parallel workers can never
  overshoot `--target-raw`; files are written atomically, so Ctrl-C never leaves a corrupt tile.
* Source: NAIP (HR, USA only) + Sentinel-2 L2A (LR) from the Microsoft Planetary Computer STAC API, no key needed.
  Region boxes are approximate; ones without a usable NAIP/S2 match in 2018-2022 are skipped automatically.

## Splits: what they do and do not prove
`clean_data.py` builds leakage-safe splits:
* tile size is read from the data (nothing hard-coded); tiles are grouped by **ground position** (UTM coordinates from
  `manifest.csv`), so overlapping tiles and the same ground seen on another date stay together;
* after the split, any train/val tile whose footprint overlaps a test (or train overlapping val) tile is **purged**
  (reported as `purged (split overlap)`); `--gap-m 500` adds a safety margin;
* every split is guaranteed non-empty when the data allows it; if train/val would end up empty the script stops with a clear
  message, and `train.py` / `evaluate.py` skip or refuse an empty test split instead of crashing.

Two different test questions:
| split | what it measures |
|---|---|
| `test` | held-out **spatial blocks inside regions seen in training** (no tile overlap with train) |
| `test_unseen` | **entirely unseen regions** = real geographic generalisation |

`test` alone does NOT show generalisation to new geography. To measure that, hold regions out completely:
```bash
python clean_data.py --holdout-regions colorado_rockies florida_everglades norcal_bay_coast   # or --holdout-auto 3
python train.py --data data/clean --out runs/exp1
python evaluate.py --ckpt runs/exp1/best.pt --split test_unseen
```
Those regions are excluded from train/val/test, and `train.py` reports `test` and `test_unseen` separately in
`test_metrics.json`. Pick held-out regions that differ from training ones (e.g. a different biome), and compare the PSNR gain
over bicubic on `test` vs `test_unseen`: a large drop means the model is memorising local appearance.

### Troubleshooting the download
* `[throttled] ... all workers pause` is normal and self-healing. If the run stops with `[STOPPED] 15 scenes failed in a row`
  and the error says rate limited/blocked, wait 15-30 min and re-run the same command with `--workers 3` (progress is kept).
  If it persists, contact planetarycomputer@microsoft.com with the reference id shown in the error.
* Every failed scene is written with a full traceback to `data/raw/download_errors.log` (file and line number are also printed).
  Send me the last traceback if you see an error that is not about throttling.

## 3. Train
```bash
python train.py --data data/clean --out runs/exp1 --epochs 300 --amp
```
Prints per-epoch val PSNR/SSIM, elapsed time and (max) remaining time. Saves `best.pt`, `history.csv`,
`test_metrics.json` (one entry per `test*` split), `samples.png`. `--model srcnn` trains the paper's SRCNN baseline.

## 4. Test / evaluate
```bash
python evaluate.py --ckpt runs/exp1/best.pt --data data/clean --split test          # or val, test_unseen, all
```
Prints PSNR / SSIM (/ LPIPS) mean+median for **bicubic vs model**, % of patches improved, and a per-region table.
Writes `runs/exp1/eval_<split>/{summary.json, per_image_metrics.csv, samples_random|best|worst.png}`.
`--data other_folder` evaluates any folder of `lr`/`hr` `.npz` pairs.

## 5. Predict on new imagery
```bash
# get a Sentinel-2 RGB crop anywhere in the world (3-band uint16 GeoTIFF)
python fetch_s2_rgb.py --bbox 14.25 46.60 14.40 46.70 --dates 2023-06-01/2023-08-31 --out klagenfurt.tif
# super-resolve it: output keeps georeferencing with 5 m pixels (+ *_preview.png and *_compare.png)
python predict.py --ckpt runs/exp1/best.pt --input klagenfurt.tif --output klagenfurt_x2.tif --compare
```
Also works on PNG/JPG files or a whole folder (`--input folder/`). Images of any shape work, including very narrow ones
(large images are processed in overlapping tiles, padding reflects as often as needed). Use `--mode`, `--bands 3 2 1`,
`--offset -0.1` if your input differs (see `python predict.py -h`).
Note: the model is trained on Sentinel-2 L2A reflectance stretched by 1/0.3; other products (L1C, TCI) may need `--mode uint8`
and will give weaker results.

## Notes
* Expect the paper's finding: a pixel-loss CNN gets the best PSNR/SSIM but looks softer than ESRGAN-type GANs (best LPIPS).
* The model is trained on US scenes; quality elsewhere (e.g. Europe) is not guaranteed. Use `test_unseen` and local reference data.
