#!/usr/bin/env python3
"""
Stage 1 - Download paired LR / HR satellite tiles automatically (large, geographically diverse).

LR : Sentinel-2 L2A, RGB (B04/B03/B02) at 10 m          (Microsoft Planetary Computer, free)
HR : NAIP aerial orthophotos (RGB, 0.6-1 m, USA)        (Microsoft Planetary Computer, free)

Recipe of Kramer et al. (2024): HR + S2 acquired close in time, S2 cloud cover < 5 %, both in the S2 UTM
CRS, HR area-averaged to 5 m  ->  LR : HR patches of 96x96 : 192x192, saved as .npz (float16, [0,1]).

New in this version
  * 20 pre-defined regions (forest, cropland, grassland, desert, mountains, urban, wetland, coast ...)
    visited round-robin, so every land-cover type is represented even if you stop early
  * scenes are processed in parallel (--workers); resumable: just re-run the same command
  * stops automatically at --target-raw pairs; live progress bar with elapsed time, REMAINING TIME and speed
  * prints the total number of raw pairs (and a per-region table) when finished

Examples
  python download_data.py --list-regions
  python download_data.py --out data/raw --target-raw 20000 --workers 6
  python download_data.py --regions oregon_cascades_forest iowa_cropland --target-raw 3000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path

import numpy as np
try:
    from tqdm import tqdm
except ImportError:          # reported by check_deps() with the install command
    tqdm = None

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
RGB_BANDS = ["B04", "B03", "B02"]          # R, G, B order
SCL_CLOUD = {0, 1, 3, 8, 9, 10, 11}        # nodata, saturated, shadow, cloud(med/high), cirrus, snow
SCL_VEGETATION = 4

# name: ((west, south, east, north), land-cover character).  NAIP covers the continental USA only.
# Boxes are approximate; if a box has no usable NAIP/S2 match in the date range it is simply skipped.
REGIONS = {
    "oregon_cascades_forest":   ((-122.20, 44.20, -121.80, 44.50), "conifer forest, mountains"),
    "washington_olympic_forest": ((-123.80, 47.70, -123.30, 48.00), "temperate rainforest"),
    "maine_north_woods":        ((-69.50, 45.60, -69.00, 46.00), "mixed forest, lakes"),
    "nc_appalachia_forest":     ((-82.00, 35.40, -81.50, 35.80), "deciduous forest, hills"),
    "minnesota_boreal_lakes":   ((-92.20, 47.60, -91.70, 47.90), "boreal forest, lakes"),
    "iowa_cropland":            ((-93.80, 41.80, -93.20, 42.30), "row-crop agriculture"),
    "california_central_valley": ((-120.30, 36.60, -119.70, 37.10), "irrigated agriculture, orchards"),
    "kansas_plains":            ((-98.50, 38.30, -98.00, 38.70), "plains, wheat, grassland"),
    "nebraska_sandhills":       ((-101.50, 42.00, -101.00, 42.40), "grassland, dunes"),
    "montana_ranchland":        ((-111.50, 46.80, -111.00, 47.20), "rangeland, foothills"),
    "arizona_sonoran_desert":   ((-111.60, 32.40, -111.10, 32.80), "desert scrub, suburbs"),
    "nevada_basin_range":       ((-116.50, 39.00, -116.00, 39.40), "arid basin and range"),
    "utah_red_rock":            ((-110.00, 38.30, -109.50, 38.70), "bare rock, canyons"),
    "colorado_rockies":         ((-106.50, 39.40, -106.00, 39.80), "alpine, conifer, snow patches"),
    "texas_houston_urban":      ((-95.70, 29.60, -95.20, 30.00), "dense urban, industrial"),
    "georgia_atlanta_suburban": ((-84.60, 33.60, -84.10, 34.00), "suburban, urban forest"),
    "florida_everglades":       ((-81.20, 25.70, -80.70, 26.10), "wetland, water"),
    "louisiana_delta":          ((-91.30, 29.60, -90.80, 30.00), "bayou, wetland, water"),
    "norcal_bay_coast":         ((-122.60, 37.70, -122.10, 38.10), "coast, urban, hills"),
    "new_york_hudson":          ((-74.20, 41.80, -73.70, 42.20), "mixed forest, farmland, towns"),
}


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/raw", help="output folder for raw tiles")
    p.add_argument("--regions", nargs="+", default=["all"], help="region names or 'all' (see --list-regions)")
    p.add_argument("--list-regions", action="store_true")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"),
                   help="custom area (lon/lat); used instead of --regions")
    p.add_argument("--target-raw", type=int, default=20000,
                   help="stop after this many raw pairs (cleaning typically removes 20-40 %%)")
    p.add_argument("--workers", type=int, default=6,
                   help="scenes processed in parallel. Planetary Computer throttles aggressive clients: keep 4-8")
    p.add_argument("--request-interval", type=float, default=0.3,
                   help="min seconds between STAC/token API calls (all workers share this)")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2022-12-31")
    p.add_argument("--scenes-per-region", type=int, default=150, help="max NAIP scenes tried per region")
    p.add_argument("--date-window", type=int, default=10, help="max |days| between NAIP and S2 acquisition")
    p.add_argument("--max-s2-cloud", type=float, default=5.0, help="scene-level S2 cloud cover limit (%%)")
    p.add_argument("--max-tile-cloud", type=float, default=0.02, help="max cloud/shadow fraction inside a tile")
    p.add_argument("--min-vegetation", type=float, default=0.0,
                   help="min fraction of SCL 'vegetation' pixels per tile (e.g. 0.5 = forest/crop focus)")
    p.add_argument("--lr-size", type=int, default=96, help="LR patch size in pixels (10 m)")
    p.add_argument("--stride", type=int, default=64,
                   help="tile step in LR pixels. 96 = no overlap (paper); 64 = ~1/3 overlap (more pairs per scene)")
    p.add_argument("--scale", type=int, default=2, help="LR->HR factor (2 => 5 m)")
    p.add_argument("--refl-max", type=float, default=0.3, help="surface reflectance mapped to 1.0")
    p.add_argument("--max-tiles-per-scene", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------- progress / shared state
# --------------------------------------------------------------------------- rate limiting / retries
class RateGate:
    """Shared by all threads: spaces out API calls and lets any thread pause EVERYONE after a 429/blocked reply."""

    def __init__(self, min_interval=0.3):
        self.lock, self.min_interval, self.next_ok, self.cool_until = threading.Lock(), min_interval, 0.0, 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            t = max(self.next_ok, self.cool_until, now)
            self.next_ok = t + self.min_interval
        if t > now:
            time.sleep(t - now)

    def penalize(self, secs):
        """Pause all workers for `secs`. Returns True if this started a new pause (used to avoid log spam)."""
        with self.lock:
            now = time.time()
            fresh = self.cool_until <= now
            self.cool_until = max(self.cool_until, now + secs)
        return fresh


GATE = RateGate()
BACKOFF_BASE = 5.0                       # seconds; doubles each retry (5, 10, 20, 40, 80 ...) with jitter
TRANSIENT = ("rate limit", "too many requests", "429", "502", "503", "504", "timed out", "timeout", "temporarily",
             "connection", "server busy", "serverbusy", "reset by peer", "slow down", "blocked")


def is_transient(e):
    msg = f"{type(e).__name__} {e}".lower()
    return any(k in msg for k in TRANSIENT)


def retry(fn, *args, what="request", tries=6):
    """Call fn(*args); on rate-limit / network errors pause ALL workers with exponential backoff and try again.
    Other errors (bugs, bad data) are raised immediately."""
    for i in range(tries):
        GATE.wait()
        try:
            return fn(*args)
        except Exception as e:
            if not is_transient(e) or i == tries - 1:
                raise
            delay = min(180.0, BACKOFF_BASE * 2 ** i) * (0.5 + random.random())
            if GATE.penalize(delay):
                tqdm.write(f"  [throttled] {what}: {str(e)[:70]!r} -> all workers pause ~{delay:.0f}s (retry {i + 1}/{tries - 1})")


_ERR_LOCK = threading.Lock()


def log_failure(out_dir, region, naip_id, e):
    tb = traceback.extract_tb(e.__traceback__)
    where = f"{Path(tb[-1].filename).name}:{tb[-1].lineno}" if tb else "?"
    with _ERR_LOCK, open(Path(out_dir) / "download_errors.log", "a") as f:
        f.write(f"--- {time.strftime('%H:%M:%S')} {region} {naip_id}\n"
                + "".join(traceback.format_exception(type(e), e, e.__traceback__)) + "\n")
    tqdm.write(f"  scene failed ({region}, ...{naip_id[-20:]}): {type(e).__name__}: {str(e)[:140]} [{where}]")


class State:
    """Shared counter. A slot is RESERVED under a lock before a tile is written, so concurrent workers can
    never push the total past the target (count = written + reserved)."""

    def __init__(self, target, initial, pbar):
        self.lock, self.target, self.count, self.pbar = threading.Lock(), target, initial, pbar
        self.scenes_done = 0
        self.fail_streak, self.max_streak, self.aborted, self.last_error = 0, 15, False, ""

    def reached(self):
        return self.aborted or self.count >= self.target

    def record_success(self):
        with self.lock:
            self.fail_streak = 0

    def record_failure(self, err):
        """Too many scenes failing in a row => something systematic is wrong: stop instead of burning through the list."""
        with self.lock:
            self.fail_streak += 1
            self.last_error = err
            if self.fail_streak >= self.max_streak:
                self.aborted = True

    def try_reserve(self):
        with self.lock:
            if self.aborted or self.count >= self.target:
                return False
            self.count += 1
            return True

    def release(self):                      # write failed -> give the slot back
        with self.lock:
            self.count -= 1

    def commit(self):                       # tile is on disk
        with self.lock:
            self.pbar.update(1)

    def scene_finished(self, total_jobs):
        with self.lock:
            self.scenes_done += 1
            self.pbar.set_postfix_str(f"scenes {self.scenes_done}/{total_jobs}")


_tls = threading.local()


def catalog():
    """One STAC client per thread. Items are NOT signed here; process_scene signs them once, right before use."""
    if not hasattr(_tls, "cat"):
        import pystac_client
        _tls.cat = retry(lambda: pystac_client.Client.open(STAC_URL), what="open STAC")
    return _tls.cat


# --------------------------------------------------------------------------- STAC search
def find_naip(region, bbox, args):
    def go():
        search = catalog().search(collections=["naip"], bbox=bbox, datetime=f"{args.start}/{args.end}",
                                  max_items=args.scenes_per_region * 4)
        return list(search.items())
    items = retry(go, what=f"NAIP search {region}")
    random.Random(args.seed).shuffle(items)
    return items[: args.scenes_per_region]


def bbox_contains(outer, inner, tol=0.0):
    return (outer[0] <= inner[0] + tol and outer[1] <= inner[1] + tol and
            outer[2] >= inner[2] - tol and outer[3] >= inner[3] - tol)


_S2_CACHE, _S2_LOCK = {}, threading.Lock()


def find_s2(naip, args):
    """Closest-in-time low-cloud S2 scene covering the NAIP quad. Quads of the same flight date and ~0.25 deg cell
    share one search, which removes most STAC requests (a main cause of throttling)."""
    d = naip.datetime
    key = (str(d.date()), math.floor(naip.bbox[0] * 4), math.floor(naip.bbox[1] * 4))

    def pick(items):
        ok = [i for i in items if bbox_contains(i.bbox, naip.bbox, tol=0.01)]
        ok.sort(key=lambda i: abs((i.datetime - d).total_seconds()))
        return ok[0] if ok else None

    with _S2_LOCK:
        cached = _S2_CACHE.get(key)
    hit = pick(cached) if cached is not None else None
    if hit is not None:
        return hit
    rng = f"{(d - timedelta(days=args.date_window)).date()}/{(d + timedelta(days=args.date_window)).date()}"

    def go():
        return list(catalog().search(collections=["sentinel-2-l2a"], bbox=naip.bbox, datetime=rng,
                                     query={"eo:cloud_cover": {"lt": args.max_s2_cloud}}).items())
    items = retry(go, what="S2 search")
    with _S2_LOCK:
        _S2_CACHE[key] = items
    return pick(items)


def band_scale_offset(item, band="B04"):
    """DN -> reflectance. Baseline >= 04.00 (Jan 2022+) carries a -1000 DN offset."""
    rb = item.assets[band].extra_fields.get("raster:bands")
    if rb and "scale" in rb[0]:
        return float(rb[0]["scale"]), float(rb[0].get("offset", 0.0))
    base = str(item.properties.get("s2:processing_baseline", "02.00"))
    return 1e-4, (-0.1 if base >= "04.00" else 0.0)


# --------------------------------------------------------------------------- one scene
def process_scene(region, naip, s2, args, out_dir, writer, wlock, state):
    """Read each scene ONCE (few big requests) and cut the tiles out of memory.

    The old version issued several small HTTP range requests per tile and re-read the same aerial pixels for
    overlapping tiles. Now: S2 bands + SCL of the footprint are read in one go (tiny), tiles are pre-filtered
    (no-data / cloud / vegetation) BEFORE any aerial pixel is fetched, and the aerial image is read once for
    the bounding window of the surviving tiles.
    """
    import planetary_computer as pc
    import rasterio
    from pyproj import Transformer
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window, from_bounds

    naip, s2 = retry(pc.sign, naip, what="sign NAIP"), retry(pc.sign, s2, what="sign S2")   # fresh SAS tokens
    scene = f"{naip.id}_{s2.datetime:%Y%m%d}"
    L, S = args.lr_size, args.scale
    gain, offset = band_scale_offset(s2)
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                       GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3", VSI_CACHE="TRUE")
    kept = 0
    with env, ExitStack() as stack:
        bands = [stack.enter_context(rasterio.open(s2.assets[b].href)) for b in RGB_BANDS]
        scl = stack.enter_context(rasterio.open(s2.assets["SCL"].href))
        tf, crs, W, H = bands[0].transform, bands[0].crs, bands[0].width, bands[0].height

        # NAIP footprint in S2 pixel coordinates
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        pts = [tr.transform(x, y) for x in (naip.bbox[0], naip.bbox[2]) for y in (naip.bbox[1], naip.bbox[3])]
        xs, ys = zip(*pts)
        c0 = max(0, int(np.ceil((min(xs) - tf.c) / tf.a)))
        c1 = min(W, int(np.floor((max(xs) - tf.c) / tf.a)))
        r0 = max(0, int(np.ceil((tf.f - max(ys)) / -tf.e)))
        r1 = min(H, int(np.floor((tf.f - min(ys)) / -tf.e)))
        rows, cols = r1 - r0, c1 - c0
        if rows < L or cols < L:
            return 0

        # ---- 1. Sentinel-2 footprint, one request per band (a few hundred KB each)
        s2_arr = np.stack([b.read(1, window=Window(int(c0), int(r0), int(cols), int(rows))) for b in bands])
        left0, top0 = tf.c + c0 * tf.a, tf.f + r0 * tf.e
        win = from_bounds(left0, top0 + rows * tf.e, left0 + cols * tf.a, top0, transform=scl.transform)
        scl_arr = scl.read(1, window=win, out_shape=(int(rows), int(cols)), resampling=Resampling.nearest)

        # ---- 2. pre-filter tiles on S2 data only (no aerial pixels fetched yet)
        tiles = [(r, c) for r in range(r0, r1 - L + 1, args.stride) for c in range(c0, c1 - L + 1, args.stride)]
        random.Random(args.seed).shuffle(tiles)
        cand, budget = [], int(args.max_tiles_per_scene * 1.3) + 1     # spare ones for later aerial no-data rejects
        for r, c in tiles:
            if len(cand) >= budget:
                break
            fname = f"{region}__{scene}_{r:05d}_{c:05d}.npz"
            if (out_dir / fname).exists():
                kept += 1
                continue
            rr, cc = r - r0, c - c0
            raw = s2_arr[:, rr:rr + L, cc:cc + L]
            if (raw == 0).all(0).mean() > 0.001:                        # S2 no-data (tile edge)
                continue
            sc = scl_arr[rr:rr + L, cc:cc + L]
            if np.isin(sc, list(SCL_CLOUD)).mean() > args.max_tile_cloud:
                continue
            veg = float((sc == SCL_VEGETATION).mean())
            if veg < args.min_vegetation:
                continue
            cand.append((r, c, fname, raw, veg))
        if not cand or state.reached():
            return kept

        # ---- 3. aerial image: ONE read covering the surviving tiles, warped + area-averaged to the 5 m grid
        nsrc = stack.enter_context(rasterio.open(naip.assets["image"].href))
        hr_tf = Affine(tf.a / S, 0, tf.c, 0, tf.e / S, tf.f)
        hr = stack.enter_context(WarpedVRT(nsrc, crs=crs, transform=hr_tf, width=int(W * S), height=int(H * S),
                                           resampling=Resampling.average))
        rmin, rmax = min(t[0] for t in cand), max(t[0] for t in cand) + L
        cmin, cmax = min(t[1] for t in cand), max(t[1] for t in cand) + L
        big = hr.read([1, 2, 3], window=Window(int(cmin * S), int(rmin * S), int((cmax - cmin) * S), int((rmax - rmin) * S)))

        # ---- 4. cut tiles from memory and save
        for r, c, fname, raw, veg in cand:
            if state.reached() or kept >= args.max_tiles_per_scene:
                break
            try:
                y0, x0 = (r - rmin) * S, (c - cmin) * S
                hr_arr = big[:, y0:y0 + L * S, x0:x0 + L * S].astype(np.float32) / 255.0
                if (hr_arr.sum(0) < 1e-3).mean() > 0.001:               # NAIP no-data
                    continue
                lr_arr = np.clip((raw.astype(np.float32) * gain + offset) / args.refl_max, 0, 1)
                if not state.try_reserve():                              # target reached by another worker
                    break
                try:                                                     # atomic write: no half-written files
                    tmp = out_dir / f".{fname}.tmp"
                    with open(tmp, "wb") as fh_:
                        np.savez(fh_, lr=lr_arr.astype(np.float16), hr=hr_arr.astype(np.float16))
                    os.replace(tmp, out_dir / fname)
                except Exception:
                    state.release()
                    raise
                left, top = tf.c + c * tf.a, tf.f + r * tf.e
                with wlock:
                    writer.writerow([fname, region, scene, naip.datetime.date(), s2.datetime.date(),
                                     round(left, 1), round(top, 1), round(veg, 3)])
                kept += 1
                state.commit()
            except Exception as e:
                tqdm.write(f"    tile ({r},{c}) skipped: {e}")
    return kept


def summarize(out):
    files = [p.name for p in Path(out).glob("*.npz")]
    per = Counter(f.split("__")[0] for f in files)
    print("\n" + "=" * 60)
    print(f"  TOTAL RAW PAIRS (HR-LR) DOWNLOADED : {len(files):,}")
    print("=" * 60)
    for k, v in sorted(per.items(), key=lambda kv: -kv[1]):
        lc = REGIONS.get(k, (None, ""))[1]
        print(f"  {k:28s} {v:6,d}   {lc}")
    (Path(out) / "raw_summary.json").write_text(json.dumps({"total_raw_pairs": len(files), "per_region": per}, indent=1))
    return len(files)


def check_deps():
    """Fail fast with the exact install command instead of 20 identical search errors."""
    import importlib
    need = {"pystac_client": "pystac-client", "planetary_computer": "planetary-computer", "rasterio": "rasterio",
            "pyproj": "pyproj", "tqdm": "tqdm"}
    missing = []
    for mod, pkg in need.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        sys.exit("Missing Python packages: " + ", ".join(missing) + "\n\nInstall them into the SAME Python you run this "
                 "script with:\n  " + f'"{sys.executable}" -m pip install ' + " ".join(missing) +
                 "\n(or everything at once:  python -m pip install -r requirements.txt)")


def main():
    a = get_args()
    if a.list_regions:
        for k, (bb, lc) in REGIONS.items():
            print(f"{k:28s} {bb}  {lc}")
        return
    check_deps()
    GATE.min_interval = a.request_interval
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "download_config.json").write_text(json.dumps(vars(a), indent=2))

    if a.bbox:
        regions = {"custom": (tuple(a.bbox), "user bbox")}
    else:
        names = list(REGIONS) if a.regions == ["all"] else a.regions
        bad = [n for n in names if n not in REGIONS]
        if bad:
            sys.exit(f"unknown region(s): {bad}. Use --list-regions")
        regions = {n: REGIONS[n] for n in names}

    # ---- 1. collect NAIP candidate scenes for every region (fast: metadata only)
    print(f"Searching NAIP scenes in {len(regions)} region(s) ...")
    queues, search_errors = {}, []
    for name, (bb, _) in regions.items():
        try:
            queues[name] = find_naip(name, bb, a)
        except ImportError:
            raise
        except Exception as e:
            print(f"  {name}: search failed ({type(e).__name__}: {e})"); queues[name] = []; search_errors.append(e)
        print(f"  {name:28s} {len(queues[name]):4d} candidate scenes")
    # round-robin interleave => all regions grow together
    jobs = []
    for i in range(max((len(q) for q in queues.values()), default=0)):
        for name, q in queues.items():
            if i < len(q):
                jobs.append((name, q[i]))
    if not jobs:
        if search_errors:
            sys.exit(f"All {len(search_errors)} STAC searches failed (last error: {search_errors[-1]!r}). "
                     "Check your internet connection / proxy / firewall (https://planetarycomputer.microsoft.com must be reachable).")
        sys.exit("No NAIP scenes found for these regions/dates - NAIP only covers the USA; adjust --regions / --start / --end.")

    existing = len(list(out.glob("*.npz")))
    if existing:
        print(f"Resuming: {existing:,} pairs already in {out}")
    t0 = time.time()
    pbar = tqdm(total=a.target_raw, initial=existing, unit="pair", desc="raw pairs", smoothing=0.05,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed {elapsed} | remaining {remaining} | {rate_fmt}]{postfix}")
    state = State(a.target_raw, existing, pbar)
    wlock = threading.Lock()
    new_manifest = not (out / "manifest.csv").exists()
    fh = open(out / "manifest.csv", "a", newline="")
    writer = csv.writer(fh)
    if new_manifest:
        writer.writerow(["file", "region", "scene", "naip_date", "s2_date", "x_left", "y_top", "vegetation_frac"])

    def job(region, naip):
        if state.reached():
            return
        try:
            s2 = find_s2(naip, a)
            if s2 is None:
                state.record_success()
                return
            n = retry(process_scene, region, naip, s2, a, out, writer, wlock, state, what=f"scene {region}", tries=4)
            fh.flush()
            state.record_success()
            if n:
                tqdm.write(f"  {region:28s} {naip.id[-22:]}  +{n} pairs (S2 {s2.datetime.date()})")
        except Exception as e:
            log_failure(out, region, naip.id, e)
            state.record_failure(f"{type(e).__name__}: {e}")
        finally:
            state.scene_finished(len(jobs))

    ex = ThreadPoolExecutor(max_workers=a.workers)
    try:
        futs = [ex.submit(job, r, n) for r, n in jobs]
        for _ in as_completed(futs):
            if state.reached():
                break
    except KeyboardInterrupt:
        print("\nInterrupted - partial data kept; re-run the same command to resume.")
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
        pbar.close(); fh.close()
    print(f"Elapsed: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))}")
    n = summarize(out)
    if state.aborted:
        print(f"\n[STOPPED] {state.max_streak} scenes failed in a row. Last error: {state.last_error[:300]}\n"
              f"Full tracebacks: {out / 'download_errors.log'}\n"
              "If this is 'rate limited / blocked': wait 15-30 min, then re-run the SAME command with --workers 3 "
              "(progress is kept). Otherwise send the last traceback from the log file.")
        sys.exit(1)
    if n < a.target_raw:
        print(f"\n[note] only {n:,} of the requested {a.target_raw:,} pairs were found. Increase --scenes-per-region, "
              f"--date-window or --max-s2-cloud, widen --start/--end, or lower --stride for more tiles per scene.")


if __name__ == "__main__":
    main()
