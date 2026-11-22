#!/usr/bin/env python3
"""
Fetch a Sentinel-2 L2A RGB crop (any place on Earth) as a 3-band uint16 GeoTIFF, ready for predict.py.

  python fetch_s2_rgb.py --bbox 14.25 46.60 14.40 46.70 --dates 2023-06-01/2023-08-31 --out klagenfurt.tif
  python predict.py --ckpt runs/exp1/best.pt --input klagenfurt.tif --compare
"""
import argparse
import math

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.windows import Window, from_bounds

from download_data import STAC_URL, band_scale_offset

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("W", "S", "E", "N"))
p.add_argument("--dates", required=True, help="e.g. 2023-06-01/2023-08-31")
p.add_argument("--max-cloud", type=float, default=10.0)
p.add_argument("--out", default="s2_rgb.tif")
a = p.parse_args()

cat = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
items = list(cat.search(collections=["sentinel-2-l2a"], bbox=a.bbox, datetime=a.dates,
                        query={"eo:cloud_cover": {"lt": a.max_cloud}}).items())
if not items:
    raise SystemExit("No scene found - widen --dates or --max-cloud")
item = pc.sign(min(items, key=lambda i: i.properties["eo:cloud_cover"]))
print(f"using {item.id}  ({item.datetime.date()}, cloud {item.properties['eo:cloud_cover']:.1f} %)")
gain, offset = band_scale_offset(item)

with rasterio.open(item.assets["B04"].href) as r0:
    l, b, r, t = Transformer.from_crs("EPSG:4326", r0.crs, always_xy=True).transform_bounds(*a.bbox)
    w = from_bounds(l, b, r, t, transform=r0.transform)
    win = Window(int(round(w.col_off)), int(round(w.row_off)), int(round(w.width)), int(round(w.height)))
    prof = dict(driver="GTiff", dtype="uint16", count=3, crs=r0.crs, width=win.width, height=win.height,
                transform=r0.window_transform(win), compress="deflate")
data = []
for band in ["B04", "B03", "B02"]:
    with rasterio.open(item.assets[band].href) as s:
        data.append(s.read(1, window=win))
with rasterio.open(a.out, "w", **prof) as o:
    o.write(np.stack(data))
    o.update_tags(S2_REFLECTANCE_OFFSET=offset, S2_SCENE=item.id, S2_DATE=str(item.datetime.date()))
print(f"saved {a.out}: {win.width}x{win.height} px at 10 m, reflectance offset {offset}")
