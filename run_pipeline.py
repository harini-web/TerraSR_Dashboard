#!/usr/bin/env python3
"""One command: download -> clean -> train -> evaluate.

  python run_pipeline.py --target-raw 20000 --workers 6 --epochs 200 --amp
  python run_pipeline.py --skip-download                 # reuse data/raw
  python run_pipeline.py --skip-download --skip-clean    # only train + evaluate
"""
import argparse
import subprocess
import sys


def run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], check=True)


p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="data")
p.add_argument("--run-dir", default="runs/exp1")
p.add_argument("--regions", nargs="+", default=["all"])
p.add_argument("--target-raw", type=int, default=20000)
p.add_argument("--workers", type=int, default=6, help="parallel download scenes")
p.add_argument("--min-vegetation", type=float, default=0.0)
p.add_argument("--model", default="resnet", choices=["resnet", "srcnn"])
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--holdout-regions", nargs="+", default=[], help="regions kept out of training for test_unseen")
p.add_argument("--amp", action="store_true")
p.add_argument("--skip-download", action="store_true")
p.add_argument("--skip-clean", action="store_true")
a = p.parse_args()

raw, clean = f"{a.data_dir}/raw", f"{a.data_dir}/clean"
if not a.skip_download:
    run(["download_data.py", "--out", raw, "--regions", *a.regions, "--target-raw", str(a.target_raw),
         "--workers", str(a.workers), "--min-vegetation", str(a.min_vegetation)])
if not a.skip_clean:
    run(["clean_data.py", "--raw", raw, "--out", clean, *(["--holdout-regions", *a.holdout_regions] if a.holdout_regions else [])])
run(["train.py", "--data", clean, "--out", a.run_dir, "--model", a.model, "--epochs", str(a.epochs),
     "--batch-size", str(a.batch_size), *(["--amp"] if a.amp else [])])
run(["evaluate.py", "--ckpt", f"{a.run_dir}/best.pt", "--data", clean, "--split", "test"])
if a.holdout_regions:
    run(["evaluate.py", "--ckpt", f"{a.run_dir}/best.pt", "--data", clean, "--split", "test_unseen"])
