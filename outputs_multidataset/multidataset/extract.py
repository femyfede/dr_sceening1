"""Resumable CUDA feature extraction over the dataset manifest.

Usage:
    python multidataset/extract.py [--config config.yaml] [--manifest outputs/manifest.csv]
                                   [--samples 10]

Runs RRWNet + DL-lesion + vascular biomarkers for every manifest image and
writes outputs/features.csv used by train.py. Skips images already done.
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

import datasets
import pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--samples", type=int, default=0,
                    help="save segmentation/lesion overlays for the first N images")
    ap.add_argument("--only-split", choices=["train", "test"], default=None)
    args = ap.parse_args()

    cfg = datasets.load_config(args.config)
    pipeline.ensure_paths(cfg)
    device = pipeline.pick_device(cfg.get("device", "auto"))
    print(f"device: {device}")

    out_dir = os.path.abspath(cfg["outputs"])
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = args.manifest or os.path.join(out_dir, "manifest.csv")
    if not os.path.isfile(manifest_path):
        manifest_path = os.path.join(out_dir, "manifest.csv")
    man = datasets.load_manifest(manifest_path)
    if args.only_split:
        man = man[man["split"] == args.only_split]
    print(f"manifest rows: {len(man)}")

    partial = os.path.join(out_dir, "features_partial.csv")
    done = set()
    if os.path.exists(partial):
        try:
            done = set(pd.read_csv(partial, usecols=["image_id"])["image_id"].astype(str))
        except Exception:
            done = set()
    todo = man[~man["image_id"].isin(done)]
    print(f"to extract: {len(todo)}  (done: {len(man) - len(todo)})")

    overlay_dir = None
    if args.samples > 0:
        overlay_dir = os.path.join(out_dir, "sample_overlays")
        os.makedirs(overlay_dir, exist_ok=True)

    model = pipeline.load_rrwnet(cfg, device)
    header_needed = not os.path.exists(partial) or os.path.getsize(partial) == 0
    start = time.time()
    n = 0
    for i, row in enumerate(todo.itertuples(index=False)):
        od = overlay_dir if i < args.samples else None
        res, err = pipeline.process_one(cfg, model, row, overlay_dir=od)
        if err:
            print(f"  [err] {row.image_id}: {err}", flush=True)
            continue
        row_df = pd.DataFrame([res])
        row_df.to_csv(partial, mode="a", header=header_needed, index=False)
        header_needed = False
        n += 1
        if n % 25 == 0 or n == len(todo):
            el = (time.time() - start) / 60
            rate = n / max(el, 1e-6)
            eta = (len(todo) - n) / rate if rate > 0 else 0
            print(f"  [{n}/{len(todo)}] {row.image_id} label={res['label_6']} "
                  f"elapsed={el:.1f}m ETA={eta:.1f}m", flush=True)

    if not os.path.exists(partial):
        print("No rows extracted.")
        return

    df = pd.read_csv(partial)
    df["image_id"] = df["image_id"].astype(str)
    df = df.drop_duplicates(subset=["image_id"], keep="last")
    df.to_csv(os.path.join(out_dir, "features.csv"), index=False)
    print(f"\nSaved features -> {os.path.join(out_dir, 'features.csv')} ({len(df)} rows)")
    print("Class distribution (label_6):")
    print(df["label_6"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
