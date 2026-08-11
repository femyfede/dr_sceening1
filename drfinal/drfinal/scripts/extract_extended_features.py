"""Rebuild biomarker CSVs with the extended feature set.

For every matched (original, RRWNet A/V segmentation) image:
  - vascular + classical-lesion features come from biomarkers.extract_all
  - DL lesion features (incl. counts, CTW area, MA area, retina-normalized
    densities) are decoded from the already-saved U-Net masks in
    outputs/lesion_masks_dl at the image's own resolution.

Writes outputs/biomarkers_train.csv / test.csv / all.csv.
Resumable via outputs/biomarkers_ext_partial.csv.
"""

import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

import biomarkers
import train_models as tm
from extract_lesions_dl import (decode_labels_from_mask, features_from_labels,
                                retina_area)

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(SCRIPTS, "..", "dr_folder1", "data", "aptos2019")
OUT = os.path.join(SCRIPTS, "..", "outputs")
PARTIAL = os.path.join(OUT, "biomarkers_ext_partial.csv")
MASK_DIR = os.path.join(OUT, "lesion_masks_dl")

SPLIT_DIRS = {"train": "train_images", "test": "test_images"}
AV_DIRS = {"train": "rrwnet/train_images", "test": "rrwnet/test_images"}

COLS = ["image_id", "split", "retina_area_px"] + tm.FEATURES


def process_one(image_id, split):
    org = cv2.imread(os.path.join(DATA, SPLIT_DIRS[split], image_id + ".png"))
    av = cv2.imread(os.path.join(DATA, AV_DIRS[split], image_id + ".png"))
    if org is None or av is None:
        return None
    res = biomarkers.extract_all(org, av)

    mask = cv2.imread(os.path.join(MASK_DIR, image_id + ".png"))
    if mask is None:
        return None
    labels = decode_labels_from_mask(mask)
    les = features_from_labels(labels, retina_area(cv2.cvtColor(org, cv2.COLOR_BGR2RGB)))

    row = {"image_id": image_id, "split": split}
    for k, v in res.items():
        if k.startswith("_"):
            continue
        row[k] = v
    for f in les:
        row[f] = les[f]
    return row


def process_batch(ids_and_splits):
    rows = []
    for image_id, split in ids_and_splits:
        try:
            r = process_one(image_id, split)
            if r is not None:
                rows.append(r)
        except Exception as e:
            sys.stderr.write(f"FAILED {split}/{image_id}: {e}\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    # authoritative list comes from the existing biomarkers_all.csv (matched ids)
    base = pd.read_csv(os.path.join(OUT, "biomarkers_all.csv"), usecols=["image_id", "split"])
    base["image_id"] = base["image_id"].astype(str)
    jobs = list(base.itertuples(index=False, name=None))

    done_ids = set()
    if os.path.exists(PARTIAL):
        try:
            done_ids = set(pd.read_csv(PARTIAL, usecols=["image_id"])["image_id"].astype(str))
        except Exception:
            done_ids = set()
    todo = [j for j in jobs if j[0] not in done_ids]
    print(f"Total matched: {len(jobs)}, done: {len(jobs) - len(todo)}, to do: {len(todo)}")

    header_needed = not os.path.exists(PARTIAL) or os.path.getsize(PARTIAL) == 0
    if args.workers > 1 and len(todo) > 0:
        chunksize = max(1, 8 * args.workers)
        chunks = [todo[i:i + chunksize] for i in range(0, len(todo), chunksize)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, chunk in enumerate(ex.map(process_batch, chunks)):
                rows = [r for r in chunk if r is not None]
                if rows:
                    pd.DataFrame(rows).to_csv(PARTIAL, mode="a",
                                              header=header_needed, index=False)
                    header_needed = False
                done = min(len(chunks), i + 1)
                if done % 5 == 0 or done == len(chunks):
                    print(f"  processed {done}/{len(chunks)} chunks", flush=True)
    else:
        rows = [r for r in process_batch(todo) if r is not None]
        if rows:
            pd.DataFrame(rows).to_csv(PARTIAL, mode="a", header=header_needed, index=False)

    if not os.path.exists(PARTIAL):
        print("No rows produced.")
        return

    df = pd.read_csv(PARTIAL)
    df["image_id"] = df["image_id"].astype(str)
    df = df.drop_duplicates(subset=["image_id"], keep="last")
    for c in COLS[2:]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[COLS]

    df_train = df[df["split"] == "train"].drop(columns=["split"])
    df_test = df[df["split"] == "test"].drop(columns=["split"])
    df_train.to_csv(os.path.join(OUT, "biomarkers_train.csv"), index=False)
    df_test.to_csv(os.path.join(OUT, "biomarkers_test.csv"), index=False)
    df.to_csv(os.path.join(OUT, "biomarkers_all.csv"), index=False)

    print(f"\nSaved biomarkers_train.csv ({len(df_train)}), "
          f"biomarkers_test.csv ({len(df_test)}), biomarkers_all.csv ({len(df)})")
    print("Missing fractions:\n", df[tm.FEATURES].isna().mean().sort_values(ascending=False).head(8))


if __name__ == "__main__":
    main()
