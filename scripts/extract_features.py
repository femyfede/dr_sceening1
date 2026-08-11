"""Extract all biomarkers for matched (original + A/V segmentation) images.

Reads original fundus images from data/aptos2019/{train,test}_images and the
RRWNet artery-vein segmentations from data/aptos2019/rrwnet/{train,test}_images.

Outputs:
    outputs/biomarkers_train.csv
    outputs/biomarkers_test.csv
    outputs/biomarkers_all.csv          (train+test stacked, with split column)
    outputs/lesion_masks/<image_id>.png (color-coded lesion masks)
"""

import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
import cv2
import pandas as pd

import biomarkers

DATA = os.path.join(os.path.dirname(__file__), "..", "dr_folder1", "data", "aptos2019")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
PARTIAL = os.path.join(OUT, "biomarkers_partial.csv")

SPLIT_DIRS = {"train": "train_images", "test": "test_images"}
AV_DIRS = {"train": "rrwnet/train_images", "test": "rrwnet/test_images"}

BIOMARKER_COLS = [
    "image_id", "split", "retina_area_px", "AVR", "CRAE", "CRVE",
    "VD", "AD", "VeD", "TI", "CI", "FD", "LA", "HA", "EA", "MAC",
]


def find_matched(split):
    orig = set(f[:-4] for f in os.listdir(os.path.join(DATA, SPLIT_DIRS[split])))
    av = set(f[:-4] for f in os.listdir(os.path.join(DATA, AV_DIRS[split])))
    return sorted(orig & av)


def process_one(image_id, split):
    orig_path = os.path.join(DATA, SPLIT_DIRS[split], image_id + ".png")
    av_path = os.path.join(DATA, AV_DIRS[split], image_id + ".png")
    original = cv2.imread(orig_path)
    av = cv2.imread(av_path)
    if original is None or av is None:
        return None
    res = biomarkers.extract_all(original, av)
    row = {"image_id": image_id, "split": split}
    for k, v in res.items():
        if k.startswith("_"):
            continue
        row[k] = v
    les = res.get("_lesions")
    if les is None:
        les = biomarkers.detect_lesions(original)
    comp = biomarkers.make_lesion_composite(les)
    mask_dir = os.path.join(OUT, "lesion_masks")
    os.makedirs(mask_dir, exist_ok=True)
    cv2.imwrite(os.path.join(mask_dir, image_id + ".png"), comp)
    return row


def process_batch(ids_and_splits):
    out = []
    for image_id, split in ids_and_splits:
        try:
            r = process_one(image_id, split)
            if r is not None:
                out.append(r)
        except Exception as e:
            sys.stderr.write(f"FAILED {split}/{image_id}: {e}\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", choices=["train", "test"], default=None)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for split in ("train", "test"):
        if args.only and split != args.only:
            continue
        for image_id in find_matched(split):
            jobs.append((image_id, split))

    done_ids = set()
    if os.path.exists(PARTIAL):
        try:
            done_ids = set(pd.read_csv(PARTIAL, usecols=["image_id"])["image_id"].astype(str))
        except Exception:
            done_ids = set()
    todo = [(i, s) for i, s in jobs if i not in done_ids]
    print(f"Total matched: {len(jobs)}, already done: {len(jobs) - len(todo)}, to do: {len(todo)}")

    header_needed = not os.path.exists(PARTIAL)
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
                if done % 10 == 0 or done == len(chunks):
                    print(f"  processed {done}/{len(chunks)} chunks")
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
    for c in BIOMARKER_COLS[3:]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[BIOMARKER_COLS]

    df_train = df[df["split"] == "train"].drop(columns=["split"])
    df_test = df[df["split"] == "test"].drop(columns=["split"])

    df_train.to_csv(os.path.join(OUT, "biomarkers_train.csv"), index=False)
    df_test.to_csv(os.path.join(OUT, "biomarkers_test.csv"), index=False)
    df.to_csv(os.path.join(OUT, "biomarkers_all.csv"), index=False)

    print(f"\nSaved:")
    print(f"  {os.path.join(OUT, 'biomarkers_train.csv')}  ({len(df_train)} rows)")
    print(f"  {os.path.join(OUT, 'biomarkers_test.csv')}   ({len(df_test)} rows)")
    print(f"  {os.path.join(OUT, 'biomarkers_all.csv')}    ({len(df)} rows)")
    print(f"\nPreview (first 5 rows, key columns):")
    print(df[["image_id", "split", "AVR", "CRAE", "CRVE", "VD", "AD", "VeD", "TI", "CI", "FD", "LA", "HA", "EA", "MAC"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
