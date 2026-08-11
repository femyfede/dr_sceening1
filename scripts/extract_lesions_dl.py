"""Replace heuristic lesion biomarkers (LA/HA/EA/MAC) with deep-learning segmentation.

Runs the pretrained fundus-lesions-toolkit U-Net (weights trained on
IDRiD+MESSIDOR+DDR+FGADR+RETLES) on every matched APTOS image.

Class map: 0=BG, 1=CTW, 2=EX, 3=HE, 4=MA
  LA  = union pixels of {CTW, EX, HE, MA}
  HA  = HE pixels
  EA  = EX + CTW pixels
  MAC = number of MA connected components with size in [3, 60]

Resumable: appends to outputs/lesions_dl_partial.csv, skips already-done ids.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import cv2
import torch

from fundus_lesions_toolkit.models.segmentation import segment
from fundus_lesions_toolkit.constants import Dataset

import biomarkers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dr_folder1", "data", "aptos2019")
OUT_DIR = os.path.join(ROOT, "outputs")
MASK_DIR = os.path.join(OUT_DIR, "lesion_masks_dl")
PARTIAL = os.path.join(OUT_DIR, "lesions_dl_partial.csv")

MIN_MA_SIZE = 3
MAX_MA_SIZE = 60

LABEL_CTW, LABEL_EX, LABEL_HE, LABEL_MA = 1, 2, 3, 4

LESION_FEATURES = [
    "LA", "HA", "EA", "MAC",
    "HE_COUNT", "EX_COUNT", "CTW_A", "MA_A", "LA_RET", "HA_RET", "EA_RET",
]

# Color composite (BGR for cv2.imwrite): CTW=purple, EX=yellow, HE=red, MA=green
COLORS = {
    LABEL_CTW: (255, 0, 255),
    LABEL_EX: (0, 255, 255),
    LABEL_HE: (0, 0, 255),
    LABEL_MA: (0, 255, 0),
}


def image_path(image_id, split):
    return os.path.join(DATA, f"{split}_images", f"{image_id}.png")


def load_images():
    with open(os.path.join(OUT_DIR, "biomarkers_all.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    return [(r["image_id"], r["split"]) for r in rows]


def load_done():
    done = set()
    if os.path.exists(PARTIAL):
        with open(PARTIAL, newline="") as f:
            for r in csv.DictReader(f):
                done.add(r["image_id"])
    return done


def retina_area(rgb_image):
    """Retina mask area (px) at the image's own resolution."""
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return float(biomarkers.retina_mask_from_original(bgr).sum())


def features_from_labels(labels, retina_area_px):
    """All DL lesion features from a per-pixel class map (0=BG,1=CTW,2=EX,3=HE,4=MA)."""
    hm = labels == LABEL_HE
    ex = labels == LABEL_EX
    ctw = labels == LABEL_CTW
    ma = labels == LABEL_MA
    la = hm | ex | ctw | ma

    def n_comp(m):
        n, comps = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        if n <= 1:
            return 0
        sizes = np.bincount(comps.ravel())
        sizes[0] = 0
        return int((sizes >= MIN_MA_SIZE).sum())

    n_ma, ma_comps = cv2.connectedComponents(ma.astype(np.uint8), connectivity=8)
    mac = 0
    if n_ma > 1:
        sizes = np.bincount(ma_comps.ravel())
        sizes[0] = 0
        mac = int(((sizes >= MIN_MA_SIZE) & (sizes <= MAX_MA_SIZE)).sum())

    la_a = int(la.sum())
    ha_a = int(hm.sum())
    ea_a = int((ex | ctw).sum())
    denom = retina_area_px if retina_area_px and retina_area_px > 0 else np.nan
    scale = 100.0 / denom if denom == denom and denom > 0 else np.nan

    return {
        "LA": la_a, "HA": ha_a, "EA": ea_a, "MAC": mac,
        "HE_COUNT": n_comp(hm), "EX_COUNT": n_comp(ex),
        "CTW_A": int(ctw.sum()), "MA_A": int(ma.sum()),
        "LA_RET": scale * la_a if scale == scale else np.nan,
        "HA_RET": scale * ha_a if scale == scale else np.nan,
        "EA_RET": scale * ea_a if scale == scale else np.nan,
    }


def decode_labels_from_mask(bgr_comp):
    """Rebuild the per-pixel class map from a saved color-composite mask (BGR)."""
    B = bgr_comp[:, :, 0].astype(int)
    G = bgr_comp[:, :, 1].astype(int)
    R = bgr_comp[:, :, 2].astype(int)
    labels = np.zeros(bgr_comp.shape[:2], dtype=np.uint8)
    labels[(R == 255) & (G == 0) & (B == 255)] = LABEL_CTW
    labels[(R == 255) & (G == 255) & (B == 0)] = LABEL_EX
    labels[(R == 255) & (G == 0) & (B == 0)] = LABEL_HE
    labels[(R == 0) & (G == 255) & (B == 0)] = LABEL_MA
    return labels


def compute_lesion_features(rgb_image):
    """DL lesion biomarkers from one RGB fundus image (any size).

    Returns dict with LA/HA/EA/MAC plus extended features and '_labels'.
    """
    pred = segment(rgb_image, train_datasets=Dataset.ALL, device="cpu", image_resolution=1024)
    labels = pred.argmax(0).cpu().numpy().astype(np.uint8)

    feats = features_from_labels(labels, retina_area(rgb_image))
    feats["_labels"] = labels
    return feats


def process_one(image_id, split):
    path = image_path(image_id, split)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return image_id, {"error": f"cannot read {path}"}
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = compute_lesion_features(img)
    labels = res.pop("_labels")

    comp = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for lbl, color in COLORS.items():
        comp[labels == lbl] = color
    os.makedirs(MASK_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(MASK_DIR, f"{image_id}.png"), comp)

    return image_id, res


def worker(args):
    image_id, split = args
    t0 = time.time()
    image_id, res = process_one(image_id, split)
    return image_id, res, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    todo = [(i, s) for i, s in load_images() if i not in load_done()]
    print(f"total={len(load_images())} done={len(load_images()) - len(todo)} todo={len(todo)}", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    write_header = not os.path.exists(PARTIAL) or os.path.getsize(PARTIAL) == 0
    f = open(PARTIAL, "a", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["image_id"] + LESION_FEATURES)

    start = time.time()
    finished = 0
    try:
        if args.workers == 1:
            iterable = map(worker, todo)
        else:
            from multiprocessing import Pool
            iterable = Pool(args.workers, maxtasksperchild=8).imap_unordered(worker, todo, chunksize=1)
        for image_id, res, dt in iterable:
            if "error" in res:
                print(f"[{finished}] {image_id} ERROR: {res['error']}", flush=True)
                continue
            writer.writerow([image_id] + [res[f] for f in LESION_FEATURES])
            f.flush()
            finished += 1
            if finished % 10 == 0 or finished == len(todo):
                el = time.time() - start
                rate = el / finished
                print(f"[{finished}/{len(todo)}] {image_id} "
                      f"(HA={res['HA']} EA={res['EA']} MAC={res['MAC']}) "
                      f"elapsed={el/60:.1f}m ETA={(len(todo)-finished)*rate/60:.1f}m", flush=True)
    finally:
        f.close()

    print(f"done: {finished}/{len(todo)} in {(time.time()-start)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
