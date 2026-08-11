"""Recompute retina area + density columns (retina_area_px, VD, AD, VeD) using the
hybrid retina area = max(resized original retina, vessel-dilated estimate).

Run after biomarkers.csv extraction to correct images whose original retina mask is
unreliable (dark/low-contrast originals).
"""

import os
import time
import sys
import cv2
import pandas as pd
import numpy as np
from scipy import ndimage

import biomarkers

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(SCRIPTS, "..", "dr_folder1", "data", "aptos2019")
OUT = os.path.join(SCRIPTS, "..", "outputs")

SPLIT_DIRS = {"train": "train_images", "test": "test_images"}
AV_DIRS = {"train": "rrwnet/train_images", "test": "rrwnet/test_images"}


def density_for(image_id, split):
    org = cv2.imread(os.path.join(DATA, SPLIT_DIRS[split], image_id + ".png"))
    av = cv2.imread(os.path.join(DATA, AV_DIRS[split], image_id + ".png"))
    if org is None or av is None:
        return None
    artery, vein = biomarkers.get_vessel_masks(av)
    vessel = artery | vein
    retina_orig = biomarkers.retina_mask_from_original(org)
    ret_resized = cv2.resize(retina_orig.astype(np.uint8), (av.shape[1], av.shape[0]),
                             interpolation=cv2.INTER_NEAREST) > 0
    retina_area = max(float(ret_resized.sum()), biomarkers.vessel_retina_area(vessel))
    return {
        "image_id": image_id,
        "retina_area_px": retina_area,
        "VD": 100.0 * float(vessel.sum()) / retina_area if retina_area > 0 else np.nan,
        "AD": 100.0 * float(artery.sum()) / retina_area if retina_area > 0 else np.nan,
        "VeD": 100.0 * float(vein.sum()) / retina_area if retina_area > 0 else np.nan,
    }


def main():
    df = pd.read_csv(os.path.join(OUT, "biomarkers_all.csv"))
    df["image_id"] = df["image_id"].astype(str)
    cols = ["retina_area_px", "VD", "AD", "VeD"]
    t0 = time.time()
    n = len(df)
    for i, r in df.iterrows():
        try:
            upd = density_for(r["image_id"], r["split"])
            if upd:
                for c in cols:
                    df.at[i, c] = upd[c]
        except Exception as e:
            sys.stderr.write(f"FAILED {r['image_id']}: {e}\n")
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)

    df.to_csv(os.path.join(OUT, "biomarkers_all.csv"), index=False)
    df[df["split"] == "train"].drop(columns=["split"]).to_csv(
        os.path.join(OUT, "biomarkers_train.csv"), index=False)
    df[df["split"] == "test"].drop(columns=["split"]).to_csv(
        os.path.join(OUT, "biomarkers_test.csv"), index=False)
    print(f"Done in {time.time() - t0:.0f}s")
    print(df[cols].describe().round(2).to_string())


if __name__ == "__main__":
    main()
