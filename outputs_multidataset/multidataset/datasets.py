"""Dataset manifest handling for the multi-dataset DR pipeline.

Builds a single table with a unified 0-5 label:
    0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative, 5 Other.

Input CSV layouts are dataset-specific (Kaggle). Fill the `datasets` and
`label_mapping` sections in config.yaml, then either:

    python multidataset/datasets.py                # build manifest CSV from config
    python multidataset/datasets.py --csv path.csv # import an existing manifest
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import yaml

MANIFEST_COLS = ["image_id", "path", "dataset", "label_native", "split", "label_6"]


def _resolve(cfg_dir, p):
    if p is None:
        return p
    return os.path.abspath(os.path.join(cfg_dir, p)) if not os.path.isabs(p) else p


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(os.path.abspath(path))
    for key in ("project_scripts", "rrwnet_module_dir", "rrwnet_weights", "outputs"):
        if key in cfg:
            cfg[key] = _resolve(cfg_dir, cfg[key])
    for name, spec in cfg.get("datasets", {}).items():
        for key in ("images_dir", "labels_csv"):
            if key in spec:
                spec[key] = _resolve(cfg_dir, spec[key])
    return cfg


def build_manifest(cfg):
    rows = []
    for name, spec in cfg.get("datasets", {}).items():
        images_dir = spec.get("images_dir")
        labels_csv = spec.get("labels_csv")
        if not images_dir or not labels_csv:
            print(f"  [skip] {name}: no images_dir/labels_csv configured")
            continue
        if not os.path.isdir(images_dir):
            print(f"  [skip] {name}: images dir not found: {images_dir}")
            continue
        if not os.path.isfile(labels_csv):
            print(f"  [skip] {name}: labels csv not found: {labels_csv}")
            continue
        id_col, label_col = spec["id_col"], spec["label_col"]
        ext = spec.get("ext", ".png")
        lab = pd.read_csv(labels_csv)
        id_col = id_col if id_col in lab.columns else lab.columns[0]
        label_col = label_col if label_col in lab.columns else lab.columns[-1]
        n0 = len(lab)
        lab = lab.dropna(subset=[id_col, label_col])
        ids = lab[id_col].astype(str).str.replace(r"\.(png|jpg|jpeg)$", "", regex=True)
        paths = [os.path.join(images_dir, i + ext) for i in ids]
        exist = [os.path.isfile(p) for p in paths]
        print(f"  {name}: {n0} labels, {int(np.sum(exist))} images found on disk")
        lab = lab[np.asarray(exist)]
        rows.append(pd.DataFrame({
            "image_id": ids[np.asarray(exist)],
            "path": np.asarray(paths)[np.asarray(exist)],
            "dataset": name,
            "label_native": lab[label_col].astype(int).to_numpy(),
            "split": spec.get("split", "train"),
        }))

    if not rows:
        print("No datasets produced any rows. Check config paths.")
        return pd.DataFrame(columns=MANIFEST_COLS)

    m = pd.concat(rows, ignore_index=True)
    mapping = cfg.get("label_mapping", {})
    out = []
    for r in m.itertuples(index=False):
        ds_map = mapping.get(r.dataset, {})
        if r.label_native not in ds_map:
            print(f"  [warn] {r.dataset} label {r.label_native} has no mapping; dropped")
            continue
        out.append({**r._asdict(), "label_6": ds_map[r.label_native]})
    return pd.DataFrame(out, columns=MANIFEST_COLS)


def load_manifest(path):
    m = pd.read_csv(path)
    for c in MANIFEST_COLS:
        if c not in m.columns:
            raise ValueError(f"manifest missing column '{c}'; found {list(m.columns)}")
    m["image_id"] = m["image_id"].astype(str)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--csv", default=None, help="import an existing manifest CSV instead of building")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.csv:
        m = load_manifest(args.csv)
    else:
        m = build_manifest(cfg)
    if m.empty:
        sys.exit("Manifest is empty.")

    out = args.out or os.path.join(cfg["outputs"], "manifest.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    m.to_csv(out, index=False)
    print(f"\nManifest saved -> {out}  ({len(m)} rows)")
    print("Class distribution (0-5):")
    print(m["label_6"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
