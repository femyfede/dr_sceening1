"""Parallel, resumable feature extraction for the combined Kaggle dataset.

Usage:
    python outputs_multidataset/run_extract.py --workers 3            # full run
    python outputs_multidataset/run_extract.py --workers 3 --limit 12 # test slice
    python outputs_multidataset/run_extract.py --merge                # merge worker files only

Each worker writes outputs_multidataset/features_worker{rank}.csv and skips
already-done ids, so interrupted runs resume automatically. After all workers
finish, run --merge to produce features.csv.
"""
import os
import sys
import time
import glob
import argparse
import multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "outputs_multidataset", "multidataset"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools", "rrwnet"))

import pandas as pd
import datasets as DS
import pipeline

CFG = os.path.join(ROOT, "outputs_multidataset", "multidataset", "config.yaml")
OUT_DIR = os.path.join(ROOT, "outputs_multidataset")
MANIFEST = os.path.join(OUT_DIR, "manifest.csv")


def write_row_retry(row_df, path, header, max_attempts=15):
    """Append one row, retrying transient file locks (OneDrive sync)."""
    for attempt in range(max_attempts):
        try:
            row_df.to_csv(path, mode="a", header=header, index=False)
            return True
        except OSError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return False


def _load_cfg():
    cfg = DS.load_config(CFG)
    return cfg


def _worker(rank, workers, limit, only_missing=False):
    import numpy as np
    cfg = _load_cfg()
    device = pipeline.pick_device(cfg.get("device", "auto"))
    model = pipeline.load_rrwnet(cfg, device)
    part = os.path.join(OUT_DIR, f"features_worker{rank}.csv")

    done = set()
    if os.path.exists(part):
        try:
            done = set(pd.read_csv(part, usecols=["image_id"])["image_id"].astype(str))
        except Exception:
            done = set()

    man = DS.load_manifest(MANIFEST)
    if only_missing:
        for f in glob.glob(os.path.join(OUT_DIR, "features_worker*.csv")) + \
                 glob.glob(os.path.join(OUT_DIR, "features_failed_worker*.csv")):
            try:
                done |= set(pd.read_csv(f, usecols=["image_id"])["image_id"].astype(str))
            except Exception:
                pass
        shard = man[~man["image_id"].isin(done)]
        todo = shard.iloc[rank::workers]
    else:
        shard = man.iloc[rank::workers]
        todo = shard[~shard["image_id"].isin(done)]
    if limit:
        todo = todo.head(limit)

    fail_part = os.path.join(OUT_DIR, f"features_failed_worker{rank}.csv")
    header = not os.path.exists(part) or os.path.getsize(part) == 0
    n = 0
    nfail = 0
    t0 = time.time()
    for row in todo.itertuples(index=False):
        try:
            res, err = pipeline.process_one(cfg, model, row._asdict())
        except Exception as e:
            res, err = None, f"{type(e).__name__}: {e}"
        if err:
            nfail += 1
            fdf = pd.DataFrame([{"image_id": row.image_id, "path": row.path,
                                 "split": row.split, "label_6": row.label_6,
                                 "error": err}])
            try:
                write_row_retry(fdf, fail_part, header=not os.path.exists(fail_part))
            except OSError as e:
                print(f"[w{rank}] failed-log write error {e}; will retry next run", flush=True)
            continue
        row_df = pd.DataFrame([res])
        try:
            write_row_retry(row_df, part, header=header)
        except OSError as e:
            print(f"[w{rank}] write error {e} on {row.image_id}; will retry next run", flush=True)
            continue
        header = False
        n += 1
        if n % 20 == 0:
            el = (time.time() - t0) / 60
            print(f"[w{rank}] {n}/{len(todo)} {row.image_id} elapsed={el:.1f}m", flush=True)
    print(f"[w{rank}] done {n}/{len(todo)} (failed {nfail})", flush=True)


def merge():
    parts = [os.path.join(OUT_DIR, f"features_worker{i}.csv")
             for i in range(32) if os.path.exists(os.path.join(OUT_DIR, f"features_worker{i}.csv"))]
    if not parts:
        print("no worker files found")
        return
    dfs = [pd.read_csv(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    df["image_id"] = df["image_id"].astype(str)
    df = df.drop_duplicates(subset=["image_id"], keep="last")
    df.to_csv(os.path.join(OUT_DIR, "features.csv"), index=False)
    if "split" in df.columns:
        tr = df[df["split"] == "train"]
        te = df[df["split"] == "test"]
        tr.drop(columns=["split"], errors="ignore").to_csv(
            os.path.join(OUT_DIR, "features_train.csv"), index=False)
        te.drop(columns=["split"], errors="ignore").to_csv(
            os.path.join(OUT_DIR, "features_test.csv"), index=False)
        print(f"train/test split files -> {len(tr)} / {len(te)}")
    print(f"merged {len(parts)} worker files -> features.csv ({len(df)} rows)")

    fparts = [os.path.join(OUT_DIR, f"features_failed_worker{i}.csv")
              for i in range(32) if os.path.exists(os.path.join(OUT_DIR, f"features_failed_worker{i}.csv"))]
    if fparts:
        fdf = pd.concat([pd.read_csv(p) for p in fparts], ignore_index=True)
        fdf = fdf.drop_duplicates(subset=["image_id"], keep="last")
        fdf.to_csv(os.path.join(OUT_DIR, "features_failed.csv"), index=False)
        print(f"failed images: {len(fdf)} -> features_failed.csv")
        print(fdf["error"].value_counts().to_string())
    else:
        print("failed images: 0")
    print(df.groupby(["split", "label_6"]).size().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--only-missing", action="store_true")
    args = ap.parse_args()

    if args.merge:
        merge()
        sys.exit(0)

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker,
                         args=(i, args.workers, args.limit, args.only_missing))
             for i in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("all workers finished")
