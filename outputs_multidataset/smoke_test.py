import os, sys, time
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "outputs_multidataset", "multidataset"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools", "rrwnet"))

import yaml
import datasets as DS
import pipeline

with open(os.path.join(ROOT, "outputs_multidataset", "multidataset", "config.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg_dir = os.path.join(ROOT, "outputs_multidataset", "multidataset")
for key in ("project_scripts", "rrwnet_module_dir", "rrwnet_weights", "outputs"):
    p = cfg.get(key)
    if p:
        cfg[key] = os.path.abspath(os.path.join(cfg_dir, p))

pipeline.ensure_paths(cfg)
device = pipeline.pick_device(cfg["device"])
print("device:", device)
model = pipeline.load_rrwnet(cfg, device)

img_dir = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub", "datasets",
                       "sehastrajits", "fundus-aptosddridirdeyepacsmessidor",
                       "versions", "1", "split_dataset", "train", "0")
img = os.listdir(img_dir)[0]
row = {"image_id": "smoke0", "dataset": "combined", "split": "train",
       "label_6": 0, "path": os.path.join(img_dir, img)}
t0 = time.time()
res, err = pipeline.process_one(cfg, model, row)
print("elapsed:", round(time.time() - t0, 2), "s, err:", err)
if res:
    feats = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k not in ("image_id", "dataset", "split", "label_6")}
    print("features:", feats)
