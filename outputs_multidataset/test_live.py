"""Headless test of the interface live path: process a real image through the
same functions the Streamlit app uses and verify the prediction matches the
bundle applied to the precomputed backend features for that image."""
import os
import sys
import tempfile
import yaml
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
MULTIDATASET = ROOT / "multidataset"
CFG_PATH = MULTIDATASET / "config.yaml"
BUNDLE_PATH = ROOT / "model_bundle.joblib"
TRAIN_FEATURES = ROOT / "features_train.csv"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "tools" / "rrwnet"))
sys.path.insert(0, str(MULTIDATASET))

import pipeline
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cfg = yaml.safe_load(open(CFG_PATH, encoding="utf-8"))
cd = os.path.dirname(os.path.abspath(CFG_PATH))
for k in ("project_scripts", "rrwnet_module_dir", "rrwnet_weights", "outputs"):
    p = cfg.get(k)
    if p and not os.path.isabs(p):
        cfg[k] = os.path.abspath(os.path.join(cd, p))
pipeline.ensure_paths(cfg)
device = pipeline.pick_device(cfg.get("device", "auto"))
model = pipeline.load_rrwnet(cfg, device)
bundle = joblib.load(BUNDLE_PATH)
features, classes = bundle["features"], bundle["classes"]

pipe = bundle["pipeline"]
rf = pipe.named_steps["clf"]
imputer = pipe.named_steps["imputer"]
scaler = pipe.named_steps["scaler"]
train = pd.read_csv(TRAIN_FEATURES)
Xbg = scaler.transform(imputer.transform(train[features]))[:100]
explainer = shap.TreeExplainer(rf, Xbg)

# pick a real test image
ft = pd.read_csv(ROOT / "features_test.csv").dropna(subset=["label_6"])
man = pd.read_csv(ROOT / "manifest.csv")
row = ft.sample(1, random_state=7).iloc[0]
row = row.to_dict()
row["path"] = man.loc[man["image_id"] == row["image_id"], "path"].iloc[0]
row["split"] = "test"
print("image:", row["image_id"], "| true label:", int(row["label_6"]))

res, err = pipeline.process_one(cfg, model, row, return_display=True)
assert err is None, f"process_one error: {err}"

Xrow = pd.DataFrame([[res[f] for f in features]], columns=features)
proba = pipe.predict_proba(Xrow)[0]
pc = int(pipe.predict(Xrow)[0])

backend_row = ft[ft["image_id"] == row["image_id"]].iloc[0]
backend_pc = int(pipe.predict(pd.DataFrame([backend_row[features]], columns=features))[0])
print("live  pred:", pc, classes[pc], "| proba:", np.round(proba, 3))
print("backend pred:", backend_pc, "| match:", pc == backend_pc)
print("av_display:", None if res.get("_av_display") is None else res["_av_display"].shape)
print("lesion_display:", None if res.get("_lesion_display") is None else res["_lesion_display"].shape)

# SHAP waterfall (same as interface)
sv = explainer.shap_values(scaler.transform(imputer.transform(Xrow)))
if isinstance(sv, list):
    sv_c = np.asarray(sv[pc]).ravel()
else:
    sv_c = np.asarray(sv)[0, :, pc]
sv_raw = sv_c / scaler.scale_
base_show = float(proba[pc]) - sv_raw.sum()
exp = shap.Explanation(values=sv_raw[None, :], base_values=np.array([base_show]),
                       data=imputer.transform(Xrow), feature_names=features)
fig = plt.figure(figsize=(9, 6.5))
shap.plots.waterfall(exp[0], max_display=15, show=False)
plt.tight_layout()
fig.savefig(ROOT / "_live_shap_test.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("SHAP waterfall OK -> _live_shap_test.png")
