"""Headless run of the Streamlit interface via AppTest:
1. First run: no upload -> resources (bundle, RRWNet, SHAP) must load cleanly.
2. Second run: with an uploaded image -> full live GPU pipeline + UI render.
"""
import os
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

APP = r"C:\Users\FEDELIKA MAXIMUS\OneDrive\Desktop\drfinal\interface\app.py"

at = AppTest.from_file(APP, default_timeout=300)
at.run()
print("== run 1 (no upload) ==")
print("exceptions:", [e.value for e in at.exception])
print("titles:", [t.value for t in at.title])
print("infos:", [i.value for i in at.info])
assert not at.exception, "resource load failed"
assert len(at.info) == 1 and "Upload an image to start." in at.info[0].value

# grab a real image from the dataset test split
man = pd.read_csv(r"outputs_multidataset\manifest.csv")
img = man[man["split"] == "test"].sample(1, random_state=3).iloc[0]
data = open(img["path"], "rb").read()
print(f"\n== run 2 (upload {img['image_id']}, true label {img['label_6']}) ==")
at.file_uploader[0].set_value([(os.path.basename(img["path"]), data, "image/jpeg")])
at.run()
print("exceptions:", [e.value for e in at.exception])
metrics = [(m.label, m.value) for m in at.metric]
print("metrics:", metrics)
assert not at.exception, f"live run failed: {[e.value for e in at.exception]}"
assert len(metrics) == 4, "expected 4 metric cards"
assert any("Predicted Stage" == m[0] for m in metrics)
print("images rendered:", len(at.image), "| subheaders:", [s.value for s in at.subheader])
print("== PASS ==")
