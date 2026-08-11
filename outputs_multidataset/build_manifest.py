"""Build outputs_multidataset/manifest.csv from the combined Kaggle folder layout.

split_dataset/{train,val,test}/{0..4}/  ->  split=train (train+val), test (test)
"""
import os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DSROOT = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub", "datasets",
                      "sehastrajits", "fundus-aptosddridirdeyepacsmessidor",
                      "versions", "1", "split_dataset")
OUT = os.path.join(ROOT, "outputs_multidataset", "manifest.csv")

rows = []
for split_dir, split_name in (("train", "train"), ("val", "train"), ("test", "test")):
    for cls in range(5):
        d = os.path.join(DSROOT, split_dir, str(cls))
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            rows.append({
                "image_id": os.path.splitext(f)[0],
                "path": os.path.join(d, f),
                "dataset": "combined",
                "label_native": cls,
                "split": split_name,
                "label_6": cls,
            })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"manifest -> {OUT}  ({len(df)} rows)")
print(df.groupby(["split", "label_6"]).size().to_string())
