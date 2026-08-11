import sys
sys.path.insert(0, "scripts")
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

b = joblib.load(r"outputs_multidataset\model_bundle.joblib")
pipe, feats, classes = b["pipeline"], b["features"], b["classes"]
print("model:", b["model_name"], "| classes:", classes)
print("n features:", len(feats))

te = pd.read_csv(r"outputs_multidataset\features_test.csv").dropna(subset=["label_6"])
y = te["label_6"].astype(int)
X = te[feats]
p = pipe.predict(X)
print("test n=%d  acc=%.4f macroF1=%.4f qwk=%.4f" % (
    len(te), accuracy_score(y, p), f1_score(y, p, average="macro"),
    cohen_kappa_score(y, p, weights="quadratic")))
print("pred dist:", np.bincount(p, minlength=5).tolist())
print("true dist:", np.bincount(y, minlength=5).tolist())
