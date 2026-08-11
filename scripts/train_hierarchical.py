"""Train a hierarchical DR severity model.

Stage 1: binary "No DR (0) vs any DR (1-4)" trained on all samples.
Stage 2: 4-class severity (Mild/Moderate/Severe/Proliferative) trained on DR
         samples only, so DR cases are never pulled toward the No DR cluster.

Interface prediction:
    p0 = P(No DR) from stage 1,  q = P(severity|DR) from stage 2
    final probs = [p0, q1*(1-p0), q2*(1-p0), q3*(1-p0), q4*(1-p0)]

Outputs (outputs/):
    hierarchical_model.joblib - bundle with both pipelines, SHAP backgrounds
                                and CV metrics used by interface/app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from train_models import FEATURES, ClassMeanImputer, load_labels

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
OUT = os.path.join(SCRIPTS, "..", "outputs")
LABELS = os.path.join(SCRIPTS, "..", "dr_folder1", "data", "aptos2019", "train (1).csv")

DR_LABELS = [1, 2, 3, 4]


def make_pipe():
    return ImbPipeline([
        ("imputer", ClassMeanImputer()),
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)),
    ])


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def main():
    labels = load_labels(LABELS)
    train = pd.read_csv(os.path.join(OUT, "biomarkers_train.csv"))
    train["image_id"] = train["image_id"].astype(str)
    m = train.merge(labels, on="image_id", how="inner")

    X = m[FEATURES]
    y = m["target"].astype(int).to_numpy()
    y_bin = (y > 0).astype(int)
    dr_mask = y > 0
    Xdr = X[dr_mask]
    ydr = y[dr_mask] - 1                       # 1-4 -> 0-3

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf_dr = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pred_full = np.full(len(X), -1, dtype=int)
    pred_bin = np.full(len(X), -1, dtype=int)
    pred_dr4 = []
    true_dr4 = []
    dr_pos = np.where(dr_mask)[0]

    for (tr, va), (trd, vad) in zip(skf.split(X, y), skf_dr.split(Xdr, ydr)):
        b = make_pipe()
        b.fit(X.iloc[tr], y_bin[tr])
        pb = b.predict(X.iloc[va])
        pred_bin[va] = pb
        pred_full[va] = pb

        d = make_pipe()
        d.fit(Xdr.iloc[trd], ydr[trd])
        pd_dr = d.predict(Xdr.iloc[vad]) + 1   # back to 1-4
        va_set = set(va)
        for orig, pr in zip(dr_pos[vad], pd_dr):
            if orig in va_set:
                pred_full[orig] = pr
        pred_dr4.extend(pd_dr)
        true_dr4.extend(ydr[vad] + 1)

    pred_full = pred_full.astype(int)
    y = y.astype(int)
    cm = confusion_matrix(y, pred_full)
    acc = accuracy_score(y, pred_full)
    f1m = f1_score(y, pred_full, average="macro")
    k = qwk(y, pred_full)

    # binary metrics
    tn, fp, fn, tp = confusion_matrix(y_bin, pred_bin).ravel()

    # DR-only 4-class metrics
    acc_dr = accuracy_score(true_dr4, pred_dr4)
    f1m_dr = f1_score(true_dr4, pred_dr4, average="macro")
    k_dr = qwk(true_dr4, pred_dr4)

    print("True dist:", np.bincount(y, minlength=5).tolist())
    print("Pred dist:", np.bincount(pred_full, minlength=5).tolist())
    print("CM (rows true, cols pred 0-4):\n", cm)
    print("binary NoDR/anyDR: tn=%d fp=%d fn=%d tp=%d  sens=%.3f spec=%.3f"
          % (tn, fp, fn, tp, tp / (tp + fn), tn / (tn + fp)))
    print("overall: acc=%.4f macroF1=%.4f QWK=%.4f" % (acc, f1m, k))
    print("DR-only 4-class: acc=%.4f macroF1=%.4f QWK=%.4f" % (acc_dr, f1m_dr, k_dr))
    print(classification_report(y, pred_full, digits=3))

    # ---- final fits on all data ------------------------------------------
    bin_pipe = make_pipe().fit(X, y_bin)
    dr_pipe = make_pipe().fit(Xdr, ydr)

    bin_bg = bin_pipe.named_steps["scaler"].transform(
        bin_pipe.named_steps["imputer"].transform(X))[:100]
    dr_bg = dr_pipe.named_steps["scaler"].transform(
        dr_pipe.named_steps["imputer"].transform(Xdr))[:100]

    joblib.dump({
        "bin_pipe": bin_pipe,
        "dr_pipe": dr_pipe,
        "features": FEATURES,
        "dr_labels": DR_LABELS,
        "name": "RandomForest (NoDR/anyDR -> 4-class)",
        "bin_bg": bin_bg,
        "dr_bg": dr_bg,
        "cv_metrics": {
            "accuracy": float(acc), "macro_f1": float(f1m), "qwk": float(k),
            "sens_any_dr": float(tp / (tp + fn)),
            "spec": float(tn / (tn + fp)),
            "dr_accuracy": float(acc_dr), "dr_macro_f1": float(f1m_dr),
            "dr_qwk": float(k_dr),
        },
        "cv_cm": cm.tolist(),
    }, os.path.join(OUT, "hierarchical_model.joblib"))
    print("Saved -> outputs/hierarchical_model.joblib")


if __name__ == "__main__":
    main()
