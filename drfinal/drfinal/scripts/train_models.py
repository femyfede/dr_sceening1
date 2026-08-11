"""Combine biomarker CSV with labels and train/compare 4 ML algorithms.

Preprocessing stages applied to every model (train folds only, no leakage):
    1. Missing values imputed with the mean of the sample's target class
       (global train mean is used as fallback for test/validation, where the
       class is not available during transform).
    2. Class imbalance corrected with SMOTE (minority oversampling).
    3. Feature scaling (StandardScaler) for distance-based models.

Usage:
    python scripts/train_models.py [--labels path/to/labels.csv] [--folds 5]

Label file detection (first match wins):
    1. --labels argument
    2. data/aptos2019/train.csv  (APTOS2019 style: id_code, diagnosis)
    3. data/aptos2019/labels.csv

The label CSV must contain an image id column (id_code / image_id / id / image)
and a target column (diagnosis / dr_grade / grade / label / severity / target).

Outputs (in outputs/):
    model_comparison.csv   - CV metrics for all 4 algorithms
    best_model.joblib      - best model (pickled sklearn pipeline)
    feature_importances.csv- importances of the best tree model (if available)
    predictions_test.csv   - best-model predictions on the test split
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(SCRIPTS, "..", "dr_folder1", "data", "aptos2019")
OUT = os.path.join(SCRIPTS, "..", "outputs")

FEATURES = [
    # vascular (RRWNet A/V segmentation)
    "AVR", "CRAE", "CRVE", "VD", "AD", "VeD", "TI", "CI", "FD",
    "ATI", "VTI", "AFD", "VFD", "JUNC", "VLEN",
    "AWID", "AWID_SD", "VWID", "VWID_SD", "WID", "WID_SD", "ADV_RATIO",
    # DL lesion segmentation
    "LA", "HA", "EA", "MAC",
    "HE_COUNT", "EX_COUNT", "CTW_A", "MA_A", "LA_RET", "HA_RET", "EA_RET",
]

ID_ALIASES = ["id_code", "image_id", "image", "id", "filename"]
TARGET_ALIASES = ["diagnosis", "dr_grade", "grade", "label", "severity", "target"]


def _has_target_col(path):
    try:
        cols = pd.read_csv(path, nrows=1).columns
        return (any(c.lower() in ID_ALIASES for c in cols)
                and any(c.lower() in TARGET_ALIASES for c in cols))
    except Exception:
        return False


def find_label_file(explicit=None):
    if explicit and os.path.exists(explicit) and _has_target_col(explicit):
        return explicit
    priority = [os.path.join(DATA, "train.csv"), os.path.join(DATA, "labels.csv")]
    for cand in priority:
        if os.path.exists(cand) and _has_target_col(cand):
            return cand
    for cand in sorted(glob.glob(os.path.join(DATA, "*.csv"))):
        if os.path.exists(cand) and "biomarker" not in cand.lower() and _has_target_col(cand):
            return cand
    return None


def load_labels(path):
    df = pd.read_csv(path)
    id_col = next((c for c in df.columns if c.lower() in ID_ALIASES), None)
    tgt_col = next((c for c in df.columns if c.lower() in TARGET_ALIASES), None)
    if id_col is None or tgt_col is None:
        raise ValueError(
            f"Could not find id/target columns in {path}. "
            f"Have columns: {list(df.columns)}. "
            f"Needed id in {ID_ALIASES} and target in {TARGET_ALIASES}.")
    labels = df[[id_col, tgt_col]].copy()
    labels.columns = ["image_id", "target"]
    labels["image_id"] = labels["image_id"].astype(str).str.replace(r"\.png$", "", regex=True)
    return labels


class ClassMeanImputer(BaseEstimator, TransformerMixin):
    """Impute missing values with the mean of the sample's target class.

    During training (fit_transform receives y) each NaN is filled with the mean
    of that feature computed over the sample's class in the training fold.
    During transform/predict (no y available) it falls back to the global
    training mean, so unseen data can still be imputed.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        df = X.copy()
        df["_y"] = np.asarray(y)
        self.class_means_ = df.groupby("_y").mean(numeric_only=True)
        self.global_mean_ = X.mean(numeric_only=True)
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=getattr(self, "columns_", None))
        return X.fillna(self.global_mean_).to_numpy()

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        X = pd.DataFrame(X, columns=self.columns_)
        fill = self.class_means_.reindex(np.asarray(y))
        fill.index = X.index
        X = X.fillna(fill)
        return X.fillna(self.global_mean_).to_numpy()


def make_models():
    return {
        "RandomForest": ImbPipeline([
            ("imputer", ClassMeanImputer()),
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, n_jobs=-1, random_state=42)),
        ]),
        "XGBoost": ImbPipeline([
            ("imputer", ClassMeanImputer()),
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.1,
                objective="multi:softprob", eval_metric="mlogloss",
                tree_method="hist", random_state=42)),
        ]),
        "SVM": ImbPipeline([
            ("imputer", ClassMeanImputer()),
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", SVC(C=1.0, gamma="scale", probability=True, random_state=42)),
        ]),
        "LogisticRegression": ImbPipeline([
            ("imputer", ClassMeanImputer()),
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0, max_iter=3000, random_state=42)),
        ]),
    }


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def evaluate_cv(models, X, y, folds):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    rows = []
    for name, pipe in models.items():
        scores = {"accuracy": [], "macro_f1": [], "weighted_f1": [], "qwk": []}
        for tr, va in skf.split(X, y):
            Xtr, ytr = X.iloc[tr], y.iloc[tr]
            Xva, yva = X.iloc[va], y.iloc[va]
            pipe.fit(Xtr, ytr)
            p = pipe.predict(Xva)
            scores["accuracy"].append(accuracy_score(yva, p))
            scores["macro_f1"].append(f1_score(yva, p, average="macro"))
            scores["weighted_f1"].append(f1_score(yva, p, average="weighted"))
            scores["qwk"].append(qwk(yva, p))
        rows.append({"model": name,
                     "accuracy": np.mean(scores["accuracy"]),
                     "accuracy_std": np.std(scores["accuracy"]),
                     "macro_f1": np.mean(scores["macro_f1"]),
                     "macro_f1_std": np.std(scores["macro_f1"]),
                     "weighted_f1": np.mean(scores["weighted_f1"]),
                     "qwk": np.mean(scores["qwk"]),
                     "qwk_std": np.std(scores["qwk"])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    train = pd.read_csv(os.path.join(OUT, "biomarkers_train.csv"))
    test = pd.read_csv(os.path.join(OUT, "biomarkers_test.csv"))
    train["image_id"] = train["image_id"].astype(str)
    test["image_id"] = test["image_id"].astype(str)

    label_path = find_label_file(args.labels)
    if label_path is None:
        print("=" * 70)
        print("No label file found. To train, provide a labels CSV with an image id")
        print("column and a target column (e.g. APTOS2019 train.csv: id_code, diagnosis).")
        print("Drop it at:", os.path.join(DATA, "train.csv"))
        print("then re-run:  python scripts/train_models.py")
        print("=" * 70)
        sys.exit(1)

    print(f"Labels loaded from: {label_path}")
    labels = load_labels(label_path)
    print(f"  label rows: {len(labels)}, target values: {sorted(labels['target'].unique())}")

    train = train.merge(labels, on="image_id", how="inner")
    print(f"Matched train images with labels: {len(train)} / {len(pd.read_csv(os.path.join(OUT, 'biomarkers_train.csv')))}")

    missing = train[FEATURES].isna().mean()
    if missing.any():
        print("Missing feature fractions:\n", missing[missing > 0].round(3))

    X = train[FEATURES]
    y = train["target"].astype(int)

    print(f"\nTraining {len(make_models())} algorithms with {args.folds}-fold stratified CV...")
    models = make_models()
    comp = evaluate_cv(models, X, y, args.folds)
    comp = comp.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    comp.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
    print("\n=== Model comparison (stratified {}-fold CV) ===".format(args.folds))
    print(comp.round(4).to_string(index=False))

    best_name = comp.iloc[0]["model"]
    print(f"\nBest model by macro-F1: {best_name}")

    best_pipe = models[best_name]
    best_pipe.fit(X, y)
    joblib.dump({"model": best_pipe, "name": best_name, "features": FEATURES},
                os.path.join(OUT, "best_model.joblib"))
    print(f"Saved best model -> outputs/best_model.joblib")

    # feature importances (tree models)
    try:
        clf = best_pipe.named_steps["clf"]
        imp = getattr(clf, "feature_importances_", None)
        if imp is not None:
            pd.DataFrame({"feature": FEATURES, "importance": imp}) \
              .sort_values("importance", ascending=False) \
              .to_csv(os.path.join(OUT, "feature_importances.csv"), index=False)
            print("Saved feature importances -> outputs/feature_importances.csv")
    except Exception:
        pass

    # predict on test
    if len(test) > 0:
        Xte = test[FEATURES]
        preds = best_pipe.predict(Xte)
        proba = best_pipe.predict_proba(Xte)
        test_out = test[["image_id"]].copy()
        test_out["predicted_grade"] = preds
        test_out = test_out.join(pd.DataFrame(proba, columns=[f"p_{int(c)}" for c in sorted(y.unique())]), on=None)
        test_out.to_csv(os.path.join(OUT, "predictions_test.csv"), index=False)
        print(f"Saved test predictions -> outputs/predictions_test.csv ({len(test_out)} rows)")

        if label_path and "test" in label_path.lower():
            pass
        test_labels = labels[labels["image_id"].isin(test_out["image_id"])]
        if len(test_labels) > 0:
            ev = test_out.merge(test_labels, on="image_id")
            if len(ev) > 0:
                acc = accuracy_score(ev["target"], ev["predicted_grade"])
                f1m = f1_score(ev["target"], ev["predicted_grade"], average="macro")
                k = qwk(ev["target"], ev["predicted_grade"])
                print(f"\nHeld-out test set: accuracy={acc:.4f}  macro-F1={f1m:.4f}  QWK={k:.4f}")
                cm = confusion_matrix(ev["target"], ev["predicted_grade"])
                print("Confusion matrix:\n", cm)


if __name__ == "__main__":
    main()
