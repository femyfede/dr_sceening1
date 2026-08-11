"""Train and evaluate the 6-class DR classifier on the extracted features.

Usage:
    python multidataset/train.py [--config config.yaml]
                                 [--features outputs/features.csv]

Loads features.csv (or features_train/features_test.csv), runs stratified CV
per algorithm, picks the best by macro-F1, retrains on all train data, and
writes plots + model bundle under outputs/.

Outputs:
    outputs/model_bundle.joblib   - tuned best model (steps=imputer,smote,scaler,clf)
    outputs/models_comparison.csv - CV metrics per algorithm
    outputs/metrics_summary.csv   - per-class metrics on held-out test
    outputs/plots/*.png           - confusion matrix, ROC, PR, per-class, etc.
"""

import os
import sys
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import yaml
import joblib

import metrics as M

# Reuse project modules when available (the interface machine / packaged scripts).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "drfinal", "scripts")))
try:
    from scripts.train_models import ClassMeanImputer, FEATURES as DEFAULT_FEATURES
except Exception:
    try:
        import train_models as tm
        ClassMeanImputer = tm.ClassMeanImputer
        DEFAULT_FEATURES = list(tm.FEATURES)
    except Exception:
        import warnings
        warnings.warn("scripts.train_models not available; using inline ClassMeanImputer")
        from sklearn.impute import SimpleImputer

        class ClassMeanImputer(SimpleImputer):
            pass

        DEFAULT_FEATURES = [
            "AVR", "CRAE", "CRVE", "VD", "AD", "VeD", "TI", "CI", "FD",
            "ATI", "VTI", "AFD", "VFD", "JUNC", "VLEN",
            "AWID", "AWID_SD", "VWID", "VWID_SD", "WID", "WID_SD", "ADV_RATIO",
            "LA", "HA", "EA", "MAC", "HE_COUNT", "EX_COUNT", "CTW_A", "MA_A",
            "LA_RET", "HA_RET", "EA_RET",
        ]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_features(cfg):
    out_dir = cfg["outputs"]
    ft = os.path.join(out_dir, "features_train.csv")
    fe = os.path.join(out_dir, "features_test.csv")
    full = os.path.join(out_dir, "features.csv")

    if os.path.exists(ft) and os.path.exists(fe):
        Xtr = pd.read_csv(ft); Xte = pd.read_csv(fe)
        Xtr = Xtr.dropna(subset=["label_6"]).reset_index(drop=True)
        Xte = Xte.dropna(subset=["label_6"]).reset_index(drop=True)
        return Xtr, Xte
    if not os.path.exists(full):
        sys.exit(f"no features found; run extract.py first ({out_dir})")
    X = pd.read_csv(full).dropna(subset=["label_6"]).reset_index(drop=True)
    test_frac = cfg.get("test_frac", 0.2)
    seed = cfg.get("seed", 42)
    tr, te = train_test_split(X, test_size=test_frac, random_state=seed,
                              stratify=X["label_6"])
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def make_models():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "XGBoost": None,
        "LogisticRegression": LogisticRegression(max_iter=2000, C=1.0, random_state=42),
        "SVM": SVC(probability=True, C=1.0, kernel="rbf", random_state=42),
    }


def make_pipeline(model):
    return ImbPipeline([
        ("imputer", ClassMeanImputer()),
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("clf", model),
    ])


def cv_evaluate(model, X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, f1s, qwks = [], [], []
    for tr_i, va_i in skf.split(X, y):
        pipe = make_pipeline(model)
        pipe.fit(X.iloc[tr_i], y.iloc[tr_i])
        p = pipe.predict(X.iloc[va_i])
        accs.append(accuracy_score(y.iloc[va_i], p))
        f1s.append(f1_score(y.iloc[va_i], p, average="macro"))
        qwks.append(cohen_kappa_score(y.iloc[va_i], p, weights="quadratic"))
    return np.mean(accs), np.mean(f1s), np.mean(qwks)


def feature_importances(model, features, X_te, y_te):
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        return clf.feature_importances_
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X_te[features], y_te, n_repeats=5,
                               random_state=42, scoring="accuracy", n_jobs=-1)
    return r.importances_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = os.path.abspath(cfg["outputs"])
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    Xtr, Xte = load_features(cfg)
    class_names = cfg["classes"]
    features = cfg.get("features") or DEFAULT_FEATURES
    features = [f for f in features if f in Xtr.columns]
    missing = [f for f in features if f not in Xte.columns]
    if missing:
        sys.exit(f"missing features in test data: {missing}")

    X_tr, y_tr = Xtr[features], Xtr["label_6"].astype(int)
    X_te, y_te = Xte[features], Xte["label_6"].astype(int)

    print(f"train {len(X_tr)} / test {len(X_te)} | features {len(features)}")

    models = make_models()
    import xgboost as xgb
    models["XGBoost"] = xgb.XGBClassifier(n_estimators=300, max_depth=6,
                                          random_state=42, eval_metric="mlogloss")

    results = []
    for name, model in models.items():
        acc, f1, qwk = cv_evaluate(model, X_tr, y_tr,
                                   n_splits=cfg.get("cv_folds", 5),
                                   seed=cfg.get("seed", 42))
        results.append({"model": name, "accuracy": acc, "macro_f1": f1, "qwk": qwk})
        print(f"  {name:20s} acc={acc:.4f} macro-F1={f1:.4f} QWK={qwk:.4f}")

    res_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    res_df.to_csv(os.path.join(out_dir, "models_comparison.csv"), index=False)
    M.plot_model_comparison(res_df, os.path.join(out_dir, "plots", "model_comparison.png"))

    best_name = res_df.iloc[0]["model"]
    print(f"\nBest: {best_name}")
    model = models[best_name]
    pipe = make_pipeline(model)
    pipe.fit(X_tr, y_tr)

    p_te = pipe.predict(X_te)
    proba_te = pipe.predict_proba(X_te)
    classes = np.unique(y_te)

    M.plot_confusion_matrix(y_te, p_te, class_names,
                            os.path.join(out_dir, "plots", "confusion_matrix.png"))
    M.plot_roc_ovr(y_te, proba_te, class_names,
                   os.path.join(out_dir, "plots", "roc_ovr.png"))
    M.plot_pr_ovr(y_te, proba_te, class_names,
                  os.path.join(out_dir, "plots", "pr_ovr.png"))
    M.plot_per_class_metrics(y_te, p_te, class_names,
                             os.path.join(out_dir, "plots", "per_class_metrics.png"))

    imp = feature_importances(pipe, features, X_te, y_te)
    M.plot_feature_importance(features, imp,
                              os.path.join(out_dir, "plots", "feature_importance.png"))

    pred_cls_proba = proba_te[np.arange(len(p_te)), p_te]
    M.plot_calibration((p_te == y_te).astype(int), pred_cls_proba,
                       os.path.join(out_dir, "plots", "calibration.png"))

    M.summary_table(y_te, p_te, proba_te, class_names,
                    os.path.join(out_dir, "metrics_summary.csv"))

    bundle = {
        "pipeline": pipe,
        "features": features,
        "classes": class_names,
        "class_index_map": {int(i): c for i, c in enumerate(class_names)},
        "model_name": best_name,
    }
    joblib.dump(bundle, os.path.join(out_dir, "model_bundle.joblib"))
    print("saved model bundle ->", os.path.join(out_dir, "model_bundle.joblib"))


if __name__ == "__main__":
    main()
