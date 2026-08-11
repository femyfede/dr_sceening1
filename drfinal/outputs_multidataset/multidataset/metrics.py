"""Metric and validation plots for the DR classifier.

Produces, under outputs/plots/:
    confusion_matrix.png   - normalized + raw count heatmap
    roc_ovr.png            - per-class ROC (one-vs-rest) with AUC
    pr_ovr.png             - per-class Precision-Recall with AP
    per_class_metrics.png  - precision / recall / F1 per class
    model_comparison.png   - accuracy / macro-F1 / QWK across algorithms
    feature_importance.png - top-20 feature importances
    calibration.png        - reliability diagram of predicted-class probability
plus metrics_summary.csv.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score, cohen_kappa_score,
                             accuracy_score, f1_score)
from sklearn.preprocessing import label_binarize


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  saved", path)


def plot_confusion_matrix(y, p, class_names, save_path):
    cm = confusion_matrix(y, p)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(class_names)) + 1, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Confusion Matrix (raw counts)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.autofmt_xdate()
    save_fig(fig, save_path)


def plot_roc_ovr(y, proba, class_names, save_path):
    classes = np.unique(y)
    yb = label_binarize(y, classes=classes)
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, c in enumerate(classes):
        if (yb[:, i].sum() == 0) or yb.shape[0] < 2 or np.unique(yb[:, i]).size < 2:
            continue
        fpr, tpr, _ = roc_curve(yb[:, i], proba[:, c])
        ax.plot(fpr, tpr, label=f"{class_names[c]} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves (one-vs-rest)")
    ax.legend(loc="lower right", fontsize=8)
    save_fig(fig, save_path)


def plot_pr_ovr(y, proba, class_names, save_path):
    classes = np.unique(y)
    yb = label_binarize(y, classes=classes)
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, c in enumerate(classes):
        if (yb[:, i].sum() == 0) or yb.shape[0] < 2 or np.unique(yb[:, i]).size < 2:
            continue
        prec, rec, _ = precision_recall_curve(yb[:, i], proba[:, c])
        ap = average_precision_score(yb[:, i], proba[:, c])
        ax.plot(rec, prec, label=f"{class_names[c]} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (one-vs-rest)")
    ax.legend(loc="best", fontsize=8)
    save_fig(fig, save_path)


def plot_per_class_metrics(y, p, class_names, save_path):
    classes = np.unique(y)
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, labels=classes,
                                                       zero_division=0)
    x = np.arange(len(classes))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(class_names)) + 1, 5))
    ax.bar(x - w, prec, w, label="Precision")
    ax.bar(x, rec, w, label="Recall")
    ax.bar(x + w, f1, w, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels([class_names[c] for c in classes], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Per-class metrics")
    save_fig(fig, save_path)


def plot_model_comparison(results, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(results["model"]))
    w = 0.27
    for i, col in enumerate(["accuracy", "macro_f1", "qwk"]):
        vals = results[col].astype(float)
        ax.bar(x + (i - 1) * w, vals, w, label=col)
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 1) * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(results["model"], rotation=15)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.set_title("Cross-validation metrics by algorithm")
    save_fig(fig, save_path)


def plot_feature_importance(features, importances, save_path, top=20):
    imp = pd.Series(importances, index=features).sort_values()
    imp = imp.tail(top)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(imp))))
    ax.barh(imp.index, imp.values, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Feature importances")
    save_fig(fig, save_path)


def plot_calibration(y, proba_pred, save_path, bins=10):
    fig, ax = plt.subplots(figsize=(6, 6))
    frac = np.zeros(bins)
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (proba_pred > edges[i]) & (proba_pred <= edges[i + 1])
        frac[i] = y[m].mean() if m.sum() > 0 else np.nan
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax.plot((edges[:-1] + edges[1:]) / 2, frac, "o-", label="Model")
    ax.set_xlabel("Mean predicted probability (of predicted class)")
    ax.set_ylabel("Observed fraction correct")
    ax.set_title("Reliability / calibration")
    ax.legend()
    save_fig(fig, save_path)


def summary_table(y, p, proba, class_names, out_csv):
    classes = np.unique(y)
    prec, rec, f1, sup = precision_recall_fscore_support(y, p, labels=classes,
                                                         zero_division=0)
    rows = []
    for i, c in enumerate(classes):
        rows.append({"class": class_names[c], "support": int(sup[i]),
                     "precision": round(prec[i], 4), "recall": round(rec[i], 4),
                     "f1": round(f1[i], 4),
                     "accuracy": np.nan, "macro_f1": np.nan, "qwk": np.nan})
    rows.append({"class": "overall", "support": len(y),
                 "precision": np.nan, "recall": np.nan, "f1": np.nan,
                 "accuracy": round(accuracy_score(y, p), 4),
                 "macro_f1": round(f1_score(y, p, average="macro"), 4),
                 "qwk": round(cohen_kappa_score(y, p, weights="quadratic"), 4)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("  saved", out_csv)
