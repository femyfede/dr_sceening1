# Multi-dataset DR pipeline (CUDA machine)

End-to-end training pipeline for a 6-class DR screening classifier on a machine
with a GPU:

    0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative DR, 5 Other

Each fundus image is processed as: preprocessing -> RRWNet A/V segmentation ->
vascular biomarkers (biomarkers.py) + DL lesion segmentation (fundus-lesions-toolkit)
-> tabular features -> train an ML classifier -> metric & validation plots.

## Files

| File              | Purpose                                              |
| ----------------- | ---------------------------------------------------- |
| `config.yaml`     | All paths, datasets, labels mapping, model settings. |
| `datasets.py`     | Builds `outputs/manifest.csv` from Kaggle datasets.  |
| `extract.py`      | Resumable per-image feature extraction (RRWNet+DL).  |
| `pipeline.py`     | Per-image plumbing (RRWNet, lesions, biomarkers).    |
| `train.py`        | CV model comparison, best-model pickling, plots.     |
| `metrics.py`      | Plot helpers (confusion, ROC, PR, calibration, ...). |
| `requirements.txt`| Python dependencies.                                 |

## Copy to the CUDA machine

Everything below, keeping the same relative layout:

```
multidataset/                  <- this folder
drfinal/scripts/               <- biomarkers.py, train_models.py,
                                  extract_lesions_dl.py, extract_extended_features.py,
                                  fix_densities.py
tools/fundus-lesions-toolkit/  <- pip install -e from here
rrwnet/                        <- RRWNet model.py, preprocessing.py, utils.py
rrwnet_RITE_1 (1).pth          <- RRWNet weights (~248 MB)
<your datasets>                <- APTOS/DDR/IDRiD/EYEPACS/MESSIDOR + an "other" set
```

`config.yaml` paths are relative to the config file unless absolute; edit them to
match the target machine.

## Setup

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r multidataset/requirements.txt
pip install -e tools/fundus-lesions-toolkit     # downloads weights on first run
```

## Dataset manifest

Each dataset needs a folder of images and a labels CSV (id + label columns).
Fill `datasets` in `config.yaml`; each `label_mapping` translates the dataset's
native grades into the unified 0-5 scheme (Messidor has 0-3). For an "other"
(no-DR-target) dataset use `other: {0: 5}`.

```
python multidataset/datasets.py --config multidataset/config.yaml
```

Writes `outputs_multidataset/manifest.csv` with the class distribution.

## Extract features

```
python multidataset/extract.py --config multidataset/config.yaml --samples 5
```

`--samples N` saves A/V segmentation and lesion overlays for the first N images
(spot-check the pipeline). Resumable: rerun to continue after interruptions.
Output: `outputs_multidataset/features.csv` (one row per image, all features).

## Train + plots

```
python multidataset/train.py --config multidataset/config.yaml
```

- Compares RandomForest / XGBoost / LogisticRegression / SVM with 5-fold
  stratified CV (imputer -> SMOTE -> scaler), picks the best by macro-F1.
- Trains the best model on all train data, evaluates on the held-out test split.
- Writes under `outputs_multidataset/`:

| Output                    | Description                              |
| ------------------------- | ---------------------------------------- |
| `model_bundle.joblib`     | Best model (steps=imputer,smote,scaler,clf) |
| `models_comparison.csv`   | CV accuracy / macro-F1 / QWK per model   |
| `metrics_summary.csv`     | Per-class precision/recall/F1, overall    |
| `plots/confusion_matrix.png` | Raw-count confusion heatmap           |
| `plots/roc_ovr.png`       | Per-class one-vs-rest ROC + AUC           |
| `plots/pr_ovr.png`        | Per-class Precision-Recall + AP           |
| `plots/per_class_metrics.png` | Precision/recall/F1 bars              |
| `plots/model_comparison.png`  | Accuracy / F1 / QWK across models      |
| `plots/feature_importance.png` | Top-20 feature importances           |
| `plots/calibration.png`   | Reliability diagram                      |

## Notes / gotchas

- `device: auto` uses CUDA when available (RRWNet + the toolkit both run on GPU).
- The lesion toolkit downloads its U-Net weights from the HuggingFace Hub on the
  first call (needs internet once).
- RRWNet runs at 512x512 and the toolkit at 1024x1024, so extraction is the slow
  step; on a GPU it is ~seconds per image.
- DDR/IDRiD need their annotation files (DDR lists `train.csv`/`test.csv`;
  IDRiD grade labels are in `a. IDRiD_Disease Grading_Testing Labels.csv` and the
  training counterpart). Point `labels_csv` at the file that has the DR grade.
