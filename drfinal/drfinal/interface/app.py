"""Diabetic Retinopathy Screening Interface (Streamlit)

Upload a retinal fundus image to get:
    1. RRWNet artery/vein/vessel segmentation image
    2. DR severity class probabilities (trained RandomForest pipeline)
    3. SHAP explanation of which biomarkers drive the prediction
       (no raw biomarker value table)

Run:  streamlit run interface/app.py
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
import joblib
import streamlit as st

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "outputs"

RRWNET_DIR = Path(r"C:\Users\samwema\Downloads\interface dr\rrwnet")
WEIGHTS_PATH = Path(r"C:\Users\samwema\Downloads\interface dr\rrwnet_RITE_1 (1).pth")
RRWNET_K = 1            # recursive-refinement iterations (K=1 ~ 2.4x faster, near-identical output)
TARGET_SIZE = 512       # RRWNet canonical resolution (matches training-time A/V segs)

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(RRWNET_DIR))

import train_models as tm                 # noqa: E402  (ClassMeanImputer needed to unpickle model)
import biomarkers as bm                   # noqa: E402
from extract_lesions_dl import compute_lesion_features  # noqa: E402
from model import RRWNet                  # noqa: E402
from preprocessing import enhance_image   # noqa: E402
from utils import pad_images_unet, to_torch_tensors  # noqa: E402
import shap                               # noqa: E402
import matplotlib                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402

FEATURES = list(tm.FEATURES)
CLASS_LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def load_rrwnet():
    model = RRWNet(iterations=RRWNET_K)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


@st.cache_resource
def load_pipeline():
    art = joblib.load(OUT / "best_model.joblib")
    return art["model"], art["name"], list(art["features"])


@st.cache_resource
def load_shap_explainer():
    pipe = load_pipeline()[0]
    rf = pipe.named_steps["clf"]
    imputer = pipe.named_steps["imputer"]
    scaler = pipe.named_steps["scaler"]
    train = pd.read_csv(OUT / "biomarkers_train.csv")
    Xbg = scaler.transform(imputer.transform(train[FEATURES]))[:100]
    return shap.TreeExplainer(rf, Xbg)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def segment_av(model, bgr):
    """RRWNet A/V/vessel segmentation. Returns (512,512,3) probs: ch0=artery, ch1=vein, ch2=vessel."""
    raw = cv2.resize(bgr, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    dummy = np.full((TARGET_SIZE, TARGET_SIZE), 255, dtype=np.uint8)
    img, mask = enhance_image(raw.copy(), dummy)
    img = img if img.max() <= 1.0 else img / 255.0
    mask = mask if mask.max() <= 1.0 else mask / 255.0

    imgs, paddings = pad_images_unet([img, mask])
    img_p, padding = imgs[0], paddings[0]
    mask_p = np.stack([imgs[1]] * 3, axis=2)

    it, mt = to_torch_tensors([img_p, mask_p])
    with torch.no_grad():
        p = torch.sigmoid(model(it.unsqueeze(0))[-1])
    p[mt.unsqueeze(0) < 0.5] = 0

    h_s = slice(padding[0][0], -padding[0][1] if padding[0][1] else None)
    w_s = slice(padding[1][0], -padding[1][1] if padding[1][1] else None)
    return p[:, :, h_s, w_s].squeeze(0).permute(1, 2, 0).numpy()


def build_av_image(pred):
    """Pred (512,512,3) probs -> uint8 BGR image with the same channel layout as the
    precomputed training A/V segmentations (R=artery, G=vein, B=vessel in RGB)."""
    rgb = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def run_pipeline(bgr):
    """Full live pipeline -> dict with display images, probabilities, SHAP figure."""
    model = load_rrwnet()
    pipe = load_pipeline()[0]
    explainer = load_shap_explainer()

    bgr_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    pred = segment_av(model, bgr)                       # RRWNet
    av_bgr = build_av_image(pred)

    res = bm.extract_all(bgr, av_bgr)                   # vascular biomarkers
    les = compute_lesion_features(bgr_rgb)              # DL lesion biomarkers

    row = {f: res.get(f, np.nan) for f in FEATURES}
    for f in ("LA", "HA", "EA", "MAC", "HE_COUNT", "EX_COUNT", "CTW_A", "MA_A",
              "LA_RET", "HA_RET", "EA_RET"):
        row[f] = les[f]
    Xrow = pd.DataFrame([row], columns=FEATURES)

    proba = pipe.predict_proba(Xrow)[0]
    pred_class = int(pipe.predict(Xrow)[0])

    # ---- SHAP in raw biomarker units -------------------------------
    imputer = pipe.named_steps["imputer"]
    scaler = pipe.named_steps["scaler"]
    rf = pipe.named_steps["clf"]
    Ximp = pd.DataFrame(imputer.transform(Xrow), columns=FEATURES)
    Xscaled = scaler.transform(Ximp)
    sv = explainer.shap_values(Xscaled)
    if isinstance(sv, list):
        sv_c = np.asarray(sv[pred_class]).ravel()
    else:
        sv_c = np.asarray(sv)[0, :, pred_class]
    sv_raw = sv_c / scaler.scale_
    base_show = float(proba[pred_class]) - sv_raw.sum()
    exp = shap.Explanation(values=sv_raw[None, :], base_values=np.array([base_show]),
                           data=Ximp.to_numpy(), feature_names=FEATURES)
    fig = plt.figure(figsize=(9, 6.5))
    shap.plots.waterfall(exp[0], max_display=15, show=False)
    plt.tight_layout()

    seg_display = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
    lesion_display = None
    if "_labels" in les:
        labels = les["_labels"]
        comp = np.zeros((*labels.shape, 3), dtype=np.uint8)
        for lbl, color in ((1, (255, 0, 255)), (2, (0, 255, 255)), (3, (0, 0, 255)), (4, (0, 255, 0))):
            comp[labels == lbl] = color
        lesion_display = comp

    return {
        "pred_class": pred_class,
        "probabilities": proba,
        "seg_display": seg_display,
        "lesion_display": lesion_display,
        "shap_fig": fig,
        "top_shap": FEATURES[int(np.argmax(np.abs(sv_raw)))],
        "model_name": load_pipeline()[1],
    }


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="DR Screening Interface", layout="wide")

    st.title("Diabetic Retinopathy Screening")
    st.markdown(
        "Upload a retinal fundus image. The pipeline runs **RRWNet** artery/vein/vessel "
        "segmentation and a **deep-learning lesion segmenter**, extracts 13 biomarkers, then "
        "predicts the DR severity class and explains it with **SHAP**."
    )
    st.caption(f"Severity model: {load_pipeline()[1]}  |  RRWNet iterations: {RRWNET_K}")

    uploaded = st.file_uploader("Upload Retinal Fundus Image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

    if uploaded is None:
        st.info("Upload an image to start.")
        st.stop()

    file_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not read the uploaded file as an image.")
        st.stop()
    bgr_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if "last_file" not in st.session_state or st.session_state.last_file != uploaded.name:
        st.session_state.last_file = uploaded.name
        with st.spinner("Running RRWNet + lesion segmentation, predicting, explaining (may take ~1 min)..."):
            st.session_state.result = run_pipeline(bgr)

    result = st.session_state.result
    pred_class = result["pred_class"]
    proba = result["probabilities"]

    # ---- Summary metrics ------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Predicted Stage", CLASS_LABELS[pred_class],
                  help="Class with the highest model probability")
    with c2:
        st.metric("Confidence", f"{proba[pred_class] * 100:.1f}%",
                  help="Probability assigned to the predicted class")
    with c3:
        st.metric("Top SHAP Biomarker", result["top_shap"],
                  help="Biomarker that most influenced this prediction")
    with c4:
        st.metric("Severity Model", result["model_name"])

    # ---- Images ----------------------------------------------------------------
    i1, i2 = st.columns(2)
    with i1:
        st.image(bgr_rgb, caption="Input image", use_container_width=True)
    with i2:
        st.image(result["seg_display"],
                 caption="RRWNet Segmentation (Red=Artery, Green=Vein, Blue=Vessel)",
                 use_container_width=True)

    if result["lesion_display"] is not None:
        with st.expander("DL lesion segmentation (Red=Hemorrhage, Yellow=Exudate, "
                         "Green=Microaneurysm, Purple=Cotton-wool)"):
            st.image(result["lesion_display"], use_container_width=True)

    # ---- Class probabilities ---------------------------------------------------
    st.subheader("Class Probabilities")
    proba_df = pd.DataFrame({
        "Stage": CLASS_LABELS,
        "Probability (%)": [p * 100 for p in proba],
    }).set_index("Stage")
    st.bar_chart(proba_df["Probability (%)"])

    # ---- SHAP explanation ------------------------------------------------------
    st.subheader("SHAP: Why This Prediction? (biomarker contribution)")
    st.pyplot(result["shap_fig"])
    st.markdown(
        "Waterfall in raw biomarker units: starting from the baseline probability, each "
        "biomarker pushes the prediction up or down to reach the model's class probability."
    )
