"""Diabetic Retinopathy Screening Interface (Streamlit)

Upload a retinal fundus image to get:
    1. RRWNet artery/vein/vessel segmentation image
    2. DR severity class probabilities (trained RandomForest pipeline)
    3. SHAP explanation of which biomarkers drive the prediction

Uses the exact same per-image GPU pipeline as training/feature-extraction
(outputs_multidataset/multidataset/pipeline.py), so live predictions match the
backend model exactly.

Run:  streamlit run interface/app.py
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import joblib
import yaml
import streamlit as st

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MULTIDATASET = ROOT / "outputs_multidataset" / "multidataset"
CFG_PATH = MULTIDATASET / "config.yaml"
BUNDLE_PATH = ROOT / "outputs_multidataset" / "model_bundle.joblib"
TRAIN_FEATURES = ROOT / "outputs_multidataset" / "features_train.csv"
SCRIPTS = ROOT / "scripts"
RRWNET_DIR = ROOT / "tools" / "rrwnet"

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(RRWNET_DIR))
sys.path.insert(0, str(MULTIDATASET))

import pipeline                     # noqa: E402
import shap                         # noqa: E402
import matplotlib                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
def _load_cfg():
    cfg = yaml.safe_load(open(CFG_PATH, encoding="utf-8"))
    cfg_dir = os.path.dirname(os.path.abspath(CFG_PATH))
    for key in ("project_scripts", "rrwnet_module_dir", "rrwnet_weights", "outputs"):
        p = cfg.get(key)
        if p and not os.path.isabs(p):
            cfg[key] = os.path.abspath(os.path.join(cfg_dir, p))
    return cfg


@st.cache_resource
def get_rrwnet():
    cfg = _load_cfg()
    pipeline.ensure_paths(cfg)
    device = pipeline.pick_device(cfg.get("device", "auto"))
    model = pipeline.load_rrwnet(cfg, device)
    return cfg, model, device


@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)


@st.cache_resource
def load_shap_explainer():
    bundle = load_bundle()
    pipe = bundle["pipeline"]
    feats = bundle["features"]
    rf = pipe.named_steps["clf"]
    imputer = pipe.named_steps["imputer"]
    scaler = pipe.named_steps["scaler"]
    train = pd.read_csv(TRAIN_FEATURES)
    Xbg = scaler.transform(imputer.transform(train[feats]))[:100]
    return shap.TreeExplainer(rf, Xbg)


def predict_and_explain(feats, pipe, explainer, features, classes):
    """Predict DR class + build SHAP waterfall. Returns dict."""
    Xrow = pd.DataFrame([feats], columns=features)
    proba = pipe.predict_proba(Xrow)[0]
    pred_class = int(pipe.predict(Xrow)[0])

    imputer = pipe.named_steps["imputer"]
    scaler = pipe.named_steps["scaler"]
    rf = pipe.named_steps["clf"]
    Ximp = pd.DataFrame(imputer.transform(Xrow), columns=features)
    Xscaled = scaler.transform(Ximp)
    sv = explainer.shap_values(Xscaled)
    if isinstance(sv, list):
        sv_c = np.asarray(sv[pred_class]).ravel()
    else:
        sv_c = np.asarray(sv)[0, :, pred_class]
    sv_raw = sv_c / scaler.scale_
    base_show = float(proba[pred_class]) - sv_raw.sum()
    exp = shap.Explanation(values=sv_raw[None, :], base_values=np.array([base_show]),
                           data=Ximp.to_numpy(), feature_names=features)
    fig = plt.figure(figsize=(9, 6.5))
    shap.plots.waterfall(exp[0], max_display=15, show=False)
    plt.tight_layout()

    return {
        "pred_class": pred_class,
        "probabilities": proba,
        "shap_fig": fig,
        "top_shap": features[int(np.argmax(np.abs(sv_raw)))],
    }


def run_live(uploaded, cfg, model, device, bundle, explainer):
    """Full live pipeline on an uploaded image. Returns display dict."""
    features = bundle["features"]
    classes = bundle["classes"]

    ext = os.path.splitext(uploaded.name)[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    try:
        row = {"image_id": uploaded.name, "dataset": "live", "split": "live",
               "label_6": 0, "path": tmp_path}
        res, err = pipeline.process_one(cfg, model, row, return_display=True)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if err:
        return {"error": err}

    feats = {f: res[f] for f in features}
    result = predict_and_explain(feats, bundle["pipeline"], explainer, features, classes)
    result["model_name"] = bundle["model_name"]
    result["av_display"] = res.get("_av_display")
    result["lesion_display"] = res.get("_lesion_display")
    result["feature_count"] = len(features)
    return result


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="DR Screening Interface", layout="wide")

    st.title("Diabetic Retinopathy Screening")
    st.markdown(
        "Upload a retinal fundus image. The pipeline runs **RRWNet** artery/vein/vessel "
        "segmentation and a **deep-learning lesion segmenter**, extracts 33 biomarkers, then "
        "predicts the DR severity class and explains it with **SHAP**."
    )

    try:
        bundle = load_bundle()
        cfg, model, device = get_rrwnet()
        explainer = load_shap_explainer()
    except Exception as e:
        st.error(f"Failed to initialise model resources: {e}")
        st.stop()

    st.caption(f"Severity model: {bundle['model_name']}  |  "
               f"features: {len(bundle['features'])}  |  device: {device}")

    uploaded = st.file_uploader("Upload Retinal Fundus Image",
                                type=["png", "jpg", "jpeg", "bmp", "tiff"])

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
        with st.spinner("Running RRWNet + lesion segmentation, predicting, explaining (a few seconds)..."):
            st.session_state.result = run_live(uploaded, cfg, model, device, bundle, explainer)

    result = st.session_state.result
    if "error" in result:
        st.error(f"Processing failed for this image: {result['error']}")
        st.stop()

    classes = bundle["classes"][:5]
    pred_class = result["pred_class"]
    proba = result["probabilities"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Predicted Stage", classes[pred_class],
                  help="Class with the highest model probability")
    with c2:
        st.metric("Confidence", f"{proba[pred_class] * 100:.1f}%",
                  help="Probability assigned to the predicted class")
    with c3:
        st.metric("Top SHAP Biomarker", result["top_shap"],
                  help="Biomarker that most influenced this prediction")
    with c4:
        st.metric("Severity Model", result["model_name"])

    i1, i2 = st.columns(2)
    with i1:
        st.image(bgr_rgb, caption="Input image", use_container_width=True)
    with i2:
        if result.get("av_display") is not None:
            st.image(result["av_display"],
                     caption="RRWNet Segmentation (Red=Artery, Green=Vein, Blue=Vessel)",
                     use_container_width=True)

    if result.get("lesion_display") is not None:
        with st.expander("DL lesion segmentation (Purple=Cotton-wool, Yellow=Exudate, "
                         "Green=Microaneurysm, Red=Hemorrhage)"):
            st.image(result["lesion_display"], use_container_width=True)

    st.subheader("Class Probabilities")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.barh(classes, proba * 100, color="steelblue")
    ax2.set_xlabel("Probability (%)")
    ax2.set_xlim(0, 100)
    ax2.invert_yaxis()
    for i, p in enumerate(proba):
        ax2.text(p * 100 + 1, i, f"{p * 100:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig2)

    st.subheader("SHAP: Why This Prediction? (biomarker contribution)")
    st.pyplot(result["shap_fig"])
    st.markdown(
        "Waterfall in raw biomarker units: starting from the baseline probability, each "
        "biomarker pushes the prediction up or down to reach the model's class probability."
    )
