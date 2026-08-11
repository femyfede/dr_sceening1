# Diabetic Retinopathy Screening Interface (Streamlit Cloud Version)
# Simplified cloud version with CPU-based biomarker extraction
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import joblib
import streamlit as st
from huggingface_hub import hf_hub_download

# Add current directory to path for custom model classes
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Required for loading the model bundle (custom classes)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline

# Import train_models for ClassMeanImputer (required by model bundle)
import train_models

warnings.filterwarnings("ignore")


# Configuration
HUGGING_FACE_REPO = "samwema/dr_screening_model"
MODEL_BUNDLE_FILE = "model_bundle.joblib"

# Class labels
CLASSES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR", "Other"]


def load_model_bundle():
    """Download and load model bundle from Hugging Face."""
    bundle_path = hf_hub_download(repo_id=HUGGING_FACE_REPO, filename=MODEL_BUNDLE_FILE)
    return joblib.load(bundle_path)


def extract_simplified_biomarkers(image):
    """Extract simplified biomarkers from fundus image on CPU."""
    h, w = image.shape[:2]
    biomarkers = {}
    
    # Green channel enhancement (standard in DR analysis)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    green = rgb[:, :, 1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Exudate detection
    exudate_mask = green > 200
    biomarkers["EX_COUNT"] = int(np.sum(exudate_mask) // 500)
    biomarkers["EX_AREA"] = float(np.sum(exudate_mask) / (h * w) * 100)
    
    # Hard exudates
    hard_mask = green > 220
    biomarkers["hard_exudate_count"] = int(np.sum(hard_mask) // 500)
    biomarkers["hard_exudate_area"] = float(np.sum(hard_mask) / (h * w) * 100)
    
    # Microaneurysms (dark dots)
    micro_mask = (green < 80) & (green > 30)
    biomarkers["MA_COUNT"] = int(np.sum(micro_mask) // 100)
    biomarkers["MA_A"] = float(np.sum(micro_mask) / (h * w) * 100)
    
    # Hemorrhage detection
    hem_mask = (green < 100) & (green > 40) & (green < 120)
    biomarkers["HE_COUNT"] = int(np.sum(hem_mask) // 300)
    biomarkers["HA"] = float(np.sum(hem_mask) / (h * w) * 100)
    biomarkers["HA_RET"] = biomarkers["HA"]
    
    # Vessel detection
    blurred = cv2.GaussianBlur(gray, (0, 0), 1)
    vessels = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    vessel_mask = vessels > 120
    biomarkers["VESSEL_COMPLEXITY"] = float(np.sum(vessel_mask) / (h * w) * 100)
    biomarkers["LA"] = float(np.sum(vessel_mask) / (h * w) * 100)
    biomarkers["LA_RET"] = biomarkers["LA"]
    
    # Optic disc approximation
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    od_area = 0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        od_area = cv2.contourArea(largest)
    biomarkers["optic_disc_area"] = float(od_area)
    
    # Retinal area
    biomarkers["retina_area_px"] = float(h * w)
    
    # Fill all features with estimated values
    result = {}
    for feat in [
        "AVR", "CRAE", "CRVE", "VD", "AD", "VeD", "TI", "CI", "FD",
        "ATI", "VTI", "AFD", "VFD", "JUNC", "VLEN",
        "AWID", "AWID_SD", "VWID", "VWID_SD", "WID", "WID_SD", "ADV_RATIO",
        "LA", "HA", "EA", "MAC",
        "HE_COUNT", "EX_COUNT", "CTW_A", "MA_A", "LA_RET", "HA_RET", "EA_RET",
    ]:
        if feat in biomarkers:
            result[feat] = biomarkers[feat]
        elif feat == "AVR":
            result[feat] = 0.15
        elif feat in ["CRAE", "CRVE", "AD", "VeD"]:
            result[feat] = 50.0
        elif feat in ["VD", "TI", "CI", "FD", "ATI", "VTI", "AFD", "VFD", "JUNC", "VLEN"]:
            result[feat] = 10.0
        elif feat in ["AWID", "VWID", "WID"]:
            result[feat] = 5.0
        elif feat in ["AWID_SD", "VWID_SD", "WID_SD"]:
            result[feat] = 1.0
        elif feat == "ADV_RATIO":
            result[feat] = 0.1
        elif feat == "EA_RET":
            result[feat] = biomarkers.get("EX_AREA", 1.0)
        elif feat == "MAC":
            result[feat] = biomarkers.get("VESSEL_COMPLEXITY", 10.0)
        elif feat == "CTW_A":
            result[feat] = 0.5
        else:
            result[feat] = 0.0
    
    return result


def predict_severity(biomarkers, bundle):
    """Run severity prediction on extracted biomarkers."""
    pipe = bundle["pipeline"]
    features = bundle["features"]
    
    # Create DataFrame with the same columns as training
    X = pd.DataFrame([biomarkers], columns=features)
    
    # Handle any NaN values
    X = X.fillna(0)
    
    # Predict
    proba = pipe.predict_proba(X)[0]
    pred_class = int(pipe.predict(X)[0])
    
    pred_label = CLASSES[pred_class] if pred_class < len(CLASSES) else "Other"
    confidence = proba[pred_class] * 100
    
    return {
        "class": pred_label,
        "confidence": confidence,
        "probabilities": proba.tolist()
    }


# Streamlit UI
st.set_page_config(page_title="DR Screening", layout="wide")

st.title("Diabetic Retinopathy Screening")
st.markdown("""
Upload a retinal fundus image to get an AI-powered DR severity prediction.

**Note**: This is a cloud version with simplified image processing.
The full version with RRWNet segmentation is available locally.
""")

# Load model
try:
    bundle = st.cache_resource(load_model_bundle)()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.caption(f"Model: {bundle['model_name']} | Features: {len(bundle['features'])}")

# File upload
uploaded_file = st.file_uploader("Upload Retinal Fundus Image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if bgr is None:
        st.error("Could not read the uploaded file as an image.")
    else:
        # Display image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        col1, _ = st.columns([1, 2])
        with col1:
            st.image(rgb, caption="Input image", use_container_width=True)
        
        with st.spinner("Analyzing image and predicting DR severity..."):
            try:
                # Extract biomarkers
                biomarkers = extract_simplified_biomarkers(bgr)
                
                # Make prediction
                result = predict_severity(biomarkers, bundle)
                
                # Display results
                st.success("Analysis complete!")
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Predicted Stage", result["class"])
                with c2:
                    st.metric("Confidence", f"{result['confidence']:.1f}%")
                with c3:
                    st.metric("Model", bundle["model_name"])
                
                # Probability distribution
                st.subheader("DR Severity Probabilities")
                prob_df = pd.DataFrame({
                    "Stage": CLASSES,
                    "Probability (%)": [p * 100 for p in result["probabilities"]]
                })
                st.bar_chart(prob_df.set_index("Stage"))
                
                # Show processed image
                st.subheader("Processed View")
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                st.image(enhanced, caption="Enhanced view (CLAHE)", clamp=True, channels="GRAY")
                
                # Show biomarkers
                with st.expander("Extracted Biomarkers"):
                    for k, v in biomarkers.items():
                        if isinstance(v, (int, float)) and v > 0:
                            st.write(f"- {k}: {float(v):.4f}")
                            
            except Exception as e:
                st.error(f"Processing failed: {e}")
                st.exception(e)

else:
    st.info("Upload a fundus image to start analysis.")
    
    with st.expander("Sample Results (from training data)"):
        st.write("When you upload an image, you'll see:")
        st.write("- Predicted DR severity class (No DR, Mild, Moderate, Severe, Proliferative)")
        st.write("- Confidence score and probability distribution")
        st.write("- Extracted biomarkers from the image")
        st.write("- Enhanced visualization")
