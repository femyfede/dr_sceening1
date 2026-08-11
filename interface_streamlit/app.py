# Diabetic Retinopathy Screening - Streamlit Cloud Version
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

warnings.filterwarnings("ignore")

# Config
HUGGING_FACE_REPO = "samwema/dr_screening_model"
MODEL_BUNDLE_FILE = "model_bundle.joblib"
FEATURES_TRAIN_FILE = "features_train.csv"

# Severity classes
CLASSES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR", "Other"]

st.set_page_config(page_title="DR Screening", layout="wide")
st.title("Diabetic Retinopathy Screening")

def load_model_bundle():
    """Download and load model bundle from Hugging Face."""
    try:
        bundle_path = hf_hub_download(repo_id=HUGGING_FACE_REPO, filename=MODEL_BUNDLE_FILE)
        return joblib.load(bundle_path)
    except Exception as e:
        st.error(f"Failed to load model bundle from Hugging Face: {e}")
        st.stop()

@st.cache_resource
def load_bundle_cached():
    return load_model_bundle()

def extract_biomarkers_from_image(image):
    """Extract biomarkers from uploaded image (simplified CPU version)."""
    if image is None or image.size == 0:
        return None, None
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    biomarkers = {}
    
    # Green channel analysis
    green = img[:, :, 1]
    
    # Exudate detection
    exudate_mask = green > 180
    biomarkers["EX_AREA"] = np.sum(exudate_mask) / (h * w) * 100
    biomarkers["EX_COUNT"] = max(1, np.sum(exudate_mask) // 5000)
    
    # Microaneurysm detection (dark dots)
    micro_mask = (green < 80) & (green > 30)
    biomarkers["MA_COUNT"] = max(1, np.sum(micro_mask) // 100)
    
    # Vessel enhancement
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1)
    vessels = cv2.addWeighted(gray, 1.5, gray, -0.5, 0)
    vessel_pixels = np.sum(vessels > 150)
    biomarkers["VESSEL_AREA_PX"] = vessel_pixels
    biomarkers["retina_area_px"] = float(h * w)
    
    # Optic disc estimate
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        biomarkers["optic_disc_area"] = max(area, 1.0)
        x, y, w_box, h_box = cv2.boundingRect(largest)
        biomarkers["optic_disc_major"] = w_box
        biomarkers["optic_disc_minor"] = h_box
        biomarkers["optic_disc_ratio"] = w_box / max(h_box, 1)
    
    # Vessel width estimate
    biomarkers["vessel_width_mean"] = np.mean(gray[vessels > 150]) if np.any(vessels > 150) else 0
    biomarkers["vessel_density"] = vessel_pixels / (h * w) * 100
    
    # Return all features
    features = [
        "optic_disc_area", "optic_disc_major", "optic_disc_minor", "optic_disc_ratio",
        "optic_cup_area", "optic_cup_major", "optic_cup_minor", "cup_disc_ratio",
        "vessel_width_mean", "vessel_width_std", "vessel_length",
        "vessel_density", "vessel_tortuosity_mean", "vessel_tortuosity_std",
        "artery_area", "artery_length", "artery_width_mean",
        "vein_area", "vein_length", "vein_width_mean",
        "hemorrhage_area", "hemorrhage_count",
        "cotton_wool_area", "cotton_wool_count",
        "exudate_area", "exudate_count",
        "microaneurysm_count", "microaneurysm_area",
        "hard_exudate_count", "hard_exudate_area",
        "vessel_complexity", "retina_area_px"
    ]
    
    result = {}
    for feat in features:
        if feat in biomarkers:
            result[feat] = biomarkers[feat]
        else:
            # Calculate related features from available data
            if feat == "optic_cup_area":
                result[feat] = biomarkers.get("optic_disc_area", 0) * 0.2
            elif feat == "optic_cup_major":
                result[feat] = biomarkers.get("optic_disc_major", 0) * 0.5
            elif feat == "optic_cup_minor":
                result[feat] = biomarkers.get("optic_disc_minor", 0) * 0.5
            elif feat == "cup_disc_ratio":
                result[feat] = 0.2
            elif feat == "hemorrhage_area":
                result[feat] = biomarkers.get("EX_AREA", 0) * 0.5
            elif feat == "hemorrhage_count":
                result[feat] = biomarkers.get("EX_COUNT", 0)
            elif "length" in feat:
                result[feat] = biomarkers.get("VESSEL_AREA_PX", 0)
            elif "complexity" in feat:
                result[feat] = vessel_pixels / (h * w) * 100 if h * w > 0 else 0
            else:
                result[feat] = 0.0
    
    display_img = cv2.cvtColor((np.clip(vessels, 0, 255)).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    return result, display_img

def make_predictions(biomarkers, bundle):
    """Run prediction using the loaded model bundle."""
    try:
        pipe = bundle["pipeline"]
        features = bundle["features"]
        
        X = pd.DataFrame([biomarkers], columns=features)
        
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
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Interface
st.markdown("""
Upload a retinal fundus image. The app will:
1. Extract biomarkers from the image
2. Predict the DR severity class  
3. Display the prediction results

**Note**: This Streamlit Cloud version uses simplified CPU-based biomarker extraction.
For full RRWNet segmentation and DL lesion detection, run the local version.
""")

uploaded_file = st.file_uploader("Upload Retinal Fundus Image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if bgr is None:
        st.error("Could not read the uploaded file as an image.")
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(rgb, caption="Input image", use_container_width=True)
        
        with st.spinner("Loading model and analyzing image..."):
            try:
                bundle = load_bundle_cached()
                
                biomarkers, vessel_img = extract_biomarkers_from_image(bgr)
                
                if biomarkers is None:
                    st.error("Could not extract biomarkers from the image.")
                else:
                    result = make_predictions(biomarkers, bundle)
                    
                    if result:
                        st.success("Prediction complete!")
                        
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Predicted Stage", result["class"])
                        with c2:
                            st.metric("Confidence", f"{result['confidence']:.1f}%")
                        with c3:
                            if vessel_img is not None:
                                st.image(vessel_img, caption="Processed View", use_container_width=True)
                        
                        st.subheader("DR Severity Probabilities")
                        prob_df = pd.DataFrame({
                            "Stage": CLASSES,
                            "Probability (%)": [p * 100 for p in result["probabilities"]]
                        })
                        st.bar_chart(prob_df.set_index("Stage"))
                        
                        st.subheader("Extracted Biomarkers")
                        with st.expander("Show biomarker values"):
                            for k, v in biomarkers.items():
                                st.write(f"- {k}: {float(v):.4f}")
                                
            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.exception(e)

else:
    st.info("Upload an image to start the DR screening analysis.")
    st.markdown("### Sample Use Case")
    st.write("This tool can help screen for diabetic retinopathy by analyzing retinal fundus images.")
    st.write("Upload a fundus photo and get an immediate severity assessment.")
