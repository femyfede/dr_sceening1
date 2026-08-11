# Diabetic Retinopathy Screening - Streamlit Cloud Version
import os
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import joblib
import yaml
import streamlit as st
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

# Config
HUGGING_FACE_REPO = "samwema/dr_screening_model"
MODEL_BUNDLE_FILE = "model_bundle.joblib"
TRAIN_FEATURES_FILE = "features_train.csv"

# Severity classes
CLASSES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

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

def extract_biomarkers_from_image(image):
    """Extract basic biomarkers from uploaded image using OpenCV.
    
    This is a simplified CPU-based version that extracts some key features.
    """
    if image is None or image.size == 0:
        return None
    
    # Convert to RGB and resize
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Simple biomarkers that can be computed on CPU
    biomarkers = {}
    
    # Green channel analysis (common in DR detection)
    green = img[:, :, 1]
    
    # Exudate detection (simplified)
    exudate_mask = green > 180
    exudate_area = np.sum(exudate_mask) / (h * w) * 100
    biomarkers["EX_AREA"] = exudate_area
    biomarkers["EX_COUNT"] = np.sum(exudate_mask) // 1000  # Simplified count
    
    # Microaneurysm detection (simplified)
    # Dark dots in green channel
    micro_mask = (green < 80) & (green > 30)
    biomarkers["MA_COUNT"] = np.sum(micro_mask) // 100
    
    # Vessel analysis
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    vessels = cv2.filter2D(gray, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))
    vessel_area = np.sum(vessels > 150) / (h * w) * 100
    biomarkers["VESSEL_AREA"] = vessel_area
    
    # Retinal area estimate
    biomarkers["retina_area_px"] = h * w
    
    # Add remaining features as zeros or defaults (would need full pipeline for accurate values)
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
    for i, feat in enumerate(features):
        if feat in biomarkers:
            result[feat] = biomarkers[feat]
        else:
            result[feat] = 0.0
    
    return result, vessel_area

def make_predictions(biomarkers, bundle):
    """Run prediction using the loaded model bundle."""
    try:
        pipe = bundle["pipeline"]
        features = bundle["features"]
        
        # Create DataFrame with the same columns as training
        X = pd.DataFrame([biomarkers], columns=features)
        
        # Predict
        proba = pipe.predict_proba(X)[0]
        pred_class = int(pipe.predict(X)[0])
        
        # Get prediction info
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
    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if bgr is None:
        st.error("Could not read the uploaded file as an image.")
    else:
        # Display image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        st.image(rgb, caption="Input image", use_container_width=True)
        
        with st.spinner("Extracting biomarkers and running prediction..."):
            try:
                # Load model (cached)
                bundle = st.cache_resource(load_model_bundle)()
                
                # Extract biomarkers
                biomarkers, vessel_area = extract_biomarkers_from_image(bgr)
                
                if biomarkers is None:
                    st.error("Could not extract biomarkers from the image.")
                else:
                    # Make prediction
                    result = make_predictions(biomarkers, bundle)
                    
                    if result:
                        st.success("Prediction complete!")
                        
                        # Display results
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("Predicted Stage", result["class"])
                        with c2:
                            st.metric("Confidence", f"{result['confidence']:.1f}%")
                        
                        # Show probabilities
                        st.subheader("DR Severity Probabilities")
                        prob_df = pd.DataFrame({
                            "Stage": CLASSES,
                            "Probability (%)": [p * 100 for p in result["probabilities"]]
                        })
                        st.bar_chart(prob_df.set_index("Stage"))
                        
                        # Display extracted biomarkers
                        st.subheader("Extracted Biomarkers")
                        with st.expander("Show biomarker values"):
                            for k, v in biomarkers.items():
                                if v != 0:
                                    st.write(f"- {k}: {v:.4f}")
                                    
            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.exception(e)

else:
    st.info("Upload an image to start the DR screening analysis.")
