# Diabetic Retinopathy Screening Interface (Streamlit Cloud Version)
# Cloud version with simplified biomarker extraction
import os
import sys
import warnings
from pathlib import Path

# Add current directory to path for custom model classes
ROOT_DIR = Path(__file__).parent
SCRIPTS_DIR = ROOT_DIR / "interface_streamlit"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from PIL import Image
import joblib
import streamlit as st
from huggingface_hub import hf_hub_download

# Required for loading the model bundle (custom classes)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline

# Import train_models for ClassMeanImputer (required by model bundle)
import train_models

# SciPy ndimage for image processing (replaces OpenCV)
from scipy import ndimage
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")


# Configuration
HUGGING_FACE_REPO = "samwema/dr_screening_model"
MODEL_BUNDLE_FILE = "model_bundle.joblib"
TRAIN_FEATURES_FILE = "features_train.csv"

# Class labels - use the actual classes from the model (5 DR grades)
CLASSES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]


def load_model_bundle():
    """Download and load model bundle from Hugging Face."""
    bundle_path = hf_hub_download(repo_id=HUGGING_FACE_REPO, filename=MODEL_BUNDLE_FILE)
    return joblib.load(bundle_path)


def load_training_features():
    """Download sample training features for reference."""
    features_path = hf_hub_download(repo_id=HUGGING_FACE_REPO, filename=TRAIN_FEATURES_FILE)
    return pd.read_csv(features_path)


def extract_simplified_biomarkers(image_array, train_features):
    """Extract simplified biomarkers from fundus image on CPU.
    
    Uses statistical methods to approximate the features used in training.
    For accurate biomarker extraction, use the local full version with RRWNet.
    """
    h, w = image_array.shape[:2]
    biomarkers = {}
    
    # Green channel (standard in DR analysis)
    if image_array.ndim == 3:
        green = image_array[:, :, 1]
        gray = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        green = image_array
        gray = image_array
    
    # Normalize
    green_f = green.astype(float)
    gray_f = gray.astype(float)
    
    # Use statistical properties from training data as baseline
    train_means = train_features.mean(numeric_only=True)
    
    # Exudate detection
    exudate_mask = green > np.percentile(green, 90)
    biomarkers["EX_COUNT"] = int(np.sum(exudate_mask) // 500)
    biomarkers["EX_AREA"] = float(np.sum(exudate_mask) / (h * w) * 100)
    
    # Hard exudates
    hard_mask = green > np.percentile(green, 95)
    biomarkers["hard_exudate_count"] = int(np.sum(hard_mask) // 500)
    biomarkers["hard_exudate_area"] = float(np.sum(hard_mask) / (h * w) * 100)
    
    # Microaneurysms (dark dots)
    micro_mask = green < np.percentile(green, 10)
    biomarkers["MA_COUNT"] = int(np.sum(micro_mask) // 100)
    biomarkers["MA_A"] = float(np.sum(micro_mask) / (h * w) * 100)
    
    # Hemorrhage detection
    hem_mask = (green < np.percentile(green, 20)) & (green > np.percentile(green, 5))
    biomarkers["HE_COUNT"] = int(np.sum(hem_mask) // 300)
    biomarkers["HA"] = float(np.sum(hem_mask) / (h * w) * 100)
    biomarkers["HA_RET"] = biomarkers["HA"]
    
    # Vessel detection using Gaussian enhancement
    blurred = gaussian_filter(gray_f / 255.0 if gray_f.max() > 1 else gray_f, sigma=1)
    vessels = (gray_f / 255.0 if gray_f.max() > 1 else gray_f) * 1.5 - blurred * 0.5
    vessels = np.clip(vessels * 255, 0, 255)
    vessel_mask = vessels > 120
    vessel_pixels = int(np.sum(vessel_mask))
    biomarkers["VESSEL_COMPLEXITY"] = float(vessel_pixels / (h * w) * 100)
    biomarkers["LA"] = float(vessel_pixels / (h * w) * 100)
    biomarkers["LA_RET"] = biomarkers["LA"]
    
    # Optic disc approximation
    thresh = gray > np.percentile(gray, 95)
    labels, num = ndimage.label(thresh)
    od_area = 0
    if num > 0:
        sizes = ndimage.sum(thresh, labels, range(num + 1))
        largest_label = np.argmax(sizes[1:]) + 1
        od_area = sizes[largest_label]
    biomarkers["optic_disc_area"] = float(od_area)
    
    # Retinal area
    biomarkers["retina_area_px"] = float(h * w)
    
    # Fill features using training data stats as guidance
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
        elif feat in train_means.index:
            # Use training mean as baseline
            result[feat] = float(train_means[feat]) if not np.isnan(train_means[feat]) else 0.0
        elif feat == "AVR":
            result[feat] = 0.15
        elif feat in ["CRAE", "CRVE"]:
            result[feat] = float(train_means.get(feat, 50.0))
        elif feat == "EA_RET":
            result[feat] = biomarkers.get("EX_AREA", 1.0)
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
    
    # Use actual number of classes from model
    n_classes = len(proba)
    class_names = CLASSES[:n_classes]
    
    pred_label = class_names[pred_class] if pred_class < len(class_names) else "Unknown"
    confidence = proba[pred_class] * 100
    
    return {
        "class": pred_label,
        "confidence": confidence,
        "probabilities": proba.tolist(),
        "class_names": class_names
    }


# Streamlit UI
st.set_page_config(page_title="DR Screening", layout="wide")

st.title("Diabetic Retinopathy Screening")
st.markdown("""
Upload a retinal fundus image to get an AI-powered DR severity prediction.

**Note**: This is a cloud version with simplified image processing.
The full version with RRWNet segmentation runs locally with GPU for best accuracy.
""")

# Load model and training data
try:
    bundle = st.cache_resource(load_model_bundle)()
    train_features = st.cache_data(load_training_features)()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

st.caption(f"Model: {bundle['model_name']} | Features: {len(bundle['features'])}")

# File upload
uploaded_file = st.file_uploader("Upload Retinal Fundus Image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

if uploaded_file is not None:
    try:
        # Read image using PIL
        pil_img = Image.open(uploaded_file)
        rgb = np.array(pil_img.convert("RGB"))
        h, w = rgb.shape[:2]
        
        # Create a processed (enhanced) view
        gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(rgb, caption="Input image", use_container_width=True)
        
        with st.spinner("Analyzing image and predicting DR severity..."):
            # Extract biomarkers using simplified method
            biomarkers = extract_simplified_biomarkers(rgb, train_features)
            
            # Make prediction
            result = predict_severity(biomarkers, bundle)
            class_names = result["class_names"]
            
            with col2:
                # Show enhanced image
                gray_f = gray.astype(float)
                if gray_f.max() > 1:
                    gray_f = gray_f / 255.0
                enhanced = np.clip((gray_f - gray_f.mean()) / (gray_f.std() + 1e-8) * 50 + 128, 0, 255).astype(np.uint8)
                st.image(enhanced, caption="Enhanced view", use_container_width=True, channels="GRAY")
            
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
                "Stage": class_names,
                "Probability (%)": [p * 100 for p in result["probabilities"]]
            })
            st.bar_chart(prob_df.set_index("Stage"))
            
            # Visualize biomarker extraction
            st.subheader("Biomarker Visualization")
            fig_col1, fig_col2, fig_col3 = st.columns(3)
            
            with fig_col1:
                green = rgb[:, :, 1]
                green_f = green.astype(float) / 255.0 if green.max() > 1 else green.astype(float)
                exudate_mask = (green_f > np.percentile(green_f, 90)) * 255
                st.image(exudate_mask.astype(np.uint8), caption="Exudates", channels="GRAY")
            
            with fig_col2:
                micro_mask = (green_f < np.percentile(green_f, 10)) * 255
                st.image(micro_mask.astype(np.uint8), caption="Microaneurysms", channels="GRAY")
            
            with fig_col3:
                gray_f = gray.astype(float) / 255.0 if gray.max() > 1 else gray.astype(float)
                blurred = gaussian_filter(gray_f, sigma=1)
                vessels = gray_f * 1.5 - blurred * 0.5
                vessel_mask = (np.clip(vessels * 255, 0, 255) > 120) * 255
                st.image(vessel_mask.astype(np.uint8), caption="Vessels", channels="GRAY")
            
            # Show biomarkers
            with st.expander("Extracted Biomarkers"):
                for k, v in biomarkers.items():
                    if isinstance(v, (int, float)) and v > 0:
                        st.write(f"- {k}: {float(v):.4f}")
                        
    except Exception as e:
        st.error(f"Processing failed: {e}")
        import traceback
        st.text(traceback.format_exc())

else:
    st.info("Upload a fundus image to start analysis.")
    
    with st.expander("How it works"):
        st.write("1. **Upload** a retinal fundus image (PNG/JPG)")
        st.write("2. **Biomarkers** are extracted using image processing")
        st.write("3. **Severity prediction** uses a RandomForest model trained on 33 biomarkers")
        st.write("4. **Results** show DR severity class and probability distribution")
        
        st.warning("The cloud version uses simplified biomarker extraction. For clinical accuracy, use the local version with RRWNet segmentation and GPU processing.")
