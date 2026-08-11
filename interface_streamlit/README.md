# Diabetic Retinopathy Screening Interface (Streamlit Cloud Version)

This Streamlit app predicts diabetic retinopathy severity from retinal fundus images.
The full-featured version with RRWNet segmentation runs locally, but this cloud version
offers a demo with simplified preprocessing.

[![Open in Streamlit Community Cloud](https://static.streamlit.io/badges/streamlit_badge_black.svg)](https://streamlit.io/cloud)

## Features

1. Upload a retinal fundus image
2. Run severity prediction model
3. View DR severity probability distribution
4. See extracted biomarkers

## Usage

Run locally:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Or open in Streamlit Community Cloud:
https://share.streamlit.io/samwema/dr_sceening1/streamlit_app.py

## Model

The model bundle is hosted on Hugging Face: https://huggingface.co/samwema/dr_screening_model
