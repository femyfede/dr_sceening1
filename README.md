# DR Screening Model

Diabetic Retinopathy Screening Pipeline

## Model Download

The trained model is hosted on Hugging Face Hub due to its large file size (>500MB).

Download via Hugging Face CLI:
```bash
pip install huggingface_hub
huggingface-cli download --repo-type model samwema/dr_screening_model
```

Or via Python:
```python
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="samwema/dr_screening_model", filename="model_bundle.joblib")
```

Direct URL: https://huggingface.co/samwema/dr_screening_model

## Project Structure

- `interface/` - Web interface for the screening tool
- `scripts/` - Feature extraction and model training scripts
- `outputs/` - Output files (predictions, visualizations)
- `outputs_multidataset/` - Multi-dataset training outputs
