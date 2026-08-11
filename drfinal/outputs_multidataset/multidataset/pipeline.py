"""CUDA-aware per-image processing: RRWNet A/V segmentation + biomarkers.

Reuses the existing project modules (biomarkers.py, train_models.py,
extract_lesions_dl.py) and the RRWNet model/weights. `extract.py` drives this
over a dataset manifest and writes outputs/features.csv for `train.py`.
"""

import os
import sys

import numpy as np
import cv2
import torch


def pick_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def ensure_paths(cfg):
    for key in ("project_scripts", "rrwnet_module_dir"):
        p = cfg.get(key)
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def load_rrwnet(cfg, device):
    from model import RRWNet
    m = RRWNet(iterations=int(cfg.get("rrwnet_iterations", 1)))
    m.load_state_dict(torch.load(cfg["rrwnet_weights"], map_location="cpu", weights_only=True))
    m.to(device)
    m.eval()
    return m


def segment_av(model, bgr, device, target_size=512):
    """RRWNet A/V/vessel probabilities (H,W,3): ch0=artery, ch1=vein, ch2=vessel."""
    from preprocessing import enhance_image
    from utils import pad_images_unet, to_torch_tensors

    raw = cv2.resize(bgr, (target_size, target_size), interpolation=cv2.INTER_AREA)
    dummy = np.full((target_size, target_size), 255, dtype=np.uint8)
    img, mask = enhance_image(raw.copy(), dummy)
    img = img if img.max() <= 1.0 else img / 255.0
    mask = mask if mask.max() <= 1.0 else mask / 255.0

    imgs, paddings = pad_images_unet([img, mask])
    img_p, padding = imgs[0], paddings[0]
    mask_p = np.stack([imgs[1]] * 3, axis=2)

    it, mt = to_torch_tensors([img_p, mask_p])
    it, mt = it.to(device), mt.to(device)
    with torch.no_grad():
        p = torch.sigmoid(model(it.unsqueeze(0))[-1])
    p[mt.unsqueeze(0) < 0.5] = 0

    h_s = slice(padding[0][0], -padding[0][1] if padding[0][1] else None)
    w_s = slice(padding[1][0], -padding[1][1] if padding[1][1] else None)
    return p[:, :, h_s, w_s].squeeze(0).permute(1, 2, 0).cpu().numpy()


def compute_lesion_features_cuda(rgb, device):
    from fundus_lesions_toolkit.models.segmentation import segment
    from fundus_lesions_toolkit.constants import Dataset
    from extract_lesions_dl import features_from_labels, retina_area

    pred = segment(rgb, train_datasets=Dataset.ALL, device=device, image_resolution=1024)
    labels = pred.argmax(0).cpu().numpy().astype(np.uint8)
    feats = features_from_labels(labels, retina_area(rgb))
    feats["_labels"] = labels
    return feats


def process_one(cfg, model, row, overlay_dir=None):
    """Compute every feature for one manifest row. Returns (dict or None, error-or-None)."""
    import biomarkers
    import train_models as tm

    if not isinstance(row, dict):
        row = row._asdict()
    path = row["path"]
    bgr = cv2.imread(path)
    if bgr is None:
        return None, f"cannot read {path}"
    device = pick_device(cfg.get("device", "auto"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    pred = segment_av(model, bgr, device)
    av_bgr = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
    av_bgr = cv2.cvtColor(av_bgr, cv2.COLOR_RGB2BGR)

    vasc = biomarkers.extract_all(bgr, av_bgr)
    les = compute_lesion_features_cuda(rgb, device)

    if overlay_dir is not None and os.path.basename(overlay_dir).startswith("sample"):
        labels = les.get("_labels")
        if labels is not None:
            comp = np.zeros((*labels.shape, 3), dtype=np.uint8)
            comp[labels == 3] = (0, 0, 255)
            comp[labels == 2] = (0, 255, 255)
            comp[labels == 4] = (0, 255, 0)
            comp[labels == 1] = (255, 0, 255)
            cv2.imwrite(os.path.join(overlay_dir, row["image_id"] + "_lesions.png"), comp)
        cv2.imwrite(os.path.join(overlay_dir, row["image_id"] + "_seg.png"), av_bgr)

    feats = {f: vasc.get(f, np.nan) for f in tm.FEATURES}
    for f in ("LA", "HA", "EA", "MAC", "HE_COUNT", "EX_COUNT", "CTW_A", "MA_A",
              "LA_RET", "HA_RET", "EA_RET"):
        feats[f] = les[f]

    out = {
        "image_id": row["image_id"],
        "dataset": row["dataset"],
        "split": row["split"],
        "label_6": row["label_6"],
        "retina_area_px": vasc.get("retina_area_px", np.nan),
    }
    out.update(feats)
    return out, None
