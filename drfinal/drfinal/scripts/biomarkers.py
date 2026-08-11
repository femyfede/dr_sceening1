"""Biomarker extraction functions for diabetic retinopathy fundus images.

Vascular biomarkers (1-9) are computed from the artery-vein (RRWNet) segmentation.
Lesion biomarkers (10-13) are computed from lesion masks that are generated from the
original fundus image via classical morphological detection (hemorrhage, exudate,
microaneurysm) and the original image.

Color convention for the A/V segmentation (RGB, 512x512):
    - red-dominant pixels   -> artery
    - blue-dominant pixels  -> vein
    - black                 -> background

Biomarkers produced:
    AVR, CRAE, CRVE, VD, AD, VeD, TI, CI, FD, LA, HA, EA, MAC
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize, remove_small_objects, disk
from skimage.measure import label
from scipy import ndimage

MIN_VESSEL_SIZE = 20
MIN_MA_SIZE = 3
MAX_MA_SIZE = 60
CONNECTED_LEN_THRESH = 25


# ---------------------------------------------------------------------------
# Retina region
# ---------------------------------------------------------------------------
def retina_mask_from_original(img):
    """Retina (fundus) region mask from an original BGR fundus image."""
    g = img[:, :, 1]
    _, rm = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rm = cv2.morphologyEx(rm, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    rm = cv2.morphologyEx(rm, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    lbl = label(rm > 0, connectivity=2)
    if lbl.max() > 0:
        sizes = np.bincount(lbl.ravel())
        rm = (lbl == (int(np.argmax(sizes[1:])) + 1))
    rm = ndimage.binary_fill_holes(rm)
    return rm.astype(bool)


# ---------------------------------------------------------------------------
# Vessel masks from A/V segmentation
# ---------------------------------------------------------------------------
def get_vessel_masks(seg):
    """Return (artery_mask, vein_mask) booleans from a BGR A/V segmentation image."""
    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
    R = seg[:, :, 0].astype(int)
    G = seg[:, :, 1].astype(int)
    B = seg[:, :, 2].astype(int)
    artery = (R > B + 8) & (R > G + 8)
    vein = (B > R + 8) & (B > G + 8)
    artery = remove_small_objects(artery, max_size=MIN_VESSEL_SIZE)
    vein = remove_small_objects(vein, max_size=MIN_VESSEL_SIZE)
    return artery, vein


def vessel_diameters(mask, skeleton):
    """Vessel width (px) at each skeleton pixel via distance transform."""
    if not mask.any():
        return np.array([])
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    diams = 2.0 * dt[skeleton]
    return diams[diams > 0]


def knudtson(diameters, k=0.88, n_largest=6):
    """Knudtson/Parr-Hubbard recursive summary: W = 0.88*sqrt(w^2 + d^2)."""
    if len(diameters) == 0:
        return np.nan
    d = np.sort(diameters)[::-1][:n_largest].astype(float)
    if len(d) == 1:
        return float(d[0])
    w = d[0]
    for x in d[1:]:
        w = k * np.sqrt(w * w + x * x)
    return float(w)


def _component_sizes(comps):
    return np.bincount(comps.ravel())


def branch_tortuosities(sk):
    """Per-branch tortuosity = arc length / chord length.

    Skeleton branches are split at junction pixels. Returns list of ratios.
    """
    sk = (sk > 0).astype(np.uint8)
    if sk.sum() < 4:
        return []
    kernel = np.ones((3, 3), np.float32)
    deg = cv2.filter2D(sk.astype(np.float32), -1, kernel) * sk.astype(np.float32)
    junc = deg >= 3.5
    branch_sk = sk & (~junc)
    n, comps = cv2.connectedComponents(branch_sk, connectivity=8)
    sizes = _component_sizes(comps)
    ratios = []
    for i in range(1, n):
        if sizes[i] < 4:
            continue
        ys, xs = np.where(comps == i)
        if len(ys) < 4:
            continue
        d = cv2.filter2D(comps.astype(np.uint8), -1, kernel) * (comps == i)
        eps = (comps == i) & (d <= 2)
        if eps.sum() == 0:
            continue
        ey, ex = np.where(eps)
        arc = float(len(ys))
        chord = 0.0
        if len(ex) >= 2:
            dists = np.hypot(ex[:, None] - ex[None, :], ey[:, None] - ey[None, :])
            chord = float(dists.max())
        if chord >= 5:
            ratios.append(arc / chord)
    return ratios


def tortuosity_index(mask):
    """TI = mean(arc/chord) over vessel branches."""
    sk = skeletonize(mask)
    r = branch_tortuosities(sk)
    return float(np.mean(r)) if r else np.nan


def continuity_index(mask, length_thresh=CONNECTED_LEN_THRESH):
    """CI = connected skeleton length / total skeleton length * 100."""
    sk = skeletonize(mask)
    if sk.sum() == 0:
        return np.nan
    n, comps = cv2.connectedComponents(sk.astype(np.uint8), connectivity=8)
    sizes = _component_sizes(comps)
    total = float(sizes[1:].sum())
    if total == 0:
        return np.nan
    connected = float(sizes[1:][sizes[1:] >= length_thresh].sum())
    return 100.0 * connected / total


def _box_count(mask, s):
    h, w = mask.shape
    m = mask[: h - h % s, : w - w % s]
    return m.reshape(h // s, s, w // s, s).any(axis=(1, 3)).sum()


def fractal_dimension(mask):
    """FD via box counting over the vessel skeleton."""
    sk = skeletonize(mask)
    if sk.sum() == 0:
        return np.nan
    sizes = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    sizes = [s for s in sizes if s <= min(sk.shape)]
    if len(sizes) < 3:
        return np.nan
    log_e = np.log([1.0 / s for s in sizes])
    log_n = np.log([max(_box_count(sk, s), 1) for s in sizes])
    return float(np.polyfit(log_e, log_n, 1)[0])


def mean_width(mask):
    """Mean and std of vessel diameter (px) over the skeleton of a mask."""
    if not mask.any():
        return np.nan, np.nan
    sk = skeletonize(mask)
    if sk.sum() == 0:
        return np.nan, np.nan
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    d = 2.0 * dt[sk]
    d = d[d > 0]
    if len(d) == 0:
        return np.nan, np.nan
    return float(d.mean()), float(d.std())


def junction_count(mask):
    """Number of junction (branching) pixels in the vessel skeleton."""
    sk = skeletonize(mask).astype(np.uint8)
    if sk.sum() == 0:
        return 0
    kernel = np.ones((3, 3), np.float32)
    deg = cv2.filter2D(sk.astype(np.float32), -1, kernel) * sk.astype(np.float32)
    return int((deg >= 3.5).sum())


def skeleton_length(mask):
    """Total vessel skeleton length in pixels."""
    sk = skeletonize(mask)
    return float(sk.sum()) if sk.any() else np.nan


# ---------------------------------------------------------------------------
# Lesion detection (classical morphology on original fundus image)
# ---------------------------------------------------------------------------
def _shade_correct(channel, radius=30):
    """Local background subtraction via fast box filter."""
    k = 2 * radius + 1
    bg = cv2.boxFilter(channel, ddepth=-1, ksize=(k, k), normalize=True)
    bg = cv2.GaussianBlur(bg, (0, 0), 5)
    bg = cv2.max(bg, 1)
    return cv2.subtract(channel.astype(np.int16), bg.astype(np.int16))


def _remove_small(mask, min_size):
    n, comps = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    sizes = _component_sizes(comps)
    keep = sizes >= min_size
    keep[0] = False
    return keep[comps]


def _adaptive_threshold(sc, sigma, low, high):
    mask = (sc > low)
    vals = sc[mask]
    if len(vals) == 0:
        return np.zeros(sc.shape, dtype=bool)
    thr = float(np.mean(vals) + sigma * np.std(vals))
    return sc > thr


def detect_lesions(img, retina=None):
    """Detect lesion masks from a BGR original fundus image.

    Returns dict with boolean masks: hemorrhage, exudate, microaneurysm, lesion.
    """
    h, w = img.shape[:2]
    green = img[:, :, 1].astype(np.int16)
    red = img[:, :, 2].astype(np.int16)

    if retina is None:
        retina = retina_mask_from_original(img)
    r = int(min(h, w) * 0.02) + 1
    retina = cv2.erode(retina.astype(np.uint8), np.ones((r, r), np.uint8)) > 0

    # ---- vessels (to suppress false microaneurysm detections) ----
    k_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    vtop = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_TOPHAT, k_v).astype(np.int16)
    vals = vtop[retina]
    vmask = np.zeros(vtop.shape, dtype=bool)
    if len(vals):
        vmask = vtop > float(np.mean(vals) + 1.5 * np.std(vals))
    vmask &= retina
    vmask = cv2.dilate(vmask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0

    # ---- exudates (bright yellow-white) ----
    sc_g = _shade_correct(green)
    sc_g[~retina] = 0
    ex = _adaptive_threshold(sc_g, 2.0, 8, 255)
    ex &= retina
    ex = _remove_small(ex, 26)
    ex = cv2.morphologyEx(ex.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0

    # ---- hemorrhages (dark red blotches) ----
    sc_r = -_shade_correct(red)
    sc_r[~retina] = 0
    hm = sc_r > 0
    vals = sc_r[hm]
    if len(vals):
        thr = float(np.mean(vals) + 2.0 * np.std(vals))
        hm = sc_r > thr
    else:
        hm = np.zeros(sc_r.shape, dtype=bool)
    hm &= retina
    hm = _remove_small(hm, 41)
    hm = ndimage.binary_fill_holes(hm)

    # ---- microaneurysms (small dark round dots, off vessels) ----
    k_ma = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bth = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_BLACKHAT, k_ma)
    bth = bth.astype(np.int16)
    bth[~retina] = 0
    ma = bth > 0
    vals = bth[ma]
    if len(vals):
        thr = float(np.mean(vals) + 2.5 * np.std(vals))
        ma = bth > thr
    else:
        ma = np.zeros(bth.shape, dtype=bool)
    ma &= retina
    ma &= ~vmask
    ma = _remove_small(ma, MIN_MA_SIZE + 1)
    ma = ma & ~ex & ~hm

    lesion = ex | hm | ma
    return {"hemorrhage": hm, "exudate": ex, "microaneurysm": ma, "lesion": lesion}


def microaneurysm_count(ma_mask, min_size=MIN_MA_SIZE, max_size=MAX_MA_SIZE):
    n, comps = cv2.connectedComponents(ma_mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return 0
    sizes = _component_sizes(comps)
    sizes[0] = 0
    return int(((sizes >= min_size) & (sizes <= max_size)).sum())


# ---------------------------------------------------------------------------
# Full per-image biomarker computation
# ---------------------------------------------------------------------------
def vessel_retina_area(vessel, radius=60):
    """Retina area estimate in A/V-seg space by dilating the vessel tree."""
    if not vessel.any():
        return 0.0
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    d = cv2.dilate(vessel.astype(np.uint8), kernel)
    return float(ndimage.binary_fill_holes(d > 0).sum())


def extract_all(original_img, av_img):
    """Compute all biomarkers for one image pair.

    original_img: BGR fundus image (480x640).
    av_img: BGR artery-vein segmentation (512x512).
    Returns dict of biomarker name -> float.
    """
    res = {}

    # retina area in A/V space (hybrid: resized original retina vs vessel-dilated estimate)
    retina_orig = retina_mask_from_original(original_img)
    retina = cv2.resize(retina_orig.astype(np.uint8), (av_img.shape[1], av_img.shape[0]),
                        interpolation=cv2.INTER_NEAREST) > 0
    retina_area = float(retina.sum())
    res["retina_area_px"] = retina_area if retina_area > 0 else np.nan

    artery, vein = get_vessel_masks(av_img)
    vessel = artery | vein
    vessel_area = float(vessel.sum())
    art_area = float(artery.sum())
    vein_area = float(vein.sum())

    retina_area = max(retina_area, vessel_retina_area(vessel))
    res["retina_area_px"] = retina_area if retina_area > 0 else np.nan
    denom = retina_area if retina_area > 0 else np.nan
    res["VD"] = 100.0 * vessel_area / denom if denom else np.nan
    res["AD"] = 100.0 * art_area / denom if denom else np.nan
    res["VeD"] = 100.0 * vein_area / denom if denom else np.nan

    # diameters
    sk_a = skeletonize(artery)
    sk_v = skeletonize(vein)
    d_a = vessel_diameters(artery, sk_a)
    d_v = vessel_diameters(vein, sk_v)
    crae = knudtson(d_a)
    crve = knudtson(d_v)
    res["CRAE"] = crae
    res["CRVE"] = crve
    res["AVR"] = crae / crve if crve and not np.isnan(crve) else np.nan

    # tortuosity / continuity / fractal dimension over full vessel tree
    res["TI"] = tortuosity_index(vessel)
    res["CI"] = continuity_index(vessel)
    res["FD"] = fractal_dimension(vessel)

    # per-artery/vein morphology, caliber and branching
    res["ATI"] = tortuosity_index(artery)
    res["VTI"] = tortuosity_index(vein)
    res["AFD"] = fractal_dimension(artery)
    res["VFD"] = fractal_dimension(vein)
    res["JUNC"] = junction_count(vessel)
    res["VLEN"] = skeleton_length(vessel)
    res["AWID"], res["AWID_SD"] = mean_width(artery)
    res["VWID"], res["VWID_SD"] = mean_width(vein)
    res["WID"], res["WID_SD"] = mean_width(vessel)
    res["ADV_RATIO"] = res["AD"] / res["VeD"] if res["VeD"] and not np.isnan(res["VeD"]) else np.nan

    # lesions (computed masks from the original image)
    les = detect_lesions(original_img, retina=retina_orig)
    res["LA"] = float(les["lesion"].sum())
    res["HA"] = float(les["hemorrhage"].sum())
    res["EA"] = float(les["exudate"].sum())
    res["MAC"] = float(microaneurysm_count(les["microaneurysm"]))

    res["_artery_px"] = art_area
    res["_vein_px"] = vein_area
    res["_vessel_px"] = vessel_area
    res["_lesions"] = les
    return res


def make_lesion_composite(lesions):
    """Color-coded composite of lesion masks: HM=red, EX=yellow, MA=green."""
    comp = np.zeros((lesions["hemorrhage"].shape[0],
                     lesions["hemorrhage"].shape[1], 3), dtype=np.uint8)
    comp[lesions["hemorrhage"]] = (0, 0, 255)
    comp[lesions["exudate"]] = (0, 255, 255)
    comp[lesions["microaneurysm"]] = (0, 255, 0)
    return comp
