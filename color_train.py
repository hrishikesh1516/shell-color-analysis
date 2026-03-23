"""
color_train.py - Enhanced Shell Color Analysis & Training System
================================================================
A standalone Python script for improved shell color detection and training.

Key features:
  - Conservative pigment extraction  (keeps light colors like cream, pale pink)
  - Selective glare removal          (only extreme glare: V>250, S<5%)
  - Hierarchical clustering          (auto-discover natural number of colors)
  - Parameter learning from data     (learn optimal thresholds from training set)
  - Auto-generated color names       (HSV-based descriptive names + hex codes)
  - Professional visualizations      (hue distribution, dendrogram, palette, report)
  - Multiple output formats          (console table, CSV, JSON)

Usage
-----
# Train on a folder of images (auto-learns parameters, saves color_model.pkl)
python color_train.py --mode train --input-folder "/path/to/images"

# Analyse a single image using the trained model
python color_train.py --mode analyze --image "/path/to/image.jpg" --use-trained-model

# Analyse a folder using the trained model
python color_train.py --mode analyze --input-folder "/path/to/images" --use-trained-model

# Analyse a single image without a trained model (fresh clustering)
python color_train.py --mode analyze --image "/path/to/image.jpg"
"""

import argparse
import csv
import json
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist

try:
    from sklearn.metrics import silhouette_score, silhouette_samples
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "STANDARD_CANVAS_SIZE": 1000,
    "MODEL_PATH": "color_model.pkl",
    "OUTPUT_FOLDER": "./output",
    # Glare removal
    "GLARE_V_THRESHOLD": 250,          # V channel (0-255) above which a pixel is potentially glare
    "GLARE_S_THRESHOLD": 13,           # S channel (0-255) below which a pixel is achromatic glare (~5%)
    "GLARE_MIN_FRACTION": 0.01,        # Skip inpainting if glare < 1 % of shell area
    # Pigment filtering
    "PIGMENT_S_MIN_PCT": 5,            # Minimum HSV saturation % to count as pigmented
    "WHITE_RGB_MIN": 240,              # Pure white threshold (all channels >= this)
    "BLACK_RGB_MAX": 20,               # Pure black threshold (all channels <= this)
    # Hierarchical clustering
    "HIER_DISTANCE_PERCENTILE": 85,    # Dendrogram cut height percentile (learned)
    "HIER_SAMPLE_SIZE": 3000,          # Max pixels sampled for clustering
    "HIER_RANDOM_STATE": 42,
    # Merge
    "COLOR_MERGE_PERCENTILE": 30,
    # Confidence
    "CONFIDENCE_SCALE_LAB": 25.0,
}

# ---------------------------------------------------------------------------
# Color naming helpers
# ---------------------------------------------------------------------------

# Hue boundaries (degrees, 0-360)
_HUE_RANGES = [
    (0,   15,  "Red"),
    (15,  30,  "Red-Orange"),
    (30,  45,  "Orange"),
    (45,  60,  "Yellow-Orange"),
    (60,  80,  "Yellow"),
    (80,  150, "Green"),
    (150, 165, "Cyan-Green"),
    (165, 195, "Cyan"),
    (195, 240, "Blue"),
    (240, 270, "Blue-Purple"),
    (270, 300, "Purple"),
    (300, 330, "Magenta"),
    (330, 345, "Pink"),
    (345, 360, "Red"),
]


def _hue_family(hue_deg: float) -> str:
    """Return the color family name for *hue_deg* (0–360)."""
    for lo, hi, name in _HUE_RANGES:
        if lo <= hue_deg < hi:
            return name
    return "Red"  # 360 wraps back to red


def _saturation_modifier(s_pct: float) -> str:
    """Return a saturation-based qualifier (percentage 0-100)."""
    if s_pct < 10:
        return "Pale"
    if s_pct < 30:
        return "Muted"
    if s_pct < 60:
        return ""          # no prefix for moderate saturation
    return "Vivid"


def _value_modifier(v_pct: float) -> str:
    """Return a brightness-based qualifier (percentage 0-100)."""
    if v_pct < 20:
        return "Very Dark"
    if v_pct < 40:
        return "Dark"
    if v_pct < 60:
        return ""          # mid-range brightness needs no qualifier
    if v_pct < 80:
        return "Light"
    return "Very Light"


def generate_color_name(rgb: tuple) -> str:
    """
    Auto-generate a descriptive color name from an RGB triple (0–255 each).

    The name is built from:
      - Hue family   : Red, Orange, Green, Blue, Purple …
      - Saturation   : Pale / Muted / (none) / Vivid
      - Value        : Very Dark / Dark / (none) / Light / Very Light

    Special-cases achromatic colours (low saturation):
      - Very Light → "White" / "Cream"
      - Very Dark  → "Black"
      - Otherwise  → "Gray"

    Examples
    --------
    >>> generate_color_name((200, 150, 100))
    'Light Orange'
    >>> generate_color_name((220, 200, 210))
    'Pale Pink'
    >>> generate_color_name((100, 50, 150))
    'Vivid Purple'
    """
    r, g, b = [int(c) for c in rgb]
    # OpenCV HSV: H 0-180, S 0-255, V 0-255
    pixel = np.array([[[r, g, b]]], dtype=np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0][0]
    h_cv, s_cv, v_cv = int(hsv[0]), int(hsv[1]), int(hsv[2])

    hue_deg = h_cv * 2.0          # convert 0-180 → 0-360
    s_pct = s_cv / 255.0 * 100.0  # 0-100
    v_pct = v_cv / 255.0 * 100.0  # 0-100

    # --- Achromatic special cases ---
    if s_pct < 10:
        if v_pct >= 80:
            return "Cream White" if v_pct < 95 else "White"
        if v_pct < 20:
            return "Black"
        return "Gray"

    family = _hue_family(hue_deg)
    sat_mod = _saturation_modifier(s_pct)
    val_mod = _value_modifier(v_pct)

    parts = [p for p in (val_mod, sat_mod, family) if p]
    return " ".join(parts)


def rgb_to_hex(rgb: tuple) -> str:
    """Convert an RGB triple (0–255) to a hex colour string (#RRGGBB)."""
    r, g, b = [int(c) for c in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


# ---------------------------------------------------------------------------
# LAB conversion helpers
# ---------------------------------------------------------------------------

def _pixels_rgb_to_lab(pixels: np.ndarray) -> np.ndarray:
    """Convert (N, 3) RGB array to (N, 3) CIELAB float array."""
    arr = np.clip(pixels, 0, 255).astype(np.uint8).reshape(1, -1, 3)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    return lab.reshape(-1, 3).astype(float)


def _rgb_to_lab(rgb: tuple) -> np.ndarray:
    """Convert a single RGB triple to CIELAB float array."""
    img = np.array([[[int(rgb[0]), int(rgb[1]), int(rgb[2])]]], dtype=np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[0][0].astype(float)


# ---------------------------------------------------------------------------
# 1. Image normalisation
# ---------------------------------------------------------------------------

def normalize_image(image: np.ndarray, target_size: int = 1000) -> np.ndarray:
    """
    Resize *image* so that its largest dimension equals *target_size*,
    padding the shorter dimension with black to produce a square canvas.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C) in RGB or RGBA.
    target_size : int
        Side length of the output square.

    Returns
    -------
    np.ndarray
        Normalised RGB image of shape (target_size, target_size, 3).
    """
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


# ---------------------------------------------------------------------------
# 2. Shell mask extraction
# ---------------------------------------------------------------------------

def extract_shell_mask(image: np.ndarray) -> np.ndarray:
    """
    Return a binary mask (uint8, 0/255) of the shell region.

    Uses *rembg* when available; falls back to Otsu thresholding on the
    grayscale image otherwise.

    Parameters
    ----------
    image : np.ndarray
        RGB image (H, W, 3).

    Returns
    -------
    np.ndarray
        Binary mask (H, W) with 255 = shell foreground, 0 = background.
    """
    if REMBG_AVAILABLE:
        pil_image = Image.fromarray(image)
        result = rembg_remove(pil_image)
        result_np = np.array(result)
        if result_np.shape[2] == 4:
            alpha = result_np[:, :, 3]
            mask = (alpha > 10).astype(np.uint8) * 255
        else:
            gray = cv2.cvtColor(result_np, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    else:
        logger.warning("rembg unavailable – using fallback Otsu mask.")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological clean-up: close small holes, remove tiny blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


# ---------------------------------------------------------------------------
# 3. Selective glare removal
# ---------------------------------------------------------------------------

def selective_glare_removal(
    image: np.ndarray,
    shell_mask: np.ndarray,
    v_threshold: int = 250,
    s_threshold: int = 13,
    min_fraction: float = 0.01,
) -> np.ndarray:
    """
    Remove only *extreme* specular glare from the shell image.

    A pixel is classified as glare only when:
      - HSV Value   > v_threshold  (very bright, default 250/255)
      - HSV Saturation < s_threshold (essentially achromatic, default ~5.1%, s_threshold=13/255)

    If the glare region is smaller than *min_fraction* of the shell area the
    image is returned unchanged (so that legitimate pale/cream pigments are
    preserved).  When inpainting is performed, edge-aware (Navier-Stokes)
    inpainting is used to respect colour boundaries.

    Parameters
    ----------
    image : np.ndarray
        RGB image to process.
    shell_mask : np.ndarray
        Binary mask (0/255) of the shell foreground.
    v_threshold : int
        HSV V channel threshold (0–255).
    s_threshold : int
        HSV S channel threshold (0–255).
    min_fraction : float
        Minimum fraction of shell area that must be glare before inpainting.

    Returns
    -------
    np.ndarray
        Processed RGB image (inpainted where glare was detected, or original).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    glare_mask = (
        (hsv[:, :, 2] > v_threshold) &
        (hsv[:, :, 1] < s_threshold) &
        (shell_mask > 0)
    ).astype(np.uint8) * 255

    shell_pixels = int(np.sum(shell_mask > 0))
    glare_pixels = int(np.sum(glare_mask > 0))

    if shell_pixels == 0 or (glare_pixels / shell_pixels) < min_fraction:
        logger.info(
            f"  Glare: {glare_pixels} px ({glare_pixels / max(shell_pixels, 1) * 100:.2f}%) "
            "– below threshold, skipping inpaint."
        )
        return image

    logger.info(
        f"  Glare: {glare_pixels} px ({glare_pixels / shell_pixels * 100:.2f}%) "
        "– inpainting."
    )
    inpainted = cv2.inpaint(image, glare_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
    return inpainted


# ---------------------------------------------------------------------------
# 4. Pigment pixel extraction
# ---------------------------------------------------------------------------

def extract_pigment_pixels(
    image: np.ndarray,
    shell_mask: np.ndarray,
    s_min_pct: float = 5.0,
    white_rgb_min: int = 240,
    black_rgb_max: int = 20,
) -> np.ndarray:
    """
    Extract pigmented pixels from the shell – conservative approach.

    **Kept** (treated as valid pigment):
      - All pixels with HSV S > *s_min_pct* %  (actual chromatic colour)
      - Light colours (cream, pale pink, light orange) even if softly saturated

    **Removed** (non-pigment):
      - Pure white:  all RGB channels ≥ *white_rgb_min*  (default 240)
      - Pure black:  all RGB channels ≤ *black_rgb_max*  (default 20)
      - Achromatic gray:  S < *s_min_pct* %  AND not a valid light colour

    Parameters
    ----------
    image : np.ndarray
        RGB image.
    shell_mask : np.ndarray
        Binary mask (0/255) of the shell region.
    s_min_pct : float
        Minimum HSV saturation (%) to be considered pigmented.
    white_rgb_min : int
        RGB threshold above which a pixel is "pure white".
    black_rgb_max : int
        RGB threshold below which a pixel is "pure black".

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing pigment RGB values.
    """
    mask_bool = shell_mask > 0
    rgb_pixels = image[mask_bool].astype(np.float32)
    if len(rgb_pixels) == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Convert to HSV for saturation filter
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_pixels = hsv_img[mask_bool].astype(np.float32)
    s_pct = hsv_pixels[:, 1] / 255.0 * 100.0

    r, g, b = rgb_pixels[:, 0], rgb_pixels[:, 1], rgb_pixels[:, 2]

    is_pure_white = (r >= white_rgb_min) & (g >= white_rgb_min) & (b >= white_rgb_min)
    is_pure_black = (r <= black_rgb_max) & (g <= black_rgb_max) & (b <= black_rgb_max)
    is_achromatic_gray = (s_pct < s_min_pct) & (~is_pure_white) & (~is_pure_black)

    # A pixel is valid pigment when it is not purely white, black, or gray
    valid = ~(is_pure_white | is_pure_black | is_achromatic_gray)
    pigment = rgb_pixels[valid]

    logger.info(
        f"  Pigment pixels: {len(pigment):,} / {len(rgb_pixels):,} shell pixels "
        f"({len(pigment) / max(len(rgb_pixels), 1) * 100:.1f}%)"
    )
    return pigment


# ---------------------------------------------------------------------------
# 5. Hierarchical clustering
# ---------------------------------------------------------------------------

def _choose_linkage_method(pixels: np.ndarray, sample_size: int, rng: int) -> str:
    """
    Choose between 'ward' and 'complete' linkage by comparing silhouette scores
    on a small sample.  Returns the name of the better method.
    """
    if not SKLEARN_AVAILABLE or len(pixels) < 10:
        return "ward"

    n = min(500, len(pixels))
    idx = np.random.RandomState(rng).choice(len(pixels), n, replace=False)
    sample = pixels[idx]
    sample_lab = _pixels_rgb_to_lab(sample)

    best_method, best_score = "ward", -2.0
    for method in ("ward", "complete"):
        try:
            Z = linkage(sample_lab, method=method)
            pw = pdist(sample_lab)
            cut = float(np.percentile(pw, 70))
            labels = fcluster(Z, t=cut, criterion="distance") - 1
            unique = np.unique(labels)
            if len(unique) < 2:
                continue
            sc = float(silhouette_score(sample_lab, labels))
            logger.info(f"    Linkage '{method}': silhouette={sc:.4f}")
            if sc > best_score:
                best_score = sc
                best_method = method
        except Exception:
            pass
    logger.info(f"  Best linkage method: '{best_method}'")
    return best_method


def hierarchical_clustering(
    pigment_pixels: np.ndarray,
    distance_percentile: int = 85,
    linkage_method: str = "ward",
    sample_size: int = 3000,
    random_state: int = 42,
) -> dict:
    """
    Perform hierarchical agglomerative clustering in CIELAB colour space.

    No manual K selection is required – the dendrogram is cut at the
    *distance_percentile*-th percentile of pairwise distances, allowing the
    data to determine the natural number of colour groups.  All clusters are
    returned (no minimum-coverage threshold).

    Parameters
    ----------
    pigment_pixels : np.ndarray
        (N, 3) RGB pigment pixel array.
    distance_percentile : int
        Percentile of pairwise CIELAB distances used as dendrogram cut height.
    linkage_method : str
        Scipy linkage method ('ward' or 'complete').
    sample_size : int
        Maximum number of pixels to subsample for clustering.
    random_state : int
        Random seed.

    Returns
    -------
    dict with keys:
        centers_rgb   : np.ndarray (K, 3) cluster centroids in RGB
        centers_lab   : np.ndarray (K, 3) cluster centroids in CIELAB
        counts        : list[int]  pixel count per cluster
        cut_height    : float      dendrogram cut height
        linkage_matrix: np.ndarray linkage matrix (Z)
        sample_lab    : np.ndarray subsample used (CIELAB)
        labels        : np.ndarray cluster label for each sample pixel
        silhouette    : float      mean silhouette score (NaN if < 2 clusters)
        n_colors      : int        number of clusters found
    """
    if len(pigment_pixels) < 2:
        return {
            "centers_rgb": np.empty((0, 3)),
            "centers_lab": np.empty((0, 3)),
            "counts": [],
            "cut_height": 0.0,
            "linkage_matrix": None,
            "sample_lab": np.empty((0, 3)),
            "labels": np.array([]),
            "silhouette": float("nan"),
            "n_colors": 0,
        }

    rng = np.random.RandomState(random_state)
    if len(pigment_pixels) > sample_size:
        idx = rng.choice(len(pigment_pixels), sample_size, replace=False)
        sample = pigment_pixels[idx]
    else:
        sample = pigment_pixels

    sample_lab = _pixels_rgb_to_lab(sample)

    logger.info(f"  Hierarchical clustering ({linkage_method}) on {len(sample)} pixels…")
    Z = linkage(sample_lab, method=linkage_method)
    pairwise = pdist(sample_lab)
    cut_height = float(np.percentile(pairwise, distance_percentile))
    raw_labels = fcluster(Z, t=cut_height, criterion="distance") - 1

    unique_labels = np.unique(raw_labels)
    centers_lab = np.array([sample_lab[raw_labels == k].mean(axis=0) for k in unique_labels])
    centers_rgb = np.array([sample[raw_labels == k].mean(axis=0) for k in unique_labels])
    counts = [int(np.sum(raw_labels == k)) for k in unique_labels]

    sil = float("nan")
    if SKLEARN_AVAILABLE and len(unique_labels) >= 2:
        try:
            sil = float(silhouette_score(sample_lab, raw_labels))
        except Exception:
            pass

    sil_str = f"{sil:.4f}" if not np.isnan(sil) else "n/a"
    logger.info(
        f"  → {len(unique_labels)} clusters found "
        f"(cut height={cut_height:.2f}, silhouette={sil_str})"
    )
    return {
        "centers_rgb": centers_rgb,
        "centers_lab": centers_lab,
        "counts": counts,
        "cut_height": cut_height,
        "linkage_matrix": Z,
        "sample_lab": sample_lab,
        "labels": raw_labels,
        "silhouette": sil,
        "n_colors": len(unique_labels),
    }


# ---------------------------------------------------------------------------
# 6. Merge similar clusters
# ---------------------------------------------------------------------------

def merge_close_clusters(
    centers_rgb: np.ndarray,
    counts: list,
    merge_threshold: float = 15.0,
) -> tuple:
    """
    Merge clusters whose CIELAB centroid distance is below *merge_threshold*.

    Returns
    -------
    centers_rgb : np.ndarray
    counts      : list[int]
    """
    if len(centers_rgb) < 2:
        return centers_rgb, counts

    rgb = list(centers_rgb.astype(float))
    cnt = list(counts)

    i = 0
    while i < len(rgb):
        j = i + 1
        while j < len(rgb):
            lab_i = _rgb_to_lab(rgb[i])
            lab_j = _rgb_to_lab(rgb[j])
            delta = float(np.linalg.norm(lab_i - lab_j))
            if delta < merge_threshold:
                total = cnt[i] + cnt[j]
                w_i, w_j = cnt[i] / total, cnt[j] / total
                rgb[i] = np.array(rgb[i]) * w_i + np.array(rgb[j]) * w_j
                cnt[i] = total
                rgb.pop(j)
                cnt.pop(j)
            else:
                j += 1
        i += 1

    return np.array(rgb), cnt


def _adaptive_merge_threshold(centers_rgb: np.ndarray, percentile: int = 30) -> float:
    """Compute an adaptive CIELAB merge threshold from pairwise inter-cluster distances."""
    if len(centers_rgb) < 2:
        return 15.0
    lab = _pixels_rgb_to_lab(centers_rgb)
    dists = [
        float(np.linalg.norm(lab[i] - lab[j]))
        for i in range(len(lab))
        for j in range(i + 1, len(lab))
    ]
    return float(np.percentile(dists, percentile))


# ---------------------------------------------------------------------------
# 7. Build colour result records
# ---------------------------------------------------------------------------

def _confidence_from_silhouette_samples(
    sample_lab: np.ndarray,
    labels: np.ndarray,
    label: int,
    scale: float = 25.0,
) -> float:
    """
    Per-cluster confidence from silhouette samples and centroid distance.

    confidence = silhouette_component × 0.7 + distance_component × 0.3
    """
    mask = labels == label
    if not SKLEARN_AVAILABLE or int(np.sum(mask)) < 2 or len(np.unique(labels)) < 2:
        return 50.0

    try:
        sil_vals = silhouette_samples(sample_lab, labels)
        sil_mean = float(np.mean(sil_vals[mask]))
        sil_conf = (sil_mean + 1.0) / 2.0 * 100.0

        centroid_lab = sample_lab[mask].mean(axis=0)
        within_dists = np.linalg.norm(sample_lab[mask] - centroid_lab, axis=1)
        mean_dist = float(np.mean(within_dists))
        dist_conf = 100.0 * float(np.exp(-mean_dist / max(scale, 1e-6)))

        return round(sil_conf * 0.7 + dist_conf * 0.3, 2)
    except Exception:
        return 50.0


def build_color_records(
    cluster_result: dict,
    total_pigment_pixels: int,
    config: dict,
) -> list:
    """
    Convert clustering output into a list of colour record dicts.

    Each record contains:
        rank, color_name, hex_code, rgb, rgb_norm,
        pct_of_pigment, confidence, lab_centroid,
        distance_to_centroid_lab, margin_of_error,
        silhouette_score
    """
    centers_rgb = cluster_result["centers_rgb"]
    centers_lab = cluster_result["centers_lab"]
    counts = cluster_result["counts"]
    sample_lab = cluster_result["sample_lab"]
    labels = cluster_result["labels"]
    sil_overall = cluster_result["silhouette"]
    scale = config.get("CONFIDENCE_SCALE_LAB", 25.0)
    n_labels = len(np.unique(labels)) if len(labels) > 0 else 0

    # Sort by count descending
    order = np.argsort(counts)[::-1]

    records = []
    for rank, idx in enumerate(order, start=1):
        rgb = tuple(int(c) for c in np.clip(centers_rgb[idx], 0, 255))
        lab = centers_lab[idx]
        count = counts[idx]
        pct = count / max(total_pigment_pixels, 1) * 100.0

        color_name = generate_color_name(rgb)
        hex_code = rgb_to_hex(rgb)

        # Confidence
        if n_labels >= 2:
            conf = _confidence_from_silhouette_samples(
                sample_lab, labels, idx, scale
            )
        else:
            conf = 50.0

        # Within-cluster mean CIELAB distance (margin of error proxy)
        mask = labels == idx
        if np.sum(mask) > 0 and len(sample_lab) > 0:
            within_dists = np.linalg.norm(sample_lab[mask] - lab, axis=1)
            mean_within = float(np.mean(within_dists))
            margin = round(float(np.std(within_dists)), 3) if len(within_dists) > 1 else 0.0
        else:
            mean_within = 0.0
            margin = 0.0

        records.append({
            "rank": rank,
            "color_name": color_name,
            "hex_code": hex_code,
            "rgb": rgb,
            "rgb_norm": tuple(round(c / 255.0, 4) for c in rgb),
            "pct_of_pigment": round(pct, 2),
            "confidence": round(conf, 2),
            "lab_centroid": tuple(round(float(v), 3) for v in lab),
            "distance_to_centroid_lab": round(mean_within, 3),
            "margin_of_error": margin,
            "pixel_count": count,
            "silhouette_score": round(float(sil_overall) if not np.isnan(sil_overall) else 0.0, 4),
        })

    return records


# ---------------------------------------------------------------------------
# 8. Parameter learning
# ---------------------------------------------------------------------------

def learn_parameters(training_results: list, config: dict) -> dict:
    """
    Derive optimal clustering parameters from a list of per-image result dicts.

    Analyses:
      - Min / max / mean / median number of colours per image
      - Optimal dendrogram cutting percentile (best silhouette across options)
      - Optimal linkage method (most frequently best)
      - Merge threshold (percentile of pairwise centroid distances)
      - Hue diversity metrics

    Parameters
    ----------
    training_results : list of dict
        Each element is the output of ``_analyze_single_for_training``.
    config : dict
        Base configuration dict.

    Returns
    -------
    dict
        Learned parameters to override defaults.
    """
    n_colors_list = [r["n_colors"] for r in training_results if r.get("n_colors", 0) > 0]
    if not n_colors_list:
        logger.warning("No valid training results – returning defaults.")
        return {}

    silhouettes_by_pct: dict = {}
    for r in training_results:
        pct = r.get("distance_percentile", config["HIER_DISTANCE_PERCENTILE"])
        sil = r.get("silhouette", float("nan"))
        if not np.isnan(sil):
            silhouettes_by_pct.setdefault(pct, []).append(sil)

    best_pct = config["HIER_DISTANCE_PERCENTILE"]
    if silhouettes_by_pct:
        best_pct = max(silhouettes_by_pct, key=lambda p: np.mean(silhouettes_by_pct[p]))

    linkage_votes = [r.get("linkage_method", "ward") for r in training_results]
    from collections import Counter
    best_linkage = Counter(linkage_votes).most_common(1)[0][0]

    # Optimal merge threshold: percentile of per-image median pairwise centroid distances
    merge_dists = [r["median_pairwise_lab"] for r in training_results if r.get("median_pairwise_lab")]
    optimal_merge = float(np.percentile(merge_dists, config["COLOR_MERGE_PERCENTILE"])) if merge_dists else 15.0

    learned = {
        "min_colors": int(np.min(n_colors_list)),
        "max_colors": int(np.max(n_colors_list)),
        "mean_colors": float(np.mean(n_colors_list)),
        "median_colors": float(np.median(n_colors_list)),
        "optimal_distance_percentile": int(best_pct),
        "optimal_linkage_method": best_linkage,
        "optimal_merge_threshold": round(optimal_merge, 3),
        "n_training_images": len(training_results),
        "silhouettes_by_percentile": {
            str(p): round(float(np.mean(s)), 4) for p, s in silhouettes_by_pct.items()
        },
    }
    logger.info("Learned parameters:")
    for k, v in learned.items():
        logger.info(f"  {k}: {v}")
    return learned


# ---------------------------------------------------------------------------
# 9. Single image analysis helper (for training loop)
# ---------------------------------------------------------------------------

def _analyze_single_for_training(
    image_path: str,
    config: dict,
    distance_percentiles: tuple = (70, 75, 80, 85, 90),
) -> dict:
    """
    Analyse one image trying multiple *distance_percentiles* and return stats.

    Returns a dict with:
        n_colors, silhouette, distance_percentile, linkage_method,
        median_pairwise_lab, hue_diversity
    """
    result = {
        "image_path": image_path,
        "n_colors": 0,
        "silhouette": float("nan"),
        "distance_percentile": config["HIER_DISTANCE_PERCENTILE"],
        "linkage_method": "ward",
        "median_pairwise_lab": None,
        "hue_diversity": 0.0,
    }
    try:
        image = _load_image(image_path)
        image = normalize_image(image, config["STANDARD_CANVAS_SIZE"])
        shell_mask = extract_shell_mask(image)
        image = selective_glare_removal(
            image, shell_mask,
            v_threshold=config["GLARE_V_THRESHOLD"],
            s_threshold=config["GLARE_S_THRESHOLD"],
            min_fraction=config["GLARE_MIN_FRACTION"],
        )
        pigment = extract_pigment_pixels(
            image, shell_mask,
            s_min_pct=config["PIGMENT_S_MIN_PCT"],
            white_rgb_min=config["WHITE_RGB_MIN"],
            black_rgb_max=config["BLACK_RGB_MAX"],
        )
        if len(pigment) < 10:
            return result

        # Test linkage methods
        linkage_method = _choose_linkage_method(pigment, config["HIER_SAMPLE_SIZE"], config["HIER_RANDOM_STATE"])
        result["linkage_method"] = linkage_method

        # Try multiple percentiles, keep best silhouette
        best_sil = -2.0
        best_pct = config["HIER_DISTANCE_PERCENTILE"]
        best_cluster = None
        for pct in distance_percentiles:
            cr = hierarchical_clustering(
                pigment,
                distance_percentile=pct,
                linkage_method=linkage_method,
                sample_size=config["HIER_SAMPLE_SIZE"],
                random_state=config["HIER_RANDOM_STATE"],
            )
            sil = cr["silhouette"]
            if not np.isnan(sil) and sil > best_sil:
                best_sil = sil
                best_pct = pct
                best_cluster = cr

        if best_cluster is None:
            best_cluster = hierarchical_clustering(
                pigment,
                distance_percentile=best_pct,
                linkage_method=linkage_method,
                sample_size=config["HIER_SAMPLE_SIZE"],
                random_state=config["HIER_RANDOM_STATE"],
            )

        # Median pairwise centroid distance in LAB
        if len(best_cluster["centers_lab"]) >= 2:
            pw = pdist(best_cluster["centers_lab"])
            result["median_pairwise_lab"] = float(np.median(pw))

        # Hue diversity: std of hue angles
        if len(pigment) > 0:
            hsv_img = cv2.cvtColor(
                np.clip(pigment, 0, 255).astype(np.uint8).reshape(1, -1, 3),
                cv2.COLOR_RGB2HSV,
            )
            hues = hsv_img[0, :, 0].astype(float) * 2.0  # 0-360
            result["hue_diversity"] = float(np.std(hues))

        result["n_colors"] = best_cluster["n_colors"]
        result["silhouette"] = best_sil if best_sil > -2.0 else float("nan")
        result["distance_percentile"] = best_pct

    except Exception as exc:
        logger.warning(f"  Skipping {os.path.basename(image_path)}: {exc}")

    return result


# ---------------------------------------------------------------------------
# 10. Training
# ---------------------------------------------------------------------------

def train_model(input_folder: str, config: dict) -> dict:
    """
    Train on all images in *input_folder*, learn optimal parameters and save
    ``color_model.pkl``.

    Parameters
    ----------
    input_folder : str
        Path to folder containing training images.
    config : dict
        Configuration dict (merged from DEFAULT_CONFIG + overrides).

    Returns
    -------
    dict
        Learned parameters + training metadata.
    """
    import glob as _glob

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(_glob.glob(os.path.join(input_folder, ext)))
        image_paths.extend(_glob.glob(os.path.join(input_folder, ext.upper())))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        raise FileNotFoundError(f"No images found in '{input_folder}'.")

    logger.info(f"Training on {len(image_paths)} images from '{input_folder}'…")

    sample_results = []
    for i, path in enumerate(image_paths, start=1):
        logger.info(f"\n[{i}/{len(image_paths)}] {os.path.basename(path)}")
        res = _analyze_single_for_training(path, config)
        sample_results.append(res)
        logger.info(
            f"  n_colors={res['n_colors']}  sil={res['silhouette']:.4f if not np.isnan(res['silhouette']) else 'n/a'}"
        )

    learned = learn_parameters(sample_results, config)

    model = {
        "learned_params": learned,
        "sample_results": sample_results,
        "training_folder": input_folder,
        "training_timestamp": datetime.now().isoformat(),
        "n_images_trained": len(image_paths),
        "config": config,
    }

    model_path = config.get("MODEL_PATH", "color_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"\n✅ Model saved to '{model_path}'.")

    # Generate training validation report
    _save_training_statistics(model, config)
    _plot_training_report(model, config)

    return model


# ---------------------------------------------------------------------------
# 11. Image loading helper
# ---------------------------------------------------------------------------

def _load_image(image_path: str) -> np.ndarray:
    """Load *image_path* as an RGB numpy array."""
    pil = Image.open(image_path).convert("RGBA")
    return np.array(pil)


# ---------------------------------------------------------------------------
# 12. Single image analysis
# ---------------------------------------------------------------------------

def analyze_image(
    image_path: str,
    config: dict,
    model: dict = None,
    output_folder: str = None,
    no_show: bool = False,
) -> list:
    """
    Analyse a single shell image and return a list of colour records.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    config : dict
        Configuration dict.
    model : dict or None
        Trained model dict (output of ``train_model``).  When *None*, fresh
        clustering is performed with the default / config parameters.
    output_folder : str or None
        Directory where output files are saved.  Defaults to config OUTPUT_FOLDER.
    no_show : bool
        If True, do not call ``plt.show()``.

    Returns
    -------
    list of dict
        Colour records sorted by pigment coverage (descending).
    """
    if output_folder is None:
        output_folder = config.get("OUTPUT_FOLDER", "./output")
    os.makedirs(output_folder, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Analysing: {image_path}")
    logger.info(f"{'=' * 60}")

    # Override config with learned parameters when model is provided
    effective_config = dict(config)
    if model and "learned_params" in model:
        lp = model["learned_params"]
        effective_config["HIER_DISTANCE_PERCENTILE"] = lp.get(
            "optimal_distance_percentile", config["HIER_DISTANCE_PERCENTILE"]
        )
        effective_config["COLOR_MERGE_THRESHOLD"] = lp.get("optimal_merge_threshold")
        linkage_method = lp.get("optimal_linkage_method", "ward")
    else:
        linkage_method = "ward"

    # Pipeline
    image_raw = _load_image(image_path)
    image = normalize_image(image_raw, effective_config["STANDARD_CANVAS_SIZE"])
    shell_mask = extract_shell_mask(image)
    image_deglared = selective_glare_removal(
        image, shell_mask,
        v_threshold=effective_config["GLARE_V_THRESHOLD"],
        s_threshold=effective_config["GLARE_S_THRESHOLD"],
        min_fraction=effective_config["GLARE_MIN_FRACTION"],
    )
    pigment = extract_pigment_pixels(
        image_deglared, shell_mask,
        s_min_pct=effective_config["PIGMENT_S_MIN_PCT"],
        white_rgb_min=effective_config["WHITE_RGB_MIN"],
        black_rgb_max=effective_config["BLACK_RGB_MAX"],
    )

    if len(pigment) < 5:
        logger.warning("Insufficient pigment pixels – cannot cluster.")
        return []

    cluster_result = hierarchical_clustering(
        pigment,
        distance_percentile=effective_config["HIER_DISTANCE_PERCENTILE"],
        linkage_method=linkage_method,
        sample_size=effective_config["HIER_SAMPLE_SIZE"],
        random_state=effective_config["HIER_RANDOM_STATE"],
    )

    # Optionally merge very close clusters
    merge_thresh = effective_config.get("COLOR_MERGE_THRESHOLD") or _adaptive_merge_threshold(
        cluster_result["centers_rgb"], effective_config["COLOR_MERGE_PERCENTILE"]
    )
    centers_m, counts_m = merge_close_clusters(
        cluster_result["centers_rgb"], cluster_result["counts"], merge_thresh
    )
    # Rebuild labels for merged clusters (re-assign each sample to nearest merged centroid)
    if len(centers_m) > 0 and len(cluster_result["sample_lab"]) > 0:
        centers_m_lab = _pixels_rgb_to_lab(centers_m)
        dists_to_centers = np.array([
            np.linalg.norm(cluster_result["sample_lab"] - c, axis=1) for c in centers_m_lab
        ])
        new_labels = np.argmin(dists_to_centers, axis=0)
    else:
        new_labels = cluster_result["labels"]

    merged_result = dict(cluster_result)
    merged_result["centers_rgb"] = centers_m
    merged_result["centers_lab"] = _pixels_rgb_to_lab(centers_m) if len(centers_m) > 0 else cluster_result["centers_lab"]
    merged_result["counts"] = list(counts_m)
    merged_result["labels"] = new_labels
    merged_result["n_colors"] = len(centers_m)

    records = build_color_records(merged_result, len(pigment), effective_config)

    # Outputs
    print_color_table(records)
    _save_csv(records, os.path.join(output_folder, f"colors_{base_name}_{ts}.csv"))
    _save_json(records, image_path, merged_result, os.path.join(output_folder, f"colors_{base_name}_{ts}.json"))

    # Visualisations
    _plot_color_palette(records, image, os.path.join(output_folder, f"color_palette_{base_name}_{ts}.png"), no_show)
    _plot_dendrogram(cluster_result, os.path.join(output_folder, f"dendrogram_{base_name}_{ts}.png"), no_show)
    _plot_hue_distribution(pigment, records, os.path.join(output_folder, f"hue_distribution_{base_name}_{ts}.png"), no_show)
    _plot_lab_scatter(cluster_result, os.path.join(output_folder, f"lab_scatter_{base_name}_{ts}.png"), no_show)

    return records


# ---------------------------------------------------------------------------
# 13. Console output
# ---------------------------------------------------------------------------

_TABLE_COLS = [
    ("Rank", 4),
    ("Color Name", 20),
    ("Hex Code", 9),
    ("RGB", 17),
    ("% Pigment", 10),
    ("Confidence (%)", 15),
    ("CIELAB Dist.", 13),
    ("Margin of Error", 16),
]


def print_color_table(records: list) -> None:
    """Print a formatted colour analysis table to stdout."""
    header = " | ".join(f"{col:<{w}}" for col, w in _TABLE_COLS)
    separator = "-+-".join("-" * w for _, w in _TABLE_COLS)
    print()
    print(header)
    print(separator)
    for r in records:
        rgb_str = f"({r['rgb'][0]}, {r['rgb'][1]}, {r['rgb'][2]})"
        row = [
            (str(r["rank"]), 4),
            (r["color_name"][:20], 20),
            (r["hex_code"], 9),
            (rgb_str, 17),
            (f"{r['pct_of_pigment']:.2f}%", 10),
            (f"{r['confidence']:.2f}", 15),
            (f"{r['distance_to_centroid_lab']:.3f}", 13),
            (f"±{r['margin_of_error']:.3f}", 16),
        ]
        print(" | ".join(f"{val:<{w}}" for val, w in row))
    print()


# ---------------------------------------------------------------------------
# 14. File output helpers
# ---------------------------------------------------------------------------

def _save_csv(records: list, path: str) -> None:
    fieldnames = [
        "rank", "color_name", "hex_code", "rgb", "rgb_norm",
        "pct_of_pigment", "confidence", "lab_centroid",
        "distance_to_centroid_lab", "margin_of_error",
        "pixel_count", "silhouette_score",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"  CSV saved: {path}")


def _save_json(records: list, image_path: str, cluster_result: dict, path: str) -> None:
    out = {
        "image": image_path,
        "timestamp": datetime.now().isoformat(),
        "n_colors": cluster_result.get("n_colors", len(records)),
        "cut_height": cluster_result.get("cut_height"),
        "silhouette": (
            float(cluster_result["silhouette"])
            if not np.isnan(cluster_result.get("silhouette", float("nan")))
            else None
        ),
        "colors": records,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"  JSON saved: {path}")


def _save_training_statistics(model: dict, config: dict) -> None:
    path = os.path.join(config.get("OUTPUT_FOLDER", "./output"), "training_statistics.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = {
        "training_folder": model.get("training_folder"),
        "training_timestamp": model.get("training_timestamp"),
        "n_images_trained": model.get("n_images_trained"),
        "learned_params": model.get("learned_params"),
        "per_image_results": [
            {
                "image": os.path.basename(r["image_path"]),
                "n_colors": r["n_colors"],
                "silhouette": r["silhouette"] if not np.isnan(r.get("silhouette", float("nan"))) else None,
                "distance_percentile": r["distance_percentile"],
                "linkage_method": r["linkage_method"],
                "hue_diversity": r["hue_diversity"],
            }
            for r in model.get("sample_results", [])
        ],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"  Training statistics saved: {path}")


# ---------------------------------------------------------------------------
# 15. Visualisations
# ---------------------------------------------------------------------------

def _plot_color_palette(records: list, image: np.ndarray, save_path: str, no_show: bool) -> None:
    """Generate a colour palette infographic with auto-named swatches."""
    n = len(records)
    if n == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, n * 0.8 + 2)))
    fig.suptitle("Shell Colour Palette", fontsize=16, fontweight="bold")

    # Left: original image
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Input Image")

    # Right: colour swatches
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n + 1)
    ax.axis("off")
    ax.set_title("Detected Colours", fontsize=13)

    for i, r in enumerate(records):
        y = n - i
        rgb_norm = tuple(c / 255.0 for c in r["rgb"])
        rect = mpatches.FancyBboxPatch(
            (0.1, y - 0.35), 1.8, 0.7,
            boxstyle="round,pad=0.05",
            facecolor=rgb_norm, edgecolor="gray", linewidth=0.5,
        )
        ax.add_patch(rect)
        label = (
            f"#{r['rank']}  {r['color_name']}  {r['hex_code']}  "
            f"{r['pct_of_pigment']:.1f}%  conf={r['confidence']:.0f}%"
        )
        ax.text(2.1, y, label, va="center", fontsize=9, family="monospace")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    logger.info(f"  Palette saved: {save_path}")
    if not no_show:
        plt.show()
    plt.close(fig)


def _plot_dendrogram(cluster_result: dict, save_path: str, no_show: bool) -> None:
    """Plot the hierarchical clustering dendrogram."""
    Z = cluster_result.get("linkage_matrix")
    if Z is None:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    cut = cluster_result.get("cut_height", 0)
    try:
        n_leaves = int(Z.shape[0]) + 1
        if n_leaves > 100:
            dendrogram(
                Z, ax=ax, no_labels=True, color_threshold=cut,
                truncate_mode="lastp", p=50,
            )
        else:
            dendrogram(Z, ax=ax, no_labels=True, color_threshold=cut)
    except Exception as exc:
        logger.warning(f"  Dendrogram rendering skipped: {exc}")
        plt.close(fig)
        return
    ax.axhline(y=cut, color="red", linestyle="--", linewidth=1.5, label=f"Cut height = {cut:.2f}")
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Pixel Samples")
    ax.set_ylabel("CIELAB Distance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    logger.info(f"  Dendrogram saved: {save_path}")
    if not no_show:
        plt.show()
    plt.close(fig)


def _plot_hue_distribution(pigment: np.ndarray, records: list, save_path: str, no_show: bool) -> None:
    """Plot a hue distribution histogram of pigment pixels."""
    if len(pigment) == 0:
        return

    hsv = cv2.cvtColor(
        np.clip(pigment, 0, 255).astype(np.uint8).reshape(1, -1, 3),
        cv2.COLOR_RGB2HSV,
    )
    hues = hsv[0, :, 0].astype(float) * 2.0  # 0-360

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Hue Distribution", fontsize=14, fontweight="bold")

    # Histogram
    ax = axes[0]
    n_bins = 72  # 5° bins
    counts, edges = np.histogram(hues, bins=n_bins, range=(0, 360))
    bin_centers = (edges[:-1] + edges[1:]) / 2
    colours = [
        colorsys_hsv_to_rgb(h / 360.0) for h in bin_centers
    ]
    ax.bar(bin_centers, counts, width=360 / n_bins, color=colours, edgecolor="none")
    ax.set_xlabel("Hue (degrees)")
    ax.set_ylabel("Pixel Count")
    ax.set_title("Hue Histogram")

    # Pie chart of detected colour families
    ax2 = axes[1]
    families = {}
    for r in records:
        rgb = r["rgb"]
        pixel = np.array([[[rgb[0], rgb[1], rgb[2]]]], dtype=np.uint8)
        h_cv = int(cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0][0][0])
        family = _hue_family(h_cv * 2.0)
        families[family] = families.get(family, 0) + r["pct_of_pigment"]

    if families:
        labels = list(families.keys())
        sizes = list(families.values())
        # Evenly space hue values around the wheel for distinct pie slice colors
        pie_colors = [
            colorsys_hsv_to_rgb((i / max(len(labels), 1)) % 1.0)
            for i in range(len(labels))
        ]
        ax2.pie(sizes, labels=labels, colors=pie_colors, autopct="%1.1f%%", startangle=90)
        ax2.set_title("Color Family Distribution")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    logger.info(f"  Hue distribution saved: {save_path}")
    if not no_show:
        plt.show()
    plt.close(fig)


def colorsys_hsv_to_rgb(h: float, s: float = 0.8, v: float = 0.8) -> tuple:
    """Thin wrapper around colorsys for matplotlib colour generation."""
    import colorsys
    return colorsys.hsv_to_rgb(h, s, v)


def _plot_lab_scatter(cluster_result: dict, save_path: str, no_show: bool) -> None:
    """Scatter plot of cluster centroids in LAB colour space (a* vs b*)."""
    centers_lab = cluster_result.get("centers_lab")
    if centers_lab is None or len(centers_lab) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    centers_rgb = cluster_result.get("centers_rgb", np.zeros_like(centers_lab))
    for i, (lab, rgb) in enumerate(zip(centers_lab, centers_rgb)):
        colour = tuple(np.clip(rgb / 255.0, 0, 1))
        ax.scatter(lab[1], lab[2], c=[colour], s=200, edgecolors="gray", linewidths=0.8, zorder=3)
        ax.annotate(f"C{i + 1}", (lab[1], lab[2]), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("a* (green ← → red)")
    ax.set_ylabel("b* (blue ← → yellow)")
    ax.set_title("Cluster Centroids – CIELAB Colour Space")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    logger.info(f"  LAB scatter saved: {save_path}")
    if not no_show:
        plt.show()
    plt.close(fig)


def _plot_training_report(model: dict, config: dict) -> None:
    """Generate a summary training infographic (training_report.png)."""
    results = model.get("sample_results", [])
    if not results:
        return

    n_colors = [r["n_colors"] for r in results if r["n_colors"] > 0]
    hue_divs = [r["hue_diversity"] for r in results if r["hue_diversity"] > 0]
    sils = [r["silhouette"] for r in results if not np.isnan(r.get("silhouette", float("nan")))]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Training Validation Report", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1: Colors per image histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(n_colors, bins=range(1, max(n_colors) + 2) if n_colors else [1, 2], color="steelblue", edgecolor="white")
    ax1.set_title("Colours Discovered per Image")
    ax1.set_xlabel("Number of Colours")
    ax1.set_ylabel("Images")

    # 2: Silhouette distribution
    ax2 = fig.add_subplot(gs[0, 1])
    if sils:
        ax2.hist(sils, bins=10, color="darkorange", edgecolor="white")
    ax2.set_title("Silhouette Score Distribution")
    ax2.set_xlabel("Silhouette Score")
    ax2.set_ylabel("Images")

    # 3: Hue diversity distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if hue_divs:
        ax3.hist(hue_divs, bins=10, color="seagreen", edgecolor="white")
    ax3.set_title("Hue Diversity Distribution")
    ax3.set_xlabel("Hue Std Dev (degrees)")
    ax3.set_ylabel("Images")

    # 4: Per-image color count line
    ax4 = fig.add_subplot(gs[1, 0:2])
    x = list(range(1, len(n_colors) + 1))
    ax4.plot(x, n_colors, marker="o", linewidth=1.5, markersize=4, color="steelblue")
    ax4.axhline(np.mean(n_colors) if n_colors else 0, color="red", linestyle="--", label=f"Mean={np.mean(n_colors):.1f}")
    ax4.set_title("Colours per Training Image")
    ax4.set_xlabel("Image Index")
    ax4.set_ylabel("Colours")
    ax4.legend()

    # 5: Learned parameter summary text box
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    lp = model.get("learned_params", {})
    summary = (
        f"Learned Parameters\n"
        f"{'─' * 28}\n"
        f"Images trained : {lp.get('n_training_images', '?')}\n"
        f"Min colours    : {lp.get('min_colors', '?')}\n"
        f"Max colours    : {lp.get('max_colors', '?')}\n"
        f"Mean colours   : {lp.get('mean_colors', '?'):.1f}\n"
        f"Median colours : {lp.get('median_colors', '?'):.1f}\n"
        f"Linkage method : {lp.get('optimal_linkage_method', '?')}\n"
        f"Dist percentile: {lp.get('optimal_distance_percentile', '?')}\n"
        f"Merge threshold: {lp.get('optimal_merge_threshold', '?'):.3f}\n"
    )
    ax5.text(0.05, 0.95, summary, transform=ax5.transAxes,
             fontsize=10, verticalalignment="top", family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    out_dir = config.get("OUTPUT_FOLDER", "./output")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "training_report.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    logger.info(f"  Training report saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 16. Command-line interface
# ---------------------------------------------------------------------------

def _build_config(args) -> dict:
    """Build effective config from defaults + CLI overrides."""
    cfg = dict(DEFAULT_CONFIG)
    if args.output_folder:
        cfg["OUTPUT_FOLDER"] = args.output_folder
    if args.model_path:
        cfg["MODEL_PATH"] = args.model_path
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="color_train.py",
        description="Enhanced Shell Colour Analysis & Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["train", "analyze"], required=True,
        help="Operation mode: 'train' to learn from a folder, 'analyze' to analyse images.",
    )
    parser.add_argument(
        "--input-folder", metavar="PATH",
        help="Folder containing input images (used in both train and analyze modes).",
    )
    parser.add_argument(
        "--image", metavar="PATH",
        help="Path to a single image to analyse (analyze mode only).",
    )
    parser.add_argument(
        "--use-trained-model", action="store_true",
        help="Load and use the trained model (color_model.pkl) for analysis.",
    )
    parser.add_argument(
        "--model-path", metavar="PATH", default="color_model.pkl",
        help="Path to the model file (default: color_model.pkl).",
    )
    parser.add_argument(
        "--output-folder", metavar="PATH", default="./output",
        help="Folder where output files are saved (default: ./output).",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not display interactive matplotlib windows.",
    )

    args = parser.parse_args(argv)
    config = _build_config(args)

    # ---- TRAIN MODE ----
    if args.mode == "train":
        if not args.input_folder:
            parser.error("--input-folder is required for train mode.")
        train_model(args.input_folder, config)
        return

    # ---- ANALYZE MODE ----
    model = None
    if args.use_trained_model:
        model_path = config["MODEL_PATH"]
        if not os.path.exists(model_path):
            logger.error(f"Trained model not found at '{model_path}'. Run with --mode train first.")
            sys.exit(1)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded trained model from '{model_path}'.")

    if args.image:
        analyze_image(args.image, config, model=model, output_folder=config["OUTPUT_FOLDER"], no_show=args.no_show)
    elif args.input_folder:
        import glob as _glob
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp")
        paths = []
        for ext in extensions:
            paths.extend(_glob.glob(os.path.join(args.input_folder, ext)))
            paths.extend(_glob.glob(os.path.join(args.input_folder, ext.upper())))
        paths = sorted(set(paths))
        if not paths:
            logger.error(f"No images found in '{args.input_folder}'.")
            sys.exit(1)
        for p in paths:
            analyze_image(p, config, model=model, output_folder=config["OUTPUT_FOLDER"], no_show=args.no_show)
    else:
        parser.error("Provide --image or --input-folder for analyze mode.")


if __name__ == "__main__":
    main()
