"""
Advanced Adaptive Color Detection Framework for Shell Organisms (Bivalves)
==========================================================================
This program performs comprehensive color identification on shell organism
images using two clustering strategies:

1. Automated K Selection   - Silhouette score, elbow method, Davies-Bouldin index
2. Hierarchical Clustering - Agglomerative with adaptive distance thresholds
3. Adaptive Merge Logic    - Percentile-based threshold computation
4. Scale-Independent Analysis - All images normalized to 1000×1000 before analysis
5. Multiple Method Comparison - Side-by-side performance metrics
6. Parameter Training      - Optimize K range and merge thresholds from sample images
7. Enhanced Visualization  - Dashboard with clustering metrics and dendrograms
8. Comprehensive Reporting - CSV and JSON export with detailed statistics

Usage
-----
Analyze all images in a folder (both clustering methods):
    python shell_color_analysis.py --folder /path/to/images

Select a specific clustering method:
    python shell_color_analysis.py --folder /path/to/images --method kmeans
    python shell_color_analysis.py --folder /path/to/images --method hierarchical

Adjust the K-Means search range:
    python shell_color_analysis.py --folder /path/to/images --k-min 3 --k-max 20

Use a fixed CIELAB merge threshold instead of the adaptive one:
    python shell_color_analysis.py --folder /path/to/images --merge-threshold 15.0

Save results without opening interactive plot windows:
    python shell_color_analysis.py --folder /path/to/images --no-show

Train on sample images to optimize clustering parameters:
    python shell_color_analysis.py --folder /path/to/training_images --train

Run analysis with trained parameters:
    python shell_color_analysis.py --folder /path/to/images --use-trained-params

Compare trained-parameter results with default results:
    python shell_color_analysis.py --folder /path/to/images --compare-trained

See COMMANDS.md for a complete quick-reference and PARAMETER_GUIDE.md for
detailed parameter tuning guidance.
"""

import argparse
import csv
import json
import logging
import os
import glob
import pickle
import sys
import warnings
from collections import Counter
from datetime import datetime

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logging.warning("rembg not available - background removal will be skipped.")

warnings.filterwarnings("ignore")

# ============================================================
# LOGGING SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_CONFIG = {
    # --- Input/Output ---
    "INPUT_FOLDER": r"./images",
    "OUTPUT_FOLDER": r"./output",

    # --- Scale-Independent Normalisation ---
    # All images are resized to a standard canvas before analysis so that
    # color-coverage percentages are comparable regardless of input resolution.
    "STANDARD_CANVAS_SIZE": 1000,  # pixels (width and height of normalized canvas)

    # --- Clustering Method ---
    # Options: "kmeans", "hierarchical", "all"
    "CLUSTERING_METHOD": "all",

    # --- K-Means Settings ---
    "NUM_CLUSTERS_MIN": 3,
    "NUM_CLUSTERS_MAX": 15,

    # --- Color Merge Settings ---
    # If None, threshold is computed adaptively from data distribution
    "COLOR_MERGE_THRESHOLD": None,
    "COLOR_MERGE_PERCENTILE": 30,  # Percentile of pairwise distances for adaptive threshold

    # --- Hierarchical Clustering ---
    "HIERARCHICAL_DISTANCE_PERCENTILE": 85,  # Percentile for adaptive distance threshold

    # --- Glare & Shadow ---
    "GLARE_THRESHOLD": 245,
    "MIN_COLOR_BRIGHTNESS": 40,

    # --- White Detection ---
    "WHITE_SENSITIVITY": 50,
    "WHITE_BRIGHTNESS": 150,

    # --- Trained Parameters ---
    "TRAINED_PARAMS_PATH": "trained_params.pkl",

    # --- Classifier ---
    "CLASSIFIER_PATH": "color_classifier.pkl",

    # --- Visualization ---
    "SHOW_DENDROGRAM": True,
    "SHOW_OPTIMIZATION_CURVES": True,
    "SAVE_FIGURES": True,
}


# ============================================================
# COLOR DICTIONARY PREPARATION (XKCD 949 colors)
# ============================================================
def build_color_dictionary():
    """Build a LAB-space dictionary from the XKCD 949-color set for perceptual matching."""
    logger.info("Loading high-accuracy perceptual color dictionary (949 XKCD colors)...")
    color_dict_lab = {}
    for name, hex_val in mcolors.XKCD_COLORS.items():
        clean_name = name.replace("xkcd:", "").title()
        rgb_float = mcolors.hex2color(hex_val)
        rgb_255 = np.uint8([[[int(c * 255) for c in rgb_float]]])
        lab_val = cv2.cvtColor(rgb_255, cv2.COLOR_RGB2LAB)[0][0]
        color_dict_lab[clean_name] = lab_val
    logger.info(f"  Loaded {len(color_dict_lab)} named colors.")
    return color_dict_lab


COLOR_DICT_LAB = build_color_dictionary()


# ============================================================
# COLOR NAMING
# ============================================================
def get_closest_color_name(rgb_tuple):
    """Match an RGB color to the closest XKCD name using CIELAB perceptual distance."""
    rgb_255 = np.uint8([[[rgb_tuple[0], rgb_tuple[1], rgb_tuple[2]]]])
    target_lab = cv2.cvtColor(rgb_255, cv2.COLOR_RGB2LAB)[0][0].astype(float)

    min_dist = float("inf")
    best_name = "Unknown"

    for name, lab_val in COLOR_DICT_LAB.items():
        dist = np.linalg.norm(target_lab - lab_val.astype(float))
        if dist < min_dist:
            min_dist = dist
            best_name = name

    return best_name


# ============================================================
# MERGE SIMILAR CLUSTERS
# ============================================================
def merge_similar_clusters(centers_rgb, counts, distance_threshold=15.0):
    """
    Merge visually similar clusters using CIELAB perceptual distance.

    Parameters
    ----------
    centers_rgb : array-like
        Cluster centers in RGB space.
    counts : list of int
        Pixel counts per cluster.
    distance_threshold : float
        Maximum CIELAB distance to consider two clusters as the same color.

    Returns
    -------
    np.ndarray, list
        Merged cluster centers (RGB) and their combined counts.
    """
    centers_rgb = [np.array(c, dtype=float) for c in centers_rgb]
    centers_rgb_img = np.uint8([[c for c in centers_rgb]])
    centers_lab = cv2.cvtColor(centers_rgb_img, cv2.COLOR_RGB2LAB)[0]

    merged_rgb = list(centers_rgb)
    merged_lab = list(centers_lab)
    merged_counts = list(counts)

    i = 0
    while i < len(merged_lab):
        j = i + 1
        while j < len(merged_lab):
            lab1 = merged_lab[i].astype(float)
            lab2 = merged_lab[j].astype(float)
            delta_e = np.linalg.norm(lab1 - lab2)

            if delta_e < distance_threshold:
                total = merged_counts[i] + merged_counts[j]
                w_i = merged_counts[i] / total
                w_j = merged_counts[j] / total
                new_rgb = merged_rgb[i] * w_i + merged_rgb[j] * w_j
                merged_rgb[i] = new_rgb

                new_rgb_img = np.uint8([[new_rgb]])
                new_lab = cv2.cvtColor(new_rgb_img, cv2.COLOR_RGB2LAB)[0][0]
                merged_lab[i] = new_lab
                merged_counts[i] = total

                merged_rgb.pop(j)
                merged_lab.pop(j)
                merged_counts.pop(j)
            else:
                j += 1
        i += 1

    return np.array(merged_rgb), merged_counts


def compute_adaptive_merge_threshold(centers_rgb, percentile=30):
    """
    Compute an adaptive CIELAB merge threshold from the distribution of
    pairwise inter-cluster distances.

    Parameters
    ----------
    centers_rgb : array-like
        Cluster centers in RGB space.
    percentile : int
        Percentile of pairwise distances to use as threshold.

    Returns
    -------
    float
        Adaptive distance threshold.
    """
    if len(centers_rgb) < 2:
        return 15.0

    centers_rgb_img = np.uint8([[c for c in centers_rgb]])
    centers_lab = cv2.cvtColor(centers_rgb_img, cv2.COLOR_RGB2LAB)[0].astype(float)

    distances = []
    for i in range(len(centers_lab)):
        for j in range(i + 1, len(centers_lab)):
            delta_e = np.linalg.norm(centers_lab[i] - centers_lab[j])
            distances.append(delta_e)

    threshold = float(np.percentile(distances, percentile))
    logger.info(
        f"  Adaptive merge threshold: {threshold:.2f} "
        f"(p{percentile} of {len(distances)} pairwise distances)"
    )
    return threshold


# ============================================================
# AUTOMATED K SELECTION
# ============================================================
def find_optimal_k(pixels, k_min=5, k_max=20, sample_size=5000, random_state=42):
    """
    Find optimal K for K-Means using silhouette score, Davies-Bouldin index,
    and the elbow method (inertia).

    Parameters
    ----------
    pixels : np.ndarray
        Array of pixel values (N, 3).
    k_min, k_max : int
        Range of K values to evaluate.
    sample_size : int
        Maximum number of pixels to sample for speed.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    int
        Optimal K value.
    dict
        Dictionary with k values and corresponding metrics.
    """
    logger.info(f"  Evaluating K in range [{k_min}, {k_max}] for optimal clustering...")

    # Sample for speed
    if len(pixels) > sample_size:
        idx = np.random.RandomState(random_state).choice(len(pixels), sample_size, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    # Ensure k_values is non-empty: upper bound is half the sample size
    k_upper = min(k_max + 1, len(sample) // 2)
    if k_min >= k_upper:
        logger.warning(
            f"  Sample size too small for requested K range [{k_min}, {k_max}]. "
            f"Using K={max(2, k_min)}."
        )
        k_upper = max(2, k_min) + 1

    k_values = list(range(k_min, k_upper))
    if not k_values:
        k_values = [max(2, k_min)]

    silhouette_scores = []
    davies_bouldin_scores = []
    inertias = []

    for k in k_values:
        clt = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = clt.fit_predict(sample)

        sil = silhouette_score(sample, labels)
        db = davies_bouldin_score(sample, labels)
        inertias.append(clt.inertia_)
        silhouette_scores.append(sil)
        davies_bouldin_scores.append(db)

    # Best K: highest silhouette score (most cohesive + separated clusters)
    best_idx = int(np.argmax(silhouette_scores))
    optimal_k = k_values[best_idx]

    metrics = {
        "k_values": k_values,
        "silhouette_scores": silhouette_scores,
        "davies_bouldin_scores": davies_bouldin_scores,
        "inertias": inertias,
        "optimal_k": optimal_k,
        "best_silhouette": silhouette_scores[best_idx],
        "best_davies_bouldin": davies_bouldin_scores[best_idx],
    }

    logger.info(
        f"  Optimal K = {optimal_k} "
        f"(Silhouette={silhouette_scores[best_idx]:.3f}, "
        f"DB={davies_bouldin_scores[best_idx]:.3f})"
    )
    return optimal_k, metrics


# ============================================================
# HIERARCHICAL CLUSTERING
# ============================================================
def hierarchical_color_clustering(pixels, distance_percentile=85, sample_size=3000, random_state=42):
    """
    Agglomerative hierarchical clustering with adaptive distance threshold.

    Parameters
    ----------
    pixels : np.ndarray
        Array of pixel values (N, 3).
    distance_percentile : int
        Percentile of pairwise distances used to set the cut height.
    sample_size : int
        Maximum number of pixels to sample (hierarchical clustering is O(n²)).
    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray, list, float, np.ndarray
        Cluster centers (RGB), counts, distance threshold, and linkage matrix.
    """
    logger.info("  Running Hierarchical (Agglomerative) Clustering...")

    if len(pixels) > sample_size:
        idx = np.random.RandomState(random_state).choice(len(pixels), sample_size, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    # Build linkage matrix for dendrogram
    Z = linkage(sample, method="ward")

    # Adaptive cut height from distance distribution
    pairwise_dists = pdist(sample)
    cut_height = float(np.percentile(pairwise_dists, distance_percentile))

    labels = fcluster(Z, t=cut_height, criterion="distance") - 1  # 0-indexed
    n_clusters = len(np.unique(labels))

    # Compute cluster centers and counts from the full pixel set
    # Assign each full pixel to the nearest sample cluster center
    # Compute cluster centers using actual unique labels to handle potential label gaps
    unique_labels = np.unique(labels)
    sample_centers = np.array([
        sample[labels == k].mean(axis=0) for k in unique_labels
    ])

    # Assign all pixels to nearest center
    diffs = pixels[:, np.newaxis, :] - sample_centers[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    full_labels = np.argmin(dists, axis=1)

    centers = []
    counts = []
    for k in range(len(sample_centers)):
        mask = full_labels == k
        if mask.any():
            centers.append(pixels[mask].mean(axis=0))
            counts.append(int(mask.sum()))

    logger.info(
        f"  Hierarchical: {n_clusters} clusters found "
        f"(cut height={cut_height:.1f})"
    )
    return np.array(centers), counts, cut_height, Z, sample



# ============================================================
# K-MEANS CLUSTERING (with optimal K)
# ============================================================
def kmeans_color_clustering(pixels, optimal_k, random_state=42):
    """
    K-Means clustering using the automatically determined optimal K.

    Parameters
    ----------
    pixels : np.ndarray
        Array of pixel values (N, 3).
    optimal_k : int
        Number of clusters.
    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray, list
        Cluster centers (RGB) and counts.
    """
    logger.info(f"  Running K-Means with K={optimal_k}...")
    clt = KMeans(n_clusters=optimal_k, n_init="auto", random_state=random_state)
    clt.fit(pixels)
    raw_counts = [int(Counter(clt.labels_)[i]) for i in range(optimal_k)]
    return clt.cluster_centers_, raw_counts



# ============================================================
# SCALE-INDEPENDENT IMAGE NORMALISATION
# ============================================================
def normalize_image_size(image, target_size=1000):
    """
    Resize an image to a square canvas of ``target_size × target_size`` pixels,
    preserving the original aspect ratio by padding with black borders.

    All images processed by this pipeline are normalized to the same canvas
    size before color analysis so that pixel-count-based coverage percentages
    are directly comparable regardless of the original image resolution.

    Parameters
    ----------
    image : np.ndarray
        Input image in any channel format (H × W × C).
    target_size : int, optional
        Side length of the square output canvas in pixels. Default: 1000.

    Returns
    -------
    np.ndarray
        Normalized image of shape ``(target_size, target_size, C)``.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    channels = image.shape[2] if image.ndim == 3 else 1
    canvas = np.zeros((target_size, target_size, channels), dtype=image.dtype)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


# ============================================================
# IMAGE PREPROCESSING
# ============================================================
def preprocess_image(file_path, config):
    """
    Load, remove background, remove glare, and create masks for a single image.

    Parameters
    ----------
    file_path : str
        Path to the image file.
    config : dict
        Configuration dictionary.

    Returns
    -------
    dict or None
        Dictionary with preprocessed image data, or None on failure.
    """
    img_bgr = cv2.imread(file_path)
    if img_bgr is None:
        logger.warning(f"  Could not read image: {file_path}")
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- Scale-independent normalisation ---
    target_size = config.get("STANDARD_CANVAS_SIZE", 1000)
    img_rgb = normalize_image_size(img_rgb, target_size)
    logger.info(f"  Normalized to {target_size}×{target_size} canvas.")

    # Background removal
    if REMBG_AVAILABLE:
        img_pil = Image.fromarray(img_rgb)
        rembg_out = rembg_remove(img_pil)
        img_rgba = np.array(rembg_out)
        alpha = img_rgba[:, :, 3]
        shell_mask = alpha > 10
        img_rgb_work = img_rgba[:, :, :3].copy()
    else:
        # Fallback: use the entire image
        alpha = np.full((img_rgb.shape[0], img_rgb.shape[1]), 255, dtype=np.uint8)
        shell_mask = np.ones((img_rgb.shape[0], img_rgb.shape[1]), dtype=bool)
        img_rgb_work = img_rgb.copy()

    img_rgb_work[~shell_mask] = [0, 0, 0]

    shell_pixel_count = int(shell_mask.sum())
    if shell_pixel_count == 0:
        logger.warning(f"  No shell detected in: {file_path}")
        return None

    # Glare removal
    glare_thresh = config["GLARE_THRESHOLD"]
    hsv_temp = cv2.cvtColor(img_rgb_work, cv2.COLOR_RGB2HSV)
    mask_glare = cv2.inRange(
        hsv_temp,
        np.array([0, 0, glare_thresh]),
        np.array([180, 40, 255]),
    )
    mask_glare = cv2.bitwise_and(mask_glare, mask_glare, mask=shell_mask.astype(np.uint8))
    img_rgb_final = cv2.inpaint(img_rgb_work, mask_glare, 3, cv2.INPAINT_TELEA)

    # HSV masks
    img_hsv = cv2.cvtColor(img_rgb_final, cv2.COLOR_RGB2HSV)

    white_brightness = config["WHITE_BRIGHTNESS"]
    white_sensitivity = config["WHITE_SENSITIVITY"]
    min_brightness = config["MIN_COLOR_BRIGHTNESS"]

    mask_white = cv2.inRange(
        img_hsv,
        np.array([0, 0, white_brightness]),
        np.array([180, white_sensitivity, 255]),
    )
    mask_dark = cv2.inRange(
        img_hsv,
        np.array([0, 0, 0]),
        np.array([180, 255, min_brightness]),
    )
    mask_exclude = cv2.bitwise_or(mask_white, mask_dark)
    mask_exclude = cv2.bitwise_and(mask_exclude, mask_exclude, mask=shell_mask.astype(np.uint8))
    mask_pigment = cv2.subtract(shell_mask.astype(np.uint8) * 255, mask_exclude)
    mask_white_final = cv2.bitwise_and(mask_white, mask_white, mask=shell_mask.astype(np.uint8))

    white_count = int(cv2.countNonZero(mask_white_final))
    pigment_count = int(cv2.countNonZero(mask_pigment))

    pigment_pixels = img_rgb_final[mask_pigment > 0]

    result_rgba = np.dstack((img_rgb_final, alpha))

    return {
        "file": file_path,
        "img_rgb_final": img_rgb_final,
        "img_hsv": img_hsv,
        "shell_mask": shell_mask,
        "mask_pigment": mask_pigment,
        "result_rgba": result_rgba,
        "shell_pixel_count": shell_pixel_count,
        "white_count": white_count,
        "pigment_count": pigment_count,
        "pigment_pixels": pigment_pixels,
    }


# ============================================================
# CLUSTERING PIPELINE
# ============================================================
def run_clustering_pipeline(pixel_stack, config):
    """
    Run the selected clustering method(s) on accumulated pigment pixels.

    Parameters
    ----------
    pixel_stack : np.ndarray
        Stacked array of pigment pixel RGB values.
    config : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Results for each method: centers, counts, metrics.
    """
    method = config["CLUSTERING_METHOD"].lower()
    results = {}
    k_metrics = None

    # --- Automated K selection (always performed for kmeans and all) ---
    if method in ("kmeans", "all"):
        optimal_k, k_metrics = find_optimal_k(
            pixel_stack,
            k_min=config["NUM_CLUSTERS_MIN"],
            k_max=config["NUM_CLUSTERS_MAX"],
        )

        raw_centers, raw_counts = kmeans_color_clustering(pixel_stack, optimal_k)
        merge_thresh = config.get("COLOR_MERGE_THRESHOLD") or compute_adaptive_merge_threshold(
            raw_centers, config["COLOR_MERGE_PERCENTILE"]
        )
        final_centers, final_counts = merge_similar_clusters(raw_centers, raw_counts, merge_thresh)
        results["kmeans"] = {
            "centers": final_centers,
            "counts": final_counts,
            "optimal_k": optimal_k,
            "merge_threshold": merge_thresh,
            "k_metrics": k_metrics,
        }

    if method in ("hierarchical", "all"):
        h_centers, h_counts, h_thresh, linkage_matrix, h_sample = hierarchical_color_clustering(
            pixel_stack,
            distance_percentile=config["HIERARCHICAL_DISTANCE_PERCENTILE"],
        )
        merge_thresh = config.get("COLOR_MERGE_THRESHOLD") or compute_adaptive_merge_threshold(
            h_centers, config["COLOR_MERGE_PERCENTILE"]
        )
        h_centers_merged, h_counts_merged = merge_similar_clusters(h_centers, h_counts, merge_thresh)
        results["hierarchical"] = {
            "centers": h_centers_merged,
            "counts": h_counts_merged,
            "distance_threshold": h_thresh,
            "linkage_matrix": linkage_matrix,
            "sample": h_sample,
            "merge_threshold": merge_thresh,
        }

    return results


# ============================================================
# RESULT FORMATTING
# ============================================================
def format_color_results(centers, counts):
    """
    Sort and enrich cluster results with color names, hex codes, and percentage
    coverage.  All output is scale-independent (percentages of the pigmented area).

    Parameters
    ----------
    centers : np.ndarray
        Cluster centers in RGB.
    counts : list of int
        Pixel counts per cluster.

    Returns
    -------
    list of dict
        Sorted list of color dictionaries.
    """
    total_pigment = sum(counts)
    sorted_pairs = sorted(zip(counts, centers), reverse=True, key=lambda x: x[0])

    colors_out = []
    for i, (count, center) in enumerate(sorted_pairs):
        pct = (count / total_pigment) * 100 if total_pigment > 0 else 0.0
        rgb = np.clip(center, 0, 255).astype(np.uint8)
        name = get_closest_color_name(rgb)
        hex_c = "#{:02x}{:02x}{:02x}".format(*rgb)
        role = "BASE" if i == 0 else "SECONDARY"
        colors_out.append({
            "rank": i + 1,
            "role": role,
            "name": name,
            "hex": hex_c,
            "rgb": rgb.tolist(),
            "count": count,
            "pct_of_pigment": round(pct, 2),
        })

    return colors_out



# ============================================================
# PARAMETER TRAINING SYSTEM
# ============================================================
def train_clustering_params(folder_path, config=None, save_path="trained_params.pkl"):
    """
    Analyze a folder of training images and optimize all tunable clustering
    parameters by finding the values that perform best on average across the
    entire training set.

    The following parameters are optimized:

    * ``NUM_CLUSTERS_MIN`` – Lower bound of the K search range.
    * ``NUM_CLUSTERS_MAX`` – Upper bound of the K search range.
    * ``COLOR_MERGE_PERCENTILE`` – Percentile used for the adaptive merge threshold.
    * ``HIERARCHICAL_DISTANCE_PERCENTILE`` – Cut-height percentile for hierarchical
      clustering.
    * ``optimal_k`` – Average best K found across training images (used as a
      suggested default when running analysis).
    * ``optimal_merge_threshold`` – Average adaptive merge threshold across training
      images.

    No manual labels are needed.  Clustering is performed automatically and the
    best parameters are inferred from the data.

    Parameters
    ----------
    folder_path : str
        Path to folder containing training shell images.
    config : dict, optional
        Base configuration dictionary.  Uses ``DEFAULT_CONFIG`` when omitted.
    save_path : str, optional
        Destination path for the saved parameter file (pickle format).
        Default: ``"trained_params.pkl"``.

    Returns
    -------
    dict
        Dictionary of optimized parameters, also persisted to ``save_path``.
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = sorted([
        f for f in glob.glob(os.path.join(folder_path, "*.*"))
        if os.path.splitext(f)[1].lower() in valid_exts
    ])
    if not image_files:
        logger.error(f"No valid images found in: {folder_path}")
        return {}

    logger.info(f"Training on {len(image_files)} image(s) in: {folder_path}")

    # Candidates to search over
    k_min_candidates = list(range(2, 8))
    k_max_candidates = list(range(8, 25))
    merge_pct_candidates = [15, 20, 25, 30, 35, 40]
    hier_pct_candidates = [70, 75, 80, 85, 90, 95]

    optimal_k_values = []
    merge_threshold_values = []
    best_merge_pct_values = []
    best_hier_pct_values = []

    for i, file in enumerate(image_files):
        fname = os.path.basename(file)
        logger.info(f"  [{i+1}/{len(image_files)}] {fname}")
        data = preprocess_image(file, config)
        if data is None or len(data["pigment_pixels"]) < 10:
            logger.warning(f"    Skipping {fname}: insufficient pigment pixels.")
            continue

        pixels = data["pigment_pixels"].astype(float)

        # --- Find best K for this image ---
        k_abs_min = min(k_min_candidates)
        k_abs_max = max(k_max_candidates)
        best_k, _ = find_optimal_k(
            pixels,
            k_min=k_abs_min,
            k_max=k_abs_max,
        )
        optimal_k_values.append(best_k)
        logger.info(f"    Best K: {best_k}")

        # --- Find best merge percentile (yields lowest intra-cluster LAB spread) ---
        best_mp_score = float("inf")
        best_mp = config["COLOR_MERGE_PERCENTILE"]
        best_merge_thresh = 15.0  # safe default if no merge candidate is selected
        _, raw_counts = kmeans_color_clustering(pixels, best_k)

        # Use K-Means centers for merge percentile search
        clt_tmp = KMeans(n_clusters=best_k, n_init="auto", random_state=42)
        clt_tmp.fit(pixels)
        raw_centers_tmp = clt_tmp.cluster_centers_

        for mp in merge_pct_candidates:
            thresh = compute_adaptive_merge_threshold(raw_centers_tmp, mp)
            merged_c, merged_n = merge_similar_clusters(raw_centers_tmp, raw_counts, thresh)
            # Prefer fewer, more distinct merged clusters
            if len(merged_c) > 0:
                score = -len(merged_c)  # fewer final clusters = tighter merge
                if score < best_mp_score:
                    best_mp_score = score
                    best_mp = mp
                    best_merge_thresh = thresh
        best_merge_pct_values.append(best_mp)
        merge_threshold_values.append(best_merge_thresh)
        logger.info(f"    Best merge percentile: {best_mp}  threshold: {best_merge_thresh:.2f}")

        # --- Find best hierarchical distance percentile ---
        best_hp_score = float("inf")
        best_hp = config["HIERARCHICAL_DISTANCE_PERCENTILE"]
        for hp in hier_pct_candidates:
            h_centers, h_counts, _, _, _ = hierarchical_color_clustering(pixels, distance_percentile=hp)
            # Aim for a reasonable number of clusters (similar to best_k)
            score = abs(len(h_centers) - best_k)
            if score < best_hp_score:
                best_hp_score = score
                best_hp = hp
        best_hier_pct_values.append(best_hp)
        logger.info(f"    Best hierarchical percentile: {best_hp}")

    if not optimal_k_values:
        logger.error("Training failed: no images yielded valid pigment data.")
        return {}

    avg_k = float(np.mean(optimal_k_values))
    avg_merge_threshold = float(np.mean(merge_threshold_values))
    avg_merge_pct = float(np.mean(best_merge_pct_values))
    avg_hier_pct = float(np.mean(best_hier_pct_values))

    # Derive k_min / k_max from the distribution of optimal K values
    trained_k_min = max(2, int(np.percentile(optimal_k_values, 10)))
    trained_k_max = int(np.percentile(optimal_k_values, 90)) + 2  # a little headroom

    trained_params = {
        "NUM_CLUSTERS_MIN": trained_k_min,
        "NUM_CLUSTERS_MAX": trained_k_max,
        "COLOR_MERGE_PERCENTILE": int(round(avg_merge_pct)),
        "HIERARCHICAL_DISTANCE_PERCENTILE": int(round(avg_hier_pct)),
        "optimal_k": int(round(avg_k)),
        "optimal_merge_threshold": round(avg_merge_threshold, 2),
        "training_images": len(optimal_k_values),
        "k_values_found": [int(v) for v in optimal_k_values],
    }

    with open(save_path, "wb") as f:
        pickle.dump(trained_params, f)

    logger.info("\n" + "=" * 55)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 55)
    logger.info(f"  Images trained on   : {trained_params['training_images']}")
    logger.info(f"  K values found      : {trained_params['k_values_found']}")
    logger.info(f"  Trained K_MIN       : {trained_params['NUM_CLUSTERS_MIN']}")
    logger.info(f"  Trained K_MAX       : {trained_params['NUM_CLUSTERS_MAX']}")
    logger.info(f"  Avg optimal K       : {avg_k:.1f}")
    logger.info(f"  Merge percentile    : {trained_params['COLOR_MERGE_PERCENTILE']}")
    logger.info(f"  Avg merge threshold : {trained_params['optimal_merge_threshold']}")
    logger.info(f"  Hier. dist. pct     : {trained_params['HIERARCHICAL_DISTANCE_PERCENTILE']}")
    logger.info(f"  Saved to            : {save_path}")
    logger.info("=" * 55)

    return trained_params


def load_trained_params(path):
    """
    Load trained clustering parameters from disk.

    Parameters
    ----------
    path : str
        Path to the pickled parameter file produced by :func:`train_clustering_params`.

    Returns
    -------
    dict or None
        Parameter dictionary, or ``None`` if the file does not exist.
    """
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        params = pickle.load(f)
    logger.info(f"  Trained parameters loaded from: {path}")
    return params


def apply_trained_params(config, trained_params):
    """
    Overlay trained parameters onto a configuration dictionary.

    Parameters
    ----------
    config : dict
        Base configuration dictionary.
    trained_params : dict
        Optimized parameters from :func:`train_clustering_params`.

    Returns
    -------
    dict
        Updated configuration dictionary.
    """
    cfg = config.copy()
    for key in ("NUM_CLUSTERS_MIN", "NUM_CLUSTERS_MAX",
                "COLOR_MERGE_PERCENTILE", "HIERARCHICAL_DISTANCE_PERCENTILE"):
        if key in trained_params:
            cfg[key] = trained_params[key]
    if "optimal_merge_threshold" in trained_params:
        cfg["COLOR_MERGE_THRESHOLD"] = trained_params["optimal_merge_threshold"]
    return cfg


def train_color_classifier(labeled_data, save_path="color_classifier.pkl"):
    """
    Train a Random Forest color classifier on labeled pixel data.

    The classifier maps CIELAB color features to color-name labels.  It can be
    trained without any hand-labeled images: run the clustering pipeline first
    to discover color groups automatically, then pass those groups as
    ``labeled_data``.

    Parameters
    ----------
    labeled_data : list of (np.ndarray, str)
        List of ``(pixel_array, label_string)`` tuples.  Each ``pixel_array``
        must have shape ``(N, 3)`` in **RGB** order.  ``label_string`` is the
        human-readable color name (e.g. ``"purple"``, ``"brown"``, ``"white"``).

    save_path : str, optional
        File path for the saved classifier (``pickle`` format).
        Default: ``"color_classifier.pkl"``.

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        The trained classifier, also persisted to ``save_path``.
    """
    logger.info("Training supervised color classifier...")
    X, y = [], []
    for pixel_array, label in labeled_data:
        for pixel in pixel_array:
            rgb_img = np.uint8([[[pixel[0], pixel[1], pixel[2]]]])
            lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)[0][0]
            X.append(lab.astype(float))
            y.append(label)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    with open(save_path, "wb") as f:
        pickle.dump(clf, f)

    logger.info(f"  Classifier saved to: {save_path}")
    return clf


def load_color_classifier(path):
    """Load a previously trained color classifier from disk."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        clf = pickle.load(f)
    logger.info(f"  Classifier loaded from: {path}")
    return clf


# ============================================================
# REPORTING: Console + CSV + JSON
# ============================================================
def print_color_table(colors, method_name="K-Means"):
    """Print a formatted color table to the console."""
    header = f"\n{'='*65}\n  {method_name.upper()} COLOR RESULTS\n{'='*65}"
    print(header)
    print(f"{'RANK':<5} {'ROLE':<12} {'COLOR NAME':<22} {'HEX':<9} {'% SHARE'}")
    print("-" * 65)
    for c in colors:
        print(
            f"{c['rank']:<5} {c['role']:<12} {c['name']:<22} {c['hex']:<9} "
            f"{c['pct_of_pigment']:.1f}%"
        )
    print("=" * 65)


def export_results(all_results, output_folder, timestamp):
    """
    Export all clustering results to CSV and JSON files.

    Parameters
    ----------
    all_results : dict
        Dictionary mapping method names to color result lists.
    output_folder : str
        Directory to write output files.
    timestamp : str
        Timestamp string for file naming.
    """
    os.makedirs(output_folder, exist_ok=True)

    for method_name, colors in all_results.items():
        # CSV export
        csv_path = os.path.join(output_folder, f"results_{method_name}_{timestamp}.csv")
        with open(csv_path, "w", newline="") as f:
            if colors:
                writer = csv.DictWriter(f, fieldnames=colors[0].keys())
                writer.writeheader()
                writer.writerows(colors)
        logger.info(f"  CSV saved: {csv_path}")

    # JSON export (all methods together)
    json_path = os.path.join(output_folder, f"results_all_{timestamp}.json")
    with open(json_path, "w") as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable = {}
        for method, colors in all_results.items():
            serializable[method] = [
                {k: convert(v) for k, v in c.items()} for c in colors
            ]

        json.dump(serializable, f, indent=2)

    logger.info(f"  JSON saved: {json_path}")


# ============================================================
# VISUALIZATION
# ============================================================
def plot_optimization_curves(k_metrics, output_folder=None, timestamp=""):
    """Plot K-selection optimization curves (silhouette, Davies-Bouldin, inertia)."""
    if k_metrics is None:
        return

    k_values = k_metrics["k_values"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(k_values, k_metrics["silhouette_scores"], "b-o", linewidth=2, markersize=6)
    axes[0].axvline(x=k_metrics["optimal_k"], color="red", linestyle="--", label=f"Optimal K={k_metrics['optimal_k']}")
    axes[0].set_title("Silhouette Score vs K\n(Higher is Better)")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(k_values, k_metrics["davies_bouldin_scores"], "g-o", linewidth=2, markersize=6)
    axes[1].axvline(x=k_metrics["optimal_k"], color="red", linestyle="--", label=f"Optimal K={k_metrics['optimal_k']}")
    axes[1].set_title("Davies-Bouldin Index vs K\n(Lower is Better)")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Davies-Bouldin Index")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(k_values, k_metrics["inertias"], "r-o", linewidth=2, markersize=6)
    axes[2].axvline(x=k_metrics["optimal_k"], color="blue", linestyle="--", label=f"Optimal K={k_metrics['optimal_k']}")
    axes[2].set_title("Inertia (Elbow Method) vs K\n(Find Elbow)")
    axes[2].set_xlabel("Number of Clusters (K)")
    axes[2].set_ylabel("Inertia")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("K-Means Optimization Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_folder and timestamp:
        fig.savefig(os.path.join(output_folder, f"optimization_curves_{timestamp}.png"), dpi=150)
    plt.show()


def plot_dendrogram(linkage_matrix, sample, cut_height, output_folder=None, timestamp=""):
    """Plot the hierarchical clustering dendrogram with cut height indicator."""
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(linkage_matrix, ax=ax, no_labels=True, color_threshold=cut_height)
    ax.axhline(y=cut_height, color="red", linestyle="--", linewidth=2, label=f"Cut height={cut_height:.1f}")
    ax.set_title("Hierarchical Clustering Dendrogram\n(Red line = adaptive cut height)")
    ax.set_xlabel("Sample Pixels")
    ax.set_ylabel("Distance")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if output_folder and timestamp:
        fig.savefig(os.path.join(output_folder, f"dendrogram_{timestamp}.png"), dpi=150)
    plt.show()


def plot_color_palette(colors, method_name, ax_pie, ax_bar):
    """Plot pie chart and bar chart for a set of color results on given axes."""
    if not colors:
        return

    labels = [
        f"{c['name']}\n{c['hex']}\n({'Base' if c['role']=='BASE' else 'Sec'})\n{c['pct_of_pigment']:.0f}%"
        for c in colors
    ]
    sizes = [c["pct_of_pigment"] for c in colors]
    face_colors = [np.array(c["rgb"]) / 255.0 for c in colors]

    ax_pie.pie(sizes, labels=labels, colors=face_colors, startangle=90,
               textprops={"fontsize": 7})
    ax_pie.set_title(f"{method_name} – Color Proportions", fontsize=9)

    x_pos = np.arange(len(colors))
    ax_bar.bar(x_pos, sizes, color=face_colors, edgecolor="gray", width=0.6)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(
        [f"{c['name']}\n{c['hex']}" for c in colors],
        rotation=45, ha="right", fontsize=7
    )
    ax_bar.set_ylabel("% of Pigmented Area")
    ax_bar.set_title(f"{method_name} – Pigmentation Histogram", fontsize=9)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)


def build_visualization_dashboard(
    processed_images,
    clustering_results,
    k_metrics,
    hierarchical_linkage,
    h_cut_height,
    h_sample,
    config,
    output_folder,
    timestamp,
):
    """
    Build and display the comprehensive multi-panel visualization dashboard.

    Panels:
    1. Processed image gallery (top row)
    2. Color palettes for each method (pie + bar)
    """
    method_names = list(clustering_results.keys())
    n_methods = len(method_names)
    n_images = len(processed_images)

    fig = plt.figure(figsize=(max(18, n_images * 5), 6 + n_methods * 5))
    outer_gs = gridspec.GridSpec(1 + n_methods, 1, figure=fig,
                                 height_ratios=[3] + [4] * n_methods)

    # ---- Row 0: Image Gallery ----
    gallery_gs = gridspec.GridSpecFromSubplotSpec(
        1, max(n_images, 1), subplot_spec=outer_gs[0]
    )
    for idx, img_rgba in enumerate(processed_images):
        ax = fig.add_subplot(gallery_gs[idx])
        ax.imshow(img_rgba)
        ax.set_title(f"Processed Image {idx + 1}", fontsize=9)
        ax.axis("off")

    # ---- Rows 1..n_methods: Color Palettes ----
    for m_idx, method in enumerate(method_names):
        method_colors = clustering_results[method]
        method_gs = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer_gs[1 + m_idx]
        )
        ax_pie = fig.add_subplot(method_gs[0])
        ax_bar = fig.add_subplot(method_gs[1])
        plot_color_palette(method_colors, method.upper(), ax_pie, ax_bar)

    plt.suptitle(
        "Shell Color Analysis Dashboard – Scale-Independent Adaptive Clustering",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    if config.get("SAVE_FIGURES") and output_folder:
        fig_path = os.path.join(output_folder, f"dashboard_{timestamp}.png")
        fig.savefig(fig_path, dpi=150)
        logger.info(f"  Dashboard saved: {fig_path}")

    plt.show()

    # Supplementary plots
    if config.get("SHOW_OPTIMIZATION_CURVES") and k_metrics:
        plot_optimization_curves(k_metrics, output_folder, timestamp)

    if config.get("SHOW_DENDROGRAM") and hierarchical_linkage is not None:
        plot_dendrogram(hierarchical_linkage, h_sample, h_cut_height, output_folder, timestamp)


# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================
def process_images(folder_path, config=None, label=None):
    """
    Main pipeline: load images, preprocess, cluster colors, visualize, and export.

    Parameters
    ----------
    folder_path : str
        Path to folder containing shell images.
    config : dict, optional
        Configuration dictionary. Uses DEFAULT_CONFIG if not provided.
    label : str, optional
        Label to distinguish runs in console output (e.g. "Default" vs "Trained").

    Returns
    -------
    dict
        Formatted clustering results keyed by method name.
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    run_label = f" [{label}]" if label else ""

    # Validate inputs
    if not folder_path or not os.path.exists(folder_path):
        logger.error(f"ERROR: Invalid INPUT_FOLDER path: '{folder_path}'")
        return {}

    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = sorted([
        f for f in glob.glob(os.path.join(folder_path, "*.*"))
        if os.path.splitext(f)[1].lower() in valid_exts
    ])

    if not image_files:
        logger.error(f"No valid images found in: {folder_path}")
        return {}

    logger.info(f"Found {len(image_files)} image(s) to process.{run_label}")

    output_folder = config.get("OUTPUT_FOLDER", "./output")
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if label:
        timestamp = f"{timestamp}_{label.lower().replace(' ', '_')}"

    total_white = 0
    total_pigment = 0
    grand_total_pixels = 0
    all_pigment_pixels = []
    processed_images_gallery = []

    # ---- Image-level processing ----
    for i, file in enumerate(image_files):
        fname = os.path.basename(file)
        logger.info(f"[{i+1}/{len(image_files)}] Processing: {fname}{run_label}")
        data = preprocess_image(file, config)
        if data is None:
            continue

        processed_images_gallery.append(data["result_rgba"])
        total_white += data["white_count"]
        total_pigment += data["pigment_count"]
        grand_total_pixels += data["shell_pixel_count"]

        if len(data["pigment_pixels"]) > 0:
            all_pigment_pixels.append(data["pigment_pixels"])

    # ---- Summary stats ----
    if grand_total_pixels == 0:
        logger.error("No shell area detected in any image.")
        return {}

    canvas = config.get("STANDARD_CANVAS_SIZE", 1000)
    total_pct = 100.0
    white_pct = (total_white / grand_total_pixels * 100) if grand_total_pixels > 0 else 0.0
    pigment_pct = (total_pigment / grand_total_pixels * 100) if grand_total_pixels > 0 else 0.0

    print("\n" + "=" * 65)
    print(f"      ANALYSIS SUMMARY{run_label}")
    print("=" * 65)
    print(f"Normalized canvas size      : {canvas}×{canvas} px")
    print(f"Total shell coverage        : {total_pct:.1f}% of canvas")
    print(f"White/Reflective coverage   : {white_pct:.1f}% of shell")
    print(f"Pigmented coverage          : {pigment_pct:.1f}% of shell")
    print("=" * 65)

    if not all_pigment_pixels:
        logger.warning("No pigmentation found in any image.")
        return {}

    pixel_stack = np.vstack(all_pigment_pixels)
    logger.info(f"Total pigment pixels collected: {len(pixel_stack):,}")

    # ---- Clustering ----
    logger.info(f"\n--- Running Adaptive Clustering Pipeline{run_label} ---")
    raw_cluster_results = run_clustering_pipeline(pixel_stack, config)

    # ---- Format results ----
    formatted_results = {}
    k_metrics = None
    hierarchical_linkage = None
    h_cut_height = None
    h_sample = None

    for method, res in raw_cluster_results.items():
        colors = format_color_results(res["centers"], res["counts"])
        formatted_results[method] = colors
        print_color_table(colors, f"{method}{run_label}")

        if method == "kmeans" and "k_metrics" in res:
            k_metrics = res["k_metrics"]
        if method == "hierarchical":
            hierarchical_linkage = res.get("linkage_matrix")
            h_cut_height = res.get("distance_threshold")
            h_sample = res.get("sample")

    # ---- Export ----
    export_results(formatted_results, output_folder, timestamp)

    # ---- Visualization ----
    logger.info(f"\n--- Building Visualization Dashboard{run_label} ---")
    build_visualization_dashboard(
        processed_images=processed_images_gallery,
        clustering_results=formatted_results,
        k_metrics=k_metrics,
        hierarchical_linkage=hierarchical_linkage,
        h_cut_height=h_cut_height,
        h_sample=h_sample,
        config=config,
        output_folder=output_folder,
        timestamp=timestamp,
    )

    logger.info(f"Analysis complete.{run_label}")
    return formatted_results


# ============================================================
# ENTRY POINT
# ============================================================
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Advanced Adaptive Color Detection Framework for Shell Organisms. "
            "Runs K-Means and/or Hierarchical clustering on scale-independent "
            "normalized shell images.  Can also train optimized clustering "
            "parameters from sample images and compare trained vs. default results. "
            "See COMMANDS.md for a quick-reference and PARAMETER_GUIDE.md for "
            "detailed parameter tuning guidance."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Analyze with both methods (default):\n"
            "    python shell_color_analysis.py --folder images --no-show\n\n"
            "  Analyze with K-Means only, custom K range:\n"
            "    python shell_color_analysis.py --folder images --method kmeans "
            "--k-min 3 --k-max 15 --no-show\n\n"
            "  Train clustering parameters on sample images (no labels needed):\n"
            "    python shell_color_analysis.py --folder training_data --train\n\n"
            "  Analyze with trained parameters:\n"
            "    python shell_color_analysis.py --folder trial --use-trained-params\n\n"
            "  Compare trained vs. default results side-by-side:\n"
            "    python shell_color_analysis.py --folder trial --compare-trained\n"
        ),
    )
    parser.add_argument(
        "--folder", type=str, default=DEFAULT_CONFIG["INPUT_FOLDER"],
        help=(
            "Path to folder containing shell images to analyze. "
            "All .jpg, .jpeg, and .png files in the folder are processed. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_CONFIG["OUTPUT_FOLDER"],
        help=(
            "Path to output folder where result files (CSV, JSON, PNG) are written. "
            "Created automatically if it does not exist. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--method", type=str, default="all",
        choices=["kmeans", "hierarchical", "all"],
        help=(
            "Clustering method(s) to use. "
            "'kmeans' selects the optimal K automatically; "
            "'hierarchical' uses agglomerative clustering and produces a dendrogram; "
            "'all' runs both and writes a combined JSON comparison report. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--k-min", type=int, default=DEFAULT_CONFIG["NUM_CLUSTERS_MIN"],
        help=(
            "Minimum number of clusters (K) for the K-Means search range. "
            "Increase if you know the shell has at least N distinct color zones. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--k-max", type=int, default=DEFAULT_CONFIG["NUM_CLUSTERS_MAX"],
        help=(
            "Maximum number of clusters (K) for the K-Means search range. "
            "Reduce to speed up analysis on simple images; increase for complex, "
            "multi-color patterns. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--merge-threshold", type=float, default=None,
        help=(
            "Fixed CIELAB distance threshold for merging similar color clusters. "
            "When omitted the threshold is computed adaptively from the percentile "
            "of pairwise cluster-center distances (recommended). "
            "Lower values keep more distinct clusters; higher values merge more "
            "aggressively (e.g. 10 = light merging, 20 = aggressive). "
            "(default: adaptive)"
        ),
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help=(
            "Do not open interactive matplotlib windows. "
            "Use this flag in server / batch environments or when running without a display."
        ),
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help=(
            "Do not write any output files (PNG, CSV, JSON) to disk. "
            "Useful for quick exploratory runs."
        ),
    )
    parser.add_argument(
        "--train", action="store_true",
        help=(
            "Train clustering parameters on the images in --folder. "
            "No manual labeling is required: the pipeline automatically finds "
            "the best K range, merge percentile, and hierarchical distance percentile "
            "from the training images, then saves them to the path set by "
            "TRAINED_PARAMS_PATH in DEFAULT_CONFIG (default: trained_params.pkl)."
        ),
    )
    parser.add_argument(
        "--use-trained-params", action="store_true",
        help=(
            "Load previously trained parameters from TRAINED_PARAMS_PATH and use "
            "them for this analysis run instead of DEFAULT_CONFIG defaults. "
            "Run --train first to generate the parameter file."
        ),
    )
    parser.add_argument(
        "--compare-trained", action="store_true",
        help=(
            "Run the analysis twice — once with default parameters and once with "
            "trained parameters — and display both results side-by-side so you can "
            "compare the improvement."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    cfg = DEFAULT_CONFIG.copy()
    cfg["INPUT_FOLDER"] = args.folder
    cfg["OUTPUT_FOLDER"] = args.output
    cfg["CLUSTERING_METHOD"] = args.method
    cfg["NUM_CLUSTERS_MIN"] = args.k_min
    cfg["NUM_CLUSTERS_MAX"] = args.k_max
    cfg["COLOR_MERGE_THRESHOLD"] = args.merge_threshold
    cfg["SAVE_FIGURES"] = not args.no_save

    if args.train:
        # ---- Parameter training mode ----
        train_clustering_params(
            folder_path=cfg["INPUT_FOLDER"],
            config=cfg,
            save_path=cfg["TRAINED_PARAMS_PATH"],
        )

    elif args.compare_trained:
        # ---- Side-by-side comparison mode ----
        trained_params = load_trained_params(cfg["TRAINED_PARAMS_PATH"])
        if trained_params is None:
            logger.error(
                f"No trained parameter file found at '{cfg['TRAINED_PARAMS_PATH']}'. "
                "Run with --train first."
            )
        else:
            logger.info("=== Run 1/2: Default parameters ===")
            process_images(cfg["INPUT_FOLDER"], config=cfg, label="Default")

            trained_cfg = apply_trained_params(cfg, trained_params)
            logger.info("=== Run 2/2: Trained parameters ===")
            process_images(cfg["INPUT_FOLDER"], config=trained_cfg, label="Trained")

    else:
        # ---- Normal analysis mode ----
        if args.use_trained_params:
            trained_params = load_trained_params(cfg["TRAINED_PARAMS_PATH"])
            if trained_params is None:
                logger.warning(
                    f"No trained parameter file found at '{cfg['TRAINED_PARAMS_PATH']}'. "
                    "Falling back to default parameters."
                )
            else:
                cfg = apply_trained_params(cfg, trained_params)
                logger.info("Using trained clustering parameters.")

        process_images(cfg["INPUT_FOLDER"], config=cfg)
