"""
Advanced Adaptive Color Detection Framework for Shell Organisms (Bivalves)
==========================================================================
This program performs comprehensive color identification and pattern analysis
on shell organism images using multiple clustering strategies:

1. Automated K Selection  - Silhouette score, elbow method, Davies-Bouldin index
2. Hierarchical Clustering - Agglomerative with adaptive distance thresholds
3. DBSCAN                  - Density-based clustering for rare color detection
4. Adaptive Merge Logic    - Percentile-based threshold computation
5. Multiple Method Comparison - Side-by-side performance metrics
6. Supervised Classifier   - Optional ML-based color classifier training
7. Enhanced Visualization  - Dashboard with clustering metrics and dendrograms
8. Comprehensive Reporting - CSV and JSON export with detailed statistics

Usage:
    python shell_color_analysis.py
    python shell_color_analysis.py --folder /path/to/images
    python shell_color_analysis.py --folder /path/to/images --method all
    python shell_color_analysis.py --train --folder /path/to/labeled_images
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
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

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
    "PIXELS_PER_UNIT": 176.0454,
    "UNIT_NAME": "cm",

    # --- Clustering Method ---
    # Options: "kmeans", "hierarchical", "dbscan", "all"
    "CLUSTERING_METHOD": "all",

    # --- K-Means Settings ---
    "NUM_CLUSTERS_MIN": 5,
    "NUM_CLUSTERS_MAX": 30,

    # --- Color Merge Settings ---
    # If None, threshold is computed adaptively from data distribution
    "COLOR_MERGE_THRESHOLD": None,
    "COLOR_MERGE_PERCENTILE": 30,  # Percentile of pairwise distances for adaptive threshold

    # --- Hierarchical Clustering ---
    "HIERARCHICAL_DISTANCE_PERCENTILE": 85,  # Percentile for adaptive distance threshold

    # --- DBSCAN Settings ---
    "DBSCAN_EPS_PERCENTILE": 10,  # Percentile of pairwise distances for eps
    "DBSCAN_MIN_SAMPLES_FRACTION": 0.002,  # Fraction of total pixels

    # --- Glare & Shadow ---
    "GLARE_THRESHOLD": 245,
    "MIN_COLOR_BRIGHTNESS": 40,

    # --- White Detection ---
    "WHITE_SENSITIVITY": 50,
    "WHITE_BRIGHTNESS": 150,

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
# DBSCAN CLUSTERING
# ============================================================
def dbscan_color_clustering(pixels, eps_percentile=10, min_samples_fraction=0.002, random_state=42):
    """
    DBSCAN density-based clustering — adapts to local color density and
    naturally detects both dominant and rare colors without pre-specifying K.

    Parameters
    ----------
    pixels : np.ndarray
        Array of pixel values (N, 3).
    eps_percentile : int
        Percentile of pairwise distances to use as eps.
    min_samples_fraction : float
        Fraction of total pixels for min_samples parameter.
    random_state : int
        Random seed for sampling.

    Returns
    -------
    np.ndarray, list, int, dict
        Cluster centers (RGB), counts, number of clusters, and DBSCAN parameters.
    """
    logger.info("  Running DBSCAN Clustering...")

    sample_size = min(3000, len(pixels))
    idx = np.random.RandomState(random_state).choice(len(pixels), sample_size, replace=False)
    sample = pixels[idx]

    # Normalize for DBSCAN
    scaler = StandardScaler()
    sample_scaled = scaler.fit_transform(sample)

    # Adaptive eps from pairwise distance distribution
    pairwise_dists = pdist(sample_scaled)
    eps = float(np.percentile(pairwise_dists, eps_percentile))
    min_samples = max(3, int(len(pixels) * min_samples_fraction))

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels_sample = db.fit_predict(sample_scaled)

    unique_labels = set(labels_sample)
    n_noise = int((labels_sample == -1).sum())
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # Compute centers from sample
    sample_centers = []
    for lbl in sorted(unique_labels):
        if lbl == -1:
            continue
        mask = labels_sample == lbl
        sample_centers.append(sample[mask].mean(axis=0))

    if not sample_centers:
        logger.warning("  DBSCAN: No clusters found. Try adjusting eps_percentile.")
        return np.array([pixels.mean(axis=0)]), [len(pixels)], 1, {}

    sample_centers = np.array(sample_centers)

    # Assign all pixels to nearest DBSCAN center
    diffs = pixels[:, np.newaxis, :] - sample_centers[np.newaxis, :, :]
    full_dists = np.linalg.norm(diffs, axis=2)
    full_labels = np.argmin(full_dists, axis=1)

    centers = []
    counts = []
    for k in range(len(sample_centers)):
        mask = full_labels == k
        if mask.any():
            centers.append(pixels[mask].mean(axis=0))
            counts.append(int(mask.sum()))

    params = {"eps": eps, "min_samples": min_samples, "n_noise_sample": n_noise}
    logger.info(
        f"  DBSCAN: {n_clusters} clusters found "
        f"(eps={eps:.3f}, min_samples={min_samples}, noise in sample={n_noise})"
    )
    return np.array(centers), counts, n_clusters, params


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
# PATTERN ANALYSIS
# ============================================================
def analyze_secondary_pattern(mask_color, shell_area_pixels):
    """
    Analyze the geometry of secondary colors to identify pattern types
    (stripes/bands, spots/dots, irregular patches, etc.).

    Parameters
    ----------
    mask_color : np.ndarray
        Binary mask of the color region.
    shell_area_pixels : int
        Total shell area in pixels (used for minimum area filtering).

    Returns
    -------
    str, int
        Pattern type label and number of detected shapes.
    """
    contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = shell_area_pixels * 0.0005
    valid = [c for c in contours if cv2.contourArea(c) > min_area]
    count = len(valid)

    if count == 0:
        return "Minor Traces", 0
    if count < 5:
        return "Large Patches", count

    elongated = 0
    for c in valid:
        if len(c) < 5:
            continue
        try:
            _, (ma, MA), _ = cv2.fitEllipse(c)
        except cv2.error:
            continue
        if ma == 0 or MA == 0 or min(MA, ma) == 0:
            continue
        if max(MA, ma) / min(MA, ma) > 3.0:
            elongated += 1

    ratio = elongated / count
    if ratio > 0.4:
        return "STRIPES / BANDS", count
    if count > 20:
        return "SPOTS / DOTS", count
    return "IRREGULAR PATCHES", count


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

    if method in ("dbscan", "all"):
        db_centers, db_counts, db_n, db_params = dbscan_color_clustering(
            pixel_stack,
            eps_percentile=config["DBSCAN_EPS_PERCENTILE"],
            min_samples_fraction=config["DBSCAN_MIN_SAMPLES_FRACTION"],
        )
        merge_thresh = config.get("COLOR_MERGE_THRESHOLD") or compute_adaptive_merge_threshold(
            db_centers, config["COLOR_MERGE_PERCENTILE"]
        )
        db_centers_merged, db_counts_merged = merge_similar_clusters(db_centers, db_counts, merge_thresh)
        results["dbscan"] = {
            "centers": db_centers_merged,
            "counts": db_counts_merged,
            "n_clusters_raw": db_n,
            "dbscan_params": db_params,
            "merge_threshold": merge_thresh,
        }

    return results


# ============================================================
# RESULT FORMATTING
# ============================================================
def format_color_results(centers, counts, total_area_cm, grand_total_pixels, unit):
    """
    Sort and enrich cluster results with color names, hex codes, areas, and roles.

    Parameters
    ----------
    centers : np.ndarray
        Cluster centers in RGB.
    counts : list of int
        Pixel counts per cluster.
    total_area_cm : float
        Total shell area in cm².
    grand_total_pixels : int
        Total shell pixel count.
    unit : str
        Unit name (e.g., "cm").

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
        area = (count / grand_total_pixels) * total_area_cm if grand_total_pixels > 0 else 0.0
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
            "area_cm2": round(area, 4),
            "unit": unit,
        })

    return colors_out


# ============================================================
# SUPERVISED CLASSIFIER (OPTIONAL)
# ============================================================
def train_color_classifier(labeled_data, save_path="color_classifier.pkl"):
    """
    Train a Random Forest color classifier on labeled pixel data.

    Parameters
    ----------
    labeled_data : list of (np.ndarray, str)
        List of (pixel_array, label_string) tuples.
    save_path : str
        Path to save the trained classifier.

    Returns
    -------
    RandomForestClassifier
        Trained classifier.
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
    header = f"\n{'='*75}\n  {method_name.upper()} COLOR RESULTS\n{'='*75}"
    print(header)
    print(f"{'RANK':<5} {'ROLE':<12} {'COLOR NAME':<22} {'HEX':<9} {'AREA':<12} {'% SHARE'}")
    print("-" * 75)
    for c in colors:
        area_str = f"{c['area_cm2']:.2f} {c['unit']}²"
        print(
            f"{c['rank']:<5} {c['role']:<12} {c['name']:<22} {c['hex']:<9} "
            f"{area_str:<12} {c['pct_of_pigment']:.1f}%"
        )
    print("=" * 75)


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
    pattern_results,
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
    3. Pattern analysis summary
    """
    method_names = list(clustering_results.keys())
    n_methods = len(method_names)
    n_images = len(processed_images)

    fig = plt.figure(figsize=(max(18, n_images * 5), 6 + n_methods * 5))
    outer_gs = gridspec.GridSpec(2 + n_methods, 1, figure=fig,
                                 height_ratios=[3] + [4] * n_methods + [2])

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

    # ---- Last Row: Pattern Analysis Text ----
    ax_pat = fig.add_subplot(outer_gs[-1])
    ax_pat.axis("off")
    if pattern_results:
        pat_text = "SECONDARY PATTERN ANALYSIS\n" + "-" * 50 + "\n"
        for method, patterns in pattern_results.items():
            pat_text += f"\n[{method.upper()}]\n"
            for entry in patterns:
                pat_text += f"  {entry['color_name']}: {entry['n_shapes']} shapes → {entry['pattern_type']}\n"
    else:
        pat_text = "No secondary pattern analysis available."
    ax_pat.text(0.01, 0.95, pat_text, transform=ax_pat.transAxes,
                verticalalignment="top", fontsize=8, family="monospace")

    plt.suptitle(
        "Shell Color Analysis Dashboard – Advanced Adaptive Clustering",
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
# PATTERN ANALYSIS (per method)
# ============================================================
def run_pattern_analysis(clustering_results, sample_data, config):
    """
    Run secondary pattern analysis for each clustering method.

    Parameters
    ----------
    clustering_results : dict
        Method name → list of color dicts.
    sample_data : dict or None
        Preprocessed data from the largest image.
    config : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Method name → list of pattern result dicts.
    """
    if sample_data is None:
        return {}

    img_hsv = sample_data["img_hsv"]
    mask_pigment = sample_data["mask_pigment"]
    shell_pixel_count = sample_data["shell_pixel_count"]

    pattern_results = {}
    for method, colors in clustering_results.items():
        method_patterns = []
        for c in colors:
            if c["role"] != "SECONDARY":
                continue

            rgb = np.array(c["rgb"], dtype=np.uint8)
            c_hsv = cv2.cvtColor(rgb[np.newaxis, np.newaxis, :], cv2.COLOR_RGB2HSV)[0][0]

            h_tol, sv_tol = 15, 60
            lower = np.array([
                max(0, int(c_hsv[0]) - h_tol),
                max(40, int(c_hsv[1]) - sv_tol),
                max(40, int(c_hsv[2]) - sv_tol),
            ])
            upper = np.array([
                min(179, int(c_hsv[0]) + h_tol),
                min(255, int(c_hsv[1]) + sv_tol),
                min(255, int(c_hsv[2]) + sv_tol),
            ])

            mask = cv2.inRange(img_hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.bitwise_and(mask, mask, mask=mask_pigment)

            pattern_type, n_shapes = analyze_secondary_pattern(mask, shell_pixel_count)
            method_patterns.append({
                "color_name": c["name"],
                "hex": c["hex"],
                "pattern_type": pattern_type,
                "n_shapes": n_shapes,
            })

        pattern_results[method] = method_patterns

    return pattern_results


# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================
def process_images(folder_path, config=None):
    """
    Main pipeline: load images, preprocess, cluster colors, analyze patterns,
    visualize, and export results.

    Parameters
    ----------
    folder_path : str
        Path to folder containing shell images.
    config : dict, optional
        Configuration dictionary. Uses DEFAULT_CONFIG if not provided.
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # Validate inputs
    if not folder_path or not os.path.exists(folder_path):
        logger.error(f"ERROR: Invalid INPUT_FOLDER path: '{folder_path}'")
        return

    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = sorted([
        f for f in glob.glob(os.path.join(folder_path, "*.*"))
        if os.path.splitext(f)[1].lower() in valid_exts
    ])

    if not image_files:
        logger.error(f"No valid images found in: {folder_path}")
        return

    logger.info(f"Found {len(image_files)} image(s) to process.")

    output_folder = config.get("OUTPUT_FOLDER", "./output")
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pixels_per_unit = config["PIXELS_PER_UNIT"]
    unit = config["UNIT_NAME"]
    scale_sq = pixels_per_unit ** 2

    total_white = 0
    total_pigment = 0
    grand_total_pixels = 0
    all_pigment_pixels = []
    processed_images_gallery = []

    best_sample_data = None
    best_sample_area = 0

    # ---- Image-level processing ----
    for i, file in enumerate(image_files):
        fname = os.path.basename(file)
        logger.info(f"[{i+1}/{len(image_files)}] Processing: {fname}")
        data = preprocess_image(file, config)
        if data is None:
            continue

        processed_images_gallery.append(data["result_rgba"])
        total_white += data["white_count"]
        total_pigment += data["pigment_count"]
        grand_total_pixels += data["shell_pixel_count"]

        if len(data["pigment_pixels"]) > 0:
            all_pigment_pixels.append(data["pigment_pixels"])

        if data["shell_pixel_count"] > best_sample_area:
            best_sample_area = data["shell_pixel_count"]
            best_sample_data = data

    # ---- Summary stats ----
    if grand_total_pixels == 0:
        logger.error("No shell area detected in any image.")
        return

    total_area_cm = grand_total_pixels / scale_sq
    white_area = total_white / scale_sq
    pigment_area = total_pigment / scale_sq

    print("\n" + "=" * 65)
    print("      ANALYSIS SUMMARY")
    print("=" * 65)
    print(f"Total Shell Surface Area: {total_area_cm:.2f} {unit}²")
    print(f"White/Reflective Area:    {white_area:.2f} {unit}²")
    print(f"Pigmented Area:           {pigment_area:.2f} {unit}²")
    print("=" * 65)

    if not all_pigment_pixels:
        logger.warning("No pigmentation found in any image.")
        return

    pixel_stack = np.vstack(all_pigment_pixels)
    logger.info(f"Total pigment pixels collected: {len(pixel_stack):,}")

    # ---- Clustering ----
    logger.info("\n--- Running Adaptive Clustering Pipeline ---")
    raw_cluster_results = run_clustering_pipeline(pixel_stack, config)

    # ---- Format results ----
    formatted_results = {}
    k_metrics = None
    hierarchical_linkage = None
    h_cut_height = None
    h_sample = None

    for method, res in raw_cluster_results.items():
        colors = format_color_results(
            res["centers"], res["counts"],
            total_area_cm, grand_total_pixels, unit
        )
        formatted_results[method] = colors
        print_color_table(colors, method)

        if method == "kmeans" and "k_metrics" in res:
            k_metrics = res["k_metrics"]
        if method == "hierarchical":
            hierarchical_linkage = res.get("linkage_matrix")
            h_cut_height = res.get("distance_threshold")
            h_sample = res.get("sample")

    # ---- Pattern Analysis ----
    logger.info("\n--- Secondary Pattern Analysis ---")
    pattern_results = run_pattern_analysis(formatted_results, best_sample_data, config)

    for method, patterns in pattern_results.items():
        print(f"\n[{method.upper()}] Pattern Analysis:")
        if not patterns:
            print("  No secondary colors found (solid/uniform shell).")
        for p in patterns:
            print(f"  {p['color_name']} ({p['hex']}): {p['n_shapes']} shapes → {p['pattern_type']}")

    # ---- Export ----
    export_results(formatted_results, output_folder, timestamp)

    # ---- Visualization ----
    logger.info("\n--- Building Visualization Dashboard ---")
    build_visualization_dashboard(
        processed_images=processed_images_gallery,
        clustering_results=formatted_results,
        pattern_results=pattern_results,
        k_metrics=k_metrics,
        hierarchical_linkage=hierarchical_linkage,
        h_cut_height=h_cut_height,
        h_sample=h_sample,
        config=config,
        output_folder=output_folder,
        timestamp=timestamp,
    )

    logger.info("Analysis complete.")


# ============================================================
# ENTRY POINT
# ============================================================
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Advanced Adaptive Color Detection Framework for Shell Organisms"
    )
    parser.add_argument(
        "--folder", type=str, default=DEFAULT_CONFIG["INPUT_FOLDER"],
        help="Path to folder containing shell images (default: ./images)"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_CONFIG["OUTPUT_FOLDER"],
        help="Path to output folder for results (default: ./output)"
    )
    parser.add_argument(
        "--method", type=str, default="all",
        choices=["kmeans", "hierarchical", "dbscan", "all"],
        help="Clustering method(s) to use (default: all)"
    )
    parser.add_argument(
        "--k-min", type=int, default=DEFAULT_CONFIG["NUM_CLUSTERS_MIN"],
        help="Minimum K for K-Means search (default: 5)"
    )
    parser.add_argument(
        "--k-max", type=int, default=DEFAULT_CONFIG["NUM_CLUSTERS_MAX"],
        help="Maximum K for K-Means search (default: 30)"
    )
    parser.add_argument(
        "--merge-threshold", type=float, default=None,
        help="Fixed CIELAB merge threshold (default: adaptive from data)"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not display plots interactively"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not save figures to output folder"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train supervised classifier (requires labeled data)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg["INPUT_FOLDER"] = args.folder
    cfg["OUTPUT_FOLDER"] = args.output
    cfg["CLUSTERING_METHOD"] = args.method
    cfg["NUM_CLUSTERS_MIN"] = args.k_min
    cfg["NUM_CLUSTERS_MAX"] = args.k_max
    cfg["COLOR_MERGE_THRESHOLD"] = args.merge_threshold
    cfg["SAVE_FIGURES"] = not args.no_save

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    if args.train:
        logger.info(
            "Training mode: prepare labeled_data as list of (pixel_array, label) "
            "and call train_color_classifier()."
        )
        # Example: labeled_data = [(pixels_array, "purple"), ...]
        # train_color_classifier(labeled_data, save_path=cfg["CLASSIFIER_PATH"])
    else:
        process_images(cfg["INPUT_FOLDER"], config=cfg)
