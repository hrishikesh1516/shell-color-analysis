"""
Self-Learning Shell Color Analysis System
==========================================
A complete self-learning color analysis framework for shell organisms
(bivalves and gastropods) using K-Means and Hierarchical clustering.

The system operates in two main modes:

TRAINING MODE  (learns color knowledge from sample images):
    python shell_color_analysis.py --mode train --training-folder ./training_samples
    python shell_color_analysis.py --mode train --training-folder ./training_samples --retrain

ANALYSIS MODE  (analyzes images using trained knowledge):
    python shell_color_analysis.py --mode analyze --input-folder ./dataset --use-trained-model
    python shell_color_analysis.py --mode analyze --input-folder ./dataset

Training produces trained_shell_model.pkl which stores:
  - Trained color centroids (RGB and CIELAB)
  - Color statistics (mean, std, min, max per centroid)
  - Optimal parameters (K_MIN, K_MAX, merge thresholds, hierarchical percentile)
  - Training metadata and validation metrics

Analysis provides two result sets per image:
  - Method 1 (Trained) : matches detected colors to trained centroids,
                          confidence based on CIELAB distance
  - Method 2 (Fresh)   : fresh clustering using learned optimal parameters,
                          confidence based on cluster cohesion
  - Combined           : mean-average of both methods with method comparison

See COMMANDS.md for a complete quick-reference.
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
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
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
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_CONFIG = {
    "INPUT_FOLDER": "./images",
    "OUTPUT_FOLDER": "./output",
    "MODEL_PATH": "trained_shell_model.pkl",
    "STANDARD_CANVAS_SIZE": 1000,
    "NUM_CLUSTERS_MIN": 3,
    "NUM_CLUSTERS_MAX": 15,
    "COLOR_MERGE_THRESHOLD": None,
    "COLOR_MERGE_PERCENTILE": 30,
    "HIERARCHICAL_DISTANCE_PERCENTILE": 85,
    "GLARE_THRESHOLD": 245,
    "MIN_COLOR_BRIGHTNESS": 40,
    "WHITE_SENSITIVITY": 50,
    "WHITE_BRIGHTNESS": 150,
    "CONFIDENCE_SCALE_LAB": 25.0,
    "SHOW_DENDROGRAM": True,
    "SHOW_OPTIMIZATION_CURVES": True,
    "SAVE_FIGURES": True,
    "MIN_PIGMENT_PIXELS": 20,             # Minimum pixels to consider a cluster valid
}

# CIELAB distance threshold for matching colors across methods (deltaE units).
# Colors within this distance are considered the same color.
_LAB_MATCH_THRESHOLD = 30.0


# ============================================================
# COLOR DICTIONARY (XKCD 949 colors -> CIELAB)
# ============================================================
def build_color_dictionary():
    """Build a LAB-space dictionary from the XKCD 949-color set."""
    logger.info("Loading perceptual color dictionary (949 XKCD colors)...")
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


def get_closest_color_name(rgb_tuple):
    """Match an RGB color to the closest XKCD name via CIELAB perceptual distance."""
    rgb_255 = np.uint8([[[int(rgb_tuple[0]), int(rgb_tuple[1]), int(rgb_tuple[2])]]])
    target_lab = cv2.cvtColor(rgb_255, cv2.COLOR_RGB2LAB)[0][0].astype(float)
    min_dist = float("inf")
    best_name = "Unknown"
    for name, lab_val in COLOR_DICT_LAB.items():
        dist = float(np.linalg.norm(target_lab - lab_val.astype(float)))
        if dist < min_dist:
            min_dist = dist
            best_name = name
    return best_name


# ============================================================
# LAB CONVERSION HELPERS
# ============================================================
def pixels_rgb_to_lab(pixels_rgb):
    """Convert array of RGB pixels (N, 3) to CIELAB (float)."""
    arr = np.clip(pixels_rgb, 0, 255).astype(np.uint8).reshape(1, -1, 3)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    return lab.reshape(-1, 3).astype(float)


def rgb_to_lab(rgb):
    """Convert a single RGB color (3,) to CIELAB float array."""
    img = np.uint8([[[int(rgb[0]), int(rgb[1]), int(rgb[2])]]])
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[0][0].astype(float)


# ============================================================
# MERGE SIMILAR CLUSTERS
# ============================================================
def merge_similar_clusters(centers_rgb, counts, distance_threshold=15.0):
    """Merge visually similar clusters using CIELAB perceptual distance."""
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
            delta_e = float(np.linalg.norm(merged_lab[i].astype(float) - merged_lab[j].astype(float)))
            if delta_e < distance_threshold:
                total = merged_counts[i] + merged_counts[j]
                w_i = merged_counts[i] / total
                w_j = merged_counts[j] / total
                new_rgb = merged_rgb[i] * w_i + merged_rgb[j] * w_j
                merged_rgb[i] = new_rgb
                merged_lab[i] = cv2.cvtColor(np.uint8([[new_rgb]]), cv2.COLOR_RGB2LAB)[0][0]
                merged_counts[i] = total
                merged_rgb.pop(j)
                merged_lab.pop(j)
                merged_counts.pop(j)
            else:
                j += 1
        i += 1

    return np.array(merged_rgb), merged_counts


def compute_adaptive_merge_threshold(centers_rgb, percentile=30):
    """Compute adaptive CIELAB merge threshold from pairwise inter-cluster distances."""
    if len(centers_rgb) < 2:
        return 15.0
    centers_lab = pixels_rgb_to_lab(np.array(centers_rgb))
    distances = [
        float(np.linalg.norm(centers_lab[i] - centers_lab[j]))
        for i in range(len(centers_lab))
        for j in range(i + 1, len(centers_lab))
    ]
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
    """Find optimal K for K-Means using silhouette score, Davies-Bouldin, and elbow."""
    logger.info(f"  Evaluating K in [{k_min}, {k_max}]...")
    if len(pixels) > sample_size:
        idx = np.random.RandomState(random_state).choice(len(pixels), sample_size, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    k_upper = min(k_max + 1, len(sample) // 2)
    if k_min >= k_upper:
        k_upper = max(2, k_min) + 1
    k_values = list(range(k_min, k_upper))
    if not k_values:
        k_values = [max(2, k_min)]

    silhouette_scores, db_scores, inertias = [], [], []
    for k in k_values:
        clt = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = clt.fit_predict(sample)
        silhouette_scores.append(float(silhouette_score(sample, labels)))
        db_scores.append(float(davies_bouldin_score(sample, labels)))
        inertias.append(float(clt.inertia_))

    best_idx = int(np.argmax(silhouette_scores))
    optimal_k = k_values[best_idx]
    metrics = {
        "k_values": k_values,
        "silhouette_scores": silhouette_scores,
        "davies_bouldin_scores": db_scores,
        "inertias": inertias,
        "optimal_k": optimal_k,
        "best_silhouette": silhouette_scores[best_idx],
        "best_davies_bouldin": db_scores[best_idx],
    }
    logger.info(
        f"  Optimal K={optimal_k} "
        f"(Sil={silhouette_scores[best_idx]:.3f}, DB={db_scores[best_idx]:.3f})"
    )
    return optimal_k, metrics


# ============================================================
# HIERARCHICAL CLUSTERING
# ============================================================
def hierarchical_color_clustering(pixels, distance_percentile=85,
                                   sample_size=3000, random_state=42):
    """Agglomerative hierarchical clustering with adaptive distance threshold."""
    logger.info("  Running Hierarchical (Agglomerative) Clustering...")
    if len(pixels) > sample_size:
        idx = np.random.RandomState(random_state).choice(len(pixels), sample_size, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    Z = linkage(sample, method="ward")
    pairwise_dists = pdist(sample)
    cut_height = float(np.percentile(pairwise_dists, distance_percentile))
    labels = fcluster(Z, t=cut_height, criterion="distance") - 1
    unique_labels = np.unique(labels)
    sample_centers = np.array([sample[labels == k].mean(axis=0) for k in unique_labels])

    diffs = pixels[:, np.newaxis, :] - sample_centers[np.newaxis, :, :]
    full_labels = np.argmin(np.linalg.norm(diffs, axis=2), axis=1)

    centers, counts = [], []
    for k in range(len(sample_centers)):
        mask = full_labels == k
        if mask.any():
            centers.append(pixels[mask].mean(axis=0))
            counts.append(int(mask.sum()))

    logger.info(
        f"  Hierarchical: {len(centers)} clusters "
        f"(cut_height={cut_height:.1f})"
    )
    return np.array(centers), counts, cut_height, Z, sample


# ============================================================
# K-MEANS CLUSTERING
# ============================================================
def kmeans_color_clustering(pixels, optimal_k, random_state=42):
    """K-Means clustering with given K. Returns centers, counts, labels."""
    logger.info(f"  Running K-Means with K={optimal_k}...")
    clt = KMeans(n_clusters=optimal_k, n_init="auto", random_state=random_state)
    clt.fit(pixels)
    raw_counts = [int(Counter(clt.labels_)[i]) for i in range(optimal_k)]
    return clt.cluster_centers_, raw_counts, clt.labels_


# ============================================================
# SCALE-INDEPENDENT IMAGE NORMALISATION
# ============================================================
def normalize_image_size(image, target_size=1000):
    """Resize image to a square canvas preserving aspect ratio (letter-box)."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
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
    """Load, normalize, remove background/glare, and extract pigment pixels."""
    img_bgr = cv2.imread(file_path)
    if img_bgr is None:
        logger.warning(f"  Could not read image: {file_path}")
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    target_size = config.get("STANDARD_CANVAS_SIZE", 1000)
    img_rgb = normalize_image_size(img_rgb, target_size)
    logger.info(f"  Normalized to {target_size}x{target_size} canvas.")

    if REMBG_AVAILABLE:
        rembg_out = rembg_remove(Image.fromarray(img_rgb))
        img_rgba = np.array(rembg_out)
        alpha = img_rgba[:, :, 3]
        shell_mask = alpha > 10
        img_rgb_work = img_rgba[:, :, :3].copy()
    else:
        alpha = np.full(img_rgb.shape[:2], 255, dtype=np.uint8)
        shell_mask = np.ones(img_rgb.shape[:2], dtype=bool)
        img_rgb_work = img_rgb.copy()

    img_rgb_work[~shell_mask] = [0, 0, 0]
    shell_pixel_count = int(shell_mask.sum())
    if shell_pixel_count == 0:
        logger.warning(f"  No shell detected in: {file_path}")
        return None

    # Glare removal
    hsv_temp = cv2.cvtColor(img_rgb_work, cv2.COLOR_RGB2HSV)
    mask_glare = cv2.inRange(
        hsv_temp,
        np.array([0, 0, config["GLARE_THRESHOLD"]]),
        np.array([180, 40, 255]),
    )
    mask_glare = cv2.bitwise_and(mask_glare, mask_glare, mask=shell_mask.astype(np.uint8))
    img_rgb_final = cv2.inpaint(img_rgb_work, mask_glare, 3, cv2.INPAINT_TELEA)

    img_hsv = cv2.cvtColor(img_rgb_final, cv2.COLOR_RGB2HSV)
    mask_white = cv2.inRange(
        img_hsv,
        np.array([0, 0, config["WHITE_BRIGHTNESS"]]),
        np.array([180, config["WHITE_SENSITIVITY"], 255]),
    )
    mask_dark = cv2.inRange(
        img_hsv,
        np.array([0, 0, 0]),
        np.array([180, 255, config["MIN_COLOR_BRIGHTNESS"]]),
    )
    mask_exclude = cv2.bitwise_or(mask_white, mask_dark)
    mask_exclude = cv2.bitwise_and(mask_exclude, mask_exclude, mask=shell_mask.astype(np.uint8))
    mask_pigment = cv2.subtract(shell_mask.astype(np.uint8) * 255, mask_exclude)
    mask_white_final = cv2.bitwise_and(mask_white, mask_white, mask=shell_mask.astype(np.uint8))

    return {
        "file": file_path,
        "img_rgb_final": img_rgb_final,
        "img_hsv": img_hsv,
        "shell_mask": shell_mask,
        "mask_pigment": mask_pigment,
        "result_rgba": np.dstack((img_rgb_final, alpha)),
        "shell_pixel_count": shell_pixel_count,
        "white_count": int(cv2.countNonZero(mask_white_final)),
        "pigment_count": int(cv2.countNonZero(mask_pigment)),
        "pigment_pixels": img_rgb_final[mask_pigment > 0],
    }


# ============================================================
# TRAINED SHELL MODEL
# ============================================================
class TrainedShellModel:
    """
    Manages trained color knowledge derived from sample shell images.

    Stores consolidated color centroids, per-centroid statistics, learned
    optimal clustering parameters, and training/validation metadata.
    All attributes are plain Python/NumPy objects for pickle serialisation.
    """

    def __init__(self):
        # Consolidated centroids across all training samples
        self.centroids_rgb = None        # np.ndarray (M, 3)
        self.centroids_lab = None        # np.ndarray (M, 3)
        self.centroid_names = []         # list of str color names

        # Per-centroid pixel statistics (list of dicts)
        self.centroid_stats = []

        # Learned optimal parameters
        self.k_min = 3
        self.k_max = 15
        self.merge_threshold = None
        self.merge_percentile = 30
        self.hierarchical_percentile = 85
        self.confidence_scale = 25.0    # CIELAB units

        # Training metadata
        self.n_training_samples = 0
        self.training_date = None
        self.k_values_found = []
        self.per_sample_colors = []     # list of per-sample results (for retraining)

        # Validation metrics (populated after training)
        self.training_accuracy = None   # % colors within threshold of a centroid
        self.consistency_score = None   # 0-100, higher = more consistent

    def save(self, path):
        """Persist the model to a pickle file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.__dict__, fh)
        logger.info(f"  Model saved to: {path}")

    @classmethod
    def load(cls, path):
        """Load a model from a pickle file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        model = cls()
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        model.__dict__.update(data)
        logger.info(f"  Model loaded from: {path}")
        return model

    def find_nearest_centroid(self, color_lab):
        """Find the nearest trained centroid to a CIELAB color. Returns (idx, distance)."""
        if self.centroids_lab is None or len(self.centroids_lab) == 0:
            return 0, float("inf")
        dists = np.linalg.norm(self.centroids_lab - color_lab, axis=1)
        idx = int(np.argmin(dists))
        return idx, float(dists[idx])

    def compute_confidence(self, lab_dist):
        """Convert CIELAB distance to confidence % (exponential decay)."""
        return float(100.0 * np.exp(-lab_dist / max(self.confidence_scale, 1e-6)))


# ============================================================
# TRAINING HELPERS
# ============================================================
def _collect_training_centroids(folder_path, config, existing_colors=None):
    """
    Process sample images and extract per-sample color centroids.

    Returns a list of per-sample result dicts.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = sorted([
        f for f in glob.glob(os.path.join(folder_path, "*.*"))
        if os.path.splitext(f)[1].lower() in valid_exts
    ])
    if not image_files:
        logger.error(f"No valid images found in: {folder_path}")
        return []

    logger.info(f"Found {len(image_files)} training image(s) in: {folder_path}")

    merge_pct_candidates = [15, 20, 25, 30, 35, 40]
    hier_pct_candidates = [70, 75, 80, 85, 90, 95]

    sample_results = []
    for i, file in enumerate(image_files):
        fname = os.path.basename(file)
        logger.info(f"  [{i+1}/{len(image_files)}] {fname}")
        data = preprocess_image(file, config)
        if data is None or len(data["pigment_pixels"]) < config.get("MIN_PIGMENT_PIXELS", 20):
            logger.warning(f"    Skipping {fname}: insufficient pigment pixels.")
            continue

        pixels = data["pigment_pixels"].astype(float)

        # Find optimal K
        best_k, k_metrics = find_optimal_k(pixels, k_min=2, k_max=20)
        logger.info(f"    Optimal K = {best_k}")

        # K-Means with optimal K
        centers, counts, labels = kmeans_color_clustering(pixels, best_k)

        # Best merge percentile
        best_mp = config["COLOR_MERGE_PERCENTILE"]
        best_thresh = 15.0
        best_mp_score = float("inf")
        for mp in merge_pct_candidates:
            thresh = compute_adaptive_merge_threshold(centers, mp)
            merged_c, _ = merge_similar_clusters(centers, counts, thresh)
            score = -len(merged_c)
            if score < best_mp_score:
                best_mp_score = score
                best_mp = mp
                best_thresh = thresh
        final_centers, final_counts = merge_similar_clusters(centers, counts, best_thresh)
        logger.info(f"    Merge pct={best_mp}, threshold={best_thresh:.2f}, "
                    f"final clusters={len(final_centers)}")

        # Best hierarchical percentile
        best_hp = config["HIERARCHICAL_DISTANCE_PERCENTILE"]
        best_hp_score = float("inf")
        for hp in hier_pct_candidates:
            h_c, _, _, _, _ = hierarchical_color_clustering(pixels, distance_percentile=hp)
            score = abs(len(h_c) - best_k)
            if score < best_hp_score:
                best_hp_score = score
                best_hp = hp
        logger.info(f"    Hierarchical percentile={best_hp}")

        sample_results.append({
            "file": file,
            "optimal_k": best_k,
            "centers_rgb": final_centers,
            "counts": final_counts,
            "merge_threshold": best_thresh,
            "merge_percentile": best_mp,
            "hier_percentile": best_hp,
            "k_metrics": k_metrics,
        })

    if existing_colors:
        sample_results = list(existing_colors) + sample_results

    return sample_results


def _consolidate_centroids(sample_results, merge_distance=20.0):
    """Merge all per-sample centroids into a single representative set."""
    all_rgb, all_counts = [], []
    for sr in sample_results:
        for rgb, cnt in zip(sr["centers_rgb"], sr["counts"]):
            all_rgb.append(rgb)
            all_counts.append(cnt)

    if not all_rgb:
        return np.empty((0, 3)), []

    merged_rgb, merged_counts = merge_similar_clusters(
        np.array(all_rgb), all_counts, distance_threshold=merge_distance
    )
    return merged_rgb, merged_counts


def _compute_confidence_scale(centroids_lab):
    """Compute confidence scale as half the median nearest-neighbour CIELAB distance."""
    if len(centroids_lab) < 2:
        return 25.0
    dists = [
        float(np.linalg.norm(centroids_lab[i] - centroids_lab[j]))
        for i in range(len(centroids_lab))
        for j in range(i + 1, len(centroids_lab))
    ]
    return max(float(np.median(dists)) / 2.0, 5.0)


def _compute_centroid_stats(centroids_rgb, sample_results):
    """Compute per-centroid pixel statistics from training data."""
    stats = []
    for c_rgb in centroids_rgb:
        c_lab = rgb_to_lab(c_rgb)
        dists_to_centroid = []
        rgb_values = []
        for sr in sample_results:
            if len(sr["centers_rgb"]) == 0:
                continue
            sr_labs = pixels_rgb_to_lab(np.array(sr["centers_rgb"]))
            nearest_idx = int(np.argmin(np.linalg.norm(sr_labs - c_lab, axis=1)))
            dists_to_centroid.append(float(np.linalg.norm(sr_labs[nearest_idx] - c_lab)))
            rgb_values.append(sr["centers_rgb"][nearest_idx])
        rgb_arr = np.array(rgb_values) if rgb_values else np.array([c_rgb])
        stats.append({
            "mean_dist": float(np.mean(dists_to_centroid)) if dists_to_centroid else 0.0,
            "std_dist": float(np.std(dists_to_centroid)) if dists_to_centroid else 0.0,
            "rgb_mean": rgb_arr.mean(axis=0).tolist(),
            "rgb_std": rgb_arr.std(axis=0).tolist(),
            "rgb_min": rgb_arr.min(axis=0).tolist(),
            "rgb_max": rgb_arr.max(axis=0).tolist(),
        })
    return stats


def _validate_model(model, sample_results, distance_threshold=_LAB_MATCH_THRESHOLD):
    """
    Validate the trained model against training samples.

    Returns (training_accuracy %, consistency_score 0-100).
    """
    if not sample_results or model.centroids_lab is None:
        return 0.0, 0.0

    match_distances = []
    per_sample_accuracies = []
    for sr in sample_results:
        if len(sr["centers_rgb"]) == 0:
            continue
        centers_lab = pixels_rgb_to_lab(np.array(sr["centers_rgb"]))
        sample_dists = []
        matched = 0
        for c_lab in centers_lab:
            _, dist = model.find_nearest_centroid(c_lab)
            sample_dists.append(dist)
            if dist <= distance_threshold:
                matched += 1
        accuracy = matched / len(centers_lab) * 100
        per_sample_accuracies.append(accuracy)
        match_distances.extend(sample_dists)

    training_accuracy = float(np.mean(per_sample_accuracies)) if per_sample_accuracies else 0.0
    if match_distances and float(np.mean(match_distances)) > 0:
        cv = float(np.std(match_distances)) / float(np.mean(match_distances))
        consistency_score = float(max(0.0, 100.0 * (1.0 - min(cv, 1.0))))
    else:
        consistency_score = 100.0

    return training_accuracy, consistency_score


# ============================================================
# TRAIN / RETRAIN
# ============================================================
def train_shell_model(folder_path, config=None, model_path="trained_shell_model.pkl",
                      retrain=False):
    """
    Train (or retrain) the TrainedShellModel on sample images.

    Parameters
    ----------
    folder_path : str
        Folder containing sample shell images.
    config : dict, optional
        Base configuration (uses DEFAULT_CONFIG if not provided).
    model_path : str
        Path to save the resulting model.
    retrain : bool
        If True, load existing model and extend it with new samples.

    Returns
    -------
    TrainedShellModel
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    existing_colors = None
    model = TrainedShellModel()

    if retrain and os.path.exists(model_path):
        logger.info("=== RETRAIN MODE: loading existing model ===")
        try:
            model = TrainedShellModel.load(model_path)
            existing_colors = model.per_sample_colors
            n_centroids = len(model.centroids_rgb) if model.centroids_rgb is not None else 0
            logger.info(
                f"  Existing model: {model.n_training_samples} samples, "
                f"{n_centroids} centroids"
            )
        except Exception as exc:
            logger.warning(f"  Could not load existing model ({exc}). Starting fresh.")
            model = TrainedShellModel()
            existing_colors = None
    elif retrain:
        logger.warning(f"  No existing model at '{model_path}'. Training from scratch.")

    logger.info(f"\n{'='*60}")
    logger.info("  TRAINING PHASE: collecting sample color centroids")
    logger.info(f"{'='*60}")

    sample_results = _collect_training_centroids(folder_path, config, existing_colors)
    if not sample_results:
        logger.error("Training failed: no valid training data.")
        return model

    model.per_sample_colors = sample_results
    model.n_training_samples = len(sample_results)
    model.training_date = datetime.now().isoformat()

    k_vals = [sr["optimal_k"] for sr in sample_results]
    model.k_values_found = k_vals
    model.k_min = max(2, int(np.percentile(k_vals, 10)))
    model.k_max = int(np.percentile(k_vals, 90)) + 2
    model.merge_percentile = int(round(float(np.mean([sr["merge_percentile"] for sr in sample_results]))))
    model.hierarchical_percentile = int(round(float(np.mean([sr["hier_percentile"] for sr in sample_results]))))
    model.merge_threshold = float(np.mean([sr["merge_threshold"] for sr in sample_results]))

    logger.info("\n  Consolidating centroids across all training samples...")
    merged_rgb, _ = _consolidate_centroids(sample_results, merge_distance=model.merge_threshold)
    model.centroids_rgb = merged_rgb
    model.centroids_lab = pixels_rgb_to_lab(merged_rgb)
    model.centroid_names = [get_closest_color_name(c) for c in merged_rgb]
    model.centroid_stats = _compute_centroid_stats(merged_rgb, sample_results)
    model.confidence_scale = _compute_confidence_scale(model.centroids_lab)

    logger.info("\n  Validating model on training samples...")
    model.training_accuracy, model.consistency_score = _validate_model(model, sample_results)

    logger.info(f"\n{'='*60}")
    logger.info("  TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Training samples    : {model.n_training_samples}")
    logger.info(f"  K values found      : {model.k_values_found}")
    logger.info(f"  Learned K_MIN       : {model.k_min}")
    logger.info(f"  Learned K_MAX       : {model.k_max}")
    logger.info(f"  Merge percentile    : {model.merge_percentile}")
    logger.info(f"  Merge threshold     : {model.merge_threshold:.2f}")
    logger.info(f"  Hier. percentile    : {model.hierarchical_percentile}")
    logger.info(f"  Confidence scale    : {model.confidence_scale:.2f} CIELAB units")
    logger.info(f"  Trained centroids   : {len(model.centroids_rgb)}")
    logger.info(f"  Training accuracy   : {model.training_accuracy:.1f}%")
    logger.info(f"  Consistency score   : {model.consistency_score:.1f}/100")
    logger.info(f"{'='*60}")

    model.save(model_path)
    return model


# ============================================================
# CONFIDENCE AND ERROR METRICS
# ============================================================
def compute_moe(pixels_rgb, center_rgb):
    """
    Compute margin of error for a color cluster.

    Returns
    -------
    moe_lab : float - std dev of CIELAB distances from pixels to centroid
    moe_rgb_pct : float - mean RGB channel std dev as % of full range (255)
    """
    if len(pixels_rgb) == 0:
        return 0.0, 0.0
    pixels_lab = pixels_rgb_to_lab(pixels_rgb)
    center_lab = rgb_to_lab(center_rgb)
    dists = np.linalg.norm(pixels_lab - center_lab, axis=1)
    moe_lab = float(np.std(dists))
    rgb_std = np.std(pixels_rgb, axis=0)
    moe_rgb_pct = float(np.mean(rgb_std) / 255.0 * 100.0)
    return moe_lab, moe_rgb_pct


def compute_cohesion_confidence(pixels_rgb, center_rgb, scale=25.0):
    """Cluster cohesion confidence: 100 * exp(-mean_intra_dist / scale)."""
    if len(pixels_rgb) == 0:
        return 0.0
    pixels_lab = pixels_rgb_to_lab(pixels_rgb)
    center_lab = rgb_to_lab(center_rgb)
    mean_dist = float(np.mean(np.linalg.norm(pixels_lab - center_lab, axis=1)))
    return float(100.0 * np.exp(-mean_dist / max(scale, 1e-6)))


# ============================================================
# SELF-LEARNING ANALYZER (two-method analysis)
# ============================================================
class SelfLearningAnalyzer:
    """
    Runs two complementary color analysis methods and combines the results.

    Method 1 (Trained):
        Clusters pixels then matches each cluster to the nearest trained centroid.
        Confidence = 100 * exp(-CIELAB_distance / confidence_scale).

    Method 2 (Fresh):
        Fresh K-Means + Hierarchical clustering using learned optimal parameters.
        Confidence = 100 * exp(-mean_intra_cluster_CIELAB_distance / scale).
    """

    def __init__(self, model=None, config=None):
        self.model = model
        self.config = config.copy() if config else DEFAULT_CONFIG.copy()
        self._apply_model_params()

    def _apply_model_params(self):
        """Overlay trained optimal parameters onto config."""
        if self.model is None:
            return
        self.config["NUM_CLUSTERS_MIN"] = self.model.k_min
        self.config["NUM_CLUSTERS_MAX"] = self.model.k_max
        self.config["COLOR_MERGE_PERCENTILE"] = self.model.merge_percentile
        self.config["HIERARCHICAL_DISTANCE_PERCENTILE"] = self.model.hierarchical_percentile
        if self.model.merge_threshold is not None:
            self.config["COLOR_MERGE_THRESHOLD"] = self.model.merge_threshold

    def analyze_trained_method(self, pixel_stack):
        """
        Method 1: K-Means clustering whose clusters are matched to trained centroids.
        Returns list of color result dicts.
        """
        if self.model is None or self.model.centroids_rgb is None:
            return []

        if self.model.k_values_found:
            optimal_k = max(
                self.config["NUM_CLUSTERS_MIN"],
                min(
                    int(round(float(np.mean(self.model.k_values_found)))),
                    self.config["NUM_CLUSTERS_MAX"],
                ),
            )
        else:
            optimal_k = self.config["NUM_CLUSTERS_MIN"]

        centers, counts, labels = kmeans_color_clustering(pixel_stack, optimal_k)
        scale = self.model.confidence_scale

        results = []
        for cluster_idx, (center, count) in enumerate(zip(centers, counts)):
            center_lab = rgb_to_lab(center)
            nearest_idx, dist = self.model.find_nearest_centroid(center_lab)
            color_name = self.model.centroid_names[nearest_idx]
            confidence = self.model.compute_confidence(dist)
            cluster_pixels = pixel_stack[labels == cluster_idx]
            moe_lab, moe_rgb_pct = compute_moe(cluster_pixels, center)
            rgb = np.clip(center, 0, 255).astype(np.uint8)
            results.append({
                "center_rgb": rgb,
                "count": int(count),
                "name": color_name,
                "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
                "confidence_trained": round(confidence, 1),
                "distance_to_centroid_lab": round(dist, 2),
                "moe_lab": round(moe_lab, 2),
                "moe_rgb_pct": round(moe_rgb_pct, 2),
            })
        return results

    def analyze_fresh_method(self, pixel_stack):
        """
        Method 2: Fresh K-Means + Hierarchical with learned optimal parameters.
        Returns (color_results_list, k_metrics, linkage_matrix, h_cut_height, h_sample).
        """
        k_min = self.config["NUM_CLUSTERS_MIN"]
        k_max = self.config["NUM_CLUSTERS_MAX"]
        optimal_k, k_metrics = find_optimal_k(pixel_stack, k_min=k_min, k_max=k_max)

        # K-Means
        km_centers, km_counts, km_labels = kmeans_color_clustering(pixel_stack, optimal_k)
        merge_thresh = (
            self.config.get("COLOR_MERGE_THRESHOLD")
            or compute_adaptive_merge_threshold(km_centers, self.config["COLOR_MERGE_PERCENTILE"])
        )
        km_centers_m, km_counts_m = merge_similar_clusters(km_centers, km_counts, merge_thresh)

        # Hierarchical
        h_centers, h_counts, h_thresh, linkage_matrix, h_sample = hierarchical_color_clustering(
            pixel_stack,
            distance_percentile=self.config["HIERARCHICAL_DISTANCE_PERCENTILE"],
        )
        h_merge_thresh = (
            self.config.get("COLOR_MERGE_THRESHOLD")
            or compute_adaptive_merge_threshold(h_centers, self.config["COLOR_MERGE_PERCENTILE"])
        )
        h_centers_m, h_counts_m = merge_similar_clusters(h_centers, h_counts, h_merge_thresh)

        # Combine K-Means + Hierarchical (merge both sets together)
        combined_rgb = list(km_centers_m) + list(h_centers_m)
        combined_counts = list(km_counts_m) + list(h_counts_m)
        if combined_rgb:
            combined_rgb_arr = np.array(combined_rgb)
            combined_rgb_arr, combined_counts = merge_similar_clusters(
                combined_rgb_arr, combined_counts, merge_thresh
            )
        else:
            return [], k_metrics, linkage_matrix, h_thresh, h_sample

        scale = (self.model.confidence_scale if self.model else
                 self.config.get("CONFIDENCE_SCALE_LAB", 25.0))

        # Assign all pixels to nearest combined centre
        combined_labs = pixels_rgb_to_lab(combined_rgb_arr)
        all_labs = pixels_rgb_to_lab(pixel_stack)
        diffs = all_labs[:, np.newaxis, :] - combined_labs[np.newaxis, :, :]
        full_assign = np.argmin(np.linalg.norm(diffs, axis=2), axis=1)

        results = []
        for idx, (center, count) in enumerate(zip(combined_rgb_arr, combined_counts)):
            mask = full_assign == idx
            cluster_pixels = pixel_stack[mask]
            confidence = compute_cohesion_confidence(cluster_pixels, center, scale)
            moe_lab, moe_rgb_pct = compute_moe(cluster_pixels, center)
            rgb = np.clip(center, 0, 255).astype(np.uint8)
            dist_to_centroid = None
            if self.model and self.model.centroids_lab is not None:
                center_lab = rgb_to_lab(rgb)
                _, dist_to_centroid = self.model.find_nearest_centroid(center_lab)
                dist_to_centroid = round(float(dist_to_centroid), 2)
            results.append({
                "center_rgb": rgb,
                "count": int(count),
                "name": get_closest_color_name(rgb),
                "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
                "confidence_fresh": round(confidence, 1),
                "distance_to_centroid_lab": dist_to_centroid,
                "moe_lab": round(moe_lab, 2),
                "moe_rgb_pct": round(moe_rgb_pct, 2),
            })
        return results, k_metrics, linkage_matrix, h_thresh, h_sample

    def combine_results(self, trained_results, fresh_results, total_pigment_pixels):
        """
        Reconcile trained and fresh results into a single ranked color list.
        Colors are matched by nearest CIELAB distance (threshold = _LAB_MATCH_THRESHOLD deltaE).
        """
        LAB_MATCH_THRESH = _LAB_MATCH_THRESHOLD

        def _to_lab(r):
            return rgb_to_lab(r["center_rgb"])

        combined = []

        if not trained_results:
            # Only fresh method available
            for rank, r in enumerate(
                sorted(fresh_results, key=lambda x: x["count"], reverse=True), 1
            ):
                pct = r["count"] / max(total_pigment_pixels, 1) * 100.0
                combined.append({
                    "rank": rank,
                    "role": "BASE" if rank == 1 else "SECONDARY",
                    "center_rgb": r["center_rgb"],
                    "name": r["name"],
                    "hex": r["hex"],
                    "count": r["count"],
                    "pct_of_pigment": round(pct, 2),
                    "confidence_trained": None,
                    "confidence_fresh": r.get("confidence_fresh"),
                    "combined_confidence": r.get("confidence_fresh"),
                    "distance_to_centroid_lab": r.get("distance_to_centroid_lab"),
                    "moe_lab": r.get("moe_lab"),
                    "moe_rgb_pct": r.get("moe_rgb_pct"),
                    "higher_confidence_method": "Fresh",
                    "suggested_method": "Fresh",
                })
            return combined

        # Match trained results to fresh results
        used_fresh = [False] * len(fresh_results)
        rows = []
        for tr in trained_results:
            tr_lab = _to_lab(tr)
            best_idx, best_dist = -1, LAB_MATCH_THRESH
            for fi, fr in enumerate(fresh_results):
                if used_fresh[fi]:
                    continue
                dist = float(np.linalg.norm(_to_lab(fr) - tr_lab))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = fi

            fr = fresh_results[best_idx] if best_idx >= 0 else None
            if fr is not None:
                used_fresh[best_idx] = True

            conf_trained = tr.get("confidence_trained") or 0.0
            conf_fresh = fr.get("confidence_fresh") or 0.0 if fr else 0.0
            combined_conf = (conf_trained + conf_fresh) / 2.0 if fr else conf_trained

            count = (tr["count"] + fr["count"]) // 2 if fr else tr["count"]
            higher_method = "Trained" if conf_trained >= conf_fresh else "Fresh"
            rows.append({
                "center_rgb": tr["center_rgb"],
                "name": tr["name"],
                "hex": tr["hex"],
                "count": count,
                "confidence_trained": round(conf_trained, 1),
                "confidence_fresh": round(conf_fresh, 1) if fr else None,
                "combined_confidence": round(combined_conf, 1),
                "distance_to_centroid_lab": tr.get("distance_to_centroid_lab"),
                "moe_lab": tr.get("moe_lab"),
                "moe_rgb_pct": tr.get("moe_rgb_pct"),
                "higher_confidence_method": higher_method,
                "suggested_method": higher_method,
            })

        # Add unmatched fresh results
        for fi, fr in enumerate(fresh_results):
            if not used_fresh[fi]:
                conf_fresh = fr.get("confidence_fresh") or 0.0
                rows.append({
                    "center_rgb": fr["center_rgb"],
                    "name": fr["name"],
                    "hex": fr["hex"],
                    "count": fr["count"],
                    "confidence_trained": None,
                    "confidence_fresh": round(conf_fresh, 1),
                    "combined_confidence": round(conf_fresh, 1),
                    "distance_to_centroid_lab": fr.get("distance_to_centroid_lab"),
                    "moe_lab": fr.get("moe_lab"),
                    "moe_rgb_pct": fr.get("moe_rgb_pct"),
                    "higher_confidence_method": "Fresh",
                    "suggested_method": "Fresh",
                })

        # Sort by count descending, assign rank
        rows = sorted(rows, key=lambda x: x["count"], reverse=True)
        for rank, row in enumerate(rows, 1):
            pct = row["count"] / max(total_pigment_pixels, 1) * 100.0
            row["rank"] = rank
            row["role"] = "BASE" if rank == 1 else "SECONDARY"
            row["pct_of_pigment"] = round(pct, 2)
        return rows


# ============================================================
# REPORTING: Console, CSV, JSON
# ============================================================
def print_color_table(colors):
    """Print formatted color result table to console."""
    if not colors:
        return
    sep = "=" * 108
    print(f"\n{sep}")
    print(
        f"  {'RNK':<4} {'ROLE':<10} {'COLOR NAME':<22} {'HEX':<9} "
        f"{'%PIGm':>6} {'CONF-T':>7} {'CONF-F':>7} {'COMB':>6} "
        f"{'DIST-dE':>8} {'MoE-dE':>7} {'MoE-RGB%':>9} {'HIGHER METHOD'}"
    )
    print(sep)
    for c in colors:
        def _f(v):
            return f"{v:.1f}" if v is not None else "  N/A"
        print(
            f"  {c['rank']:<4} {c['role']:<10} {c['name']:<22} {c['hex']:<9} "
            f"{c['pct_of_pigment']:>5.1f}% "
            f"{_f(c.get('confidence_trained')):>7} "
            f"{_f(c.get('confidence_fresh')):>7} "
            f"{_f(c.get('combined_confidence')):>6} "
            f"{_f(c.get('distance_to_centroid_lab')):>8} "
            f"{_f(c.get('moe_lab')):>7} "
            f"{_f(c.get('moe_rgb_pct')):>9} "
            f"{c.get('higher_confidence_method', 'N/A')}"
        )
    print(sep)


def export_results(combined_colors, output_folder, timestamp):
    """Export combined results to CSV and JSON."""
    os.makedirs(output_folder, exist_ok=True)

    fieldnames = [
        "rank", "role", "name", "hex", "rgb",
        "pct_of_pigment",
        "confidence_trained", "confidence_fresh", "combined_confidence",
        "distance_to_centroid_lab", "moe_lab", "moe_rgb_pct",
        "higher_confidence_method", "suggested_method",
    ]

    csv_path = os.path.join(output_folder, f"results_combined_{timestamp}.csv")
    if combined_colors:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for c in combined_colors:
                row = {k: c.get(k) for k in fieldnames}
                rgb_val = c.get("center_rgb")
                row["rgb"] = str(rgb_val.tolist() if hasattr(rgb_val, "tolist") else rgb_val)
                writer.writerow(row)
        logger.info(f"  CSV saved: {csv_path}")

    json_path = os.path.join(output_folder, f"results_combined_{timestamp}.json")

    def _cvt(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(json_path, "w") as f:
        json.dump([{k: _cvt(v) for k, v in c.items()} for c in combined_colors], f, indent=2)
    logger.info(f"  JSON saved: {json_path}")


# ============================================================
# VISUALIZATION - TRAINING INFOGRAPHIC
# ============================================================
def plot_training_infographic(model, sample_results, output_folder, timestamp):
    """
    Professional training summary infographic:
    - K values found per sample
    - K distribution histogram
    - Trained centroids in CIELAB a*b* scatter plot
    - Color swatches of all trained centroids
    - Training validation metrics
    """
    dark_bg = "#1a1a2e"
    panel_bg = "#16213e"
    title_kw = dict(color="white", fontsize=11, fontweight="bold", pad=8)
    label_kw = dict(color="#cccccc", fontsize=9)
    tick_kw = dict(colors="#aaaaaa", labelsize=8)

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(dark_bg)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.92, bottom=0.06, left=0.06, right=0.97)

    # 1. K values per sample (bar chart)
    ax1 = fig.add_subplot(gs[0, :2])
    k_vals = [sr["optimal_k"] for sr in sample_results]
    fnames = [os.path.basename(sr["file"])[:15] for sr in sample_results]
    bar_colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(k_vals)))
    bars = ax1.bar(range(len(k_vals)), k_vals, color=bar_colors, edgecolor="#444444")
    ax1.set_xticks(range(len(k_vals)))
    ax1.set_xticklabels(fnames, rotation=45, ha="right", **label_kw)
    ax1.set_ylabel("Optimal K", **label_kw)
    ax1.set_title("Optimal K Found per Training Sample", **title_kw)
    ax1.set_facecolor(panel_bg)
    ax1.tick_params(**tick_kw)
    mean_k = float(np.mean(k_vals)) if k_vals else 0
    ax1.axhline(y=mean_k, color="#ff6b6b", linestyle="--",
                linewidth=1.5, label=f"Mean K={mean_k:.1f}")
    ax1.legend(fontsize=8, facecolor=panel_bg, labelcolor="white")
    for bar, kv in zip(bars, k_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(kv), ha="center", va="bottom", color="white", fontsize=8)

    # 2. K distribution histogram
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(k_vals, bins=max(3, len(set(k_vals))), color="#e94560",
             edgecolor=panel_bg, alpha=0.85)
    ax2.set_xlabel("Optimal K", **label_kw)
    ax2.set_ylabel("Frequency", **label_kw)
    ax2.set_title("K Distribution\nacross Samples", **title_kw)
    ax2.set_facecolor(panel_bg)
    ax2.tick_params(**tick_kw)

    # 3. LAB a*b* scatter
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor(panel_bg)
    if model.centroids_lab is not None and len(model.centroids_lab) > 0:
        a_vals = model.centroids_lab[:, 1] - 128.0
        b_vals = model.centroids_lab[:, 2] - 128.0
        colors_norm = np.clip(model.centroids_rgb / 255.0, 0, 1)
        ax3.scatter(a_vals, b_vals, c=colors_norm,
                    s=180, edgecolors="white", linewidths=0.8, zorder=3)
        for a, b, name in zip(a_vals, b_vals, model.centroid_names):
            ax3.annotate(name[:12], (a, b), textcoords="offset points",
                         xytext=(6, 4), fontsize=6.5, color="#dddddd")
    ax3.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax3.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax3.set_xlabel("a* (Green-Red)", **label_kw)
    ax3.set_ylabel("b* (Blue-Yellow)", **label_kw)
    ax3.set_title("Trained Color Centroids - CIELAB a*b* Plane", **title_kw)
    ax3.tick_params(**tick_kw)
    ax3.grid(True, color="#333355", linewidth=0.5, alpha=0.6)

    # 4. Centroid color swatches
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(panel_bg)
    if model.centroids_rgb is not None and len(model.centroids_rgb) > 0:
        n = len(model.centroids_rgb)
        for idx, (rgb, name) in enumerate(zip(model.centroids_rgb, model.centroid_names)):
            y = 1.0 - (idx + 0.5) / n
            h = 0.85 / n
            rect = mpatches.FancyBboxPatch(
                (0.05, y - h / 2), 0.3, h,
                boxstyle="round,pad=0.005",
                facecolor=np.clip(rgb / 255.0, 0, 1),
                edgecolor="white", linewidth=0.5,
            )
            ax4.add_patch(rect)
            ax4.text(0.42, y, name[:18], va="center", ha="left",
                     color="#dddddd", fontsize=7)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    n_centroids = len(model.centroids_rgb) if model.centroids_rgb is not None else 0
    ax4.set_title(f"Trained Color Palette\n({n_centroids} centroids)", **title_kw)
    ax4.axis("off")

    # 5. Parameter summary table
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.set_facecolor(panel_bg)
    ax5.axis("off")
    summary_lines = [
        ("Training Samples", str(model.n_training_samples)),
        ("K Values Discovered", str(model.k_values_found)),
        ("Learned K_MIN", str(model.k_min)),
        ("Learned K_MAX", str(model.k_max)),
        ("Merge Percentile", str(model.merge_percentile)),
        ("Merge Threshold (dE)", f"{model.merge_threshold:.2f}" if model.merge_threshold else "N/A"),
        ("Hierarchical Percentile", str(model.hierarchical_percentile)),
        ("Confidence Scale (dE)", f"{model.confidence_scale:.2f}"),
        ("Training Accuracy", f"{model.training_accuracy:.1f}%" if model.training_accuracy is not None else "N/A"),
        ("Consistency Score", f"{model.consistency_score:.1f}/100" if model.consistency_score is not None else "N/A"),
        ("Training Date", str(model.training_date)[:19] if model.training_date else "N/A"),
    ]
    for row_idx, (label, value) in enumerate(summary_lines):
        y = 0.95 - row_idx * 0.082
        ax5.text(0.02, y, label + ":", transform=ax5.transAxes,
                 color="#aaaaaa", fontsize=9, va="top")
        ax5.text(0.35, y, value, transform=ax5.transAxes,
                 color="#f0f0f0", fontsize=9, va="top", fontweight="bold")
    ax5.set_title("Parameter Summary", **title_kw)

    # 6. Validation gauge
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(panel_bg)
    metrics_names = ["Training\nAccuracy", "Consistency\nScore"]
    metrics_vals = [
        model.training_accuracy if model.training_accuracy is not None else 0.0,
        model.consistency_score if model.consistency_score is not None else 0.0,
    ]
    hbars = ax6.barh(metrics_names, metrics_vals, color=["#00b4d8", "#90e0ef"],
                     edgecolor=panel_bg, height=0.4)
    ax6.set_xlim(0, 105)
    ax6.set_xlabel("Score (%)", **label_kw)
    ax6.set_title("Validation Metrics", **title_kw)
    ax6.set_facecolor(panel_bg)
    ax6.tick_params(**tick_kw)
    for bar, val in zip(hbars, metrics_vals):
        ax6.text(val + 1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", color="white", fontsize=9)

    fig.suptitle(
        "Shell Color Analysis - Training Summary Report",
        fontsize=16, fontweight="bold", color="white", y=0.97
    )

    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, f"training_summary_{timestamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"  Training infographic saved: {path}")
    return fig


# ============================================================
# VISUALIZATION - ANALYSIS DASHBOARD
# ============================================================
def plot_analysis_dashboard(
    combined_colors,
    trained_colors_pie,
    fresh_colors_pie,
    processed_images,
    model,
    k_metrics,
    output_folder,
    timestamp,
):
    """
    Professional multi-panel analysis dashboard:
    Row 0: Processed image gallery
    Row 1: Multi-method comparison pie charts (Trained / Fresh / Combined)
    Row 2: Confidence distribution histogram + CIELAB centroid scatter plot
    Row 3: K-optimization curves (if k_metrics provided)
    """
    dark_bg = "#1a1a2e"
    panel_bg = "#16213e"
    title_kw = dict(color="white", fontsize=10, fontweight="bold", pad=6)
    label_kw = dict(color="#cccccc", fontsize=8)
    tick_kw = dict(colors="#aaaaaa", labelsize=7)

    n_rows = 3 + (1 if k_metrics else 0)
    height_ratios = [2.5, 3.5, 3.5] + ([3] if k_metrics else [])
    fig = plt.figure(figsize=(22, 5 * n_rows))
    fig.patch.set_facecolor(dark_bg)
    outer_gs = gridspec.GridSpec(
        n_rows, 1, figure=fig, hspace=0.4,
        height_ratios=height_ratios, top=0.95, bottom=0.04,
        left=0.04, right=0.97,
    )

    # Row 0: Image Gallery
    n_imgs = max(len(processed_images), 1)
    gallery_gs = gridspec.GridSpecFromSubplotSpec(1, n_imgs, subplot_spec=outer_gs[0])
    for idx, img_rgba in enumerate(processed_images[:n_imgs]):
        ax = fig.add_subplot(gallery_gs[idx])
        ax.imshow(img_rgba)
        ax.set_title(f"Processed Image {idx+1}", **title_kw)
        ax.axis("off")

    # Row 1: Multi-method comparison (3 pie charts)
    compare_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1], wspace=0.3)
    pie_datasets = [
        ("Method 1\n(Trained)", trained_colors_pie),
        ("Method 2\n(Fresh)", fresh_colors_pie),
        ("Combined\nResult", combined_colors),
    ]
    for col_idx, (label, color_list) in enumerate(pie_datasets):
        ax = fig.add_subplot(compare_gs[col_idx])
        ax.set_facecolor(panel_bg)
        if color_list:
            sizes = [max(c.get("pct_of_pigment") or c.get("count", 1) / 1000.0, 0.1) for c in color_list]
            total = sum(sizes) or 1.0
            sizes_pct = [s / total * 100 for s in sizes]
            face_colors = [np.clip(c["center_rgb"] / 255.0, 0, 1) for c in color_list]
            lbl_strs = [
                f"{c['name'][:12]}\n{c['hex']}\n{s:.0f}%"
                for c, s in zip(color_list, sizes_pct)
            ]
            ax.pie(sizes_pct, labels=lbl_strs, colors=face_colors,
                   startangle=90, textprops={"fontsize": 6, "color": "#dddddd"},
                   wedgeprops=dict(linewidth=0.8, edgecolor=dark_bg))
        ax.set_title(label, **title_kw)

    # Row 2: Confidence distribution + LAB scatter
    row2_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[2], wspace=0.35)

    ax_hist = fig.add_subplot(row2_gs[0])
    ax_hist.set_facecolor(panel_bg)
    conf_t = [c["confidence_trained"] for c in combined_colors if c.get("confidence_trained") is not None]
    conf_f = [c["confidence_fresh"] for c in combined_colors if c.get("confidence_fresh") is not None]
    conf_c = [c["combined_confidence"] for c in combined_colors if c.get("combined_confidence") is not None]
    bins = np.linspace(0, 100, 21)
    if conf_t:
        ax_hist.hist(conf_t, bins=bins, alpha=0.7, color="#e94560", label="Trained")
    if conf_f:
        ax_hist.hist(conf_f, bins=bins, alpha=0.7, color="#00b4d8", label="Fresh")
    if conf_c:
        ax_hist.hist(conf_c, bins=bins, alpha=0.7, color="#90e0ef", label="Combined")
    ax_hist.set_xlabel("Confidence (%)", **label_kw)
    ax_hist.set_ylabel("Count", **label_kw)
    ax_hist.set_title("Confidence Score Distribution", **title_kw)
    ax_hist.legend(fontsize=8, facecolor=panel_bg, labelcolor="white")
    ax_hist.tick_params(**tick_kw)
    ax_hist.grid(True, color="#333355", linewidth=0.5, alpha=0.5)

    ax_lab = fig.add_subplot(row2_gs[1])
    ax_lab.set_facecolor(panel_bg)
    if model and model.centroids_lab is not None and len(model.centroids_lab) > 0:
        a_t = model.centroids_lab[:, 1] - 128.0
        b_t = model.centroids_lab[:, 2] - 128.0
        fc_t = np.clip(model.centroids_rgb / 255.0, 0, 1)
        ax_lab.scatter(a_t, b_t, c=fc_t, s=200, marker="*",
                       edgecolors="white", linewidths=0.8, zorder=4,
                       label="Trained Centroids")
    if combined_colors:
        det_rgb = np.array([np.clip(c["center_rgb"], 0, 255) for c in combined_colors])
        det_lab = pixels_rgb_to_lab(det_rgb)
        a_d = det_lab[:, 1] - 128.0
        b_d = det_lab[:, 2] - 128.0
        fc_d = det_rgb / 255.0
        sizes = [max(40, (c.get("combined_confidence") or 50.0) * 2.5) for c in combined_colors]
        ax_lab.scatter(a_d, b_d, c=fc_d, s=sizes,
                       edgecolors="#888888", linewidths=0.6, zorder=3,
                       label="Detected Colors", marker="o")
    ax_lab.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax_lab.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax_lab.set_xlabel("a* (Green-Red)", **label_kw)
    ax_lab.set_ylabel("b* (Blue-Yellow)", **label_kw)
    ax_lab.set_title("CIELAB Centroid Distribution\n(star=trained, circle=detected)", **title_kw)
    ax_lab.legend(fontsize=7, facecolor=panel_bg, labelcolor="white")
    ax_lab.tick_params(**tick_kw)
    ax_lab.grid(True, color="#333355", linewidth=0.5, alpha=0.5)

    # Row 3 (optional): K-optimization curves
    if k_metrics:
        row3_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[3], wspace=0.35)
        k_vals = k_metrics["k_values"]
        curve_data = [
            ("Silhouette Score\n(Higher = Better)", k_metrics["silhouette_scores"], "#e94560"),
            ("Davies-Bouldin Index\n(Lower = Better)", k_metrics["davies_bouldin_scores"], "#00b4d8"),
            ("Inertia - Elbow Method", k_metrics["inertias"], "#90e0ef"),
        ]
        for ci, (ctitle, scores, color) in enumerate(curve_data):
            ax = fig.add_subplot(row3_gs[ci])
            ax.set_facecolor(panel_bg)
            ax.plot(k_vals, scores, "-o", color=color, linewidth=2, markersize=5)
            ax.axvline(x=k_metrics["optimal_k"], color="#f0c040",
                       linestyle="--", linewidth=1.5,
                       label=f"Optimal K={k_metrics['optimal_k']}")
            ax.set_title(ctitle, **title_kw)
            ax.set_xlabel("Number of Clusters K", **label_kw)
            ax.legend(fontsize=7, facecolor=panel_bg, labelcolor="white")
            ax.tick_params(**tick_kw)
            ax.grid(True, color="#333355", linewidth=0.5, alpha=0.5)

    fig.suptitle(
        "Shell Color Analysis - Multi-Method Dashboard",
        fontsize=15, fontweight="bold", color="white", y=0.98
    )

    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, f"analysis_dashboard_{timestamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"  Dashboard saved: {path}")
    return fig


def plot_color_palette_pie(combined_colors, output_folder, timestamp):
    """Color palette pie chart with confidence-coded bar chart."""
    if not combined_colors:
        return None

    dark_bg = "#1a1a2e"
    panel_bg = "#16213e"
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(dark_bg)

    sizes = [max(c.get("pct_of_pigment", 0), 0.1) for c in combined_colors]
    face_colors = [np.clip(c["center_rgb"] / 255.0, 0, 1) for c in combined_colors]
    labels = [
        f"{c['name']}\n{c['hex']}\n{c.get('pct_of_pigment', 0):.1f}%\n"
        f"Conf: {c.get('combined_confidence') or '?'}"
        for c in combined_colors
    ]

    # Pie chart
    axes[0].set_facecolor(panel_bg)
    axes[0].pie(sizes, labels=labels, colors=face_colors, startangle=90,
                textprops={"fontsize": 7, "color": "#dddddd"},
                wedgeprops=dict(linewidth=1, edgecolor=dark_bg))
    axes[0].set_title("Color Palette Pie Chart\n(with Confidence)",
                       color="white", fontsize=11, fontweight="bold")

    # Bar chart colored by confidence level
    axes[1].set_facecolor(panel_bg)
    x_pos = np.arange(len(combined_colors))
    bars = axes[1].bar(x_pos, sizes, color=face_colors, edgecolor=dark_bg, width=0.7)
    for bar, c in zip(bars, combined_colors):
        conf = c.get("combined_confidence") or 50.0
        edge_color = "#00ff00" if conf >= 75 else "#ffaa00" if conf >= 50 else "#ff4444"
        bar.set_edgecolor(edge_color)
        bar.set_linewidth(2.5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(
        [f"{c['name'][:12]}\n{c.get('combined_confidence') or '?'}%" for c in combined_colors],
        rotation=30, ha="right", fontsize=8, color="#dddddd",
    )
    axes[1].set_ylabel("% of Pigmented Area", color="#cccccc", fontsize=9)
    axes[1].set_title(
        "Color Distribution Bar Chart\n"
        "(Edge: green>=75%, orange>=50%, red<50% confidence)",
        color="white", fontsize=10, fontweight="bold",
    )
    axes[1].tick_params(colors="#aaaaaa", labelsize=7)
    axes[1].grid(axis="y", color="#333355", linewidth=0.5, alpha=0.5)
    legend_patches = [
        mpatches.Patch(color="#00ff00", label="High Confidence (>=75%)"),
        mpatches.Patch(color="#ffaa00", label="Medium Confidence (50-75%)"),
        mpatches.Patch(color="#ff4444", label="Low Confidence (<50%)"),
    ]
    axes[1].legend(handles=legend_patches, fontsize=8,
                   facecolor=panel_bg, labelcolor="white", loc="upper right")

    fig.suptitle("Shell Color Palette with Confidence Indicators",
                 fontsize=14, fontweight="bold", color="white", y=1.02)
    plt.tight_layout()

    path = os.path.join(output_folder, f"color_palette_{timestamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"  Color palette chart saved: {path}")
    return fig


# ============================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================
def _load_and_preprocess(folder_path, config):
    """Preprocess all images in folder_path. Returns (pixel_stack, gallery, stats)."""
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = sorted([
        f for f in glob.glob(os.path.join(folder_path, "*.*"))
        if os.path.splitext(f)[1].lower() in valid_exts
    ])
    if not image_files:
        logger.error(f"No valid images found in: {folder_path}")
        return None, [], {}

    logger.info(f"Found {len(image_files)} image(s) to process.")

    all_pigment_pixels = []
    gallery = []
    total_white, total_pigment, grand_total = 0, 0, 0

    for i, file in enumerate(image_files):
        fname = os.path.basename(file)
        logger.info(f"[{i+1}/{len(image_files)}] Processing: {fname}")
        data = preprocess_image(file, config)
        if data is None:
            continue
        gallery.append(data["result_rgba"])
        total_white += data["white_count"]
        total_pigment += data["pigment_count"]
        grand_total += data["shell_pixel_count"]
        if len(data["pigment_pixels"]) > 0:
            all_pigment_pixels.append(data["pigment_pixels"])

    if grand_total == 0:
        logger.error("No shell area detected in any image.")
        return None, [], {}

    canvas = config.get("STANDARD_CANVAS_SIZE", 1000)
    stats = {
        "canvas_size": canvas,
        "white_pct": total_white / grand_total * 100,
        "pigment_pct": total_pigment / grand_total * 100,
    }
    print(f"\n{'='*65}\n  ANALYSIS SUMMARY\n{'='*65}")
    print(f"Normalized canvas size      : {canvas}x{canvas} px")
    print(f"White/Reflective coverage   : {stats['white_pct']:.1f}% of shell")
    print(f"Pigmented coverage          : {stats['pigment_pct']:.1f}% of shell")
    print("=" * 65)

    if not all_pigment_pixels:
        logger.warning("No pigmentation found in any image.")
        return None, gallery, stats

    pixel_stack = np.vstack(all_pigment_pixels)
    logger.info(f"Total pigment pixels: {len(pixel_stack):,}")
    return pixel_stack, gallery, stats


def run_analysis_mode(folder_path, config, model=None, output_folder="./output", no_show=False):
    """
    Run full analysis on images in folder_path using both methods.

    Parameters
    ----------
    folder_path : str   Folder with shell images.
    config : dict       Configuration dictionary.
    model : TrainedShellModel or None   Trained model (if available).
    output_folder : str Folder for output files.
    no_show : bool      Suppress interactive plot windows.
    """
    pixel_stack, gallery, _ = _load_and_preprocess(folder_path, config)
    if pixel_stack is None:
        return {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_folder, exist_ok=True)

    analyzer = SelfLearningAnalyzer(model=model, config=config)

    logger.info("\n--- Running Self-Learning Analysis ---")

    # Method 1 (Trained)
    trained_results = []
    if model is not None:
        logger.info("\n  [Method 1] Trained centroid matching...")
        trained_results = analyzer.analyze_trained_method(pixel_stack)

    # Method 2 (Fresh)
    logger.info("\n  [Method 2] Fresh clustering with trained parameters...")
    fresh_out = analyzer.analyze_fresh_method(pixel_stack)
    fresh_results, k_metrics, linkage_matrix, h_cut_height, h_sample = fresh_out

    total_pix = len(pixel_stack)

    # Build display-ready color lists (add pct_of_pigment)
    def _enrich(result_list, conf_key):
        enriched = []
        for rank, r in enumerate(
            sorted(result_list, key=lambda x: x["count"], reverse=True), 1
        ):
            r2 = dict(r)
            r2["pct_of_pigment"] = round(r["count"] / max(total_pix, 1) * 100, 2)
            r2["role"] = "BASE" if rank == 1 else "SECONDARY"
            enriched.append(r2)
        return enriched

    trained_colors_pie = _enrich(trained_results, "confidence_trained")
    fresh_colors_pie = _enrich(fresh_results, "confidence_fresh")

    # Combined results
    combined = analyzer.combine_results(trained_results, fresh_results, total_pix)

    # Console output
    print_color_table(combined)

    # Export CSV + JSON
    if config.get("SAVE_FIGURES", True):
        export_results(combined, output_folder, timestamp)

    # Visualize
    logger.info("\n--- Building Visualization Dashboard ---")
    figs = []

    if config.get("SAVE_FIGURES", True):
        fig_dash = plot_analysis_dashboard(
            combined_colors=combined,
            trained_colors_pie=trained_colors_pie,
            fresh_colors_pie=fresh_colors_pie,
            processed_images=gallery,
            model=model,
            k_metrics=k_metrics,
            output_folder=output_folder,
            timestamp=timestamp,
        )
        figs.append(fig_dash)

        fig_pie = plot_color_palette_pie(combined, output_folder, timestamp)
        figs.append(fig_pie)

    if no_show:
        for f in figs:
            if f is not None:
                plt.close(f)
    else:
        plt.show()

    logger.info("Analysis complete.")
    return {"combined": combined, "trained": trained_results, "fresh": fresh_results}


# ============================================================
# ENTRY POINT
# ============================================================
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Self-Learning Shell Color Analysis System.\n"
            "Supports two modes:\n"
            "  train   - learn color knowledge from sample images\n"
            "  analyze - analyze dataset images using learned knowledge\n\n"
            "Examples:\n"
            "  python shell_color_analysis.py --mode train "
            "--training-folder ./training_samples\n"
            "  python shell_color_analysis.py --mode train "
            "--training-folder ./training_samples --retrain\n"
            "  python shell_color_analysis.py --mode analyze "
            "--input-folder ./dataset --use-trained-model\n"
            "  python shell_color_analysis.py --mode analyze "
            "--input-folder ./dataset\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=str, default="analyze",
        choices=["train", "analyze"],
        help="Operation mode: 'train' or 'analyze'. (default: analyze)",
    )
    parser.add_argument(
        "--training-folder", type=str, default="./training_samples",
        help="Folder of sample training images (used in train mode). "
             "(default: ./training_samples)",
    )
    parser.add_argument(
        "--input-folder", "--folder", type=str, default="./dataset",
        dest="input_folder",
        help="Folder of dataset images to analyze (used in analyze mode). "
             "(default: ./dataset)",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_CONFIG["OUTPUT_FOLDER"],
        help="Output folder for results (CSV, JSON, PNG). (default: %(default)s)",
    )
    parser.add_argument(
        "--model-path", type=str, default=DEFAULT_CONFIG["MODEL_PATH"],
        help="Path to trained model file. (default: %(default)s)",
    )
    parser.add_argument(
        "--use-trained-model", action="store_true",
        help="Load and use the trained model for analysis. "
             "Without this flag only Method 2 (Fresh) runs.",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="In train mode: extend existing model with new samples "
             "(incremental learning).",
    )
    parser.add_argument(
        "--k-min", type=int, default=DEFAULT_CONFIG["NUM_CLUSTERS_MIN"],
        help="Minimum K for K-Means search range. (default: %(default)s)",
    )
    parser.add_argument(
        "--k-max", type=int, default=DEFAULT_CONFIG["NUM_CLUSTERS_MAX"],
        help="Maximum K for K-Means search range. (default: %(default)s)",
    )
    parser.add_argument(
        "--merge-threshold", type=float, default=None,
        help="Fixed CIELAB merge threshold (adaptive if omitted).",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not open interactive matplotlib windows (batch/server use).",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write output files to disk.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    cfg = DEFAULT_CONFIG.copy()
    cfg["OUTPUT_FOLDER"] = args.output
    cfg["MODEL_PATH"] = args.model_path
    cfg["NUM_CLUSTERS_MIN"] = args.k_min
    cfg["NUM_CLUSTERS_MAX"] = args.k_max
    cfg["SAVE_FIGURES"] = not args.no_save
    if args.merge_threshold is not None:
        cfg["COLOR_MERGE_THRESHOLD"] = args.merge_threshold

    if args.mode == "train":
        logger.info(f"\n{'='*65}")
        logger.info("  MODE: TRAINING")
        logger.info(f"{'='*65}")
        model = train_shell_model(
            folder_path=args.training_folder,
            config=cfg,
            model_path=args.model_path,
            retrain=args.retrain,
        )
        if model and model.centroids_rgb is not None and cfg.get("SAVE_FIGURES"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_training_infographic(model, model.per_sample_colors, cfg["OUTPUT_FOLDER"], ts)
            if not args.no_show:
                plt.show()
            else:
                plt.close("all")

    elif args.mode == "analyze":
        logger.info(f"\n{'='*65}")
        logger.info("  MODE: ANALYSIS")
        logger.info(f"{'='*65}")

        model = None
        if args.use_trained_model:
            if os.path.exists(args.model_path):
                try:
                    model = TrainedShellModel.load(args.model_path)
                    n_centroids = len(model.centroids_rgb) if model.centroids_rgb is not None else 0
                    logger.info(
                        f"  Loaded model with {n_centroids} centroids "
                        f"from '{args.model_path}'"
                    )
                except Exception as exc:
                    logger.error(f"  Failed to load model: {exc}")
            else:
                logger.warning(
                    f"  --use-trained-model specified but no model found at "
                    f"'{args.model_path}'. Running fresh analysis only."
                )

        run_analysis_mode(
            folder_path=args.input_folder,
            config=cfg,
            model=model,
            output_folder=cfg["OUTPUT_FOLDER"],
            no_show=args.no_show,
        )
