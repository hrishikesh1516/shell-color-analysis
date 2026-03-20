# Shell Color Analysis – Parameter Guide

Detailed documentation for every configuration parameter in `DEFAULT_CONFIG`,
including recommended ranges, tuning tips, and performance guidance.

---

## Overview

All parameters live in the `DEFAULT_CONFIG` dictionary near the top of
`shell_color_analysis.py`.  Command-line flags override the most commonly tuned
ones; the rest must be edited directly in the source file.

```python
DEFAULT_CONFIG = {
    "INPUT_FOLDER":                    "./images",
    "OUTPUT_FOLDER":                   "./output",
    "PIXELS_PER_UNIT":                 176.0454,
    "UNIT_NAME":                       "cm",
    "CLUSTERING_METHOD":               "all",
    "NUM_CLUSTERS_MIN":                5,
    "NUM_CLUSTERS_MAX":                30,
    "COLOR_MERGE_THRESHOLD":           None,      # None = adaptive
    "COLOR_MERGE_PERCENTILE":          30,
    "HIERARCHICAL_DISTANCE_PERCENTILE": 85,
    "DBSCAN_EPS_PERCENTILE":           10,
    "DBSCAN_MIN_SAMPLES_FRACTION":     0.002,
    "GLARE_THRESHOLD":                 245,
    "MIN_COLOR_BRIGHTNESS":            40,
    "WHITE_SENSITIVITY":               50,
    "WHITE_BRIGHTNESS":                150,
    "CLASSIFIER_PATH":                 "color_classifier.pkl",
    "SAVE_FIGURES":                    True,
}
```

---

## Input / Output Parameters

### `INPUT_FOLDER`
- **Default:** `"./images"`
- **CLI flag:** `--folder`
- **Description:** Path to the directory containing shell images.  All `.jpg`,
  `.jpeg`, and `.png` files found in this folder are processed.
- **Tips:** Use absolute paths to avoid ambiguity, especially on Windows.

### `OUTPUT_FOLDER`
- **Default:** `"./output"`
- **CLI flag:** `--output`
- **Description:** Directory where all result files (CSV, JSON, PNG) are written.
  Created automatically if it does not exist.

### `PIXELS_PER_UNIT`
- **Default:** `176.0454`
- **Description:** Camera / microscope calibration value in pixels per unit.
  Used to convert pixel areas into physical measurements.
- **How to calibrate:** Photograph a ruler or calibration target at the same
  zoom level as your shell images, then measure the pixel length of a known
  distance (e.g. 1 cm) and enter that value here.
- **Recommended range:** Depends entirely on your setup; typical values range
  from 50 (low magnification) to 500 (high magnification).

### `UNIT_NAME`
- **Default:** `"cm"`
- **Description:** Label appended to area values in reports (e.g. `"cm"` →
  `3.52 cm²`).  Change to `"mm"` or `"px"` as appropriate.

### `SAVE_FIGURES`
- **Default:** `True`
- **CLI flag:** `--no-save` (sets to `False`)
- **Description:** When `True` all visualization figures are written to
  `OUTPUT_FOLDER`.  Set to `False` for quick exploratory runs.

---

## Clustering Method

### `CLUSTERING_METHOD`
- **Default:** `"all"`
- **CLI flag:** `--method`
- **Choices:** `"kmeans"` | `"hierarchical"` | `"dbscan"` | `"all"`
- **Description:** Which clustering algorithm(s) to run.

| Value | Best For |
|-------|----------|
| `"kmeans"` | Fast analysis; known approximate number of colors |
| `"hierarchical"` | Understanding color relationships; producing dendrograms |
| `"dbscan"` | Detecting rare / subtle pigment spots alongside dominant colors |
| `"all"` | Comprehensive comparison and combined JSON report (recommended) |

---

## K-Means Parameters

### `NUM_CLUSTERS_MIN` / `K_MIN`
- **Default:** `5`
- **CLI flag:** `--k-min`
- **Description:** Lower bound of the K search range.
- **When to increase:** You know the shell has at least N distinct color zones.
- **When to decrease:** Images are very simple (e.g. monochrome shells) and a
  smaller search range speeds up analysis.
- **Recommended range:** 2–10

### `NUM_CLUSTERS_MAX` / `K_MAX`
- **Default:** `30`
- **CLI flag:** `--k-max`
- **Description:** Upper bound of the K search range.
- **When to increase:** Shells have very complex, multi-color patterns and you
  want the optimizer to explore a wider space.
- **When to decrease:** Analysis is too slow, or images are relatively simple.
  Reducing from 30 to 15 roughly halves computation time.
- **Recommended range:** 10–50

**Relationship between `K_MIN` and `K_MAX`:**
The optimizer evaluates every integer K from `K_MIN` to `K_MAX` and picks the
best score.  A wider range yields a more thorough search at the cost of runtime.

---

## Color Merge Parameters

### `COLOR_MERGE_THRESHOLD`
- **Default:** `None` (adaptive)
- **CLI flag:** `--merge-threshold`
- **Description:** After clustering, nearby color centers (in CIELAB space) are
  merged if their distance is below this threshold.

#### Adaptive Mode (recommended)

When set to `None`, the threshold is computed automatically:

```
threshold = percentile(pairwise_distances(cluster_centers),
                       COLOR_MERGE_PERCENTILE)
```

This scales with the actual color complexity of each image — images with
subtly different hues get a small threshold; images with very different colors
get a larger one.

#### Fixed Mode

Set a specific float value when you need reproducible, consistent thresholds
across a batch regardless of per-image color distribution.

```bash
python shell_color_analysis.py --folder images --merge-threshold 15.0
```

| Fixed Value | Effect |
|-------------|--------|
| `5.0` | Very conservative — keeps nearly all clusters separate |
| `10.0` | Light merging — removes only near-duplicate colors |
| `15.0` | Moderate merging — good general-purpose fixed value |
| `20.0` | Aggressive merging — fewer, broader color groups |
| `30.0+` | Very aggressive — may collapse distinct colors together |

### `COLOR_MERGE_PERCENTILE`
- **Default:** `30`
- **Description:** Percentile of the pairwise CIELAB distance distribution
  used to compute the adaptive merge threshold.  Only active when
  `COLOR_MERGE_THRESHOLD` is `None`.
- **When to increase:** Too many near-duplicate colors survive after merging
  (raise the percentile → larger adaptive threshold → more merging).
- **When to decrease:** Distinct colors are being merged together
  (lower the percentile → smaller threshold → less merging).
- **Recommended range:** 20–50

---

## Hierarchical Clustering Parameters

### `HIERARCHICAL_DISTANCE_PERCENTILE`
- **Default:** `85`
- **Description:** The linkage tree is cut at the height corresponding to this
  percentile of all linkage distances, determining how many clusters are formed.
- **When to increase:** Want fewer, broader clusters (e.g. 90–95 → very coarse
  grouping).
- **When to decrease:** Want finer color distinctions with more clusters (e.g.
  70–80).
- **Recommended range:** 70–95

**Intuition:** A high percentile cuts near the top of the dendrogram, producing
few broad clusters.  A low percentile cuts lower, producing many fine clusters.

---

## DBSCAN Parameters

### `DBSCAN_EPS_PERCENTILE`
- **Default:** `10`
- **Description:** The neighborhood radius `eps` for DBSCAN is set to the value
  at this percentile of all pairwise CIELAB distances between sampled pixels.
- **When to increase:** DBSCAN finds too few clusters or marks most pixels as
  noise (raise the percentile → larger `eps` → more inclusive neighborhoods).
- **When to decrease:** DBSCAN merges distinct colors into one cluster (lower
  the percentile → smaller `eps` → stricter neighborhood definition).
- **Recommended range:** 5–20

### `DBSCAN_MIN_SAMPLES_FRACTION`
- **Default:** `0.002` (0.2 % of total pixels)
- **Description:** Fraction of the total pixel count used to set DBSCAN's
  `min_samples` parameter (minimum points required to form a core point).
- **When to increase:** Too many tiny, noisy clusters appear (raise the
  fraction → more points required → fewer, denser clusters).
- **When to decrease:** Rare pigment spots (< 0.2 % of the shell) are being
  discarded as noise (lower the fraction → more sensitive to small clusters).
- **Recommended range:** 0.001–0.010

---

## Glare and Shadow Filtering

### `GLARE_THRESHOLD`
- **Default:** `245`
- **Description:** V-channel (HSV) threshold above which pixels are considered
  specular glare and removed / inpainted before clustering.
- **When to increase:** Bright but valid shell colors are wrongly removed
  (raise toward 255 → only the very brightest highlights are masked).
- **When to decrease:** Glare is still visible in results (lower → more
  aggressive glare removal).
- **Recommended range:** 230–255

### `MIN_COLOR_BRIGHTNESS`
- **Default:** `40`
- **Description:** Minimum HSV V-channel value for a pixel to be included as a
  pigment pixel.  Darker pixels are treated as shadow and excluded.
- **When to increase:** Shadow regions are contaminating color results.
- **When to decrease:** Valid dark-colored shells (e.g. very dark brown / black)
  are being excluded.
- **Recommended range:** 20–80

---

## White Detection Parameters

### `WHITE_SENSITIVITY`
- **Default:** `50`
- **Description:** HSV S-channel threshold below which a pixel is classified as
  "white" (low saturation).
- **When to increase:** Off-white or cream pixels are not being classified as
  white.
- **When to decrease:** Slightly saturated pixels (light pink, pale blue) are
  being wrongly classified as white.
- **Recommended range:** 30–80

### `WHITE_BRIGHTNESS`
- **Default:** `150`
- **Description:** HSV V-channel threshold above which a low-saturation pixel is
  classified as "white" (bright enough to be visually white rather than grey).
- **When to increase:** Mid-grey pixels are being classified as white.
- **When to decrease:** True white pixels are not reaching this threshold (e.g.
  underexposed images).
- **Recommended range:** 120–200

---

## Classifier Parameters

### `CLASSIFIER_PATH`
- **Default:** `"color_classifier.pkl"`
- **Description:** File path for saving and loading the trained Random Forest
  classifier.  Change this if you want to maintain multiple classifiers for
  different shell species or datasets.

---

## Performance Tuning Summary

| Goal | Action |
|------|--------|
| Faster analysis | Reduce `K_MAX` (e.g. 30 → 15) |
| Find more subtle colors | Increase `K_MAX`; lower `COLOR_MERGE_PERCENTILE` |
| Fewer near-duplicate colors | Increase `COLOR_MERGE_PERCENTILE` or use a larger fixed `COLOR_MERGE_THRESHOLD` |
| More hierarchical clusters | Decrease `HIERARCHICAL_DISTANCE_PERCENTILE` |
| Fewer hierarchical clusters | Increase `HIERARCHICAL_DISTANCE_PERCENTILE` |
| DBSCAN finds rare pigments | Decrease `DBSCAN_EPS_PERCENTILE`; decrease `DBSCAN_MIN_SAMPLES_FRACTION` |
| DBSCAN less noisy | Increase `DBSCAN_EPS_PERCENTILE`; increase `DBSCAN_MIN_SAMPLES_FRACTION` |
| Remove more glare | Decrease `GLARE_THRESHOLD` |
| Include dark pixels | Decrease `MIN_COLOR_BRIGHTNESS` |
| Accurate area measurements | Calibrate `PIXELS_PER_UNIT` to your camera setup |

---

## Default Values at a Glance

| Parameter | Default | CLI Flag |
|-----------|---------|----------|
| `INPUT_FOLDER` | `./images` | `--folder` |
| `OUTPUT_FOLDER` | `./output` | `--output` |
| `CLUSTERING_METHOD` | `all` | `--method` |
| `NUM_CLUSTERS_MIN` | `5` | `--k-min` |
| `NUM_CLUSTERS_MAX` | `30` | `--k-max` |
| `COLOR_MERGE_THRESHOLD` | `None` (adaptive) | `--merge-threshold` |
| `COLOR_MERGE_PERCENTILE` | `30` | — |
| `HIERARCHICAL_DISTANCE_PERCENTILE` | `85` | — |
| `DBSCAN_EPS_PERCENTILE` | `10` | — |
| `DBSCAN_MIN_SAMPLES_FRACTION` | `0.002` | — |
| `GLARE_THRESHOLD` | `245` | — |
| `MIN_COLOR_BRIGHTNESS` | `40` | — |
| `WHITE_SENSITIVITY` | `50` | — |
| `WHITE_BRIGHTNESS` | `150` | — |
| `PIXELS_PER_UNIT` | `176.0454` | — |
| `UNIT_NAME` | `cm` | — |
| `CLASSIFIER_PATH` | `color_classifier.pkl` | — |
| `SAVE_FIGURES` | `True` | `--no-save` |
