# Shell Color Analysis – Command Reference

Quick reference for all command-line arguments and common workflows.

---

## Synopsis

```
python shell_color_analysis.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--folder FOLDER` | path | `./images` | Input folder containing shell images |
| `--output OUTPUT` | path | `./output` | Output folder for results |
| `--method METHOD` | choice | `all` | Clustering method: `kmeans` \| `hierarchical` \| `dbscan` \| `all` |
| `--k-min K_MIN` | int | `5` | Minimum K for K-Means cluster search |
| `--k-max K_MAX` | int | `30` | Maximum K for K-Means cluster search |
| `--merge-threshold T` | float | adaptive | Fixed CIELAB merge threshold (omit for adaptive) |
| `--no-show` | flag | — | Do not open interactive plot windows |
| `--no-save` | flag | — | Do not write output files to disk |
| `--train` | flag | — | Train the supervised Random Forest classifier |

---

## Clustering Methods

### K-Means
Finds the optimal number of clusters automatically using silhouette score, elbow
method, and Davies-Bouldin index.

```bash
python shell_color_analysis.py --folder images --method kmeans
python shell_color_analysis.py --folder images --method kmeans --k-min 3 --k-max 15
python shell_color_analysis.py --folder images --method kmeans --no-show
```

### Hierarchical
Agglomerative clustering with an adaptive distance threshold derived from a
configurable percentile of the linkage heights.  Also produces a dendrogram.

```bash
python shell_color_analysis.py --folder images --method hierarchical
python shell_color_analysis.py --folder images --method hierarchical --no-show
```

### DBSCAN
Density-based clustering that discovers dominant colors and rare / subtle pigments
without requiring a pre-specified cluster count.

```bash
python shell_color_analysis.py --folder images --method dbscan
python shell_color_analysis.py --folder images --method dbscan --no-show
```

### All Methods (recommended)
Runs all three methods and writes a combined JSON comparison report.

```bash
python shell_color_analysis.py --folder images --method all
python shell_color_analysis.py --folder images --method all --no-show
```

---

## Training Workflows

### Train Without Pre-Labeled Images (recommended)

No manual annotation is required.  Clustering automatically assigns color labels
which are then used to train the Random Forest classifier.

```bash
# Linux / macOS
python shell_color_analysis.py --folder training_data --train

# Windows
python shell_color_analysis.py --folder "C:\Users\YourName\Project\training_data" --train
```

Aim for **5–10 diverse images** that cover the range of colors, patterns, and
lighting conditions present in your full dataset.

### Train + Suppress Plot Windows

```bash
python shell_color_analysis.py --folder training_data --train --no-show
```

### Train With a Specific Clustering Method

```bash
python shell_color_analysis.py --folder training_data --train --method kmeans
python shell_color_analysis.py --folder training_data --train --method hierarchical
python shell_color_analysis.py --folder training_data --train --method dbscan
```

### Train With Custom K Range

```bash
python shell_color_analysis.py --folder training_data --train --k-min 3 --k-max 20
```

### Train With Labeled Data (Python API)

If you have hand-labeled pixel arrays you can call the training function directly:

```python
from shell_color_analysis import train_color_classifier
import numpy as np

labeled_data = [
    (purple_pixels, "purple"),   # purple_pixels: np.ndarray of shape (N, 3) in RGB
    (brown_pixels,  "brown"),
    (white_pixels,  "white"),
]
clf = train_color_classifier(labeled_data, save_path="color_classifier.pkl")
```

### Apply the Trained Classifier

After training, run analysis normally — the saved model is loaded automatically:

```bash
python shell_color_analysis.py --folder trial --no-show
```

---

## Output Options

```bash
# Save results without opening plot windows (server / batch use)
python shell_color_analysis.py --folder images --no-show

# Run analysis but do not write any files to disk
python shell_color_analysis.py --folder images --no-save

# Write results to a custom folder
python shell_color_analysis.py --folder images --output /path/to/results

# Combine: save to custom folder, no pop-up windows
python shell_color_analysis.py --folder images --output results --no-show
```

---

## Parameter Tuning

### Adjust K-Search Range

```bash
# Broader search (slower, finds more colors)
python shell_color_analysis.py --folder images --k-min 5 --k-max 40

# Narrow search (faster, good for simple shells)
python shell_color_analysis.py --folder images --k-min 3 --k-max 10
```

### Color Merge Threshold

```bash
# Adaptive threshold – scales per image (default)
python shell_color_analysis.py --folder images

# Fixed threshold – same CIELAB distance for every image
python shell_color_analysis.py --folder images --merge-threshold 10.0   # light merging – keeps more colors separate
python shell_color_analysis.py --folder images --merge-threshold 15.0   # moderate
python shell_color_analysis.py --folder images --merge-threshold 25.0   # aggressive merging – fewer, broader groups
```

A lower value merges similar colors more aggressively; a higher value preserves
more distinct color clusters.

> See [PARAMETER_GUIDE.md](PARAMETER_GUIDE.md) for full details on all parameters
> including `COLOR_MERGE_PERCENTILE`, `HIERARCHICAL_DISTANCE_PERCENTILE`, and
> `DBSCAN_EPS_PERCENTILE`.

---

## Advanced Usage

### Full Pipeline: Train Then Analyze

```bash
# Step 1 – Train on representative images
python shell_color_analysis.py --folder training_data --train --no-show

# Step 2 – Analyze all new images with every method
python shell_color_analysis.py --folder trial --method all --no-show

# Step 3 – Analyze with a fine-grained K search
python shell_color_analysis.py --folder trial --method all --k-min 3 --k-max 20 --no-show
```

### Combine Multiple Options

```bash
# K-Means, custom K range, fixed merge threshold, save only
python shell_color_analysis.py \
  --folder images \
  --method kmeans \
  --k-min 4 \
  --k-max 20 \
  --merge-threshold 12.0 \
  --output results \
  --no-show

# Windows equivalent (use ^ for line continuation)
python shell_color_analysis.py ^
  --folder "C:\Project\images" ^
  --method kmeans ^
  --k-min 4 ^
  --k-max 20 ^
  --merge-threshold 12.0 ^
  --output "C:\Project\results" ^
  --no-show
```

---

## Windows Quick Reference

```bat
REM Navigate to project folder
cd "C:\Users\YourName\PythonProject"

REM Analyze with all methods
python shell_color_analysis.py --folder trial --method all --no-show

REM Analyze with K-Means
python shell_color_analysis.py --folder trial --method kmeans --no-show

REM Analyze with Hierarchical
python shell_color_analysis.py --folder trial --method hierarchical --no-show

REM Analyze with DBSCAN
python shell_color_analysis.py --folder trial --method dbscan --no-show

REM Train the classifier
python shell_color_analysis.py --folder training_data --train

REM Apply the trained classifier to new images
python shell_color_analysis.py --folder trial --no-show

REM Adjust K search range
python shell_color_analysis.py --folder trial --k-min 3 --k-max 15 --no-show

REM Use a fixed merge threshold
python shell_color_analysis.py --folder trial --merge-threshold 15.0 --no-show
```

---

## Output Files

| File | Description |
|------|-------------|
| `dashboard_<timestamp>.png` | Multi-panel visualization dashboard |
| `optimization_curves_<timestamp>.png` | K-selection optimization curves (K-Means) |
| `dendrogram_<timestamp>.png` | Hierarchical clustering dendrogram |
| `results_kmeans_<timestamp>.csv` | K-Means color table |
| `results_hierarchical_<timestamp>.csv` | Hierarchical color table |
| `results_dbscan_<timestamp>.csv` | DBSCAN color table |
| `results_all_<timestamp>.json` | All methods combined (JSON) |
| `color_classifier.pkl` | Saved Random Forest classifier (after `--train`) |
