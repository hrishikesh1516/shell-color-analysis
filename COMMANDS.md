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
| `--method METHOD` | choice | `all` | Clustering method: `kmeans` \| `hierarchical` \| `all` |
| `--k-min K_MIN` | int | `3` | Minimum K for K-Means cluster search |
| `--k-max K_MAX` | int | `15` | Maximum K for K-Means cluster search |
| `--merge-threshold T` | float | adaptive | Fixed CIELAB merge threshold (omit for adaptive) |
| `--no-show` | flag | — | Do not open interactive plot windows |
| `--no-save` | flag | — | Do not write output files to disk |
| `--train` | flag | — | Train clustering parameters from sample images |
| `--use-trained-params` | flag | — | Load trained parameters for this analysis run |
| `--compare-trained` | flag | — | Run default + trained parameters side-by-side |

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

### Both Methods (recommended)
Runs both methods and writes a combined JSON comparison report.

```bash
python shell_color_analysis.py --folder images --method all
python shell_color_analysis.py --folder images --method all --no-show
```

---

## Scale-Independent Analysis

All images are automatically normalized to a **1000×1000 px canvas** regardless of
their original resolution.  No calibration is needed — results are color percentages
that are directly comparable across your entire dataset.

```bash
# Works for any image size (500 px, 3000 px, etc.)
python shell_color_analysis.py --folder images --no-show
```

---

## Training Workflows

### Train Clustering Parameters (no labels needed)

Analyses sample images to optimize K range, merge percentile, and hierarchical
distance percentile.  The best parameters are saved to `trained_params.pkl`.

```bash
# Linux / macOS
python shell_color_analysis.py --folder training_data --train

# Windows
python shell_color_analysis.py --folder "C:\Users\YourName\Project\training_data" --train
```

Aim for **5–10 diverse images** that cover the range of colors and lighting
conditions present in your full dataset.

### Train + Suppress Plot Windows

```bash
python shell_color_analysis.py --folder training_data --train --no-show
```

### Train With a Specific Method

```bash
python shell_color_analysis.py --folder training_data --train --method kmeans
python shell_color_analysis.py --folder training_data --train --method hierarchical
```

### Analyse With Trained Parameters

```bash
python shell_color_analysis.py --folder trial --use-trained-params --no-show
```

### Compare Trained vs. Default Results

```bash
python shell_color_analysis.py --folder trial --compare-trained --no-show
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
python shell_color_analysis.py --folder images --k-min 5 --k-max 25

# Narrow search (faster, good for simple shells)
python shell_color_analysis.py --folder images --k-min 3 --k-max 10
```

### Color Merge Threshold

```bash
# Adaptive threshold – scales per image (default)
python shell_color_analysis.py --folder images

# Fixed threshold – same CIELAB distance for every image
python shell_color_analysis.py --folder images --merge-threshold 10.0   # light merging
python shell_color_analysis.py --folder images --merge-threshold 15.0   # moderate
python shell_color_analysis.py --folder images --merge-threshold 25.0   # aggressive
```

A lower value keeps more distinct color clusters; a higher value merges them more
aggressively.

> See [PARAMETER_GUIDE.md](PARAMETER_GUIDE.md) for full details on all parameters
> including `COLOR_MERGE_PERCENTILE` and `HIERARCHICAL_DISTANCE_PERCENTILE`.

---

## Advanced Usage

### Full Pipeline: Train Then Analyse

```bash
# Step 1 – Train on representative images
python shell_color_analysis.py --folder training_data --train --no-show

# Step 2 – Analyse new images with trained parameters
python shell_color_analysis.py --folder trial --use-trained-params --no-show

# Step 3 – Compare trained vs. default (optional)
python shell_color_analysis.py --folder trial --compare-trained --no-show
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

REM Analyse with both methods
python shell_color_analysis.py --folder trial --method all --no-show

REM Analyse with K-Means
python shell_color_analysis.py --folder trial --method kmeans --no-show

REM Analyse with Hierarchical
python shell_color_analysis.py --folder trial --method hierarchical --no-show

REM Train clustering parameters
python shell_color_analysis.py --folder training_data --train

REM Analyse with trained parameters
python shell_color_analysis.py --folder trial --use-trained-params --no-show

REM Compare trained vs. default
python shell_color_analysis.py --folder trial --compare-trained --no-show

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
| `results_kmeans_<timestamp>.csv` | K-Means color table (percentages) |
| `results_hierarchical_<timestamp>.csv` | Hierarchical color table (percentages) |
| `results_all_<timestamp>.json` | Both methods combined (JSON) |
| `trained_params.pkl` | Optimized parameters (after `--train`) |
