# Shell Color Analysis – Command Reference

Quick reference for all command-line arguments and common workflows.

---

## Synopsis

```
python shell_color_analysis.py --mode {train|analyze} [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mode MODE` | choice | `analyze` | Operation mode: `train` or `analyze` |
| `--training-folder FOLDER` | path | `./training_samples` | Sample images folder (train mode) |
| `--input-folder FOLDER` | path | `./dataset` | Dataset images folder (analyze mode) |
| `--output OUTPUT` | path | `./output` | Output folder for results |
| `--model-path PATH` | path | `trained_shell_model.pkl` | Path to trained model file |
| `--use-trained-model` | flag | — | Load trained model for Method 1 analysis |
| `--retrain` | flag | — | Extend existing model incrementally (train mode) |
| `--k-min K_MIN` | int | `3` | Minimum K for K-Means cluster search |
| `--k-max K_MAX` | int | `15` | Maximum K for K-Means cluster search |
| `--merge-threshold T` | float | adaptive | Fixed CIELAB merge threshold (omit for adaptive) |
| `--no-show` | flag | — | Do not open interactive plot windows |
| `--no-save` | flag | — | Do not write output files to disk |

---

## Training Workflows

### Initial Training (no labels needed)

Analyses ~20 sample images to learn: optimal K range, color centroids (RGB and CIELAB),
merge thresholds, and hierarchical distance percentile.  Saves `trained_shell_model.pkl`.

```bash
# Linux / macOS
python shell_color_analysis.py --mode train --training-folder ./training_samples

# Windows
python shell_color_analysis.py --mode train --training-folder "C:\Project\training_samples"

# Suppress interactive plot windows (server / batch use)
python shell_color_analysis.py --mode train --training-folder ./training_samples --no-show
```

Aim for **~20 diverse images** covering the full range of shell colors and lighting
conditions in your dataset.

### Incremental Retraining

Add new sample images without losing previously learned knowledge:

```bash
python shell_color_analysis.py --mode train --training-folder ./new_samples --retrain
```

### Custom Model Path

```bash
python shell_color_analysis.py --mode train \
  --training-folder ./training_samples \
  --model-path ./models/my_shell_model.pkl
```

---

## Analysis Workflows

### Two-Method Analysis (with trained model)

Runs **Method 1 (Trained)** + **Method 2 (Fresh)** and combines the results:

```bash
python shell_color_analysis.py --mode analyze \
  --input-folder ./dataset \
  --use-trained-model

# With custom model path
python shell_color_analysis.py --mode analyze \
  --input-folder ./dataset \
  --use-trained-model \
  --model-path ./models/my_shell_model.pkl \
  --no-show
```

### Fresh Analysis Only (no trained model)

Runs only **Method 2 (Fresh)** using learned optimal parameters if a model exists:

```bash
python shell_color_analysis.py --mode analyze --input-folder ./dataset

# Windows
python shell_color_analysis.py --mode analyze --input-folder "C:\Project\dataset"
```

---

## Scale-Independent Analysis

All images are automatically normalized to a **1000×1000 px canvas** regardless of
their original resolution.  Results are color percentages directly comparable across
your entire dataset.

---

## Output Options

```bash
# Save results without opening plot windows (server / batch use)
python shell_color_analysis.py --mode analyze --input-folder images --no-show

# Run analysis but do not write any files to disk
python shell_color_analysis.py --mode analyze --input-folder images --no-save

# Write results to a custom folder
python shell_color_analysis.py --mode analyze --input-folder images --output /path/to/results
```

---

## Parameter Tuning

### Adjust K-Search Range

```bash
# Broader search (slower, finds more colors)
python shell_color_analysis.py --mode analyze --input-folder images --k-min 5 --k-max 25

# Narrow search (faster, good for simple shells)
python shell_color_analysis.py --mode analyze --input-folder images --k-min 3 --k-max 10
```

### Color Merge Threshold

```bash
# Adaptive threshold – scales per image (default)
python shell_color_analysis.py --mode analyze --input-folder images

# Fixed threshold – same CIELAB distance for every image
python shell_color_analysis.py --mode analyze --input-folder images --merge-threshold 10.0
python shell_color_analysis.py --mode analyze --input-folder images --merge-threshold 15.0
python shell_color_analysis.py --mode analyze --input-folder images --merge-threshold 25.0
```

---

## Full Pipeline (Recommended Workflow)

```bash
# Step 1 – Train on ~20 representative sample images (first time)
python shell_color_analysis.py --mode train \
  --training-folder ./training_samples \
  --no-show

# Step 2 – Analyze dataset with trained model
python shell_color_analysis.py --mode analyze \
  --input-folder ./dataset \
  --use-trained-model \
  --no-show

# Step 3 – Retrain with additional samples (future)
python shell_color_analysis.py --mode train \
  --training-folder ./new_training_samples \
  --retrain \
  --no-show
```

---

## Windows Quick Reference

```bat
REM Navigate to project folder
cd "C:\Users\YourName\PythonProject"

REM Train on sample images
python shell_color_analysis.py --mode train --training-folder training_samples --no-show

REM Retrain with new samples
python shell_color_analysis.py --mode train --training-folder new_samples --retrain --no-show

REM Analyze with trained model
python shell_color_analysis.py --mode analyze --input-folder dataset --use-trained-model --no-show

REM Analyze without trained model (fresh only)
python shell_color_analysis.py --mode analyze --input-folder dataset --no-show

REM Adjust K search range
python shell_color_analysis.py --mode analyze --input-folder dataset --k-min 3 --k-max 15 --no-show

REM Use a fixed merge threshold
python shell_color_analysis.py --mode analyze --input-folder dataset --merge-threshold 15.0 --no-show
```

---

## Output Files

| File | Description |
|------|-------------|
| `trained_shell_model.pkl` | Self-learning trained model (after `--mode train`) |
| `training_summary_<timestamp>.png` | Training infographic with centroids, K values, validation |
| `analysis_dashboard_<timestamp>.png` | Multi-method dashboard (Trained vs Fresh vs Combined) |
| `color_palette_<timestamp>.png` | Color palette pie chart with confidence indicators |
| `results_combined_<timestamp>.csv` | Combined results with all confidence and error metrics |
| `results_combined_<timestamp>.json` | Combined results in JSON format |

### CSV / JSON Columns

| Column | Description |
|--------|-------------|
| `rank` | Color rank by pigment coverage |
| `role` | `BASE` (dominant) or `SECONDARY` |
| `name` | Nearest XKCD color name |
| `hex` | Hex color code |
| `rgb` | RGB values |
| `pct_of_pigment` | % of pigmented shell area |
| `confidence_trained` | Confidence % from Method 1 (CIELAB distance to trained centroid) |
| `confidence_fresh` | Confidence % from Method 2 (cluster cohesion) |
| `combined_confidence` | Mean of both confidence scores |
| `distance_to_centroid_lab` | CIELAB distance to nearest trained centroid (deltaE units) |
| `moe_lab` | Margin of error: std dev of CIELAB distances within cluster |
| `moe_rgb_pct` | Margin of error: mean RGB channel std dev as % of 255 |
| `higher_confidence_method` | Which method had higher confidence (`Trained` or `Fresh`) |
| `suggested_method` | Recommended method based on confidence scores |
