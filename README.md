# Shell Color Analysis

Self-learning adaptive color detection framework for shell organisms (bivalves and
gastropods) using K-Means and Hierarchical clustering with a two-method confidence
system.

## Features

- **Self-Learning Training** – Automatically learns color centroids, optimal K range,
  merge thresholds, and hierarchical parameters from ~20 unlabeled sample images
- **Two-Method Analysis** –
  - **Method 1 (Trained)**: matches detected colors to learned centroids, confidence
    based on CIELAB distance
  - **Method 2 (Fresh)**: fresh clustering with learned optimal parameters, confidence
    based on cluster cohesion
  - **Combined**: mean-average of both methods with method comparison
- **Incremental Retraining** – Extend the model with new samples without losing
  previous knowledge
- **Confidence Scoring** – Per-color confidence % based on how closely each detected
  color matches the trained palette
- **Margin of Error** – Both ±CIELAB units and ±RGB % variation reported per color
- **Scale-Independent Analysis** – All images normalized to 1000×1000 canvas;
  color percentages are directly comparable across your entire dataset
- **Professional Infographics** – Dark-themed multi-panel dashboards saved as PNG:
  multi-method comparison, confidence distribution, CIELAB centroid scatter, training
  validation report
- **Comprehensive Reporting** – CSV and JSON export with all confidence and error
  metrics; XKCD 949-color perceptual naming via CIELAB distance
- **Background Removal** – Automatic shell isolation using `rembg`
- **Glare Detection** – Inpainting-based glare removal before color analysis

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `rembg` requires a compatible Python environment. On first run it will
> download the background-removal model (~175 MB).

---

## Quick Start

### Step 1 – Train on sample images (~20 images, no labels needed)

```bash
python shell_color_analysis.py --mode train --training-folder ./training_samples
```

The system will:
- Auto-discover the optimal K for each sample
- Extract and store color centroids (RGB and CIELAB)
- Auto-tune all parameters (K_MIN, K_MAX, merge thresholds, hierarchical percentile)
- Generate a training validation report with accuracy and consistency metrics
- Save `trained_shell_model.pkl`
- Create a training summary infographic

### Step 2 – Analyze dataset images with trained model

```bash
python shell_color_analysis.py --mode analyze \
  --input-folder ./dataset \
  --use-trained-model
```

### Step 3 – Retrain with additional samples (future)

```bash
python shell_color_analysis.py --mode train \
  --training-folder ./new_training_samples \
  --retrain
```

---

## Usage

### Training Mode

```bash
# Initial training
python shell_color_analysis.py --mode train --training-folder ./training_samples

# Incremental retraining (adds new samples to existing model)
python shell_color_analysis.py --mode train --training-folder ./training_samples --retrain

# Custom model path, no pop-up windows
python shell_color_analysis.py --mode train \
  --training-folder ./training_samples \
  --model-path ./models/my_model.pkl \
  --no-show
```

### Analysis Mode

```bash
# Two-method analysis with trained model (Method 1 + Method 2 + Combined)
python shell_color_analysis.py --mode analyze \
  --input-folder ./dataset \
  --use-trained-model

# Fresh analysis only (Method 2, no trained model required)
python shell_color_analysis.py --mode analyze --input-folder ./dataset

# Batch / server use (no interactive windows)
python shell_color_analysis.py --mode analyze \
  --input-folder ./dataset \
  --use-trained-model \
  --no-show
```

---

## Command-Line Options

```
python shell_color_analysis.py --mode {train|analyze} [OPTIONS]

options:
  --mode {train,analyze}     Operation mode (default: analyze)
  --training-folder FOLDER   Sample images for training (default: ./training_samples)
  --input-folder FOLDER      Dataset images to analyze (default: ./dataset)
  --output OUTPUT            Output folder for results (default: ./output)
  --model-path PATH          Trained model file path (default: trained_shell_model.pkl)
  --use-trained-model        Load trained model for Method 1 + 2 analysis
  --retrain                  Extend existing model incrementally
  --k-min K_MIN              Minimum K for K-Means search (default: 3)
  --k-max K_MAX              Maximum K for K-Means search (default: 15)
  --merge-threshold T        Fixed CIELAB merge threshold (default: adaptive)
  --no-show                  Do not open interactive plot windows
  --no-save                  Do not write output files to disk
```

---

## Output

### Console Table

```
RANK  ROLE        COLOR NAME        HEX      %PIGm  CONF-T  CONF-F  COMB  DIST-dE  MoE-dE  MoE-RGB%  HIGHER METHOD
1     BASE        Sandy Brown       #c89664  45.0%   82.9    70.5   76.7     3.7     7.0      5.3     Trained
2     SECONDARY   Cerulean Blue     #326496  35.0%   95.0    60.0   77.5     1.2     5.5      4.1     Trained
3     SECONDARY   Olive Yellow      #b4b450  20.0%    N/A    65.0   65.0     N/A     6.0      4.5     Fresh
```

### CSV / JSON Columns

| Column | Description |
|--------|-------------|
| `rank` | Color rank by pigment coverage |
| `role` | `BASE` (dominant) or `SECONDARY` |
| `name` | Nearest XKCD color name |
| `hex` | Hex color code |
| `rgb` | RGB values |
| `pct_of_pigment` | % of pigmented shell area |
| `confidence_trained` | Confidence % – Method 1 (CIELAB distance to trained centroid) |
| `confidence_fresh` | Confidence % – Method 2 (cluster cohesion) |
| `combined_confidence` | Mean of both confidence scores |
| `distance_to_centroid_lab` | CIELAB distance to nearest trained centroid (ΔE units) |
| `moe_lab` | Margin of error: std dev of CIELAB distances within cluster |
| `moe_rgb_pct` | Margin of error: mean RGB channel std dev as % of 255 |
| `higher_confidence_method` | Which method had higher confidence (`Trained` or `Fresh`) |
| `suggested_method` | Recommended method based on confidence scores |

### Output Files

| File | Description |
|------|-------------|
| `trained_shell_model.pkl` | Self-learning model (after `--mode train`) |
| `training_summary_<ts>.png` | Training infographic with centroids, K values, validation |
| `analysis_dashboard_<ts>.png` | Multi-method dashboard (Trained / Fresh / Combined) |
| `color_palette_<ts>.png` | Color palette pie chart with confidence indicators |
| `results_combined_<ts>.csv` | Combined results with all confidence and error metrics |
| `results_combined_<ts>.json` | Combined results in JSON format |

---

## Trained Model Contents (`trained_shell_model.pkl`)

| Field | Description |
|-------|-------------|
| `centroids_rgb` | Consolidated color centroids (RGB) |
| `centroids_lab` | Consolidated color centroids (CIELAB) |
| `centroid_names` | Perceptual color names for each centroid |
| `centroid_stats` | Per-centroid mean, std, min, max statistics |
| `k_min` / `k_max` | Learned optimal K range |
| `merge_threshold` | Learned optimal CIELAB merge distance |
| `merge_percentile` | Learned optimal merge percentile |
| `hierarchical_percentile` | Learned optimal hierarchical cut percentile |
| `confidence_scale` | Scale factor for confidence decay (CIELAB units) |
| `n_training_samples` | Number of images trained on |
| `training_date` | ISO timestamp of last training |
| `k_values_found` | Best K discovered for each training image |
| `per_sample_colors` | Per-sample color data (for incremental retraining) |
| `training_accuracy` | Validation accuracy on training images (%) |
| `consistency_score` | Color consistency across training samples (0–100) |

---

## Configuration

All parameters can be tuned in `DEFAULT_CONFIG` at the top of `shell_color_analysis.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INPUT_FOLDER` | `./images` | Folder containing shell images |
| `OUTPUT_FOLDER` | `./output` | Folder for output files |
| `MODEL_PATH` | `trained_shell_model.pkl` | Trained model file path |
| `STANDARD_CANVAS_SIZE` | `1000` | Normalized canvas size (pixels) |
| `NUM_CLUSTERS_MIN` | `3` | Minimum K for K-Means search |
| `NUM_CLUSTERS_MAX` | `15` | Maximum K for K-Means search |
| `COLOR_MERGE_THRESHOLD` | `None` (adaptive) | Fixed CIELAB merge distance |
| `COLOR_MERGE_PERCENTILE` | `30` | Percentile for adaptive merge threshold |
| `HIERARCHICAL_DISTANCE_PERCENTILE` | `85` | Percentile for hierarchical cut height |
| `CONFIDENCE_SCALE_LAB` | `25.0` | CIELAB units for confidence decay |
| `GLARE_THRESHOLD` | `245` | V-channel threshold for glare removal |
| `MIN_COLOR_BRIGHTNESS` | `40` | Minimum brightness for pigment pixels |
| `WHITE_SENSITIVITY` | `50` | Saturation below which pixel is "white" |
| `WHITE_BRIGHTNESS` | `150` | Brightness above which pixel is "white" |

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
```

---

## Dependencies

- [OpenCV](https://opencv.org/) – Image processing and color space conversions
- [scikit-learn](https://scikit-learn.org/) – KMeans, metrics
- [SciPy](https://scipy.org/) – Hierarchical linkage
- [rembg](https://github.com/danielgatis/rembg) – Background removal
- [Matplotlib](https://matplotlib.org/) – Visualization
- [NumPy](https://numpy.org/) – Numerical operations
- [Pillow](https://python-pillow.org/) – Image loading

See [COMMANDS.md](COMMANDS.md) for the full command quick-reference.
