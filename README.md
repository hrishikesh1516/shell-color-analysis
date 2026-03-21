# Shell Color Analysis

Advanced adaptive color detection framework for shell organisms (bivalves) using
scale-independent image normalisation and unsupervised clustering with automatic
parameter optimisation.

## Features

- **Automated K Selection** – Silhouette score, elbow method, and Davies-Bouldin index determine the optimal cluster count automatically
- **Hierarchical Clustering** – Agglomerative clustering with adaptive distance thresholds and dendrogram visualization
- **Adaptive Merge Logic** – Percentile-based analysis computes merge thresholds from the data distribution
- **Scale-Independent Analysis** – All images are normalized to a 1000×1000 canvas before analysis so that color-coverage percentages are directly comparable regardless of the original image resolution
- **Parameter Training** – Optimize K range, merge percentile, and hierarchical distance percentile automatically from sample images (no manual labels needed)
- **Trained vs. Default Comparison** – Run both parameter sets side-by-side to evaluate training improvement
- **Enhanced Visualization** – Dashboard with clustering metrics, dendrograms, and optimization curves
- **Comprehensive Reporting** – Detailed console output plus CSV and JSON export with color percentages
- **Background Removal** – Automatic shell isolation using `rembg`
- **Glare Detection** – Inpainting-based glare removal before color analysis
- **XKCD 949-Color Dictionary** – Perceptual color naming using CIELAB distance

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `rembg` requires a compatible Python environment. On first run it will download the background-removal model (~175 MB).

---

## Usage

### Basic – Analyse all images in a folder

```bash
# Linux / macOS
python shell_color_analysis.py --folder /path/to/images

# Windows
python shell_color_analysis.py --folder "C:\Users\YourName\PythonProject\images"
```

---

### Clustering Methods

Both methods can be run individually or together.

#### K-Means (optimal K selected automatically)
```bash
python shell_color_analysis.py --folder /path/to/images --method kmeans
```

#### Hierarchical (agglomerative with dendrogram)
```bash
python shell_color_analysis.py --folder /path/to/images --method hierarchical
```

#### Both methods together (default – recommended)
```bash
python shell_color_analysis.py --folder /path/to/images --method all
```

---

### Adjust K-Search Range (K-Means)

Control the minimum and maximum number of clusters the optimizer searches over:

```bash
python shell_color_analysis.py --folder /path/to/images --k-min 3 --k-max 15
python shell_color_analysis.py --folder /path/to/images --k-min 5 --k-max 20
```

---

### Color Merge Threshold

By default the merge threshold is computed **adaptively** from the pairwise distance
distribution of the found cluster centers.  You can override it with a fixed CIELAB
distance:

```bash
# Use adaptive threshold (default – recommended)
python shell_color_analysis.py --folder /path/to/images

# Fix the threshold at a specific CIELAB distance
python shell_color_analysis.py --folder /path/to/images --merge-threshold 15.0
python shell_color_analysis.py --folder /path/to/images --merge-threshold 20.0
```

---

### Output Options

```bash
# Save outputs without opening interactive plot windows
python shell_color_analysis.py --folder /path/to/images --no-show

# Run analysis but do not write any files to disk
python shell_color_analysis.py --folder /path/to/images --no-save

# Specify a custom output folder
python shell_color_analysis.py --folder /path/to/images --output /path/to/results
```

---

### Scale-Independent Image Analysis

All images are automatically normalized to a **1000×1000 pixel canvas** before any
analysis takes place.  This means:

- Images of any resolution (500 px or 5000 px) produce **identical percentage outputs**
- No camera calibration or `PIXELS_PER_UNIT` value is required
- Results are directly comparable across your entire dataset

```
Input: any resolution (e.g. 3000×2000 px)
        ↓
Resize to fit 1000×1000 canvas (aspect ratio preserved, padded with black)
        ↓
All analysis runs on the 1000×1000 image
        ↓
Output: color percentages (scale-independent ✓)
```

---

### Training Optimized Clustering Parameters

The framework can learn the best K range, merge percentile, and hierarchical distance
percentile for **your** shells by analysing a folder of training images.  No manual
labeling is required.

```bash
# Train on sample images
python shell_color_analysis.py --folder training_data --train

# Windows
python shell_color_analysis.py --folder "C:\Users\YourName\PythonProject\training_data" --train
```

The optimized parameters are saved to `trained_params.pkl` (path configurable via
`TRAINED_PARAMS_PATH` in `DEFAULT_CONFIG`).

**What gets optimized:**

| Parameter | What it controls |
|-----------|-----------------|
| `NUM_CLUSTERS_MIN` | Lower bound of the K search |
| `NUM_CLUSTERS_MAX` | Upper bound of the K search |
| `COLOR_MERGE_PERCENTILE` | Merge aggressiveness |
| `HIERARCHICAL_DISTANCE_PERCENTILE` | Hierarchical cut height |
| `optimal_merge_threshold` | Fixed merge distance derived from training |

---

### Using Trained Parameters for Analysis

```bash
# Analyse new images with the trained parameter set
python shell_color_analysis.py --folder trial --use-trained-params --no-show
```

---

### Comparing Trained vs. Default Results

Run both parameter sets back-to-back and compare their outputs:

```bash
python shell_color_analysis.py --folder trial --compare-trained --no-show
```

This produces two sets of output files (labelled `default` and `trained`) so you can
evaluate the improvement the training step provides.

---

### Complete Step-by-Step Workflow

```
Step 1 – Prepare your images
─────────────────────────────
project/
├── training_data/        ← Representative images for parameter optimisation
│   ├── shell_01.jpg      (any resolution – will be normalized automatically)
│   ├── shell_02.jpg
│   └── ...
├── trial/                ← New images to analyse
│   └── purple_shell.jpg
└── shell_color_analysis.py

Step 2 – Train clustering parameters on representative images
──────────────────────────────────────────────────────────────
python shell_color_analysis.py --folder training_data --train

Step 3 – Analyse new images with trained parameters
─────────────────────────────────────────────────────
python shell_color_analysis.py --folder trial --use-trained-params --no-show

Step 4 – Compare trained vs. default (optional)
────────────────────────────────────────────────
python shell_color_analysis.py --folder trial --compare-trained --no-show
```

---

### Output Files

Running `--method all` produces:

| File | Description |
|------|-------------|
| `dashboard_<timestamp>.png` | Multi-panel visualization dashboard |
| `optimization_curves_<timestamp>.png` | K-selection optimization curves (K-Means only) |
| `dendrogram_<timestamp>.png` | Hierarchical clustering dendrogram |
| `results_kmeans_<timestamp>.csv` | K-Means color table (percentages) |
| `results_hierarchical_<timestamp>.csv` | Hierarchical color table (percentages) |
| `results_all_<timestamp>.json` | Both methods combined (JSON) |
| `trained_params.pkl` | Optimized parameters (after `--train`) |

---

### All Command-Line Options

```
usage: shell_color_analysis.py [-h] [--folder FOLDER] [--output OUTPUT]
                                [--method {kmeans,hierarchical,all}]
                                [--k-min K_MIN] [--k-max K_MAX]
                                [--merge-threshold MERGE_THRESHOLD]
                                [--no-show] [--no-save]
                                [--train] [--use-trained-params]
                                [--compare-trained]

options:
  -h, --help                Show this help message and exit
  --folder FOLDER           Input folder containing shell images (default: ./images)
  --output OUTPUT           Output folder for results (default: ./output)
  --method METHOD           Clustering method: kmeans | hierarchical | all (default: all)
  --k-min K_MIN             Minimum K for K-Means cluster search (default: 3)
  --k-max K_MAX             Maximum K for K-Means cluster search (default: 15)
  --merge-threshold T       Fixed CIELAB merge threshold; omit for adaptive (default: adaptive)
  --no-show                 Do not open interactive plot windows
  --no-save                 Do not save output files to disk
  --train                   Train clustering parameters on --folder images
  --use-trained-params      Load trained parameters from trained_params.pkl
  --compare-trained         Run both default and trained parameters and compare
```

> See [COMMANDS.md](COMMANDS.md) for a full quick-reference card and
> [PARAMETER_GUIDE.md](PARAMETER_GUIDE.md) for detailed parameter tuning guidance.

---

## Parameter Optimization

### Key Parameters and When to Change Them

| Parameter | Default | When to Increase | When to Decrease |
|-----------|---------|-----------------|-----------------|
| `K_MIN` | `3` | Always want at least N colors | Images are very simple |
| `K_MAX` | `15` | Shells have many subtle colors | Analysis is too slow |
| `COLOR_MERGE_THRESHOLD` | adaptive | Colors are over-split | Colors are merged too aggressively |
| `HIERARCHICAL_DISTANCE_PERCENTILE` | `85` | Want fewer, broader clusters | Want finer color distinctions |

**Adaptive vs fixed `COLOR_MERGE_THRESHOLD`**

Leave `COLOR_MERGE_THRESHOLD` as `None` (adaptive) in most cases.  The program
computes the threshold from the 30th percentile of pairwise cluster-center distances,
which scales automatically with the color complexity of each image.  Only set a fixed
value when you need strictly reproducible thresholds across a batch.

See [PARAMETER_GUIDE.md](PARAMETER_GUIDE.md) for a complete parameter reference with
recommended ranges, performance tips, and troubleshooting examples.

---

## Configuration

All parameters can be tuned in the `DEFAULT_CONFIG` dictionary at the top of
`shell_color_analysis.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INPUT_FOLDER` | `./images` | Folder containing shell images |
| `OUTPUT_FOLDER` | `./output` | Folder for output files |
| `STANDARD_CANVAS_SIZE` | `1000` | Normalized canvas size (pixels) |
| `CLUSTERING_METHOD` | `all` | Method(s) to run |
| `NUM_CLUSTERS_MIN` | `3` | Minimum K for K-Means search |
| `NUM_CLUSTERS_MAX` | `15` | Maximum K for K-Means search |
| `COLOR_MERGE_THRESHOLD` | `None` (adaptive) | Fixed CIELAB merge distance |
| `COLOR_MERGE_PERCENTILE` | `30` | Percentile for adaptive merge threshold |
| `HIERARCHICAL_DISTANCE_PERCENTILE` | `85` | Percentile for hierarchical cut height |
| `GLARE_THRESHOLD` | `245` | V-channel threshold for glare removal |
| `MIN_COLOR_BRIGHTNESS` | `40` | Minimum brightness for pigment pixels |
| `WHITE_SENSITIVITY` | `50` | Saturation below which pixel is "white" |
| `WHITE_BRIGHTNESS` | `150` | Brightness above which pixel is "white" |
| `TRAINED_PARAMS_PATH` | `trained_params.pkl` | Path for saving/loading trained params |

---

## Windows Quick Reference

```bat
REM Open Command Prompt and navigate to project folder
cd "C:\Users\YourName\PythonProject"

REM Analyse with both methods
python shell_color_analysis.py --folder trial --method all --no-show

REM Analyse with K-Means only
python shell_color_analysis.py --folder trial --method kmeans --no-show

REM Analyse with Hierarchical only
python shell_color_analysis.py --folder trial --method hierarchical --no-show

REM Train clustering parameters on representative images
python shell_color_analysis.py --folder training_data --train

REM Analyse new images with trained parameters
python shell_color_analysis.py --folder trial --use-trained-params --no-show

REM Compare trained vs. default results
python shell_color_analysis.py --folder trial --compare-trained --no-show

REM Adjust K search range
python shell_color_analysis.py --folder trial --k-min 3 --k-max 15 --no-show

REM Use fixed merge threshold
python shell_color_analysis.py --folder trial --merge-threshold 15.0 --no-show
```

---

## References and Best Practices

- Use **at least 5–10 training images** covering the full range of colors in your dataset.
- Images can be **any resolution** – the 1000×1000 normalisation is applied automatically.
- Run `--method all` first to see which clustering method performs best for your images.
- Use `--no-show` in automated or server environments to avoid GUI dependencies.
- Keep `COLOR_MERGE_THRESHOLD` adaptive unless you need perfectly reproducible thresholds.
- Consult [COMMANDS.md](COMMANDS.md) for a condensed command reference.
- Consult [PARAMETER_GUIDE.md](PARAMETER_GUIDE.md) for detailed parameter tuning.

---

## Dependencies

- [OpenCV](https://opencv.org/) – Image processing and color space conversions
- [scikit-learn](https://scikit-learn.org/) – KMeans, AgglomerativeClustering, RandomForest, metrics
- [SciPy](https://scipy.org/) – Hierarchical linkage and dendrogram
- [rembg](https://github.com/danielgatis/rembg) – Background removal
- [Matplotlib](https://matplotlib.org/) – Visualization
- [NumPy](https://numpy.org/) – Numerical operations
- [Pillow](https://python-pillow.org/) – Image loading
