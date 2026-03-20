# Shell Color Analysis

Advanced adaptive color detection framework for shell organisms (bivalves) using
multiple unsupervised and supervised clustering strategies.

## Features

- **Automated K Selection** – Silhouette score, elbow method, and Davies-Bouldin index determine the optimal cluster count automatically
- **Hierarchical Clustering** – Agglomerative clustering with adaptive distance thresholds and dendrogram visualization
- **DBSCAN Integration** – Density-based clustering detects both dominant and rare colors without pre-specifying K
- **Adaptive Merge Logic** – Percentile-based analysis computes merge thresholds from the data distribution
- **Multiple Method Comparison** – K-Means, Hierarchical, and DBSCAN run together with side-by-side performance metrics
- **Supervised Classifier** – Optional Random Forest color classifier for standardized detection across a dataset
- **Enhanced Visualization** – Dashboard with clustering metrics, dendrograms, optimization curves, and method comparisons
- **Comprehensive Reporting** – Detailed console output plus CSV and JSON export including area statistics and pattern analysis
- **Background Removal** – Automatic shell isolation using `rembg`
- **Glare Detection** – Inpainting-based glare removal before color analysis
- **Pattern Analysis** – Geometric analysis of secondary color patterns (stripes, spots, patches)
- **XKCD 949-Color Dictionary** – Perceptual color naming using CIELAB distance

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `rembg` requires a compatible Python environment. On first run it will download the background-removal model (~175 MB).

---

## Usage

### Basic – Analyze all images in a folder

```bash
# Linux / macOS
python shell_color_analysis.py --folder /path/to/images

# Windows
python shell_color_analysis.py --folder "C:\Users\YourName\PythonProject\images"
```

---

### Clustering Methods

All three methods can be run individually or together.

#### K-Means (optimal K selected automatically)
```bash
python shell_color_analysis.py --folder /path/to/images --method kmeans
```

#### Hierarchical (agglomerative with dendrogram)
```bash
python shell_color_analysis.py --folder /path/to/images --method hierarchical
```

#### DBSCAN (density-based, detects rare colors)
```bash
python shell_color_analysis.py --folder /path/to/images --method dbscan
```

#### All methods together (default – recommended)
```bash
python shell_color_analysis.py --folder /path/to/images --method all
```

---

### Adjust K-Search Range (K-Means)

Control the minimum and maximum number of clusters the optimizer searches over:

```bash
python shell_color_analysis.py --folder /path/to/images --k-min 3 --k-max 15
python shell_color_analysis.py --folder /path/to/images --k-min 5 --k-max 25
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

### Complete Step-by-Step Workflow

```
Step 1 – Prepare your images
─────────────────────────────
project/
├── training_data/        ← Images used for training the classifier
│   ├── shell_01.jpg
│   ├── shell_02.jpg
│   └── ...
├── trial/                ← New images to analyze
│   └── purple_shell.jpg
└── shell_color_analysis.py

Step 2 – Analyze new images with all clustering methods
────────────────────────────────────────────────────────
python shell_color_analysis.py --folder trial --method all --no-show

Step 3 – Train the classifier on representative images (no labels needed)
──────────────────────────────────────────────────────────────────────────
python shell_color_analysis.py --folder training_data --train

Step 4 – Evaluate training results
────────────────────────────────────
python shell_color_analysis.py --folder training_data --train --no-show

Step 5 – Apply the trained model to all new images
────────────────────────────────────────────────────
python shell_color_analysis.py --folder trial --no-show
```

---

### Scale-Independent Image Analysis

The framework converts pixel areas to physical units using the `PIXELS_PER_UNIT` and
`UNIT_NAME` configuration values.  Adjust these in `DEFAULT_CONFIG` (or pass the
corresponding values) to match your camera / microscope calibration:

```python
# In DEFAULT_CONFIG:
"PIXELS_PER_UNIT": 176.0454,   # pixels per cm (calibrate to your setup)
"UNIT_NAME": "cm",             # reported unit label
```

All area outputs in the CSV / JSON reports are then expressed in the chosen unit
regardless of image resolution.

---

### Model Comparison

Running `--method all` automatically compares K-Means, Hierarchical, and DBSCAN
side by side and writes a combined JSON report:

```bash
python shell_color_analysis.py --folder /path/to/images --method all --no-show
```

Output files produced:

| File | Description |
|------|-------------|
| `dashboard_<timestamp>.png` | Multi-panel visualization dashboard |
| `optimization_curves_<timestamp>.png` | K-selection optimization curves (K-Means only) |
| `dendrogram_<timestamp>.png` | Hierarchical clustering dendrogram |
| `results_kmeans_<timestamp>.csv` | K-Means color table |
| `results_hierarchical_<timestamp>.csv` | Hierarchical color table |
| `results_dbscan_<timestamp>.csv` | DBSCAN color table |
| `results_all_<timestamp>.json` | All methods combined (JSON) |

---

### All Command-Line Options

```
usage: shell_color_analysis.py [-h] [--folder FOLDER] [--output OUTPUT]
                                [--method {kmeans,hierarchical,dbscan,all}]
                                [--k-min K_MIN] [--k-max K_MAX]
                                [--merge-threshold MERGE_THRESHOLD]
                                [--no-show] [--no-save] [--train]

options:
  -h, --help            Show this help message and exit
  --folder FOLDER       Input folder containing shell images (default: ./images)
  --output OUTPUT       Output folder for results (default: ./output)
  --method METHOD       Clustering method: kmeans | hierarchical | dbscan | all
                        (default: all)
  --k-min K_MIN         Minimum K for K-Means cluster search (default: 5)
  --k-max K_MAX         Maximum K for K-Means cluster search (default: 30)
  --merge-threshold T   Fixed CIELAB merge threshold; omit for adaptive (default: adaptive)
  --no-show             Do not open interactive plot windows
  --no-save             Do not save output files to disk
  --train               Train the supervised Random Forest color classifier
```

> See [COMMANDS.md](COMMANDS.md) for a full quick-reference card and
> [PARAMETER_GUIDE.md](PARAMETER_GUIDE.md) for detailed parameter tuning guidance.

---

## Training the Supervised Classifier

### How It Works (No Manual Labels Needed)

The system uses an **unsupervised-to-supervised hybrid** approach.  You supply a
folder of representative shell images — **no manual pixel marking required**.  The
clustering algorithms automatically detect color groups, which then serve as labels
for training the Random Forest classifier.

```
Your Training Images
        ↓
Automatic Color Detection (Clustering)
├── K-Means  – finds dominant color clusters
├── Hierarchical – captures color relationships
└── DBSCAN   – catches rare / subtle colors
        ↓
System Automatically Labels Clusters
├── "This cluster = Purple"
├── "This cluster = Pink"
└── "This cluster = Orange"
        ↓
Random Forest Classifier Trains on CIELAB Features
        ↓
Model saved → color_classifier.pkl  ✓
```

### Training Without Pre-Labeled Images

```bash
# Linux / macOS
python shell_color_analysis.py --folder training_data --train

# Windows
python shell_color_analysis.py --folder "C:\Users\YourName\PythonProject\training_data" --train
```

Aim for **5–10 diverse images** covering:
- Different species (bivalves and gastropods)
- Different dominant colors (purple, brown, orange, white, etc.)
- Different patterns (solid, striped, spotted)
- Different lighting conditions

### Training With Labeled Data (Advanced)

If you already have hand-labeled pixel arrays, you can call the training function
directly from Python:

```python
from shell_color_analysis import train_color_classifier
import numpy as np

# labeled_data is a list of (pixel_array, label_string) tuples
labeled_data = [
    (purple_pixel_array, "purple"),   # shape (N, 3) in RGB
    (brown_pixel_array,  "brown"),
    (white_pixel_array,  "white"),
    # ...
]
clf = train_color_classifier(labeled_data, save_path="color_classifier.pkl")
```

### Applying the Trained Classifier

After training, run the normal analysis — the saved classifier is loaded automatically:

```bash
python shell_color_analysis.py --folder trial --no-show
```

---

## Parameter Optimization

### Key Parameters and When to Change Them

| Parameter | Default | When to Increase | When to Decrease |
|-----------|---------|-----------------|-----------------|
| `K_MIN` | `5` | Always want at least N colors | Images are very simple |
| `K_MAX` | `30` | Shells have many subtle colors | Analysis is too slow |
| `COLOR_MERGE_THRESHOLD` | adaptive | Colors are over-split | Colors are merged too aggressively |
| `HIERARCHICAL_DISTANCE_PERCENTILE` | `85` | Want fewer, broader clusters | Want finer color distinctions |

**Adaptive vs fixed `COLOR_MERGE_THRESHOLD`**

Leave `COLOR_MERGE_THRESHOLD` as `None` (adaptive) in most cases.  The program
computes the threshold from the 30th percentile of pairwise cluster-center distances,
which scales automatically with the color complexity of each image.  Only set a fixed
value when you need strictly reproducible thresholds across a batch.

```bash
# Adaptive (default – scales per image)
python shell_color_analysis.py --folder images

# Fixed threshold of 15 CIELAB units
python shell_color_analysis.py --folder images --merge-threshold 15.0
```

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
| `CLUSTERING_METHOD` | `all` | Method(s) to run |
| `NUM_CLUSTERS_MIN` | `5` | Minimum K for K-Means search |
| `NUM_CLUSTERS_MAX` | `30` | Maximum K for K-Means search |
| `COLOR_MERGE_THRESHOLD` | `None` (adaptive) | Fixed CIELAB merge distance |
| `COLOR_MERGE_PERCENTILE` | `30` | Percentile for adaptive merge threshold |
| `HIERARCHICAL_DISTANCE_PERCENTILE` | `85` | Percentile for hierarchical cut height |
| `DBSCAN_EPS_PERCENTILE` | `10` | Percentile of distances for DBSCAN eps |
| `DBSCAN_MIN_SAMPLES_FRACTION` | `0.002` | Fraction of pixels for min_samples |
| `GLARE_THRESHOLD` | `245` | V-channel threshold for glare removal |
| `MIN_COLOR_BRIGHTNESS` | `40` | Minimum brightness for pigment pixels |
| `WHITE_SENSITIVITY` | `50` | Saturation below which pixel is "white" |
| `WHITE_BRIGHTNESS` | `150` | Brightness above which pixel is "white" |
| `PIXELS_PER_UNIT` | `176.0454` | Scale for area calculation (pixels/cm) |

---

## Windows Quick Reference

```bat
REM Open Command Prompt and navigate to project folder
cd "C:\Users\YourName\PythonProject"

REM Analyze with all methods
python shell_color_analysis.py --folder trial --method all --no-show

REM Analyze with K-Means only
python shell_color_analysis.py --folder trial --method kmeans --no-show

REM Analyze with Hierarchical only
python shell_color_analysis.py --folder trial --method hierarchical --no-show

REM Analyze with DBSCAN only
python shell_color_analysis.py --folder trial --method dbscan --no-show

REM Train the classifier on representative images
python shell_color_analysis.py --folder training_data --train

REM Apply the trained classifier to new images
python shell_color_analysis.py --folder trial --no-show

REM Adjust K search range
python shell_color_analysis.py --folder trial --k-min 3 --k-max 15 --no-show

REM Use fixed merge threshold
python shell_color_analysis.py --folder trial --merge-threshold 15.0 --no-show
```

---

## References and Best Practices

- Use **at least 5 training images** covering the full range of colors in your dataset.
- Run `--method all` first to see which clustering method performs best for your images.
- Use `--no-show` in automated or server environments to avoid GUI dependencies.
- Keep `COLOR_MERGE_THRESHOLD` adaptive unless you need perfectly reproducible thresholds.
- Calibrate `PIXELS_PER_UNIT` to your camera setup for accurate area measurements.
- Consult [COMMANDS.md](COMMANDS.md) for a condensed command reference.
- Consult [PARAMETER_GUIDE.md](PARAMETER_GUIDE.md) for detailed parameter tuning.

---

## Dependencies

- [OpenCV](https://opencv.org/) – Image processing and color space conversions
- [scikit-learn](https://scikit-learn.org/) – KMeans, AgglomerativeClustering, DBSCAN, RandomForest, metrics
- [SciPy](https://scipy.org/) – Hierarchical linkage and dendrogram
- [rembg](https://github.com/danielgatis/rembg) – Background removal
- [Matplotlib](https://matplotlib.org/) – Visualization
- [NumPy](https://numpy.org/) – Numerical operations
- [Pillow](https://python-pillow.org/) – Image loading
