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

## Usage

### Basic (analyze all images in a folder)
```bash
python shell_color_analysis.py --folder /path/to/images
```

### Select a specific clustering method
```bash
python shell_color_analysis.py --folder /path/to/images --method kmeans
python shell_color_analysis.py --folder /path/to/images --method hierarchical
python shell_color_analysis.py --folder /path/to/images --method dbscan
python shell_color_analysis.py --folder /path/to/images --method all
```

### Adjust K-search range
```bash
python shell_color_analysis.py --folder /path/to/images --k-min 5 --k-max 25
```

### Fix CIELAB merge threshold (instead of adaptive)
```bash
python shell_color_analysis.py --folder /path/to/images --merge-threshold 20.0
```

### Save outputs without displaying interactive plots
```bash
python shell_color_analysis.py --folder /path/to/images --no-show
```

### All options
```
usage: shell_color_analysis.py [-h] [--folder FOLDER] [--output OUTPUT]
                                [--method {kmeans,hierarchical,dbscan,all}]
                                [--k-min K_MIN] [--k-max K_MAX]
                                [--merge-threshold MERGE_THRESHOLD]
                                [--no-show] [--no-save] [--train]
```

## Output

Results are written to `./output/` (configurable via `--output`):

| File | Description |
|------|-------------|
| `dashboard_<timestamp>.png` | Multi-panel visualization dashboard |
| `optimization_curves_<timestamp>.png` | K-selection optimization curves (K-Means only) |
| `dendrogram_<timestamp>.png` | Hierarchical clustering dendrogram |
| `results_kmeans_<timestamp>.csv` | K-Means color table (CSV) |
| `results_hierarchical_<timestamp>.csv` | Hierarchical color table (CSV) |
| `results_dbscan_<timestamp>.csv` | DBSCAN color table (CSV) |
| `results_all_<timestamp>.json` | All methods combined (JSON) |

## Configuration

All parameters can be tuned in the `DEFAULT_CONFIG` dictionary at the top of
`shell_color_analysis.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INPUT_FOLDER` | `./images` | Folder containing shell images |
| `OUTPUT_FOLDER` | `./output` | Folder for output files |
| `CLUSTERING_METHOD` | `all` | Method(s) to run |
| `NUM_CLUSTERS_MIN` | `5` | Minimum K for search |
| `NUM_CLUSTERS_MAX` | `30` | Maximum K for search |
| `COLOR_MERGE_THRESHOLD` | `None` (adaptive) | Fixed CIELAB merge distance |
| `COLOR_MERGE_PERCENTILE` | `30` | Percentile for adaptive threshold |
| `HIERARCHICAL_DISTANCE_PERCENTILE` | `85` | Percentile for hierarchical cut height |
| `DBSCAN_EPS_PERCENTILE` | `10` | Percentile of distances for DBSCAN eps |
| `DBSCAN_MIN_SAMPLES_FRACTION` | `0.002` | Fraction of pixels for min_samples |
| `GLARE_THRESHOLD` | `245` | V-channel threshold for glare removal |
| `MIN_COLOR_BRIGHTNESS` | `40` | Minimum brightness for pigment pixels |
| `WHITE_SENSITIVITY` | `50` | Saturation below which pixel is "white" |
| `WHITE_BRIGHTNESS` | `150` | Brightness above which pixel is "white" |
| `PIXELS_PER_UNIT` | `176.0454` | Scale for area calculation (pixels/cm) |

## Supervised Classifier (Optional)

To train a classifier on your labeled dataset:

```python
from shell_color_analysis import train_color_classifier

labeled_data = [
    (purple_pixel_array, "purple"),
    (brown_pixel_array, "brown"),
    # ...
]
clf = train_color_classifier(labeled_data, save_path="color_classifier.pkl")
```

## Dependencies

- [OpenCV](https://opencv.org/) – Image processing and color space conversions
- [scikit-learn](https://scikit-learn.org/) – KMeans, AgglomerativeClustering, DBSCAN, RandomForest, metrics
- [SciPy](https://scipy.org/) – Hierarchical linkage and dendrogram
- [rembg](https://github.com/danielgatis/rembg) – Background removal
- [Matplotlib](https://matplotlib.org/) – Visualization
- [NumPy](https://numpy.org/) – Numerical operations
- [Pillow](https://python-pillow.org/) – Image loading
