# DPM Medical Anomaly Detection — Claude Code Build Plan (v2)

## Project Title
**Interpretable Medical Anomaly Detection using Discriminatively Trained Part-Based Models**

---

## What Changed From v1 (Decision Log)

| # | Topic | v1 | v2 (this plan) |
|---|---|---|---|
| 1 | Image source | Raw DICOM (~200 GB) | **Resized PNG 1024×1024** (`xhlulu/vinbigdata-chest-xray-resized-png-1024x1024`) + `train_meta.csv` with original dimensions |
| 2 | Box scaling | Not handled (major bug) | Boxes mapped from original pixels to cached image space using `train_meta.csv`; verified visually and by assertion |
| 3 | Aspect ratio | Squashed to 512×512 | Original aspect **restored** from meta dims, then letterboxed |
| 4 | Negatives | Only class 14 "No finding" | **Every image without a Cardiomegaly annotation** (includes images with other findings) |
| 5 | Label noise | Any radiologist = positive | Positive needs **≥ 2 radiologist votes**; 1-vote images are ambiguous and excluded |
| 6 | Train cap | `MAX_TRAIN_IMGS = 500` | **Cap removed**; all positives, subsampled negatives + hard-negative mining |
| 7 | Compute | CPU only, no PyTorch | **GPU allowed**; PyTorch used only as a tensor library (no neural nets) |
| 8 | Window size | 64×64 px window (smaller than a heart) | **Heart-sized root window** derived from box statistics |
| 9 | Feature extraction | HOG re-extracted per patch | **HOG computed once per pyramid level**, filters slide over the cell grid |
| 10 | Parts | Fixed anchors, deformation weights never learned | **True latent parts**: finer-resolution part filters, max over displacements, learned deformation cost |
| 11 | Part naming | "Left" at image-left | Corrected to radiological convention (image-left = patient's right) |
| 12 | Baselines | Baseline 1 mixed patch/image data | Baseline 1 is genuinely image-level; added a no-deformation ablation |

---

## Your Task
Build a **complete, end-to-end, well-commented** Jupyter notebook named
`DPM_Cardiomegaly_Pipeline.ipynb` that implements the full DPM pipeline for
detecting cardiomegaly on chest X-rays from the **VinBigData Chest X-Ray
Abnormalities Detection** dataset (resized-PNG release).

The notebook must be:
- Beginner-friendly with a markdown explanation cell before every code cell
- Fully executable **top-to-bottom** without manual intervention
- Runnable on a CUDA GPU, and **still runnable on CPU** (slower) via the `DEVICE` constant
- Producing a complete structured inference report in the final section

---

## Dataset

| Property | Detail |
|---|---|
| Images | PNG, 1024×1024, 8-bit single channel, aspect-**squashed** from originals |
| Source | `kaggle.com/datasets/xhlulu/vinbigdata-chest-xray-resized-png-1024x1024` |
| Annotations | `train.csv` from the competition (one row per radiologist box, **original pixel coordinates**) |
| Size metadata | `train_meta.csv` (`image_id, dim0, dim1`), 15,000 rows. **`dim0` = original height, `dim1` = original width** (verified against a DICOM header: Rows = dim0, Columns = dim1) |
| Test metadata | `test_meta.csv` with the same columns, if shipped with the dataset (see Assumptions) |
| Splits provided | `train/` (15,000) and `test/` (3,000, **no labels**). No val split |
| Classes | 14 findings + class 14 "No finding" |

### train.csv columns
```
image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max
```
- Coordinates are absolute pixels in the **original** DICOM space (roughly 900–3400 px per side), **not** in the 1024×1024 PNG space.
- Three radiologists annotate each training image; rows share an `image_id` but differ in `rad_id`.

### Known data facts (measured from the supplied `train_meta.csv`)
- Only ~13% of originals are square. Width/height has mean 0.88 (IQR 0.82–0.94, range 0.63–1.58). The PNGs were stretched to square, so the heart's aspect ratio is distorted by a different amount per image. This plan **undoes** the stretch.
- Original size range: heights 927–3408, widths 823–3320.

### Target class
**Cardiomegaly** (`class_name = 'Cardiomegaly'`, `class_id = 3`)

### Label rules (important)
- **Positive:** image has Cardiomegaly boxes from **≥ `MIN_RAD_VOTES` (2)** distinct radiologists.
- **Ambiguous:** exactly 1 radiologist marked Cardiomegaly. These images are **excluded from training and validation** and listed in a table in Section 2 (count, and how many have other findings).
- **Negative:** image with **zero** Cardiomegaly rows. This deliberately includes images with other findings (effusion, nodules, etc.) and "No finding" images.
- The master list of image IDs is `train_meta.csv` (not the unique IDs in `train.csv`).

### Val split
- 80/20 stratified split on `has_target` over positives + negatives (ambiguous excluded)
- `RANDOM_SEED = 42`
- Val keeps its **natural class ratio** (no subsampling)
- `test/` is used only in Section 14

---

## Assumptions to Verify at Runtime (fail loudly with a clear message)
1. PNGs are 1024×1024, mode `L` (single channel). If 3-channel, convert to grayscale.
2. `train_meta.csv` has `image_id, dim0, dim1` and covers every training PNG.
3. After scaling, **no Cardiomegaly box exceeds the image bounds** (`x_max ≤ dim1`, `y_max ≤ dim0` in original space; tolerance of a few px). If many fail, `dim0`/`dim1` may be swapped; print a diagnostic.
4. `test_meta.csv` exists. If not, Section 14 must fall back to a validation image and say so clearly (aspect restoration is impossible without original dims).

---

## File & Folder Structure (Expected Before Running)

```
project/
├── DPM_Cardiomegaly_Pipeline.ipynb
├── data/
│   ├── train/            ← <image_id>.png (1024×1024)
│   ├── test/             ← <image_id>.png
│   ├── train.csv         ← competition annotations (original-pixel boxes)
│   ├── train_meta.csv    ← image_id, dim0 (height), dim1 (width)
│   └── test_meta.csv     ← same, for test images
├── cache/                ← created by notebook: restored-aspect uint8 arrays + cache_index.csv
├── pkl/                  ← created at training time
│   ├── root_filter_Cardiomegaly.pkl
│   └── lsvm_Cardiomegaly.pkl
└── output/
    ├── plots/
    └── reports/
```

---

## Global Configuration Constants
Define all of these at the top of Section 1 and reference them everywhere.

**Hardcoding policy.** Anything algorithmic or tunable must be a constant. These
may stay as literals inside the code that uses them: HOG internals (gradient
kernels, epsilon in block normalisation), plotting cosmetics (figure sizes,
colormaps, fonts, DPI), and ASCII-bar width in the report.

```python
# ---- Paths ----
DATA_DIR        = Path('./data')
TRAIN_DIR       = DATA_DIR / 'train'
TEST_DIR        = DATA_DIR / 'test'
CSV_PATH        = DATA_DIR / 'train.csv'
TRAIN_META_PATH = DATA_DIR / 'train_meta.csv'
TEST_META_PATH  = DATA_DIR / 'test_meta.csv'
CACHE_DIR       = Path('./cache')
PKL_DIR         = Path('./pkl')
OUTPUT_DIR      = Path('./output')
PLOTS_DIR       = OUTPUT_DIR / 'plots'
REPORTS_DIR     = OUTPUT_DIR / 'reports'

# ---- Device ----
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- Target & labels ----
TARGET_CLASS    = 'Cardiomegaly'
TARGET_ID       = 3
MIN_RAD_VOTES   = 2               # radiologists needed for a positive label

# ---- Image ----
SRC_SIZE        = 1024            # PNG side length (squashed square)
CACHE_SIZE      = 512             # aspect-restored, letterboxed cache size
DET_DOWNSCALE   = 2               # root-filter level = CACHE_SIZE / 2 (256 px)
USE_CLAHE       = True
CLAHE_CLIP      = 2.0
CLAHE_TILE      = (8, 8)
PAD_VALUE       = 0               # letterbox fill

# ---- HOG ----
HOG_ORIENT      = 9
HOG_PPC         = (8, 8)          # pixels per cell
HOG_CPB         = (2, 2)          # cells per block

# ---- Root window (derived from data at runtime, bounded by these) ----
ROOT_SIZE_PERCENTILE = 25         # window sized to the small end of heart sizes
ROOT_CONTEXT_MARGIN  = 0.10       # extra lung-field context around the box
ROOT_MIN_CELLS       = 8
ROOT_MAX_CELLS       = 20

# ---- Pyramid ----
N_SCALES        = 5
SCALE_FACTOR    = 0.85            # 0.85^4 ≈ 0.52 → covers ≈1.9× size range

# ---- Detection ----
STRIDE_CELLS    = 1               # dense: every HOG cell
ROOT_TOP_K      = 20              # root-score candidates per image sent to part rescoring
THRESHOLD       = 0.0             # decision threshold on final DPM score
IOU_NMS         = 0.3
USE_POSITION_PRIOR   = True       # restrict search to where hearts occur in training
PRIOR_PERCENTILES    = (1, 99)    # of positive box centres

# ---- Model / training ----
N_PARTS         = 4
PART_RES_FACTOR = 2               # parts use HOG at 2× the root resolution
PART_SEARCH_CELLS = 2             # max part displacement (in part-level cells)
DEFORM_INIT_QUAD  = 0.01          # initial quadratic deformation weight
SVM_C           = 0.01
SVM_MAX_ITER    = 200
LSVM_ITER       = 5
NEG_PER_IMAGE   = 5               # random negative windows per negative image
INIT_NEG_IMAGES = 4000            # initial negative images (None = all)
HARD_NEG_ROUNDS = 3
HARD_NEG_PER_IMAGE = 5
POS_JITTER_CELLS = 1              # positive augmentation (shift only; NO horizontal flip)
VAL_SPLIT       = 0.2
RANDOM_SEED     = 42

# ---- Reporting ----
HIGH_DEFORM_THRESH = 0.05
CONF_HIGH       = 0.5             # final score ≥ this → HIGH confidence
BENCH_IMAGES    = 20
MODEL_VERSION   = 'DPM-v2.0'
REPORT_ID       = datetime.now().strftime('%Y%m%d_%H%M%S')
```

---

## Coordinate Conventions (read before building)

Three coordinate spaces exist. Mixing them up is the project's biggest risk.

1. **Original space** — pixel coordinates in `train.csv`, size `(dim0 = H, dim1 = W)` from the meta CSV.
2. **PNG space** — 1024×1024, the original squashed to a square.
3. **Cache space** — `CACHE_SIZE × CACHE_SIZE`, aspect-restored and letterboxed. **All modelling happens here.**

**Conversion (original → cache), using one uniform scale:**
```
s      = CACHE_SIZE / max(H, W)
new_w  = round(W * s),  new_h = round(H * s)
pad_x  = (CACHE_SIZE - new_w) // 2,  pad_y = (CACHE_SIZE - new_h) // 2
x_cache = x_orig * s + pad_x
y_cache = y_orig * s + pad_y
```
**Image:** resize the 1024×1024 PNG directly to `(new_h, new_w)` with `cv2.INTER_AREA`
(this simultaneously undoes the squash and downsamples), then paste into a
`CACHE_SIZE` canvas filled with `PAD_VALUE` at `(pad_x, pad_y)`.

**Never** scale the image back to original size, and **never** scale boxes twice.
Store `(H, W, s, pad_x, pad_y)` per image in `cache/cache_index.csv` so boxes
can be mapped back to original space for the report.

---

## Notebook Sections — Build Exactly in This Order

---

### Section 1 — Setup & Imports

**Markdown cell:** Explain what the notebook does, the dataset, and the DPM approach in 3–4 sentences.

**Code cell:**
- pip install if missing: `torch opencv-python scikit-image scikit-learn matplotlib numpy pandas joblib Pillow tqdm psutil`
- Imports (add `torch`, `psutil` to the v1 list; keep `skimage.feature.hog` for visualisation/parity testing only)
- Define all constants above; create directories
- Print device info (GPU name, VRAM if CUDA) and a path-existence table
- Set seeds for `numpy` and `torch`

---

### Section 2 — Data Loading, Labels & Box-Scaling Verification

**Markdown cell:** Explain the CSV format, what cardiomegaly is (enlarged cardiac silhouette; clinically a cardiothoracic ratio above about 0.5), how multi-radiologist labels work, and why the vote threshold exists.

**Code cell — Load and explore:**
- Read `train.csv` and `train_meta.csv`; print row counts, unique images, classes
- Class distribution bar chart → `output/plots/class_distribution.png`
- Compute, per image, the number of distinct radiologists who marked Cardiomegaly
- Print the label table: positives (≥ `MIN_RAD_VOTES`), ambiguous (1 vote), negatives; for negatives show how many have other findings vs "No finding"

**Code cell — Consensus box (original space):**
- `get_consensus_bbox(df, image_id)`: per radiologist, merge multiple Cardiomegaly boxes into their enclosing box, then average across radiologists
- Build `records`: `image_id, png_path, has_target, bbox_orig (or None), n_rads, H, W`
- Print mean/percentile box width and height **as a fraction of the original image** (this drives root-window sizing later)

**Code cell — Verify scaling (hard gate):**
- Assert boxes lie within `(W, H)`; print fraction passing; stop with a diagnostic if below 99%
- Draw consensus boxes on 8 positive PNGs after mapping original → PNG space (`x * 1024/W`, `y * 1024/H`). The box must sit on the heart
- Save `output/plots/box_scaling_check.png`

**Code cell — Train/Val split:** stratified, `VAL_SPLIT`, `RANDOM_SEED`; print counts. Ambiguous excluded.

**Code cell — Sample visualisation:** 2×4 grid (4 pos, 4 neg) with boxes → `output/plots/sample_images.png`.

---

### Section 3 — PNG Loading, Aspect Restoration & Caching

**Markdown cell:** Explain why the PNGs are squashed, why that distorts a HOG template, and what letterboxing does. Note that windowing and MONOCHROME1 inversion were already applied by the dataset authors, so this notebook does not repeat them.

**Code cell — class `Preprocessor`:**
```python
class Preprocessor:
    def load_png(self, path) -> np.ndarray          # uint8 (1024,1024), grayscale
    def restore_aspect(self, img, H, W) -> tuple    # returns (cache_img, s, pad_x, pad_y)
    def apply_clahe(self, img) -> np.ndarray        # only if USE_CLAHE
    def preprocess(self, path, H, W) -> tuple       # full pipeline + metadata
    def map_box(self, bbox_orig, s, pad_x, pad_y)   # original → cache space
```
- Edge cases: missing meta row, unexpected channel count, empty array

**Code cell — Build cache (resumable):**
- Process all train/test PNGs with `tqdm`; write uint8 `.npy` per image to `cache/`; write `cache_index.csv`
- Skip images already cached; optional `joblib` parallel decode
- After caching, compute and **store scaled boxes** for all records

**Code cell — Visualise:**
- 3 positives × 3 columns: PNG (squashed) | aspect-restored | after CLAHE; boxes overlaid on the restored image
- Save `output/plots/preprocessing_comparison.png`

---

### Section 4 — HOG Feature Extraction (GPU)

**Markdown cell:** Explain HOG (orientations, cells, blocks), why 8×8 cells, and how to compute dimensionality: a window of `rh × rw` cells gives `(rh−1)(rw−1) × 2 × 2 × 9` features.

**Code cell — class `HOGExtractor` (PyTorch implementation):**
```python
class HOGExtractor:
    def __init__(self, orientations=HOG_ORIENT, pixels_per_cell=HOG_PPC,
                 cells_per_block=HOG_CPB, device=DEVICE): ...
    def block_map(self, img) -> torch.Tensor      # (n_block_y, n_block_x, 36), computed ONCE per image/level
    def window(self, block_map, cy, cx, ch, cw)   # contiguous sub-grid of blocks → flat vector
    def visualise(self, img) -> np.ndarray        # uses skimage hog(..., visualize=True) for display only
```
- Gradients via convolution, unsigned orientation binning with bilinear voting, cell pooling, L2-Hys block normalisation
- Batched over images where possible
- Output must be a **dense block map**; a detection window is just a sub-grid of it. No per-patch resizing + re-extraction
- Parity test: compare against `skimage.feature.hog` on 5 images; print correlation (expect > 0.95, differences allowed from interpolation)

**Code cell — HOG visualisation:** 4 images (2 pos, 2 neg): original | HOG heatmap → `output/plots/hog_visualization.png`. Print feature dimensionality.

---

### Section 5 — Image Pyramid & Root Window Sizing

**Markdown cell:** Explain multiscale detection. Explain that the root window is sized to the **small end** of heart sizes, and the pyramid only shrinks the image, so larger hearts are matched at coarser levels.

**Code cell — Derive the root window (from training positives only):**
- Compute positive box sizes in cache space at the root level (`CACHE_SIZE / DET_DOWNSCALE`)
- Window size (cells) = `ROOT_SIZE_PERCENTILE`-th percentile area at the median aspect ratio, expanded by `ROOT_CONTEXT_MARGIN`, clamped to `[ROOT_MIN_CELLS, ROOT_MAX_CELLS]`
- Print root window in pixels and cells, expected feature dimension
- **Coverage check:** print what fraction of positive boxes have a size within `[window, window / SCALE_FACTOR^(N_SCALES−1)]`. If below 95%, print a recommendation to raise `N_SCALES`
- Compute and store the position prior (box-centre percentile region) if `USE_POSITION_PRIOR`

**Code cell — class `ImagePyramid`:** `build(img) → list[(scale, img)]`; stops when the image would be smaller than the root window.

**Code cell — Visualise:** all levels in one row, title `Level N scale=X.XX HxW` → `output/plots/image_pyramid.png`.

---

### Section 6 — Training Data Construction

**Markdown cell:** Explain positives (heart box resized to the root window), negatives (random windows from negative images and from outside-the-box regions of positive images), and why hard-negative mining is needed.

**Code cell — `build_feature_matrix`:**
- **Positives (all of them, no cap):** for each positive, rescale the image so the box (plus margin) maps to the root window size, compute the block map once, extract the cell-aligned window. Add shift jitter of ±`POS_JITTER_CELLS`. **No horizontal flip** (the heart is asymmetric)
- **Initial negatives:** from `INIT_NEG_IMAGES` negative images (or all), `NEG_PER_IMAGE` random windows each at random pyramid levels, plus windows with IoU < 0.3 against the box in positive images
- Return `X` (float16/32 on CPU or GPU), `y`, `ids`
- Print `X.shape`, class balance, memory footprint, summary stats

---

### Section 7 — Root Filter Training (GPU, Hard-Negative Mining)

**Markdown cell:** Explain the root filter as a global template of cardiomegaly appearance, trained discriminatively with a linear SVM; explain hard-negative mining.

**Code cell — class `RootFilter`:**
```python
class RootFilter:
    def __init__(self, C=SVM_C): ...                # weights w, bias b, standardisation stats, window size
    def fit(self, X, y): ...                        # squared hinge + L2, class-weighted, L-BFGS on DEVICE
    def score_map(self, block_map) -> Tensor        # dense scores via conv2d over a pyramid level
    def decision_function(self, X): ...
    def save(self, path) / load(cls, path): ...     # joblib
```
- Training: standardise features, optimise in PyTorch; use `SVM_MAX_ITER`
- Sanity check: on a 2k-sample subset, compare against `sklearn.svm.LinearSVC` (accuracy within a few points)

**Code cell — Train with hard-negative mining:**
- Round 0: initial matrix. Rounds 1…`HARD_NEG_ROUNDS`: densely score **all** negative training images across the pyramid, add the top `HARD_NEG_PER_IMAGE` highest-scoring false positives per image, retrain
- Save `pkl/root_filter_Cardiomegaly.pkl`
- Print per-round training accuracy, number of hard negatives, feature dim
- Plot filter weights reshaped to `(blocks_y, blocks_x, 36)` summed over orientations → `output/plots/root_filter_weights.png`

---

### Section 8 — Latent SVM with Learned Part Filters

**Markdown cell:** Explain the four parts, the quadratic spring model, and why the problem is "latent": part positions are not annotated, so training alternates between inferring them and re-fitting the SVM (coordinate descent). Explain the radiological convention: **image-left is the patient's right**.

**Code cell — Part definitions:**
```python
# (rel_x, rel_y, width_factor, height_factor) relative to the root box (cache space)
PART_ANCHORS = [
    (0.0, 0.2, 0.3, 0.6),   # Part 0: image-left edge  = PATIENT'S RIGHT heart border
    (0.7, 0.2, 0.3, 0.6),   # Part 1: image-right edge = PATIENT'S LEFT heart border
    (0.1, 0.0, 0.8, 0.3),   # Part 2: superior contour (great vessels / aortic arch region)
    (0.1, 0.7, 0.8, 0.3),   # Part 3: inferior contour (diaphragmatic surface)
]
PART_NAMES  = ['Right cardiac border (pt. right)', 'Left cardiac border (pt. left)',
               'Superior contour', 'Inferior contour']
PART_COLORS = ['cyan', 'lime', 'orange', 'magenta']
```
Anchors are only **initial priors**; the latent displacement lets each part move.

**Model and scoring:**
```
score(window) = w_root · φ_root
              + Σ_i  max over δ ∈ D  [ w_i · φ_part_i(anchor_i + δ)  −  d_i · φ_d(δ) ]
              + b
φ_d(δ) = [dx, dx², dy, dy²]     D = ±PART_SEARCH_CELLS (part-level cells)
```
- Part HOG is computed at `PART_RES_FACTOR`× the root resolution (finer border detail)
- Quadratic deformation weights are constrained ≥ 0 (projected after each update); initialise to `DEFORM_INIT_QUAD`

**Code cell — class `LatentSVM`:**
```python
class LatentSVM:
    def __init__(self, n_parts=N_PARTS, C=SVM_C, max_iter=LSVM_ITER): ...
    def infer_parts(self, root_window, part_maps) -> (best_deltas, part_scores, deform_costs)
    def build_vector(self, root_feat, part_feats_at_best_delta, best_deltas) -> Tensor
        # concat [φ_root, φ_part_1..N, −φ_d_1..N, 1] so the SVM LEARNS filters AND deformation costs
    def fit(self, train_records, preprocessor, hog, root_filter): ...
        # init: part filters from positive part crops at anchor positions
        # loop LSVM_ITER:
        #   E-step: best δ per part for every positive (and for mined negatives) under current weights
        #   M-step: retrain on concatenated vectors (GPU)
        # tqdm over iterations; record loss per iteration
    def score_window(self, ...) -> dict   # returns total, root, per-part, deformation, bias (for the report)
    def save(self, path) / load(cls, path): ...
```
- Part displacement search is vectorised on the GPU (small window, brute-force max)
- Tracks and returns the **full decomposition** (root, per-part, per-part deformation, bias) for interpretability

**Code cell — Train and save:** save to `pkl/lsvm_Cardiomegaly.pkl`; print combined feature dim, iterations run, loss per iteration (formatted table).

---

### Section 9 — Dense Detection Engine

**Markdown cell:** Explain the two-stage detector: (1) dense root scoring over every pyramid level, (2) part rescoring of the best candidates, then NMS.

**Code cell — `detect(img, root_filter, lsvm=None)`:**
```
1. Build pyramid; compute ONE block map per level (root level and part level)
2. Root score map per level via conv2d; apply position prior mask; STRIDE_CELLS stride
3. Keep top ROOT_TOP_K candidates over all levels (not a hard threshold)
4. NMS on root detections                      → root_dets
5. If lsvm: rescore candidates with full DPM score (max over part displacements)
6. NMS on DPM detections (IOU_NMS)             → final_dets
7. Map boxes back to cache space (and optionally to original space via cache_index)
Return: dicts with root_score, dpm_score, bbox, level, part boxes, deltas
```
- Image-level score = max DPM score over candidates (or max root score for root-only mode)
- Edge cases: zero candidates, windows touching image borders

**Code cells — `nms(detections, iou_threshold=IOU_NMS)` and `compute_iou(boxA, boxB)`** (both with docstrings).

---

### Section 10 — Validation & Metrics

**Markdown cell:** Explain each metric and why sensitivity (missing a sick patient) usually matters more than accuracy, and why PR-AUC is more informative under class imbalance.

**Code cell — Run on full val set** (`tqdm`): for each val record run `detect`; store `image_id, gt_label, pred_label (score > THRESHOLD), pred_score, pred_bbox, gt_bbox, iou` → `val_results_df`.

**Code cell — Metrics table (all of these):**
Accuracy, Precision/PPV, Recall/Sensitivity, Specificity, NPV, F1, F2, MCC, ROC-AUC, PR-AUC, Miss Rate, FPR per image, Mean IoU on true positives, **localisation hit-rate (IoU ≥ 0.5 among positives)**.
- Print as a formatted table. Also report the Youden-optimal threshold **clearly labelled as tuned on val (optimistic)**

**Code cell — Evaluation curves:** ROC, PR, confusion matrix, score histogram (TP green vs FP red, threshold line) in 2×2 → `output/plots/evaluation_curves.png`.

**Code cell — Failure gallery:** 4 FP + 4 FN → `output/plots/failure_cases.png`.

---

### Section 11 — Interpretability Visualisation

**Markdown cell:** Explain each panel and how a radiologist would read it.

**Code cell — `build_response_map(img, root_filter)`:** dense root response over the single root-level scale; return the 2D map (reuse the Section 9 convolution).

**Code cell — 4-panel figure** for the top-scoring true positive:
1. Preprocessed CXR
2. Full-image HOG, `hot` colormap
3. Root response heatmap (`RdYlGn`, alpha 0.5, colorbar)
4. DPM detection: root box RED (lw 3), part boxes in `PART_COLORS` (lw 2, dashed), labels P0–P3, white deformation arrows from each part's **anchor to its inferred position**, score annotation `Root: X.XX | DPM: X.XX`, legend for all 5 boxes
- Suptitle `Interpretable DPM — {TARGET_CLASS}` → `output/plots/interpretability_4panel.png`

**Code cell — Score decomposition bar chart:** root, parts 0–3, total deformation (red, negative), final score (bold) with value labels → `output/plots/score_decomposition.png`.

**Code cell — Part activation grid:** 1×4, one panel per part showing the cropped region with HOG overlay; title = part name, deformation cost, activation → `output/plots/part_activations.png`.

---

### Section 12 — Baselines & Ablations

**Markdown cell:** Explain why baselines are needed to justify the interpretability trade-off, and what each ablation isolates.

**Code cells:**
- **Baseline 1 — global HOG + LinearSVC (image-level):** HOG of the whole cached image downsampled to a fixed size, labelled by image-level `has_target`, trained on the **train images** (not on patches). Evaluate on val; no IoU (N/A)
- **Baseline 2 — root only:** the Section 7 filter with the same dense pyramid and NMS, no parts
- **Ablation — DPM without deformation:** parts used but displacement fixed at 0 (isolates the value of the latent search)
- **DPM (full)**
- Print a 4-column comparison table (Accuracy, Sensitivity, Specificity, F1, ROC-AUC, PR-AUC, Mean IoU, Hit-rate@0.5) and a grouped bar chart (F1, AUC, Sensitivity) → `output/plots/baseline_comparison.png`
- Interpret honestly in a markdown cell: if the parts do not beat the root filter, say so

---

### Section 13 — Computational Profile

**Markdown cell:** Why inference speed and memory matter for clinical edge deployment.

**Code cell — Benchmark on `BENCH_IMAGES` random val images:** per-image times (ms) for preprocessing, HOG block maps, pyramid, dense root scoring, NMS, part scoring, total. Use `torch.cuda.synchronize()` around GPU timings. Peak host RAM via `psutil`, peak GPU memory via `torch.cuda.max_memory_allocated` (note that `tracemalloc` does not see tensors). Print mean ± std table; also report a CPU-only timing for a subset for comparison.

**Code cell — Model size:** pkl sizes (KB), total, and estimated FLOPs per image from **actual** filter dimensions × windows × levels (not a fixed formula). Print the GPU name used.

---

### Section 14 — Full Structured Inference Report

**Markdown cell:** Runs the complete pipeline on a single test PNG and writes the report.

**Code cell — Pick a test image:** first file in `TEST_DIR`, or a user-set `TEST_IMAGE_PATH`. Look up original `(H, W)` in `test_meta.csv`. If missing, print a clear message and use a validation image instead.

**Code cell — Run full pipeline with all timings and peak memory:**
1. `Preprocessor.preprocess`
2. HOG block maps
3. Pyramid
4. Dense root scoring → raw candidates
5. NMS
6. Part rescoring → DPM scores and score decomposition
7. Response map
8. 4-panel interpretability figure

**Code cell — Generate and print the full report**, also saved to `output/reports/report_{TARGET_CLASS}_{REPORT_ID}.txt`:

```
Block 0 — REPORT HEADER
═══════════════════════════════════════════════════════════════════════
        MEDICAL ANOMALY DETECTION — INFERENCE REPORT
               Part-Based Deformable Model (DPM)
═══════════════════════════════════════════════════════════════════════
Report ID, Image File (PNG), Original Size (H×W), Working Size, Device,
Scan Type: Chest X-Ray, Target Class, Date Processed, Model Version
(MODEL_VERSION), Dataset: VinBigData Chest X-Ray Abnormalities Detection
═══════════════════════════════════════════════════════════════════════

Block 1 — DETECTION SUMMARY
    Anomaly Detected, Anomaly Type, Confidence Score, Detection Status,
    Bounding Box (x,y,w,h) in ORIGINAL image pixels, Box Size (px),
    Pyramid Levels Built, Raw Detections, NMS Suppressed, Final Detections

Block 2 — SCORE DECOMPOSITION
    Root Filter Score + contribution % + ASCII bar
    Part 0..3 Score   + contribution % + ASCII bar
    ──────────────────────────────────────────────
    Total Part Score, Total Deformation Cost, Bias Term
    ──────────────────────────────────────────────
    FINAL DPM SCORE

Block 3 — PART-LEVEL DETAILS
    Part#, Name, BBox(x,y,x2,y2), Displacement (dx,dy), Deformation Cost,
    HIGH flag if |cost| > HIGH_DEFORM_THRESH

Block 4 — VISUALISATION OUTPUT INDEX
    All saved plots with descriptions

Block 5 — COMPUTATIONAL PROFILE
    Timing phases (ms), peak host RAM (MB), peak GPU memory (MB),
    model size (KB), FLOPs

Block 6 — INTERPRETABILITY SUMMARY
    Most Discriminative Part, Least Active Part, Highest Deformation Part,
    Deformation Arrows Drawn, Root Heatmap Generated, Score Breakdown Available
    + Plain-language explanation (3–4 sentences) of WHY the image was
      flagged or not flagged, referencing the actual part contributions

Block 7 — ALL SCORED DETECTIONS
    Det#, BBox(x,y,w,h), Root Score, DPM Score, Classification label

Block 8 — CLINICAL FLAG
═══════════════════════════════════════════════════════════════════════
⚠⚠  CLINICAL FLAG — ANOMALY DETECTED  ⚠⚠   (or ✓ NORMAL)
═══════════════════════════════════════════════════════════════════════
    Finding, Confidence Level (HIGH if score ≥ CONF_HIGH, MODERATE if
    ≥ THRESHOLD, else LOW), Priority (URGENT/ROUTINE/LOW), Recommended Action
    DISCLAIMER: This output is a research pre-filtering tool only.
    All flagged cases must be reviewed by a qualified radiologist.
    This output does not constitute a clinical diagnosis.

FOOTER
═══════════════════════════════════════════════════════════════════════
Model | Hardware (CPU/GPU name) | RAM | Time | Scales | Detections | Saved path
═══════════════════════════════════════════════════════════════════════
```

---

## Code Quality Requirements

- Every class and function has a docstring
- Every section has a markdown explanation cell before its code cell
- `tqdm` for any loop over more than 10 items
- Formatted tables for all numerical outputs, never raw dict dumps
- All plots: title, axis labels, legend where applicable, saved to file
- Handle edge cases: empty detections, missing meta rows, zero-size windows, windows at image borders, `DEVICE = 'cpu'`
- No magic numbers in algorithmic code; use the constants (see the hardcoding policy above)
- `pkl/` files saved with `joblib.dump` and loadable with `joblib.load`; model objects must save CPU tensors and reload onto `DEVICE`
- The notebook must run top-to-bottom on a fresh kernel without errors; the cache build is resumable

---

## Do NOT Include

- Neural-network models, pretrained backbones or transfer learning. **PyTorch is allowed only as a tensor/optimisation library** (HOG, convolution scoring, SVM training)
- DICOM reading code (the dataset is PNG)
- Docker or containerisation setup
- Any reference to the old BoneFractureYOLO8 dataset
- Horizontal-flip augmentation

---

## Expected Behaviour & Honest Caveats (include a short version in the notebook's final markdown)

- Cardiomegaly is defined by a global measurement (heart vs thorax width). A local HOG template can capture it only partly; the context margin helps, but expect modest performance compared with deep models. The comparison in Section 12 is the honest test of whether the parts add value.
- Cached images are 512 px and the root filter runs at 256 px. If part detail looks too coarse, raise `PART_RES_FACTOR`; if memory is tight, reduce it.
- Per-image aspect restoration is exact only as far as `train_meta.csv` is correct (verified in Section 2).

---

## Final Checklist Before Finishing

- [ ] All 14 sections present and in order
- [ ] Constants defined in Section 1 and used throughout (exempt literals only as listed)
- [ ] `train_meta.csv` joined; **boxes verified inside bounds and visually on the heart**
- [ ] Aspect ratio restored; boxes converted original → cache space exactly once
- [ ] Positives need ≥ 2 votes; ambiguous excluded and tabulated; other-finding images are negatives
- [ ] 80/20 stratified split; val at natural ratio; test used only in Section 14
- [ ] No `MAX_TRAIN_IMGS` cap; hard-negative mining implemented
- [ ] HOG computed once per pyramid level (no per-patch resizing)
- [ ] Root window derived from data; coverage check printed
- [ ] LSVM learns part filters **and** deformation weights; parts use finer resolution; part names follow radiological convention
- [ ] RootFilter, LatentSVM and HOGExtractor fully implemented, GPU-capable, CPU fallback works
- [ ] All Section 10 metrics computed (incl. localisation hit-rate)
- [ ] Baselines and no-deformation ablation reported honestly
- [ ] All plots saved to `output/plots/`
- [ ] Report printed AND saved to `output/reports/`; clinical disclaimer present in Block 8
