# Complete File Guide — Well-LAB-ALL- Pipeline

This guide explains EVERY file in the pipeline — what it does, what input it needs, and what output it produces.

---

## TASK 1: Canal Detection & Boundary Segmentation

**Location:** `pipeline/task1_canal_detection/`

### File 1: `extract_patches_v2.py`
**What it does:** Extracts 512x512 image patches from large orthomosaics for manual labeling.

**When to run:** First step — when you have new drone orthomosaics and need to create training data.

**Input:**
- Drone orthomosaics from `data/raw_drone_orthos/`:
  - `TLBC_D95_ATTANUR_1_ortho.tif`
  - `TLBC_D95_ATTANUR_4_ortho.tif`

**How it works:**
- Scans orthomosaic with 1024px stride (grid pattern)
- Uses Hough transform to detect linear features (canal candidates)
- Ranks patches by number of detected lines
- Saves top 50 patches per image

**Output:**
- `labels/annotation_v2/ATTANUR_1_v2_001.jpg` — JPEG for viewing
- `labels/annotation_v2/ATTANUR_1_v2_001_raw.npy` — Raw numpy array (512x512x3)
- Prints: "Found X candidates" → "Saving top 50 patches"

**Next step:** Open these images in LabelMe for annotation.

---

### File 2: `extract_patches_for_labeling.py`
**What it does:** Alternative patch extractor (older version) — extracts from specific regions.

**When to run:** If you need patches from specific areas (left half, center, etc.).

**Input:** Same orthomosaics as above

**How it works:**
- Similar to v2 but with manual region selection
- Extracts from predefined regions (left 50%, center 25-75%)

**Output:** Patches in `labels/annotation_v2/`

**Use:** Legacy file — prefer `extract_patches_v2.py` for new work.

---

### File 3: `label_full_image.py`
**What it does:** Prepares full-resolution orthomosaic tiles for labeling (not patches).

**When to run:** When you want to label full-resolution sections instead of small patches.

**Input:** Full orthomosaics

**Output:** Full tiles in labeling directory

**Use:** Rarely used — patches are preferred for training.

---

### File 4: `extract_from_labels.py`
**What it does:** Extracts training data FROM already labeled LabelMe JSON files.

**When to run:** After you have LabelMe annotations and want to create training dataset.

**Input:**
- `labels/annotation_v2/*.json` — LabelMe annotation files
- `labels/annotation_v2/*_raw.npy` — Corresponding raw patches

**How it works:**
- Reads JSON polygons
- Creates binary masks (canal=1, background=0)
- Saves mask images

**Output:**
- `labels/annotation_v2/masks/*_mask.png` — Binary masks

**Next step:** Use these masks for training in `05_binary_training.ipynb`.

---

### File 5: `03_canal_extraction.py` (and `.ipynb`)
**What it does:** Classical (non-ML) canal extraction using image processing.

**When to run:** For quick canal detection without training a model.

**Input:**
- `data/raw_drone_orthos/TLBC_D95_ATTANUR_1_ortho.tif`

**How it works (algorithm):**
1. **VARI calculation:** (G-R)/(G+R-B) — detects vegetation
2. **Concrete detection:** Grey + bright + low VARI
3. **FIC detection:** Earthen features (warm tones, moderate brightness)
4. **Skeletonization:** Thins canals to centerlines
5. **Width measurement:** Distance transform from skeleton
6. **FIC vs bund separation:** Connectivity to concrete + width thresholds
7. **Blockage flags:** High VARI inside canal = vegetation blockage

**Output:**
- `outputs/03_extraction/ATTANUR1_canal_extraction.png` — 4-panel visualization
- `outputs/03_extraction/ATTANUR1_skeleton_crop.png` — Detail view
- `outputs/03_extraction/extraction_stats.json` — Statistics

**Metrics saved:**
- concrete_pct, fic_pct, field_bund_pct
- veg_blockage_pct, dark_blockage_pct
- Width medians in pixels

---

### File 6: `04_dataset.ipynb`
**What it does:** Creates training dataset from annotated patches.

**When to run:** After labeling patches and before training.

**Input:**
- `labels/annotation_v2/*.json` — Annotations
- `labels/annotation_v2/*_raw.npy` — Raw patches

**How it works:**
- Splits data into train/validation sets
- Applies augmentations (rotation, flip, color jitter)
- Creates PyTorch Dataset class

**Output:**
- Dataset ready for training
- Visualizations of sample batches

---

### File 7: `04_patch_tiling.ipynb`
**What it does:** Handles tiling strategy for large orthomosaics.

**When to run:** When planning how to process full-resolution images.

**Input:** Full orthomosaics

**How it works:**
- Calculates tile sizes for GPU memory limits
- Plans overlap between tiles
- Prevents edge artifacts

**Output:** Tiling configuration

---

### File 8: `05_binary_training.ipynb`
**What it does:** Trains UNet model for canal segmentation.

**When to run:** After creating labeled dataset.

**Input:**
- `labels/annotation_v2/` — Labeled patches
- Masks from `extract_from_labels.py`

**How it works (Training pipeline):**
1. **Cell 1:** Installs segmentation-models-pytorch
2. **Cell 2:** Configures paths, device (CUDA/CPU), image normalization
3. **Cell 3:** Converts LabelMe JSON → binary masks
   - "canal", "distributary" → mask=1 (positive)
   - "fic", "not canal" → mask=0 (negative)
4. **Cell 4:** Training loop
   - Model: UNet with ResNet50 encoder
   - Loss: Focal Loss + Dice Loss + Boundary Loss
   - Augmentation: Copy-paste, rotation, flip, color jitter, cutout
   - Weighted sampling: Canal positives 8x, FIC negatives 3x
   - Round 1: 30 epochs with early stopping
   - Round 2: Self-training with pseudo-labels
5. **Cell 5:** Full-mosaic inference with TTA
   - TTA: Test-time augmentation (4 rotations)
   - Sliding window with overlap
   - Threshold 0.4 for binary mask
6. **Cell 6:** Patch-level predictions for validation

**Output:**
- `config/models/canal_segmentation.pth` — Best model weights
- `outputs/05b_binary/` — Training logs, validation images
- `outputs/05b_binary/full_mosaic_inference_v5.png` — Full prediction

**Training metrics:**
- IoU (Intersection over Union)
- Precision, Recall, F1 score
- Best model saved at highest validation IoU

---

### File 9: `predict.py` (Task 1)
**What it does:** Runs trained model on new orthomosaics.

**When to run:** After training is complete, to process new images.

**Input:**
- Orthomosaics: `data/raw_drone_orthos/*.tif`
- Model: `config/models/canal_segmentation.pth`

**How it works:**
- Loads trained UNet model
- Processes full mosaic with sliding window
- Applies TTA (4 rotations)
- Generates probability map
- Thresholds to binary mask
- Creates visualizations

**Output:**
- `outputs/1_canal_masks/ATTANUR1_canal_extraction.png`
- Canal masks, stats JSON

---

### File 10: `inject_corridor.py`
**What it does:** Utility to inject corridor filter code into notebook cells.

**When to run:** Automatically — modifies `07_blockage_detection.ipynb`.

**Input:** `07_blockage_detection.ipynb`

**Output:** Modified notebook with corridor filter

**Use:** One-time utility script — not run manually.

---

## TASK 2: Blockage Percentage Classification

**Location:** `pipeline/task2_blockage_percentage/`

### File 11: `line_blockage.py`
**What it does:** Creates 50m reaches along canal and classifies by vegetation.

**When to run:** First step for blockage detection — creates baseline reaches.

**Input:**
- Canal alignment: `data/alignments/TLBC_D95/TLBC_D95.shp`
- Orthomosaics: `data/raw_drone_orthos/*.tif`

**How it works:**
1. Reprojects canal lines to orthomosaic CRS
2. Orients lines (chainage starts at junction)
3. Interpolates points every 2m along line
4. Creates 50m reaches
5. Extracts corridor (±2.5m from centerline)
6. Calculates vegetation % in corridor
7. Classifies: clear/partial/blocked

**Output:**
- `outputs/2_blockage_maps/tlbc_blockage_reaches.shp`
- `outputs/2_blockage_maps/tlbc_blockage_chainage_summary.json`
- `outputs/2_blockage_maps/*_bbox_chainage_map.png` — Visual maps

**Modes:** `python line_blockage.py tlbc` OR `python line_blockage.py nrbc`

---

### File 12: `extract_features.py`
**What it does:** Extracts 18 features and trains ML classifier.

**When to run:** After manually labeling some reaches.

**Input:**
- Labeled reaches: `outputs/2_blockage_maps/tlbc_blockage_reaches.shp`
- All segments: `data/alignments/TLBC_alignment/tlbc_segments.shp`
- Orthomosaics

**18 Features extracted:**
1. veg (vegetation fraction in main corridor)
2. water (water fraction)
3. bright (bright pixels fraction)
4. soil (soil fraction)
5. exg_mean (mean Excess Green index)
6. exg_std (std Excess Green)
7. sat_mean (mean saturation)
8. val_mean (mean value)
9. gray_std (std grayscale)
10. vari_mean (mean VARI)
11. veg_c1 (veg in inner corridor)
12. open_c1 (non-veg in inner corridor)
13. water_c1 (water in inner corridor)
14. veg_c2 (veg in outer corridor)
15. veg_diff (veg difference outer-inner)
16. edges_mean (edge density)
17. sobel_mean (gradient magnitude)
18. hue_std (hue variation)

**Models trained:**
- Random Forest (400 trees)
- Extra Trees (400 trees)
- Histogram Gradient Boosting
- MLP Neural Net (64,32 hidden layers)
- Voting Ensemble (soft voting)

**Output:**
- `config/models/tlbc_blockage_classifier.pkl` — Best model
- `outputs/2_blockage_maps/_tlbc_feats.npz` — Cached features
- Updated `data/alignments/TLBC_alignment/tlbc_segments.shp` with predictions

---

### File 13: `predict_segments.py`
**What it does:** Simpler vegetation-based prediction (no ML training).

**When to run:** Quick prediction without training — uses simple thresholds.

**Input:**
- Segments: `data/alignments/TLBC_alignment/tlbc_segments.shp`
- Orthomosaics

**How it works:**
- Calculates vegetation % in corridor
- Classifies by thresholds:
  - ≤5%: clear
  - 6-30%: partial
  - 31-80%: moderate
  - >80%: full/blocked

**Output:** Updated segments shapefile with `predicted` field.

---

### File 14: `train_blockage.py`
**What it does:** Trains classifier for NRBC reaches (15 features).

**When to run:** For NRBC canal blockage classification.

**Input:**
- `data/alignments/NRBC_alignment/nrbc_segments.shp`
- Orthomosaics

**Output:**
- `outputs/2_blockage_maps/nrbc_blockage_trained.shp`
- Trained model

**Difference from extract_features.py:** Uses 15 features (simpler set) for field-verified NRBC data.

---

### File 15: `train_predict_segments.py`
**What it does:** Combined training + prediction for TLBC.

**When to run:** End-to-end training with cross-validation.

**Input:** Labeled reaches + all segments

**Output:** Predicted segments with 4-class labels

---

### File 16: `zoom_nrbc.py`
**What it does:** Diagnostic tool — zooms into NRBC canal sections.

**When to run:** To verify corridor sampling is working correctly.

**Input:** NRBC orthomosaic + alignment

**Output:** `outputs/2_blockage_maps/_nrbc_zoom_check.png` — Grid of zoomed crops with corridor overlay

---

### File 17: `box_images.py` (and `box_images_nrbc_v2.py`)
**What it does:** Creates numbered box images for manual review.

**When to run:** To generate images for validation/QGIS.

**Input:** Segments + orthomosaics

**Output:**
- `outputs/5_production_packages/*/images/*.jpg`
- Numbered crops with status overlay

**Note:** Merged functionality into `export_production_pack.py`.

---

### File 18: `07_blockage_detection.ipynb` (and `blockage_new.ipynb`)
**What it does:** Notebook version of blockage detection pipeline.

**When to run:** Interactive development and experimentation.

**Input:** Reaches, segments, orthomosaics

**Output:** Classified reaches, visualizations

---

## TASK 3: Concrete Damage Detection

**Location:** `pipeline/task3_concrete_damage/`

### File 19: `predict.py` (Task 3)
**What it does:** Detects structural damage (rubble/broken concrete).

**When to run:** After canal segmentation to find damaged areas.

**Input:**
- Segments: `data/alignments/TLBC_alignment/tlbc_segments.shp`
- Orthomosaics

**How it works:**
1. Measures local texture (standard deviation) in 7x7 window
2. Detects rocky areas: bright, low saturation, high texture
3. Calculates vegetation + rocky fractions
4. Flags damage if rocky fraction > 95th percentile threshold

**Output:**
- Updated `data/alignments/TLBC_alignment/tlbc_segments.shp` with `damage` field
- Prints: blockage distribution + damage count

---

### File 20: `06_fic_training.ipynb`
**What it does:** Trains model for FIC (Field Irrigation Channel) detection.

**When to run:** To detect small earthen channels (not concrete canals).

**Input:** `labels/annotation_v2/` with "fic" labels

**Output:**
- `config/models/concrete_damage.pth` (or fic model)
- `outputs/06_fic/` — Training curves, predictions

**Difference from Task 1:** FICs are earthen (tan/brown), not concrete.

---

## TASK 4: Silt Depth Calculation

**Location:** `pipeline/task4_silt_depth/`

### File 21: `silt_calculator.py`
**What it does:** Calculates silt depth from DSM - DTM.

**When to run:** When you have elevation data and need silt depth.

**Input:**
- DSM: `data/raw_elevation/*_dsm.tif`
- DTM: `data/raw_elevation/*_dtm.tif`
- Canal mask: `outputs/1_canal_masks/*.tif`

**Usage:**
```bash
python pipeline/task4_silt_depth/silt_calculator.py \
  data/raw_elevation/ATTANUR_2_dsm.tif \
  data/raw_elevation/ATTANUR_2_dtm.tif \
  outputs/1_canal_masks/ATTANUR_2_mask.tif \
  outputs/3_silt_depth/ATTANUR_2_silt.tif
```

**How it works:**
- CHM = DSM - DTM (Canopy Height Model = silt depth)
- Classifies:
  - Class 1 (low): < 0.5m
  - Class 2 (medium): 0.5-1.0m
  - Class 3 (high): > 1.0m

**Output:**
- `outputs/3_silt_depth/*_silt.tif` — Classified raster (1,2,3)

---

### File 22: `refine_silt.py`
**What it does:** Removes off-canal silt detections.

**When to run:** After silt calculation to clean up false positives.

**Input:**
- Silt rasters: `outputs/3_silt_depth/*_silt.tif`
- Canal alignment: `data/alignments/TLBC_D95/TLBC_D95.shp`

**How it works:**
- Creates 25m buffer around canal centerline
- Keeps only silt pixels within buffer
- Discards off-canal detections

**Output:**
- `outputs/3_silt_depth/*_silt_refined.tif`
- `outputs/3_silt_depth/*_silt_refined_map.png` — Before/after comparison

---

### File 23: `fill_silt.py`
**What it does:** Fills gaps in silt raster using DSM-DTM.

**When to run:** When silt detection missed areas within canal.

**Input:**
- Refined silt: `outputs/3_silt_depth/*_silt_refined.tif`
- DSM/DTM
- Canal alignment

**How it works:**
- Finds gaps (0 values) in silt raster along canal
- Fills from CHM (DSM-DTM)
- Uses class thresholds learned from existing silt

**Output:**
- `outputs/3_silt_depth/*_silt_filled.tif`
- `outputs/3_silt_depth/*_silt_filled.png`

---

### File 24: `08_silt_detection.ipynb` (and `08B_validate_silt.ipynb`)
**What it does:** Notebook versions of silt detection.

**When to run:** Interactive silt processing and validation.

**Input:** DSM, DTM, masks

**Output:** Silt rasters, validation reports

---

## TASK 5: Farm Pond Detection

**Location:** `pipeline/task5_farm_ponds/`

### File 25: `build_dataset.py` (and `01_build_dataset.py`)
**What it does:** Builds training dataset from Sentinel-2.

**When to run:** First step for pond detection.

**Input:**
- Sentinel-2: `data/raw_satellite/S2*.SAFE/`
- Pond polygons: `data/training_labels/Farmponds/Farmpond_D95.shp`

**How it works:**
- Reads 6 bands: B2, B3, B4, B8 (10m) + B11, B12 (20m resampled to 10m)
- Creates 64x64 patches (640m × 640m)
- 80/20 split by pond ID (no leakage — same pond never in train+test)
- 6 offset crops per pond (augmentation)
- 150 negative samples per tile (random locations, no pond)

**Output:**
- `data/pond_dataset/dataset.npz` — X (patches), Y (masks), split, metadata

**Stats printed:**
- Total patches
- Train/test split
- Pond pixel percentage

---

### File 26: `train.py` (and `02_train.py`)
**What it does:** Trains UNet for pond segmentation.

**When to run:** After building dataset.

**Input:** `data/pond_dataset/dataset.npz`

**How it works:**
- Model: UNet with ResNet34 encoder
- Input: 6-band Sentinel-2
- Loss: Dice Loss + BCE with pos_weight=20 (handles class imbalance)
- Optimizer: AdamW, lr=3e-4
- Augmentation: Rotation, flip
- Evaluates every 5 epochs at multiple thresholds (0.3-0.7)
- Saves best model by F1 score

**Output:**
- `config/models/farm_pond_unet.pt` — Best weights
- `config/normalization_stats.npz` — Mean/std for 6 bands

**Metrics:**
- F1 score, IoU, Precision, Recall
- Ponds found (object-based metric)

---

### File 27: `03_inference.py`
**What it does:** Runs pond detection on full Sentinel-2 tiles.

**When to run:** After training to detect ponds in new areas.

**Input:**
- Sentinel-2: `data/raw_satellite/S2*.SAFE/`
- Model: `config/models/farm_pond_unet.pt`
- Normalization: `config/normalization_stats.npz`

**Usage:**
```bash
python pipeline/task5_farm_ponds/03_inference.py 0.5
```
(0.5 = probability threshold)

**How it works:**
- 1024x1024 sliding window
- Pad to multiple of 32 for UNet
- Cloud masking (SCL band)
- Vectorizes predictions to polygons
- Filters by area (>800 m²)

**Output:**
- `outputs/4_farm_ponds/<tile>/ponds.shp` — Detected ponds
- `outputs/4_farm_ponds/<tile>/overview.png` — Visualization

**Attributes:**
- id, area_m2, area_ha, circum_m

---

## Common Utilities

**Location:** `pipeline/common/`

### File 28: `utils.py`
**What it does:** Shared helper functions.

**Functions:**
- `get_project_root()` — Returns project root path
- `get_data_dir()`, `get_models_dir()`, `get_outputs_dir()` — Path helpers
- `get_raw_ortho_path(filename)` — Path to orthomosaic
- `get_alignment_path(filename)` — Path to shapefile
- `ensure_dir(path)` — Create directory if not exists
- `COLORS` — BGR color constants for visualization
- `classify_veg()`, `to_4class()` — Classification helpers

---

### File 29: `01_inspect_tifs.py`
**What it does:** Inspects TIFF metadata.

**When to run:** First step when you get new drone data.

**Input:** Orthomosaics in `data/raw_drone_orthos/`

**Output:**
- `outputs/01_inspect/metadata_summary.json`
- Thumbnail PNGs for each tile

**Shows:** Dimensions, bands, CRS, pixel size, coverage area, compression

---

### File 30: `02_spectral_analysis.py`
**What it does:** Analyzes spectral properties.

**When to run:** To understand spectral signatures of features.

**Input:** Orthomosaics

**Output:**
- `outputs/02_spectral/` — VARI maps, 3-class overlays, histograms
- RGB vs VARI scatter plots

**Detects:**
- Concrete canal (bright, grey, low VARI)
- FIC/earthen (warm tones, moderate brightness)
- Vegetation (high VARI)

---

### File 31: `build_report.py`
**What it does:** Generates HTML reports.

**When to run:** To create summary reports.

**Input:** Various outputs

**Output:** HTML report with embedded images

---

### File 32: `make_pngs.py`
**What it does:** Converts GeoTIFFs to PNGs for viewing.

**When to run:** To create web-friendly images.

**Input:** GeoTIFF rasters

**Output:** PNG images

---

## Production Export

### File 33: `export_production_pack.py`
**What it does:** Creates final client deliverable package.

**When to run:** After all processing is complete.

**Usage:**
```bash
python pipeline/export_production_pack.py tlbc
# OR
python pipeline/export_production_pack.py nrbc
```

**Input:**
- Segments: `data/alignments/TLBC_alignment/tlbc_segments.shp`
- Must have: `status` (clear/partial/moderate/blocked), `damage`, `blockpct` fields
- Orthomosaics

**Output (in `outputs/5_production_packages/<project>/`):**

| File | Description |
|------|-------------|
| `images/0001.jpg` | Numbered zoomed reach images (1100px max) |
| `blocks.shp` | QGIS-ready shapefile with all attributes |
| `boxes.geojson` | Web-ready GeoJSON with clickable boxes |
| `index.csv` | Spreadsheet: id, canal, type, chainage, status, blockpct, lon, lat, image |
| `index.json` | Same data in JSON format |
| `overview_ATTANUR_*.png` | Overview maps with numbered boxes |

**Process:**
1. Reads segments shapefile
2. For each 50m reach:
   - Crops orthomosaic around reach
   - Overlays colored line (green/orange/red) for status
   - Adds label bar: "#ID STATUS BLOCK% CHAINAGE"
   - Resizes to max 1100px
   - Saves as numbered JPG
3. Creates bounding box polygons
4. Reprojects to WGS84 for GeoJSON
5. Generates index files

---

## Summary Table: Input → File → Output

| Task | If you have... | Run this file... | You get... |
|------|---------------|------------------|-----------|
| **Task 1** | Raw drone TIFFs | `extract_patches_v2.py` → Label → `predict.py` | Canal masks |
| **Task 2** | Canal shapefile | `line_blockage.py` → Label → `extract_features.py` | Classified reaches |
| **Task 3** | Segments + TIFFs | `predict.py` (task3) | Damage flags |
| **Task 4** | DSM + DTM | `silt_calculator.py` | Silt depth raster |
| **Task 5** | Sentinel-2 | `build_dataset.py` → `train.py` → `03_inference.py` | Pond shapefile |
| **Export** | Final segments | `export_production_pack.py` | Client package |
