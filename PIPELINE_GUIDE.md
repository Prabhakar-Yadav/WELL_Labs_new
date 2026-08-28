# Well-LAB-ALL- Pipeline Guide

This guide explains how to run each task in the pipeline, what inputs are needed, and what outputs are produced.

---

## Task 1: Canal Detection & Boundary Segmentation

**Purpose:** Detect concrete canals, FICs (Field Irrigation Channels), and field bunds from drone orthomosaics.

### Input Data Required
- Drone orthomosaic TIFFs (ATTANUR_1, ATTANUR_2, ATTANUR_3, ATTANUR_4, SHAKAPUR)
- Location: `data/raw_drone_orthos/`

### Pipeline Flow

#### Step 1: Inspect Raw Data
```bash
python pipeline/common/01_inspect_tifs.py
```
- **Input:** Orthomosaics in `data/raw_drone_orthos/`
- **Output:** `outputs/01_inspect/` — metadata JSON + thumbnail images
- **What it does:** Reads TIFF headers, shows resolution, bands, CRS, coverage area

#### Step 2: Spectral Analysis
```bash
python pipeline/common/02_spectral_analysis.py
```
- **Input:** Orthomosaics
- **Output:** `outputs/02_spectral/` — VARI maps, 3-class overlays (concrete/FIC/veg), RGB histograms
- **What it does:** Analyzes spectral signatures to distinguish concrete canals from vegetation

#### Step 3: Extract Patches for Labeling
```bash
python pipeline/task1_canal_detection/extract_patches_v2.py
```
- **Input:** Orthomosaics
- **Output:** `labels/annotation_v2/` — 512x512 patches saved as JPG + raw NPY
- **What it does:** Extracts patches likely to contain canals using linear feature detection

#### Step 4: Label Patches (Manual)
```bash
python -m labelme labels/annotation_v2/
```
- **Input:** Extracted patches
- **Output:** LabelMe JSON files with polygons for "canal", "fic", "not canal"
- **What it does:** Interactive GUI for drawing polygons around canal features

#### Step 5: Train Canal Detection Model
```bash
python pipeline/task1_canal_detection/05_binary_training.ipynb
# OR run as script:
python pipeline/task1_canal_detection/train.py
```
- **Input:** `labels/annotation_v2/` (labeled patches)
- **Output:** `config/models/canal_segmentation.pth` (trained UNet model)
- **What it does:** Trains UNet to segment canals vs background

#### Step 6: Run Canal Detection Prediction
```bash
python pipeline/task1_canal_detection/predict.py
```
- **Input:** Full orthomosaic TIFFs
- **Output:** `outputs/1_canal_masks/` — canal masks, extraction stats, visualization PNGs
- **What it does:** Applies trained model to detect canals across entire orthomosaic

---

## Task 2: Blockage Percentage Classification

**Purpose:** Classify 50m canal reaches as clear/partial/moderate/blocked using ML.

### Input Data Required
- Drone orthomosaics: `data/raw_drone_orthos/`
- Canal alignment shapefile: `data/alignments/TLBC_D95/TLBC_D95.shp`
- Hand-labeled reaches (optional): `outputs/2_blockage_maps/tlbc_blockage_reaches.shp`

### Pipeline Flow

#### Step 1: Generate Initial Blockage Reaches
```bash
python pipeline/task2_blockage_percentage/line_blockage.py tlbc
# OR for NRBC:
python pipeline/task2_blockage_percentage/line_blockage.py nrbc
```
- **Input:** Canal alignment shapefile + orthomosaics
- **Output:** `outputs/2_blockage_maps/tlbc_blockage_reaches.shp` + chainage maps
- **What it does:** Creates 50m reaches along canal, classifies by vegetation using rule-based thresholds

#### Step 2: Label Reaches in QGIS (Manual)
- Open `outputs/2_blockage_maps/tlbc_blockage_reaches.shp` in QGIS
- Edit `status` field: clear / partial / blocked
- Save edits

#### Step 3: Train Classifier with 18 Features
```bash
python pipeline/task2_blockage_percentage/extract_features.py
```
- **Input:** Labeled reaches + orthomosaics
- **Output:** 
  - `config/models/tlbc_blockage_classifier.pkl` (best model)
  - `outputs/2_blockage_maps/_tlbc_feats.npz` (cached features)
  - Updated `data/alignments/TLBC_alignment/tlbc_segments.shp` with predictions
- **What it does:** 
  - Extracts 18 features (veg%, water%, texture, edges, etc.) for each reach
  - Trains RF/ExtraTrees/HistGB/MLP/Ensemble
  - Compares models via 5-fold CV
  - Predicts 4-class blockage for all segments

#### Step 4: Predict Segments (Alternative/Refinement)
```bash
python pipeline/task2_blockage_percentage/predict_segments.py
```
- **Input:** Aligned segments shapefile + orthomosaics
- **Output:** Updated `data/alignments/TLBC_alignment/tlbc_segments.shp` with predicted field
- **What it does:** Simpler vegetation-fraction based prediction (no ML training)

#### Step 5: Train NRBC Blockage (for NRBC canals)
```bash
python pipeline/task2_blockage_percentage/train_blockage.py
```
- **Input:** `data/alignments/NRBC_alignment/nrbc_segments.shp` + orthomosaics
- **Output:** `outputs/2_blockage_maps/nrbc_blockage_trained.shp` + trained model
- **What it does:** 15-feature classifier for NRBC field-verified reaches

---

## Task 3: Concrete Damage Detection

**Purpose:** Detect structural damage (rubble/broken rock) on canal banks.

### Input Data Required
- Drone orthomosaics: `data/raw_drone_orthos/`
- Aligned segments: `data/alignments/TLBC_alignment/tlbc_segments.shp`

### Pipeline Flow

#### Step 1: Detect Damage
```bash
python pipeline/task3_concrete_damage/predict.py
```
- **Input:** Orthomosaics + tlbc_segments.shp
- **Output:** Updated `data/alignments/TLBC_alignment/tlbc_segments.shp` with `damage` field
- **What it does:** 
  - Measures texture (local standard deviation) along canal corridor
  - Detects rocky/rubble areas
  - Flags segments with `damage = 'Structural damage'` if rocky fraction > threshold

#### Step 2: Train FIC Model (Optional - for FIC detection)
```bash
python pipeline/task3_concrete_damage/06_fic_training.ipynb
```
- **Input:** `labels/annotation_v2/` patches with "fic" labels
- **Output:** `config/models/concrete_damage.pth` (Mask R-CNN or UNet)
- **What it does:** Trains model specifically for Field Irrigation Channel detection

---

## Task 4: Silt Depth Calculation

**Purpose:** Calculate silt depth from DSM-DTM difference.

### Input Data Required
- DSM (Digital Surface Model): `data/raw_elevation/`
- DTM (Digital Terrain Model): `data/raw_elevation/`
- Canal mask: `outputs/1_canal_masks/`

### Pipeline Flow

#### Step 1: Calculate Silt Depth
```bash
python pipeline/task4_silt_depth/silt_calculator.py <dsm.tif> <dtm.tif> <canal_mask.tif> <output.tif>
```
- **Input:** DSM, DTM, canal mask TIFFs
- **Output:** `outputs/3_silt_depth/<name>_silt.tif` — classified silt raster (1=low, 2=medium, 3=high)
- **What it does:** 
  - CHM = DSM - DTM
  - Classifies: <0.5m=low, 0.5-1m=medium, >1m=high
  - Masks to canal corridor only

#### Step 2: Refine Silt (Remove Off-Canal)
```bash
python pipeline/task4_silt_depth/refine_silt.py
```
- **Input:** `outputs/3_silt_depth/*_silt.tif` + canal alignment shapefile
- **Output:** `outputs/3_silt_depth/*_silt_refined.tif` + before/after maps
- **What it does:** Clips silt detections to canal corridor (±25m from centerline)

#### Step 3: Fill Silt Gaps
```bash
python pipeline/task4_silt_depth/fill_silt.py
```
- **Input:** Refined silt + DSM/DTM
- **Output:** `outputs/3_silt_depth/*_silt_filled.tif` + filled maps
- **What it does:** Fills gaps in silt raster using DSM-DTM along canal buffer

#### Step 4: Validate Silt (Optional)
```bash
python pipeline/task4_silt_depth/08B_validate_silt.ipynb
```
- **Input:** Silt rasters
- **Output:** Validation report
- **What it does:** Visual validation of silt classifications

---

## Task 5: Farm Pond Detection

**Purpose:** Detect farm ponds from Sentinel-2 satellite imagery.

### Input Data Required
- Sentinel-2 .SAFE folders: `data/raw_satellite/`
- Pond polygons: `data/training_labels/Farmponds/Farmpond_D95.shp`

### Pipeline Flow

#### Step 1: Build Dataset
```bash
python pipeline/task5_farm_ponds/build_dataset.py
```
- **Input:** Sentinel-2 tiles + pond shapefile
- **Output:** `data/pond_dataset/dataset.npz` — 64x64 patches with labels
- **What it does:**
  - Reads 6 bands (B2,B3,B4,B8,B11,B12) at 10m
  - Creates 80/20 train/test split by pond ID (no leakage)
  - Extracts patches around pond polygons + random negatives

#### Step 2: Train Pond Model
```bash
python pipeline/task5_farm_ponds/train.py [epochs]
```
- **Input:** `data/pond_dataset/dataset.npz`
- **Output:** 
  - `config/models/farm_pond_unet.pt` (best model)
  - `config/normalization_stats.npz` (mean/std for 6 bands)
- **What it does:** Trains UNet+ResNet34 on 6-band Sentinel-2 data

#### Step 3: Run Inference
```bash
python pipeline/task5_farm_ponds/03_inference.py [threshold]
```
- **Input:** Sentinel-2 .SAFE folders + trained model
- **Output:** `outputs/4_farm_ponds/<tile>/ponds.shp` + overview PNGs
- **What it does:**
  - Slides window over full Sentinel-2 tile
  - Predicts pond probability for each pixel
  - Vectorizes polygons, filters by area (>800 m²)
  - Saves shapefile with area, perimeter attributes

#### Step 4: Generate Report
```bash
python pipeline/task5_farm_ponds/helpers/04_report.py
```
- **Input:** Prediction results
- **Output:** `outputs/4_farm_ponds/report.html`
- **What it does:** Creates HTML report with metrics and visualizations

---

## Production Export

**Purpose:** Generate final deliverable packages for clients.

### For TLBC (Attanur)
```bash
python pipeline/export_production_pack.py tlbc
```
- **Input:** `data/alignments/TLBC_alignment/tlbc_segments.shp` (with predicted + damage fields)
- **Output:** `outputs/5_production_packages/attanur/`
  - `images/0001.jpg` — numbered zoomed reach images
  - `boxes.geojson` — clickable web map
  - `blocks.shp` — QGIS-ready shapefile
  - `index.csv` — spreadsheet with all attributes
  - `overview_*.png` — overview maps

### For NRBC
```bash
python pipeline/export_production_pack.py nrbc
```
- **Input:** `data/alignments/NRBC_alignment/nrbc_segments.shp`
- **Output:** `outputs/5_production_packages/nrbc/` — same structure as above

---

## Quick Reference: File Purposes

| File | Task | Input | Output | Purpose |
|------|------|-------|--------|---------|
| `01_inspect_tifs.py` | Common | Orthomosaics | Metadata JSON | Inspect TIFF metadata |
| `02_spectral_analysis.py` | Common | Orthomosaics | VARI maps | Spectral analysis |
| `extract_patches_v2.py` | Task 1 | Orthomosaics | JPG patches | Extract for labeling |
| `05_binary_training.ipynb` | Task 1 | Labeled patches | Model .pth | Train canal UNet |
| `predict.py` (task1) | Task 1 | Orthomosaics | Canal masks | Detect canals |
| `line_blockage.py` | Task 2 | Alignments + orthos | Reaches SHP | Create 50m reaches |
| `extract_features.py` | Task 2 | Reaches + orthos | Classifier .pkl | Train 18-feature ML |
| `predict.py` (task3) | Task 3 | Segments + orthos | Damage flags | Detect rubble |
| `silt_calculator.py` | Task 4 | DSM+DTM+mask | Silt raster | Calculate silt depth |
| `refine_silt.py` | Task 4 | Silt raster | Refined silt | Clip to corridor |
| `build_dataset.py` | Task 5 | Sentinel-2 | dataset.npz | Build training data |
| `train.py` (task5) | Task 5 | dataset.npz | Model .pt | Train pond UNet |
| `03_inference.py` | Task 5 | Sentinel-2 | ponds.shp | Detect ponds |
| `export_production_pack.py` | Export | Segments SHP | Production folder | Client deliverables |

---

## Directory Structure Summary

```
data/
  raw_drone_orthos/     ← Drone TIFFs (junction)
  raw_elevation/        ← DSM, DTM
  raw_satellite/        ← Sentinel-2 (junction)
  alignments/           ← Canal shapefiles
  training_labels/      ← Labels, farm ponds, train lines

labels/
  annotation/           ← Original LabelMe annotations
  annotation_v2/        ← v2 annotations

pipeline/
  task1_canal_detection/     ← Canal extraction scripts + notebooks
  task2_blockage_percentage/ ← Blockage ML scripts + notebooks
  task3_concrete_damage/     ← Damage detection scripts + notebooks
  task4_silt_depth/          ← Silt calculation scripts + notebooks
  task5_farm_ponds/          ← Pond detection scripts + notebooks
  common/                    ← Shared utilities
  export_production_pack.py  ← Final export

config/
  models/               ← Trained model weights (.pth, .pt, .pkl)

outputs/
  1_canal_masks/        ← Canal detection outputs
  2_blockage_maps/      ← Blockage classification outputs
  3_silt_depth/         ← Silt calculation outputs
  4_farm_ponds/         ← Pond detection outputs
  5_production_packages/ ← Final client deliverables
```
