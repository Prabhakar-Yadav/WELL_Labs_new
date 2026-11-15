# File Guide (Execution Order)

Files listed in the order you run them for each task — from input to output.

---

## TASK 1: Canal Detection — Step by Step

### Step 1: Inspect Raw Data
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `common/01_inspect_tifs.py` | `data/raw_drone_orthos/*.tif` | `outputs/01_inspect/metadata.json, thumbnails` | Check orthomosaic metadata, resolution, CRS |

### Step 2: Spectral Analysis (Optional)
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `common/02_spectral_analysis.py` | Orthomosaics | `outputs/02_spectral/VARI_maps.png` | Understand spectral signatures |

### Step 3: Extract Patches for Labeling
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `task1/extract_patches_v2.py` | ATTANUR_1, ATTANUR_4 orthos | `labels/annotation_v2/*.jpg, *.npy` | Extract 512x512 patches |

### Step 4: Label Patches (Manual)
| Tool | Input | Output | Purpose |
|------|-------|--------|---------|
| `labelme labels/annotation_v2/` | Extracted patches | `labels/annotation_v2/*.json` | Draw polygons: canal, fic, not_canal |

### Step 5: Create Masks from Labels
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `task1/extract_from_labels.py` | JSONs + raw patches | `labels/annotation_v2/masks/*.png` | Convert polygons to binary masks |

### Step 6: Train Canal Model
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `task1/05_binary_training.ipynb` | Labels + masks | `config/models/canal_segmentation.pth` | Train UNet (ResNet50) |

### Step 7: Predict on New Images
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `task1/predict.py` | Orthomosaics + model | `outputs/1_canal_masks/` | Run trained model |

---

## TASK 2: Blockage Classification — Step by Step

### Step 1: Create Reaches Along Canal
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `task2/line_blockage.py tlbc` | `data/alignments/TLBC_D95.shp` + orthos | `outputs/2_blockage_maps/tlbc_blockage_reaches.shp` | Create 50m reaches |

### Step 2: Label Reaches in QGIS (Manual)
| Tool | Input | Output | Purpose |
|------|-------|--------|---------|
| QGIS | `tlbc_blockage_reaches.shp` | Edited SHP with `status` field | Label as clear/partial/blocked |

### Step 3: Train ML Classifier
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `task2/extract_features.py` | Labeled reaches + orthos | `config/models/tlbc_blockage_classifier.pkl` | Extract 18 features, train RF/GB/MLP |

### Alternative: Simple Prediction (Skip Training)
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `task2/predict_segments.py` | `data/alignments/tlbc_segments.shp` + orthos | Updated segments SHP | Simple veg% thresholds |

### For NRBC Canals
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `task2/train_blockage.py` | `data/alignments/nrbc_segments.shp` + orthos | `outputs/2_blockage_maps/nrbc_blockage_trained.shp` | NRBC 15-feature classifier |

---

## TASK 3: Concrete Damage — Step by Step

### Step 1: Detect Damage
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `task3/predict.py` | `data/alignments/tlbc_segments.shp` + orthos | Updated segments with `damage` field | Texture analysis for rubble |

### Optional: Train FIC Model
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `task3/06_fic_training.ipynb` | `labels/annotation_v2/` with "fic" | `config/models/concrete_damage.pth` | Train for earthen FICs |

---

## TASK 4: Silt Depth — Step by Step

### Step 1: Calculate Silt
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `task4/silt_calculator.py` | DSM + DTM + canal_mask | `outputs/3_silt_depth/*_silt.tif` | DSM-DTM = silt depth |

### Step 2: Refine to Canal Corridor
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `task4/refine_silt.py` | Silt raster + alignment | `outputs/3_silt_depth/*_silt_refined.tif` | Remove off-canal |

### Step 3: Fill Gaps
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `task4/fill_silt.py` | Refined silt + DSM/DTM | `outputs/3_silt_depth/*_silt_filled.tif` | Fill gaps using CHM |

---

## TASK 5: Farm Ponds — Step by Step

### Step 1: Build Dataset
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `task5/build_dataset.py` | Sentinel-2 SAFE + pond SHP | `data/pond_dataset/dataset.npz` | 64x64 patches, 6 bands |

### Step 2: Train Model
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `task5/train.py` | `data/pond_dataset/dataset.npz` | `config/models/farm_pond_unet.pt` | Train UNet |

### Step 3: Run Inference
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `task5/03_inference.py` | Sentinel-2 + model | `outputs/4_farm_ponds/<tile>/ponds.shp` | Detect ponds |

---

## PRODUCTION EXPORT — Final Step

| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `export_production_pack.py tlbc/nrbc` | Final segments SHP + orthos | `outputs/5_production_packages/<project>/` | Client deliverable |

**Output includes:**
- `images/0001.jpg` — Numbered reach images
- `boxes.geojson` — Web map
- `blocks.shp` — QGIS file
- `index.csv` — Spreadsheet
- `overview_*.png` — Maps

---

## Other Files (Utilities/Alternatives)

| File | When to Use |
|------|-------------|
| `task1/extract_patches_for_labeling.py` | Alternative patch extractor |
| `task1/label_full_image.py` | Full-res tiles (rarely used) |
| `task1/03_canal_extraction.py` | Classical detection (no ML) |
| `task1/04_dataset.ipynb` | Dataset prep notebook |
| `task1/04_patch_tiling.ipynb` | Tiling strategy |
| `task1/inject_corridor.py` | Auto-modify notebook |
| `task2/train_predict_segments.py` | Combined train+predict |
| `task2/zoom_nrbc.py` | Diagnostic zoom views |
| `task2/box_images.py` | Create review images |
| `task4/08_silt_detection.ipynb` | Notebook silt calc |
| `task5/01_build_dataset.py` | Same as build_dataset.py |
| `task5/02_train.py` | Same as train.py |
| `task5/helpers/*.py` | Reports, galleries |
| `common/utils.py` | Shared functions |
| `common/build_report.py` | HTML reports |
| `common/make_pngs.py` | TIFF to PNG |
