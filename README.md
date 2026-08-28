# Well-LAB-ALL- Production Pipeline

**AI-powered drone and satellite imagery analysis for canal health monitoring and water body detection.**

This system automatically detects:
- **Canal boundaries** (concrete + earthen channels)
- **Vegetation blockages** (clear / partial / moderate / blocked)
- **Structural damage** (cracks, erosion, rubble on banks)
- **Silt depth** (low / medium / high accumulation)
- **Farm ponds** (from Sentinel-2 satellite imagery)

Built for the TLBC D95 (Attanur) and NRBC canal networks in Karnataka, India — but adaptable to any canal system.

---

## Quick Start for New Users

### What You Need (Input Data)

| Data Type | File Format | Where to Put It |
|-----------|-------------|-----------------|
| **Drone orthomosaics** | `.tif` files | `data/raw_drone_orthos/` |
| **Canal alignment** | `.shp` shapefile | `data/alignments/` |
| **DSM/DTM** (for silt) | `.tif` elevation | `data/raw_elevation/` |
| **Sentinel-2** (for ponds) | `.SAFE` folders | `data/raw_satellite/` |

> **No coding required** for running trained models — just command-line commands.

---

## Pipeline Overview (5 Tasks)

```
┌─────────────────────────────────────────────────────────────────────┐
│  TASK 1: Canal Detection                                            │
│  Input: Drone orthomosaic (.tif)                                    │
│  Output: Canal masks, extraction stats                              │
│  Command: python pipeline/task1_canal_detection/predict.py          │
├─────────────────────────────────────────────────────────────────────┤
│  TASK 2: Blockage Classification                                    │
│  Input: Canal shapefile + orthomosaic                               │
│  Output: 4-class blockage (clear/partial/moderate/blocked)          │
│  Command: python pipeline/task2_blockage_percentage/line_blockage.py│
├─────────────────────────────────────────────────────────────────────┤
│  TASK 3: Structural Damage                                          │
│  Input: Canal segments + orthomosaic                                │
│  Output: Damage flags per segment                                   │
│  Command: python pipeline/task3_concrete_damage/predict.py          │
├─────────────────────────────────────────────────────────────────────┤
│  TASK 4: Silt Depth                                                 │
│  Input: DSM + DTM + canal mask                                      │
│  Output: Silt depth raster (low/medium/high)                        │
│  Command: python pipeline/task4_silt_depth/silt_calculator.py       │
├─────────────────────────────────────────────────────────────────────┤
│  TASK 5: Farm Pond Detection                                        │
│  Input: Sentinel-2 satellite imagery                                │
│  Output: Pond shapefile with area & perimeter                       │
│  Command: python pipeline/task5_farm_ponds/03_inference.py          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  PRODUCTION EXPORT                                                  │
│  Input: Final processed segments                                    │
│  Output: Client-ready package (images, maps, spreadsheet)           │
│  Command: python pipeline/export_production_pack.py tlbc            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
Well-LAB-ALL-/
│
├── config/
│   └── models/              ← Trained model weights (.pth, .pt, .pkl)
│
├── data/
│   ├── raw_drone_orthos/    ← Drone TIFFs (junction to _keep_locally)
│   ├── raw_elevation/       ← DSM, DTM files
│   ├── raw_satellite/       ← Sentinel-2 .SAFE folders
│   ├── alignments/          ← Canal shapefiles (.shp)
│   └── training_labels/     ← Labels, farm ponds, train lines
│
├── labels/
│   ├── annotation/          ← Original LabelMe annotations
│   └── annotation_v2/       ← v2 annotations + masks
│
├── pipeline/
│   ├── common/              ← Shared utilities (inspect, spectral, reports)
│   ├── task1_canal_detection/      ← Canal extraction scripts + notebooks
│   ├── task2_blockage_percentage/  ← Blockage ML + classification
│   ├── task3_concrete_damage/      ← Damage detection
│   ├── task4_silt_depth/           ← Silt calculation
│   ├── task5_farm_ponds/           ← Pond detection + helpers/
│   └── export_production_pack.py   ← Final client export
│
├── outputs/
│   ├── 1_canal_masks/       ← Canal detection outputs
│   ├── 2_blockage_maps/     ← Blockage classification
│   ├── 3_silt_depth/        ← Silt depth rasters
│   ├── 4_farm_ponds/        ← Pond detection results
│   └── 5_production_packages/  ← Final deliverables
│
├── requirements.txt         ← Python dependencies
├── FILE_GUIDE_SHORT.md      ← What each file does (execution order)
└── README.md                ← This file
```

---

## Step-by-Step Instructions

### Installation (One-Time Setup)

```bash
# 1. Install Python 3.9 or 3.10

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify GPU support (optional but recommended)
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True (if GPU available)
```

---

### Task 1: Detect Canals (No Training Required)

**If you already have a trained model:**

```bash
python pipeline/task1_canal_detection/predict.py
```

**To train a new model on your data:**

1. **Extract patches** from your orthomosaic:
   ```bash
   python pipeline/task1_canal_detection/extract_patches_v2.py
   ```

2. **Label the patches** using LabelMe:
   ```bash
   python -m labelme labels/annotation_v2/
   ```
   - Draw polygons around canals
   - Labels: "canal", "fic" (earthen), "not_canal"

3. **Create masks** from labels:
   ```bash
   python pipeline/task1_canal_detection/extract_from_labels.py
   ```

4. **Train the model** (run in Jupyter):
   ```bash
   jupyter notebook pipeline/task1_canal_detection/05_binary_training.ipynb
   ```
   - Follow cells in order
   - Best model saves to: `config/models/canal_segmentation.pth`

**Output:** `outputs/1_canal_masks/` — canal masks and statistics

---

### Task 2: Classify Blockages

**Step 1: Create canal reaches**
```bash
# For TLBC/Attanur canals:
python pipeline/task2_blockage_percentage/line_blockage.py tlbc

# For NRBC canals:
python pipeline/task2_blockage_percentage/line_blockage.py nrbc
```

**Step 2: Label some reaches in QGIS**
- Open `outputs/2_blockage_maps/tlbc_blockage_reaches.shp` in QGIS
- Edit the `status` field: clear / partial / blocked
- Save edits

**Step 3: Train classifier**
```bash
python pipeline/task2_blockage_percentage/extract_features.py
```
- Extracts 18 features per reach
- Trains Random Forest / Gradient Boosting / MLP
- Best model saves to: `config/models/tlbc_blockage_classifier.pkl`
- Predictions saved to: `data/alignments/TLBC_alignment/tlbc_segments.shp`

**Alternative (no training):** Simple vegetation thresholding
```bash
python pipeline/task2_blockage_percentage/predict_segments.py
```

---

### Task 3: Detect Structural Damage

```bash
python pipeline/task3_concrete_damage/predict.py
```

**What it does:**
- Analyzes texture along canal banks
- Detects rubble/broken concrete
- Adds `damage` field to segments shapefile

**Output:** Updated `data/alignments/TLBC_alignment/tlbc_segments.shp`

---

### Task 4: Calculate Silt Depth

**Prerequisites:** DSM (Digital Surface Model) and DTM (Digital Terrain Model)

```bash
# Step 1: Calculate silt
python pipeline/task4_silt_depth/silt_calculator.py \
  data/raw_elevation/ATTANUR_2_dsm.tif \
  data/raw_elevation/ATTANUR_2_dtm.tif \
  outputs/1_canal_masks/ATTANUR_2_mask.tif \
  outputs/3_silt_depth/ATTANUR_2_silt.tif

# Step 2: Refine to canal corridor
python pipeline/task4_silt_depth/refine_silt.py

# Step 3: Fill gaps
python pipeline/task4_silt_depth/fill_silt.py
```

**Output:** 
- `outputs/3_silt_depth/*_silt.tif` — Classified (1=low, 2=medium, 3=high)

---

### Task 5: Detect Farm Ponds (Satellite)

**Step 1: Build training dataset**
```bash
python pipeline/task5_farm_ponds/build_dataset.py
```

**Step 2: Train model**
```bash
python pipeline/task5_farm_ponds/train.py 40
```
(40 = number of epochs)

**Step 3: Run inference on new areas**
```bash
python pipeline/task5_farm_ponds/03_inference.py 0.5
```
(0.5 = probability threshold)

**Output:** `outputs/4_farm_ponds/<tile>/ponds.shp` — Detected ponds with area

---

### Production Export (Final Deliverable)

After all processing is complete, create the client package:

```bash
# For TLBC/Attanur:
python pipeline/export_production_pack.py tlbc

# For NRBC:
python pipeline/export_production_pack.py nrbc
```

**Output in `outputs/5_production_packages/<project>/`:**

| File | What It Is |
|------|------------|
| `images/0001.jpg` | Numbered zoomed images of each reach |
| `boxes.geojson` | Web-ready clickable map |
| `blocks.shp` | QGIS shapefile with all attributes |
| `index.csv` | Spreadsheet with ID, status, %, GPS coords |
| `overview_*.png` | Overview maps |

---

## Viewing Results

### In QGIS

1. **Load orthomosaic:** Layer → Add Raster Layer
2. **Load blockage shapefile:** `outputs/2_blockage_maps/*.shp`
3. **Style by status:**
   - Green = clear
   - Orange/Yellow = partial
   - Red = blocked

### Web Map

Open `boxes.geojson` in:
- [geojson.io](https://geojson.io) — instant web map
- Google Earth Pro — File → Import

---

## File Reference Guide

For detailed explanation of every file:

| Document | What's Inside |
|----------|---------------|
| `FILE_GUIDE_SHORT.md` | Every file explained in execution order |
| `pipeline/common/utils.py` | Shared helper functions |

---

## Troubleshooting

### "File not found" errors
- Check that orthomosaics are in `data/raw_drone_orthos/`
- Check that shapefiles are in `data/alignments/`

### "CUDA out of memory"
- Reduce batch size in training notebooks
- Use smaller tile sizes

### Model not loading
- Check that model files exist in `config/models/`
- Verify file paths in scripts

---

## Performance Metrics

| Task | Metric | Value |
|------|--------|-------|
| Canal Detection | IoU | 87.6% |
| Blockage Classification | Exact accuracy | ~85% |
| Blockage Classification | Within-1-class | ~95% |
| Farm Pond Detection | F1 Score | 0.739 |
| Farm Pond Detection | Ponds found | 85% |

---

## Author

**Prabhakar Yadav**  
prabhakaryadav0505@gmail.com  
Wells Lab — Canal Survey Project, Karnataka, India  
2026

---

## License

This project is for research and production use within Wells Lab. Contact the author for external licensing.
