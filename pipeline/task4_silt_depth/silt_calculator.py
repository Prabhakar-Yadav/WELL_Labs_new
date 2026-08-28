#!/usr/bin/env python3
"""Silt depth calculator from DSM - DTM. Usage: python silt_calculator.py [dsm.tif] [dtm.tif] [canal_mask.tif] [output.tif]"""
import sys, os, numpy as np, rasterio
from rasterio.enums import Resampling
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from utils import get_output_subdir

if len(sys.argv) < 5:
    print("Usage: python silt_calculator.py <dsm.tif> <dtm.tif> <canal_mask.tif> <output.tif>")
    sys.exit(1)

DSM_PATH, DTM_PATH, MASK_PATH, OUT_PATH = sys.argv[1:5]
THRESHOLD_LOW = 0.5
THRESHOLD_HIGH = 1.0

print(f"Loading DSM: {DSM_PATH}")
with rasterio.open(DSM_PATH) as dsm_src:
    dsm = dsm_src.read(1).astype(np.float32)
    dsm_nodata = dsm_src.nodata if dsm_src.nodata is not None else -9999
    dsm[dsm == dsm_nodata] = np.nan
    profile = dsm_src.profile

print(f"Loading DTM: {DTM_PATH}")
with rasterio.open(DTM_PATH) as dtm_src:
    dtm = dtm_src.read(1, out_shape=dsm.shape, resampling=Resampling.bilinear).astype(np.float32)
    dtm_nodata = dtm_src.nodata if dtm_src.nodata is not None else -9999
    dtm[dtm == dtm_nodata] = np.nan

print(f"Loading canal mask: {MASK_PATH}")
with rasterio.open(MASK_PATH) as mask_src:
    mask = mask_src.read(1, out_shape=dsm.shape, resampling=Resampling.nearest).astype(bool)

print("Calculating silt depth (DSM - DTM)...")
chm = np.where(np.isfinite(dsm) & np.isfinite(dtm), dsm - dtm, np.nan)

silt_class = np.zeros(dsm.shape, dtype=np.uint8)
silt_class[(chm > 0) & (chm < THRESHOLD_LOW) & mask] = 1
silt_class[(chm >= THRESHOLD_LOW) & (chm < THRESHOLD_HIGH) & mask] = 2
silt_class[(chm >= THRESHOLD_HIGH) & mask] = 3

print(f"Silt distribution: Class 1 (low): {(silt_class==1).sum()} px, Class 2 (medium): {(silt_class==2).sum()} px, Class 3 (high): {(silt_class==3).sum()} px")

profile.update(dtype=rasterio.uint8, count=1, compress='deflate', nodata=0)
with rasterio.open(OUT_PATH, 'w', **profile) as dst:
    dst.write(silt_class, 1)
print(f"Saved silt classification to: {OUT_PATH}")
