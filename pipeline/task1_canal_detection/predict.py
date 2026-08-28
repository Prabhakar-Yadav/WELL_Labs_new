"""
Stage 3 — Classical canal extraction on ATTANUR_1
Steps:
  1. Load overview + full-res patches for dev
  2. Detect concrete canals (grey + bright + low VARI)
  3. Detect earthen candidates (FIC + field bunds)
  4. Skeletonize + distance transform -> measure width
  5. Connectivity filter -> separate FIC from field bunds
  6. Flag initial blockage candidates (VARI spike / width anomaly inside canal)
  7. Save annotated overlay + GeoTIFF masks
"""

import os, json, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from utils import get_project_root, get_raw_ortho_path, get_output_subdir

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
import cv2
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = get_project_root()
FOCUS_FILE = "TLBC_D95_ATTANUR_1_ortho.tif"
OUT_DIR = get_output_subdir("1_canal_masks")
FPATH = get_raw_ortho_path(FOCUS_FILE)

THUMB_PX = 2000
MIN_CANAL_AREA = 200

WIDTH_CONCRETE_MIN = 20
WIDTH_FIC_MIN = 4
WIDTH_FIC_MAX = 35
FIC_CONNECT_DIST = 30

def vari(r, g, b):
    d = g + r - b
    d = np.where(np.abs(d) < 1e-6, 1e-6, d)
    return np.clip((g - r) / d, -1, 1)

def stretch(rgb_hw3, lo=2, hi=98):
    out = np.zeros_like(rgb_hw3, dtype=np.uint8)
    for c in range(3):
        ch = rgb_hw3[:,:,c].astype(float)
        v = ch[ch > 0]
        if v.size == 0: continue
        pl, ph = np.percentile(v, [lo, hi])
        if ph > pl:
            out[:,:,c] = np.clip((ch-pl)/(ph-pl)*255, 0, 255).astype(np.uint8)
    return out

def read_overview(fpath, target_px=THUMB_PX):
    with rasterio.open(fpath) as src:
        W, H = src.width, src.height
        scale = max(1, max(W, H) // target_px)
        ow, oh = W // scale, H // scale
        data = src.read([1,2,3,4], out_shape=(4, oh, ow)).astype(np.float32)
        transform = src.transform
        crs = src.crs
    return data, scale, transform, crs

print("Loading overview ...")
data, scale, src_transform, crs = read_overview(FPATH)
r, g, b, a = data[0], data[1], data[2], data[3]
H, W = r.shape
valid = a > 128

rgb_s = stretch(np.stack([r,g,b], axis=2).astype(np.uint8))
vi = vari(r, g, b)
vi[~valid] = np.nan
bright = (r + g + b) / 3.0
grey = (np.abs(r-g) + np.abs(g-b)).astype(float)

print(f"  Overview size: {W}x{H}  (scale 1:{scale})")

print("Detecting concrete canals ...")
concrete_raw = (
    valid &
    (np.nan_to_num(vi) < 0.06) &
    (bright > 120) &
    (grey < 22)
).astype(np.uint8)

kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
concrete = cv2.morphologyEx(concrete_raw, cv2.MORPH_OPEN, kernel3)
concrete = cv2.morphologyEx(concrete, cv2.MORPH_CLOSE, kernel5)

concrete_bool = remove_small_objects(concrete.astype(bool), MIN_CANAL_AREA)
concrete = concrete_bool.astype(np.uint8)

print("Detecting earthen features (FIC + bunds) ...")
earthen_raw = (
    valid &
    (np.nan_to_num(vi) < 0.08) &
    (bright > 70) & (bright < 210) &
    (r > b + 4) &
    (grey >= 12) &
    ~concrete_bool
).astype(np.uint8)

earthen = cv2.morphologyEx(earthen_raw, cv2.MORPH_OPEN, kernel3)
earthen = cv2.morphologyEx(earthen, cv2.MORPH_CLOSE, kernel3)
earthen_bool = remove_small_objects(earthen.astype(bool), 100)
earthen = earthen_bool.astype(np.uint8)

print("Skeletonizing and measuring widths ...")

concrete_skel = skeletonize(concrete_bool)
concrete_dist = cv2.distanceTransform(concrete.astype(np.uint8)*255, cv2.DIST_L2, 5)
concrete_width = concrete_dist * 2

earthen_skel = skeletonize(earthen_bool)
earthen_dist = cv2.distanceTransform(earthen.astype(np.uint8)*255, cv2.DIST_L2, 5)
earthen_width = earthen_dist * 2

print("Separating FICs from field bunds ...")

connect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FIC_CONNECT_DIST*2+1, FIC_CONNECT_DIST*2+1))
concrete_dilated = cv2.dilate(concrete.astype(np.uint8), connect_kernel)

earthen_labels = label(earthen_bool, connectivity=2)
fic_mask = np.zeros_like(earthen_bool)
bund_mask = np.zeros_like(earthen_bool)

for region in regionprops(earthen_labels):
    comp = earthen_labels == region.label
    skel_comp = comp & earthen_skel
    if skel_comp.sum() == 0:
        bund_mask |= comp
        continue
    widths = earthen_width[skel_comp]
    med_w = np.median(widths)
    touches_concrete = bool((comp & concrete_dilated.astype(bool)).any())
    
    if touches_concrete and (WIDTH_FIC_MIN <= med_w <= WIDTH_FIC_MAX):
        fic_mask |= comp
    else:
        bund_mask |= comp

print("Flagging blockage candidates ...")
vi_filled = np.nan_to_num(vi, nan=0.0)
veg_blockage = concrete_bool & (vi_filled > 0.08)
bright_norm = bright / 255.0
dark_in_canal = concrete_bool & (bright_norm < 0.45)
fic_veg_blockage = fic_mask & (vi_filled > 0.10)

def pct(m): return round(m.sum() / valid.sum() * 100, 2) if valid.sum() else 0

stats = {
    "concrete_pct": pct(concrete_bool),
    "fic_pct": pct(fic_mask),
    "field_bund_pct": pct(bund_mask),
    "veg_blockage_pct": pct(veg_blockage),
    "dark_blockage_pct": pct(dark_in_canal),
    "fic_veg_block_pct": pct(fic_veg_blockage),
    "concrete_width_median_px": round(float(np.median(concrete_width[concrete_skel & (concrete_width > 0)])), 1) if (concrete_skel & (concrete_width > 0)).any() else 0,
    "fic_width_median_px": round(float(np.median(earthen_width[earthen_skel & fic_mask & (earthen_width > 0)])), 1) if (earthen_skel & fic_mask & (earthen_width > 0)).any() else 0,
}

print("\n" + "="*50)
print("EXTRACTION STATS")
print("="*50)
for k, v in stats.items():
    print(f"  {k:<35}: {v}")

with open(OUT_DIR / "extraction_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\nGenerating plots ...")

fig, axes = plt.subplots(2, 2, figsize=(18, 18))
fig.suptitle(f"Stage 3 — Canal Extraction  |  {FOCUS_FILE}", fontsize=13)

axes[0,0].imshow(rgb_s)
axes[0,0].set_title("RGB overview", fontsize=10); axes[0,0].axis("off")

overlay = rgb_s.copy()
overlay[concrete_bool] = [180, 180, 180]
overlay[fic_mask] = [210, 130, 40]
overlay[bund_mask] = [160, 80, 20]
axes[0,1].imshow(overlay)
axes[0,1].set_title("Canal classes\nGrey=concrete  Orange=FIC  Brown=field bund", fontsize=9)
axes[0,1].axis("off")
axes[0,1].legend(handles=[
    Patch(facecolor="#b4b4b4", label=f"Concrete ({stats['concrete_pct']:.1f}%)"),
    Patch(facecolor="#d28228", label=f"FIC ({stats['fic_pct']:.1f}%)"),
    Patch(facecolor="#a05014", label=f"Field bund ({stats['field_bund_pct']:.1f}%)"),
], loc="lower right", fontsize=8)

width_vis = np.zeros_like(concrete_width)
width_vis[concrete_bool] = concrete_width[concrete_bool]
width_vis[fic_mask] = earthen_width[fic_mask]
im = axes[1,0].imshow(width_vis, cmap="plasma", vmin=0, vmax=80)
axes[1,0].set_title(f"Width map (px)\nConcrete median: {stats['concrete_width_median_px']} px  ({stats['concrete_width_median_px']*scale*0.0462:.2f} m)\nFIC median: {stats['fic_width_median_px']} px  ({stats['fic_width_median_px']*scale*0.0462:.2f} m)", fontsize=8)
axes[1,0].axis("off")
plt.colorbar(im, ax=axes[1,0], fraction=0.04)

block_vis = rgb_s.copy()
block_vis[veg_blockage] = [0, 200, 0]
block_vis[dark_in_canal] = [255, 0, 0]
block_vis[fic_veg_blockage] = [255, 200, 0]
axes[1,1].imshow(block_vis)
axes[1,1].set_title("Blockage candidates\nGreen=veg in canal  Red=sediment/dark  Yellow=veg in FIC", fontsize=9)
axes[1,1].axis("off")
axes[1,1].legend(handles=[
    Patch(facecolor="#00c800", label=f"Veg in concrete ({stats['veg_blockage_pct']:.1f}%)"),
    Patch(facecolor="#ff0000", label=f"Sediment/dark ({stats['dark_blockage_pct']:.1f}%)"),
    Patch(facecolor="#ffc800", label=f"Veg in FIC ({stats['fic_veg_block_pct']:.1f}%)"),
], loc="lower right", fontsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR / "ATTANUR1_canal_extraction.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT_DIR / 'ATTANUR1_canal_extraction.png'}")

cy, cx = H//2, W//2
crop_h, crop_w = min(400, H), min(400, W)
sy, ey = cy - crop_h//2, cy + crop_h//2
sx, ex = cx - crop_w//2, cx + crop_w//2

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle("Centre crop — skeleton detail", fontsize=11)

axes[0].imshow(rgb_s[sy:ey, sx:ex])
axes[0].set_title("RGB"); axes[0].axis("off")

skel_vis = rgb_s[sy:ey, sx:ex].copy()
skel_vis[concrete_skel[sy:ey, sx:ex]] = [255, 255, 0]
skel_vis[earthen_skel[sy:ey, sx:ex] & fic_mask[sy:ey, sx:ex]] = [255, 128, 0]
axes[1].imshow(skel_vis)
axes[1].set_title("Skeleton\nYellow=concrete  Orange=FIC"); axes[1].axis("off")

block_crop = rgb_s[sy:ey, sx:ex].copy()
block_crop[veg_blockage[sy:ey, sx:ex]] = [0, 220, 0]
block_crop[dark_in_canal[sy:ey, sx:ex]] = [220, 0, 0]
axes[2].imshow(block_crop)
axes[2].set_title("Blockage candidates\nGreen=veg  Red=sediment"); axes[2].axis("off")

plt.tight_layout()
fig.savefig(OUT_DIR / "ATTANUR1_skeleton_crop.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT_DIR / 'ATTANUR1_skeleton_crop.png'}")

print("\nStage 3 complete.")
