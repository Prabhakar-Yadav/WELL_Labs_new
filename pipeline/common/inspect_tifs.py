#!/usr/bin/env python3
"""Step 1 — Inspect TIF files: metadata, bands, resolution, spatial coverage."""
import os, sys, json, numpy as np, rasterio
from rasterio.enums import ColorInterp
from rasterio.windows import Window
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from utils import get_project_root, get_raw_ortho_path, get_output_subdir

ROOT = get_project_root()
OUT_DIR = get_output_subdir('01_inspect')
FILES = [
    "TLBC_D95_ATTANUR_1_ortho.tif",
    "TLBC_D95_ATTANUR_2_ortho.tif",
    "TLBC_D95_ATTANUR_3_ortho.tif",
    "TLBC_D95_ATTANUR_4_ortho.tif",
    "TLBC_D95_SHAKAPUR_ortho.tif",
]

def band_stats_window(src, band_idx=1, step=10):
    data = src.read(band_idx, out_shape=(max(1, src.height // step), max(1, src.width // step)))
    valid = data[data != src.nodata] if src.nodata is not None else data.ravel()
    if valid.size == 0: return {}
    return {"min": float(valid.min()), "max": float(valid.max()), "mean": float(valid.mean()), "std": float(valid.std()), "dtype": str(data.dtype)}

def read_overview_rgb(src, target_px=1024):
    scale = max(1, max(src.width, src.height) // target_px)
    out_h, out_w = src.height // scale, src.width // scale
    n = src.count
    if n >= 3: rgb = src.read([1, 2, 3], out_shape=(3, out_h, out_w))
    elif n == 1: grey = src.read(1, out_shape=(out_h, out_w)); rgb = np.stack([grey, grey, grey])
    else: rgb = src.read(out_shape=(n, out_h, out_w)); rgb = np.stack([rgb[0], rgb[1 % n], rgb[2 % n]])
    out = np.zeros_like(rgb, dtype=np.uint8)
    for i in range(3):
        ch = rgb[i].astype(float); lo, hi = np.percentile(ch[ch > 0] if ch.any() else ch, [2, 98])
        if hi > lo: out[i] = np.clip((ch - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return np.transpose(out, (1, 2, 0))

summary = {}
for fname in FILES:
    fpath = get_raw_ortho_path(fname)
    tag = fname.replace("_ortho.tif", "")
    print(f"\n{'='*60}\n  {fname}\n{'='*60}")
    if not os.path.exists(fpath): print(f"  FILE NOT FOUND: {fpath}"); continue
    with rasterio.open(fpath) as src:
        meta = {"file": fname, "width_px": src.width, "height_px": src.height, "num_bands": src.count,
                "dtypes": [str(d) for d in src.dtypes], "nodata": src.nodata, "crs": str(src.crs),
                "driver": src.driver, "compression": src.compression.name if src.compression else "none",
                "overviews": src.overviews(1), "pixel_size_x_m": abs(src.transform.a), "pixel_size_y_m": abs(src.transform.e),
                "bounds": {"left": src.bounds.left, "bottom": src.bounds.bottom, "right": src.bounds.right, "top": src.bounds.top},
                "color_interp": [ColorInterp(src.colorinterp[i]).name for i in range(src.count)]}
        dx, dy = abs(src.transform.a), abs(src.transform.e)
        meta["coverage_km2"] = round(src.width * dx * src.height * dy / 1e6, 3)
        meta["band_stats"] = {f"band_{b}": band_stats_window(src, b, step=20) for b in range(1, src.count + 1)}
        thumb = read_overview_rgb(src, target_px=800)
    print(f"  Dimensions: {meta['width_px']} x {meta['height_px']} px")
    print(f"  Bands: {meta['num_bands']} -> {meta['color_interp']}")
    print(f"  CRS: {meta['crs']}")
    print(f"  Pixel size: {meta['pixel_size_x_m']:.4f} m")
    print(f"  Coverage: {meta['coverage_km2']} km²")
    summary[tag] = meta
    fig, ax = plt.subplots(figsize=(10, 8)); ax.imshow(thumb)
    ax.set_title(f"{tag}\n{meta['width_px']}x{meta['height_px']} px | {meta['pixel_size_x_m']:.4f} m/px | {meta['coverage_km2']} km²", fontsize=9)
    ax.axis("off"); plt.tight_layout()
    plt.savefig(OUT_DIR / f"{tag}_thumb.png", dpi=150); plt.close()

with open(OUT_DIR / "metadata_summary.json", "w") as f: json.dump(summary, f, indent=2)

fig, axes = plt.subplots(1, len(FILES), figsize=(5 * len(FILES), 6))
for ax, fname in zip(axes, FILES):
    tag = fname.replace("_ortho.tif", "")
    fpath = get_raw_ortho_path(fname)
    if not os.path.exists(fpath): continue
    with rasterio.open(fpath) as src: thumb = read_overview_rgb(src, target_px=400)
    m = summary[tag]; ax.imshow(thumb)
    ax.set_title(f"{tag}\n{m['pixel_size_x_m']:.4f} m/px | {m['coverage_km2']} km²", fontsize=7)
    ax.axis("off")
plt.suptitle("TLBC D95 Canal Survey — All Tiles Overview", fontsize=12, y=1.01)
plt.tight_layout(); plt.savefig(OUT_DIR / "all_tiles_overview.png", dpi=150, bbox_inches="tight"); plt.close()
print(f"\nCombined overview saved -> {OUT_DIR / 'all_tiles_overview.png'}")
print("\nDone.")
