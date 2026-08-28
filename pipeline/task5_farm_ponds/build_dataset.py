#!/usr/bin/env python3
"""Build farm pond dataset from Sentinel-2."""
import os, sys, glob, numpy as np, rasterio, shapefile
from rasterio.warp import transform as wt, reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from utils import get_project_root, get_data_dir

ROOT = get_project_root()
PATCH, N_OFF, TEST_FRAC, NEG_PER_TILE = 64, 6, 0.20, 150
OUT = str(get_data_dir() / 'pond_dataset')
os.makedirs(OUT, exist_ok=True)
SDIR = str(get_data_dir() / 'raw_satellite')

TILES = [
    ('June_T43QGU', glob.glob(f'{SDIR}/S2A_*T43QGU*.SAFE')),
    ('Dec_T43QGU', glob.glob(f'{SDIR}/S2C_*T43QGU*.SAFE')),
    ('Dec_T43PGT', glob.glob(f'{SDIR}/S2C_*T43PGT*.SAFE')),
]

r = shapefile.Reader(str(get_data_dir() / 'training_labels/Farmponds/Farmpond_D95.shp'))
shapes = r.shapes(); NP = len(shapes)
rng = np.random.default_rng(0)
perm = rng.permutation(NP)
n_test = int(NP * TEST_FRAC)
test_ids = set(perm[:n_test].tolist())
print(f'{NP} ponds -> train {NP-n_test} / test {n_test} (split by pond id)', flush=True)

def read6(SAFE, L, B, R, T):
    G10 = glob.glob(f'{SAFE}/GRANULE/*/IMG_DATA/R10m')[0]
    G20 = glob.glob(f'{SAFE}/GRANULE/*/IMG_DATA/R20m')[0]
    PFX = os.path.basename(glob.glob(f'{G10}/*_B02_10m.jp2')[0]).replace('_B02_10m.jp2', '')
    bands = []
    with rasterio.open(f'{G10}/{PFX}_B02_10m.jp2') as s0:
        win = from_bounds(L, B, R, T, s0.transform)
        wtr = s0.window_transform(win)
        H = int(round((T - B) / 10)); W = int(round((R - L) / 10))
    for c in ['B02', 'B03', 'B04', 'B08']:
        with rasterio.open(f'{G10}/{PFX}_{c}_10m.jp2') as s:
            bands.append(s.read(1, window=from_bounds(L, B, R, T, s.transform), out_shape=(H, W), boundless=True, fill_value=0).astype('float32'))
    for c in ['B11', 'B12']:
        with rasterio.open(f'{G20}/{PFX}_{c}_20m.jp2') as s:
            bands.append(s.read(1, window=from_bounds(L, B, R, T, s.transform), out_shape=(H, W), boundless=True, fill_value=0).astype('float32'))
    return np.stack(bands) / 10000.0, wtr, rasterio.open(f'{G10}/{PFX}_B02_10m.jp2').crs

X, Y, SPLIT, META = [], [], [], []
for tname, safes in TILES:
    if not safes: print(f'{tname}: no SAFE, skip', flush=True); continue
    SAFE = safes[0]
    G10 = glob.glob(f'{SAFE}/GRANULE/*/IMG_DATA/R10m')[0]
    with rasterio.open(glob.glob(f'{G10}/*_B02_10m.jp2')[0]) as s:
        tb, crs = s.bounds, s.crs
    polys = []
    for i, sh in enumerate(shapes):
        p = np.array(sh.points)
        ux, uy = wt('EPSG:4326', crs.to_string(), p[:,0].tolist(), p[:,1].tolist())
        ux, uy = np.array(ux), np.array(uy)
        if tb.left <= ux.mean() <= tb.right and tb.bottom <= uy.mean() <= tb.top:
            polys.append((i, ux, uy))
    if not polys: print(f'{tname}: 0 ponds', flush=True); continue
    axs = np.concatenate([a for _, a, b in polys]); ays = np.concatenate([b for _, a, b in polys])
    pad = PATCH * 10
    L, B, R, T = axs.min() - pad, ays.min() - pad, axs.max() + pad, ays.max() + pad
    img, wtr, _ = read6(SAFE, L, B, R, T)
    Himg, Wimg = img.shape[1:]
    inv = ~wtr
    geoms = [{'type': 'Polygon', 'coordinates': [[[a, b] for a, b in zip(ux, uy)]]} for _, ux, uy in polys]
    fullmask = geometry_mask(geoms, out_shape=(Himg, Wimg), transform=wtr, invert=True).astype('uint8')
    ntr = nte = 0
    for i, ux, uy in polys:
        split = 'test' if i in test_ids else 'train'
        pcx, pcy = inv * (ux.mean(), uy.mean()); pcx, pcy = int(pcx), int(pcy)
        for k in range(N_OFF):
            ox = 0 if k == 0 else int(rng.integers(-PATCH // 3, PATCH // 3))
            oy = 0 if k == 0 else int(rng.integers(-PATCH // 3, PATCH // 3))
            r0 = np.clip(pcy - PATCH // 2 + oy, 0, Himg - PATCH); c0 = np.clip(pcx - PATCH // 2 + ox, 0, Wimg - PATCH)
            patch = img[:, r0:r0 + PATCH, c0:c0 + PATCH]; ym = fullmask[r0:r0 + PATCH, c0:c0 + PATCH]
            if patch.shape[1:] != (PATCH, PATCH) or patch.mean() < 0.005 or ym.sum() == 0: continue
            X.append((patch * 10000).astype('uint16')); Y.append(ym); SPLIT.append(split)
            META.append((tname, i)); ntr += (split == 'train'); nte += (split == 'test')
    added = 0; tries = 0
    while added < NEG_PER_TILE and tries < NEG_PER_TILE * 8:
        tries += 1
        r0 = int(rng.integers(0, Himg - PATCH)); c0 = int(rng.integers(0, Wimg - PATCH))
        patch = img[:, r0:r0 + PATCH, c0:c0 + PATCH]; ym = fullmask[r0:r0 + PATCH, c0:c0 + PATCH]
        if patch.mean() < 0.02 or ym.sum() > 0: continue
        X.append((patch * 10000).astype('uint16')); Y.append(ym)
        SPLIT.append('train' if rng.random() < 0.8 else 'test'); META.append((tname, -1)); added += 1
    print(f'{tname}: {len(polys)} ponds -> train patches {ntr}, test patches {nte}, +{added} neg', flush=True)

X = np.stack(X); Y = np.stack(Y); SPLIT = np.array(SPLIT)
print(f'\nTOTAL {len(X)} patches | train {int((SPLIT=="train").sum())} | test {int((SPLIT=="test").sum())}', flush=True)
np.savez_compressed(f'{OUT}/dataset.npz', X=X, Y=Y, split=SPLIT, meta=np.array(META, dtype=object), test_ids=np.array(sorted(test_ids)))
print(f'DONE -> {OUT}/dataset.npz', flush=True)
