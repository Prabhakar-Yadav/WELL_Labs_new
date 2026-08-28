#!/usr/bin/env python3
"""Export production package for TLBC/NRBC. Usage: python export_production_pack.py [tlbc|nrbc]"""
import sys, os, glob, json, csv, collections, numpy as np, rasterio, cv2, shapefile
from rasterio import Affine
from rasterio.transform import xy as rxy
from rasterio.warp import transform as wt
from rasterio.windows import Window
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))
from utils import get_project_root, get_raw_ortho_path, get_alignment_path, get_output_subdir, COLORS

MODE = sys.argv[1] if len(sys.argv) > 1 else 'tlbc'
ROOT = get_project_root()
CFG = {
    'nrbc': dict(
        shp=str(get_alignment_path('NRBC_alignment/nrbc_segments.shp')),
        out=str(get_output_subdir('5_production_packages/nrbc')),
        tiles={'NRBC_D10_Survey_1': str(get_raw_ortho_path('NRBC_D10_Survey_1_ortho.tif')),
               'NRBC_D10_MISSION_1': str(get_raw_ortho_path('NRBC_D10_MISSION_1_ortho.tif'))},
        verified=str(get_output_subdir('2_blockage_maps') / 'nrbc_blockage_reaches.shp')
    ),
    'tlbc': dict(
        shp=str(get_alignment_path('TLBC_D95/TLBC_D95.shp')),
        out=str(get_output_subdir('5_production_packages/attanur')),
        tiles={'ATTANUR_2': str(get_raw_ortho_path('TLBC_D95_ATTANUR_2_ortho.tif')),
               'ATTANUR_3': str(get_raw_ortho_path('TLBC_D95_ATTANUR_3_ortho.tif'))},
        verified=str(get_output_subdir('2_blockage_maps') / 'tlbc_blockage_reaches.shp')
    ),
}
c = CFG[MODE]; SHP, OUT, TILES, VER = c['shp'], c['out'], c['tiles'], c['verified']
HALF_W_M, REACH_M, SAMP_M, MARGIN_M, MAXPX, VIS = 2.5, 50.0, 2.0, 25.0, 1100, 2400
COL = {'blocked': COLORS['blocked'], 'partial': COLORS['partial'], 'clear': COLORS['clear']}
INSPECT = {'clear', 'blocked', 'partial'}
FT = cv2.FONT_HERSHEY_SIMPLEX
os.makedirs(OUT + '/images', exist_ok=True)

def kind(nm):
    n = nm.lower()
    return 'lateral' if 'lateral' in n else 'main' if 'canal' in n else 'distributary'

def _norm(s):
    s = str(s).strip().lower()
    return 'blocked' if 'block' in s else 'partial' if 'part' in s else 'clear' if 'clear' in s else None

VST = {}
if VER and glob.glob(VER):
    rr = shapefile.Reader(VER); ff = [x[0] for x in rr.fields[1:]]
    for rec in rr.records():
        d = dict(zip(ff, rec))
        if d['chainage'] is not None:
            t = _norm(d['status'])
            if t: VST[(str(d['canal']).strip(), round(float(d['chainage']), 1))] = t

def feats(rgb, mask):
    cp = mask > 0
    if cp.sum() < 20: return 0.0, 0.0
    r, g, b = [rgb[:,:,i].astype(np.float32) for i in range(3)]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = [hsv[:,:,i].astype(np.float32) for i in range(3)]
    exg = 2*g - r - b
    vari = (g - r) / (g + r - b + 1e-6)
    green = (h >= 35) & (h <= 85) & (s > 40) & (v > 50)
    veg = ((exg > 25) | ((vari > 0.1) & green)) & cp
    water = (v < 120) & (b > r) & (s > 15) & cp
    n = max(1, int(cp.sum()))
    return float(veg.sum()/n), float(water.sum()/n)

def classify(veg, water):
    if veg > 0.55 and water < 0.20: return 'blocked'
    if veg > 0.35: return 'partial'
    return 'clear'

def enhance(rgb, valid):
    f = rgb.astype(np.float32)
    for ch in range(3):
        pv = f[:,:,ch][valid] if valid is not None else f[:,:,ch].ravel()
        if pv.size: lo, hi = np.percentile(pv, [2,98]); f[:,:,ch] = np.clip((f[:,:,ch]-lo)/(hi-lo+1e-6)*255, 0, 255)
    return np.ascontiguousarray(f.astype(np.uint8))

raw = [(sr.record[0] if sr.record else '?', np.array(sr.shape.points, float)) for sr in shapefile.Reader(SHP).iterShapeRecords()]
CRS = rasterio.open(next(iter(TILES.values()))).crs
idx, ID, seen = [], 0, set()
w = shapefile.Writer(f'{OUT}/blocks', shapeType=shapefile.POLYGON)
w.field('id', 'N', 6); w.field('area', 'C', 24); w.field('canal', 'C', 40); w.field('type', 'C', 14)
w.field('chainage', 'N', 10, 1); w.field('status', 'C', 10); w.field('blockpct', 'N', 5, 0); w.field('image', 'C', 40)

for tag, op in TILES.items():
    if not glob.glob(op): print(tag, 'ortho missing', flush=True); continue
    with rasterio.open(op) as src:
        H, W = src.height, src.width; tr = src.transform; px_m = abs(tr.a)
        hw = max(2, int(HALF_W_M/px_m)); mg = max(4, int(MARGIN_M/px_m)); vs = max(1, max(H,W)//VIS); vh, vw = H//vs, W//vs
        thumb = src.read([1,2,3,4], out_shape=(4, vh, vw)); canvas = enhance(np.transpose(thumb[:3], (1,2,0)), thumb[3]==255).copy()
        ovboxes = []
        for nm, pts in raw:
            xu, yu = wt('EPSG:4326', CRS, pts[:,0].tolist(), pts[:,1].tolist()); xu, yu = np.array(xu), np.array(yu)
            chain = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xu), np.diff(yu)))]); total = chain[-1]
            if total < REACH_M: continue
            sc = np.arange(0, total, SAMP_M); sx = np.interp(sc, chain, xu); sy = np.interp(sc, chain, yu)
            srr, scl = rasterio.transform.rowcol(tr, sx.tolist(), sy.tolist()); srr, scl = np.array(srr), np.array(scl)
            for cc0 in np.arange(0, total, REACH_M):
                key = (str(nm).strip(), round(float(cc0), 1))
                if key in seen: continue
                m = (sc >= cc0) & (sc < cc0 + REACH_M); rr = srr[m]; cl = scl[m]
                insm = (rr >= 0) & (rr < H) & (cl >= 0) & (cl < W); rr, cl = rr[insm], cl[insm]
                if len(rr) < 3: continue
                r0 = max(0, rr.min()-mg); r1 = min(H, rr.max()+mg); a0 = max(0, cl.min()-mg); a1 = min(W, cl.max()+mg)
                if r1-r0 < 8 or a1-a0 < 8: continue
                d = src.read([1,2,3,4], window=Window(a0, r0, a1-a0, r1-r0)); valid = d[3] == 255
                if valid.mean() < 0.3: continue
                rgb = np.transpose(d[:3], (1,2,0)); mk = np.zeros(rgb.shape[:2], np.uint8)
                loc = np.array([(cl[i]-a0, rr[i]-r0) for i in range(len(rr))], np.int32)
                cv2.polylines(mk, [loc], False, 255, 2*hw); mk = ((mk>0) & valid).astype(np.uint8)
                veg, water = feats(rgb, mk); st = VST.get(key) if VST else classify(veg, water)
                if st is None: st = classify(veg, water)
                seen.add(key); ovboxes.append((int(cl.min()/vs), int(rr.min()/vs), int(cl.max()/vs), int(rr.max()/vs), st, None))
                if st not in INSPECT: continue
                ID += 1; img = f'images/{ID:04d}.jpg'
                crop = enhance(rgb, valid); cv2.polylines(crop, [loc], False, COL[st], 3)
                hh, ww = crop.shape[:2]; s2 = min(1.0, MAXPX/max(hh, ww))
                if s2 < 1: crop = cv2.resize(crop, (int(ww*s2), int(hh*s2)))
                crop = cv2.copyMakeBorder(crop, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
                cv2.putText(crop, f'#{ID}  {st.upper()}  {nm} {cc0:.0f}m  veg{int(veg*100)}%', (6,21), FT, 0.55, (255,255,255), 1)
                cv2.imwrite(f'{OUT}/{img}', cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
                ovboxes[-1] = (ovboxes[-1][0], ovboxes[-1][1], ovboxes[-1][2], ovboxes[-1][3], st, ID)
                rows, cols = [r0,r0,r1,r1], [a0,a1,a1,a0]; xs, ys = rxy(tr, rows, cols)
                lons, lats = wt(CRS, 'EPSG:4326', list(xs), list(ys)); corners = [[float(a), float(bb)] for a, bb in zip(lons, lats)]
                cxs, cys = rxy(tr, [(r0+r1)/2], [(a0+a1)/2]); clon, clat = wt(CRS, 'EPSG:4326', list(cxs), list(cys))
                w.poly([corners+[corners[0]]]); w.record(ID, tag, str(nm), kind(str(nm)), round(float(cc0), 1), st, int(veg*100), img)
                idx.append(dict(id=ID, area=tag, canal=str(nm), type=kind(str(nm)), chainage_m=round(float(cc0), 1), status=st, blockage_pct=int(veg*100), lon=round(float(clon[0]), 7), lat=round(float(clat[0]), 7), bbox_lonlat=corners, image=img))
        for x1,y1,x2,y2,st,bid in ovboxes:
            cv2.rectangle(canvas, (x1-2,y1-2), (x2+2,y2+2), COL[st][::-1], 2)
        for x1,y1,x2,y2,st,bid in ovboxes:
            if bid is not None:
                cv2.putText(canvas, str(bid), (x1, max(10,y1-3)), FT, 0.42, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(canvas, str(bid), (x1, max(10,y1-3)), FT, 0.42, (0,0,0), 1, cv2.LINE_AA)
        fig, ax = plt.subplots(figsize=(13, int(13*vh/vw)+1)); ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        nb = sum(1 for b in ovboxes if b[4]=='blocked'); npa = sum(1 for b in ovboxes if b[4]=='partial')
        ax.set_title(f'{tag} — numbered blockage boxes ({nb} blocked, {npa} partial)', fontsize=11); ax.axis('off')
        ax.legend(handles=[mpatches.Patch(color=np.array(COL[k])/255, label=k) for k in COL], loc='lower right')
        plt.tight_layout(); plt.savefig(f'{OUT}/overview_{tag}.png', dpi=150, bbox_inches='tight'); plt.close()
        print(f'{tag}: {len(ovboxes)} reaches, {ID} boxes so far', flush=True)
w.close()
open(f'{OUT}/blocks.prj', 'w').write('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')
json.dump(idx, open(f'{OUT}/index.json', 'w'), indent=2)
with open(f'{OUT}/index.csv', 'w', newline='') as f:
    cw = csv.writer(f); cw.writerow(['id', 'area', 'canal', 'type', 'chainage_m', 'status', 'blockage_pct', 'lon', 'lat', 'image'])
    for r in idx: cw.writerow([r['id'], r['area'], r['canal'], r['type'], r['chainage_m'], r['status'], r['blockage_pct'], r['lon'], r['lat'], r['image']])
fc = {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [r['bbox_lonlat']+[r['bbox_lonlat'][0]]]}, 'properties': {k: r[k] for k in ['id', 'area', 'canal', 'type', 'chainage_m', 'status', 'blockage_pct', 'lon', 'lat', 'image']}} for r in idx]}
json.dump(fc, open(f'{OUT}/boxes.geojson', 'w'), indent=2)
print(f'DONE {MODE}: {ID} box images -> {OUT}/images/ , boxes.geojson , index.json/csv , blocks.shp , overview_*.png')
