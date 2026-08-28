#!/usr/bin/env python3
"""Detect structural damage (rubble/broken rock on the bank) per TLBC segment."""
import os, sys, glob, numpy as np, rasterio, cv2, shapefile, collections
from rasterio.windows import Window
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from utils import get_project_root, get_raw_ortho_path, get_alignment_path

ROOT = get_project_root()
ORTHOS = [
    get_raw_ortho_path('TLBC_D95_ATTANUR_2_ortho.tif'),
    get_raw_ortho_path('TLBC_D95_ATTANUR_3_ortho.tif'),
    get_raw_ortho_path('TLBC_D95_ATTANUR_1_ortho.tif'),
    get_raw_ortho_path('TLBC_D95_ATTANUR_4_ortho.tif'),
    get_raw_ortho_path('TLBC_D95_SHAKAPUR_ortho.tif'),
]

def cls(p): return 'clear' if p <= 5 else 'partial' if p <= 40 else 'moderate' if p <= 70 else 'full'

def measure(rgb, mk):
    cp = mk > 0
    if cp.sum() < 15: return None
    r, g, b = [rgb[:,:,i].astype(np.float32) for i in range(3)]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = [hsv[:,:,i].astype(np.float32) for i in range(3)]
    exg = 2*g - r - b
    vari = (g - r) / (g + r - b + 1e-6)
    green = (h >= 35) & (h <= 85) & (s > 40) & (v > 50)
    gray = (0.299*r + 0.587*g + 0.114*b)
    g32 = gray.astype(np.float32)
    mean = cv2.boxFilter(g32, -1, (7,7)); sq = cv2.boxFilter(g32*g32, -1, (7,7))
    lstd = np.sqrt(np.maximum(sq - mean*mean, 0.0))
    greenish = ((exg > 25) | ((vari > 0.1) & green))
    veg = greenish & (lstd > 14)
    water = (greenish & (lstd <= 14)) | ((v < 130) & (b > r) & (s > 15))
    base = ((exg < 12) & (v > 95) & (v < 235) & (s < 95) & (~water) & (~veg) & (lstd > 22)).astype(np.uint8)
    rocky = cv2.morphologyEx(base, cv2.MORPH_OPEN, np.ones((3,3), np.uint8)) > 0
    fr = lambda c, m: float((c & (m > 0)).sum()) / max(1, int((m > 0).sum()))
    return fr(veg, mk), fr(rocky, mk)

r = shapefile.Reader(str(get_alignment_path('TLBC_alignment/tlbc_segments.shp')))
flds = [x[0] for x in r.fields[1:]]
def _ct(d): return 'main' if 'tlbc' in str(d.get('canal', '')).lower() else 'distributary'
def _wd(d): return 14.0 if _ct(d) == 'main' else 5.0
segs = [dict(rec=dd, pts=np.array(sr.shape.points, float), width=_wd(dd), ctype=_ct(dd), vegc=None, rock=None) for sr in r.shapeRecords() for dd in [dict(zip(flds, sr.record))] if len(sr.shape.points) >= 2]
r.close()

for op in ORTHOS:
    if not glob.glob(str(op)) or not [s for s in segs if s['vegc'] is None]: continue
    with rasterio.open(op) as src:
        H, W = src.height, src.width; tr = src.transform; px_m = abs(tr.a); b = src.bounds
        for it in [s for s in segs if s['vegc'] is None]:
            pts = it['pts']; cx, cy = pts[:,0].mean(), pts[:,1].mean()
            if not (b.left <= cx <= b.right and b.bottom <= cy <= b.top): continue
            hw = max(2, int((it['width']/2) / px_m))
            rr, cc = rasterio.transform.rowcol(tr, pts[:,0].tolist(), pts[:,1].tolist()); rr, cc = np.array(rr), np.array(cc)
            ins = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W); rr, cc = rr[ins], cc[ins]
            if len(rr) < 2: continue
            r0 = max(0, rr.min()-hw); r1 = min(H, rr.max()+hw); a0 = max(0, cc.min()-hw); a1 = min(W, cc.max()+hw)
            if r1-r0 < 4 or a1-a0 < 4: continue
            d = src.read([1,2,3,4], window=Window(a0, r0, a1-a0, r1-r0)); valid = d[3] == 255
            if valid.mean() < 0.3: continue
            rgb = np.transpose(d[:3], (1,2,0)); loc = np.array([(cc[i]-a0, rr[i]-r0) for i in range(len(rr))], np.int32)
            mk = np.zeros(rgb.shape[:2], np.uint8)
            cv2.polylines(mk, [loc], False, 255, 2*hw); mk = ((mk>0) & valid).astype(np.uint8)
            m = measure(rgb, mk)
            if m is not None: it['vegc'], it['rock'] = m
    print(f'{op.name}: sampled {sum(1 for s in segs if s["vegc"] is not None)}/{len(segs)}', flush=True)

rocks = [s['rock'] for s in segs if s['rock'] is not None]
if rocks:
    T = max(0.04, float(np.percentile(rocks, 95)))
else:
    T = 0.08
w = shapefile.Writer(str(get_alignment_path('TLBC_alignment/tlbc_segments')), shapeType=shapefile.POLYLINE)
for fx in ['id', 'canal', 'ctype', 'width', 'chainage', 'predicted', 'actual', 'blockpct', 'damage']:
    if fx == 'id': w.field('id', 'N', 6)
    elif fx in ('width', 'chainage'): w.field(fx, 'N', 10, 1)
    elif fx == 'blockpct': w.field('blockpct', 'N', 5, 0)
    elif fx == 'damage': w.field('damage', 'C', 20)
    else: w.field(fx, 'C', 40 if fx == 'canal' else 14)
dist = collections.Counter(); byc = {}; nd = 0
for s in segs:
    d = s['rec']
    if s['vegc'] is None: pred = ''; pct = 0; dmg = ''
    else:
        pct = int(round(s['vegc']*100)); pred = cls(pct); dist[pred] += 1
        byc.setdefault(str(d.get('ctype', '')).strip(), collections.Counter())[pred] += 1
        dmg = 'Structural damage' if s['rock'] >= T else ''; nd += (dmg != '')
    w.line([[[float(x), float(y)] for x, y in s['pts']]])
    w.record(d.get('id'), d.get('canal'), s['ctype'], s['width'], d.get('chainage'), pred, pred, pct, dmg)
w.close()
print(f'threshold rocky>= {T:.3f} | blockage {dict(dist)} | DAMAGE flagged: {nd}')
for c in byc: print(f'  {c}: {dict(byc[c])}')
print('done -> tlbc_segments.shp (+damage field)')
