"""Compare front-view width ratios of the aligned mesh with the front reference photo."""
import sys, json, numpy as np, trimesh
sys.path.insert(0, '/home/claude/plush-ai/pipeline')
import align_measure as am
from PIL import Image, ImageDraw
mesh_path, ref_path, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
m = trimesh.load(mesh_path); v = np.asarray(m.vertices)
W = int(np.ptp(v[:, 0])) + 20; H = int(v[:, 2].max()) + 20
img = Image.new('L', (W, H), 0); d = ImageDraw.Draw(img)
xz = np.c_[v[:, 0] - v[:, 0].min() + 10, H - 10 - v[:, 2]]
for f in m.faces: d.polygon([tuple(xz[i]) for i in f], fill=255)
mm = np.array(img) > 0
ref = am.photo_mask(ref_path)
def widths(mask):
    rows = np.where(mask.any(1))[0]; top, bot = rows.min(), rows.max(); h = bot - top
    out = {}
    for name, (a, b) in {'head_width_band_80_95': (0.80, 0.95), 'chest_width_band_45_60': (0.45, 0.60), 'legs_band_10_25': (0.10, 0.25)}.items():
        r0 = int(bot - b * h); r1 = int(bot - a * h)
        w = [(np.ptp(np.where(mask[r])[0]) if mask[r].any() else 0) for r in range(r0, r1)]
        out[name] = round(float(max(w)) / h, 3)
    return out
a, b = widths(mm), widths(ref)
ok = True; res = {}
for k in a:
    dlt = a[k] / b[k] - 1; good = abs(dlt) <= 0.07; ok &= good
    res[k] = {"mesh": a[k], "reference": b[k], "diff_pct": round(dlt * 100, 1)}
    print(f"{k}: mesh {a[k]} vs ref {b[k]} ({dlt*100:+.1f} %) {'PASS' if good else 'FAIL'} (limit 7 %)")
json.dump(res, open(out_json, 'w'), indent=1)
Image.fromarray((~mm).astype(np.uint8) * 255).save(out_json.replace('.json', '_mesh.png'))
Image.fromarray((~ref).astype(np.uint8) * 255).save(out_json.replace('.json', '_ref.png'))
print('RESULT:', 'PASS' if ok else 'FAIL')
