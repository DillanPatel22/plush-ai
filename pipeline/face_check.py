"""Face likeness check: the template head's side and front outlines against the reference photos, at true scale.

Side: photo scaled by withers height, nose tip and ground aligned. Front: photo scaled by eye spacing, eyes
aligned (ears are excluded from the mesh, so the front check uses the face between the ears and the head top).
Ratios compared (face weighted, limit 7 %): muzzle length / head length, head depth at the stop / head length,
muzzle depth / head length, skull width at eye level / eye spacing, muzzle width at mouth corner / eye spacing.

Usage: python3 face_check.py work_dir template.obj face_points.json k out_prefix
"""
import json, sys
import numpy as np, trimesh
from PIL import Image, ImageDraw
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import align_measure as am

W, tpl, fpj, k, pre = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
tr = json.load(open(f"{W}/ref/face_trace.json")); fp = json.load(open(fpj))
T = trimesh.load(tpl, process=False); V = T.vertices; F = np.asarray(T.faces)
tri = np.r_[F[:, :3], F[:, [0, 2, 3]]] if F.shape[1] == 4 else F

def raster(P2, w, h):
    im = Image.new("L", (w, h), 0); d = ImageDraw.Draw(im)
    for t in tri: d.polygon([tuple(P2[i]) for i in t], fill=255)
    return np.array(im) > 0

# ---- side: 1 px = 1 mm, image x = Y - nose, image row = top - Z
y0 = V[:, 1].min(); H = 800; Wd = 1200
mesh_side = raster(np.c_[V[:, 1] - y0, H - V[:, 2]], Wd, H)
ph = am.photo_mask(f"{W}/{tr['side_image']}"); rs, _ = am.measure_mask(ph, 1.0)
s = tr["withers_mm"] / rs["withers_height_mm"]; nose_px = np.where(ph.any(0))[0].min(); ground = rs["landmarks_px"]["ground"]
rr, cc = np.nonzero(ph)
ph_side = np.zeros((H, Wd), bool)
yy = ((cc - nose_px) * s).astype(int); zz = (H - (ground - rr) * s).astype(int)
ok = (yy >= 0) & (yy < Wd) & (zz >= 0) & (zz < H); ph_side[zz[ok], yy[ok]] = True
from scipy import ndimage
ph_side = ndimage.binary_closing(ph_side, iterations=2)

def head_profile(mask):
    """top and bottom of the head outline for Y in 0..260 mm (head region: above 420 mm)."""
    top = np.full(261, np.nan); bot = np.full(261, np.nan)
    for y in range(261):
        col = np.where(mask[: H - 420, y])[0]
        if len(col): top[y] = H - col.min(); bot[y] = H - col.max()
    return top, bot
tm, bm = head_profile(mesh_side); tp, bp = head_profile(ph_side)
def stop_y(top):           # steepest rise of the top line between 40 and 160 mm from the nose
    g = np.gradient(ndimage.uniform_filter1d(np.nan_to_num(top), 9)); return 40 + int(np.argmax(g[40:160]))
def ratios(top, bot):
    sy = stop_y(top); L = 240.0
    return {"stop_from_nose_over_L": sy / L, "depth_at_stop_over_L": (top[sy] - bot[sy]) / L,
            "muzzle_depth_mid_over_L": (top[sy // 2] - bot[sy // 2]) / L}
side_m, side_p = ratios(tm, bm), ratios(tp, bp)

# ---- front: scale photo by eye spacing; image col = X, row = top - Z
eye_sp_mm = 2 * fp["eye_spacing_mm_plush"] / 2 / k
fm = am.photo_mask(f"{W}/{tr['front_image']}")
(ex0, ey), (ex1, _) = tr["eye_front_px"]; sf = eye_sp_mm / abs(ex1 - ex0); cx = (ex0 + ex1) / 2
eye_z = fp["eye_R"][2]
def front_width_photo(z):
    r = int(round(ey - (z - eye_z) / sf)); c = np.where(fm[r])[0]
    runs = np.split(c, np.where(np.diff(c) > 1)[0] + 1); run = [u for u in runs if u.min() <= cx <= u.max()]
    return (run[0].max() - run[0].min()) * sf if run else np.nan
def front_width_mesh(z, ymax=230):
    sel = (np.abs(V[:, 2] - z) < 2.5) & (V[:, 1] < y0 + ymax); return 2 * np.abs(V[sel, 0]).max() if sel.any() else np.nan
mouth_z = eye_z - 0.9 * eye_sp_mm              # mouth-corner height (front photo: ~0.9 eye spacings under the eyes)
front_m = {"muzzle_width_over_eye_spacing": front_width_mesh(mouth_z, 120) / eye_sp_mm}
# face width between the ears at eye level, read off the front photo by eye (ear inner edges), see face_trace
front_p = {"muzzle_width_over_eye_spacing": tr.get("muzzle_width_front_px", np.nan) / abs(ex1 - ex0)}
skull_p = tr.get("face_width_eye_level_front_px", np.nan) / abs(ex1 - ex0)
front_m["skull_width_eye_level_over_eye_spacing"] = front_width_mesh(eye_z) / eye_sp_mm
front_p["skull_width_eye_level_over_eye_spacing"] = skull_p

res = {"side_mesh": side_m, "side_photo": side_p, "front_mesh": front_m, "front_photo": front_p, "dev_pct": {}}
for a, b in [(side_m, side_p), (front_m, front_p)]:
    for key in a:
        res["dev_pct"][key] = round(100 * (a[key] / b[key] - 1), 1) if np.isfinite(b[key]) else None
bad = [k_ for k_, v in res["dev_pct"].items() if v is not None and abs(v) > 7]
res["result"] = "PASS" if not bad else f"FAIL {bad}"
json.dump(res, open(pre + "_face_check.json", "w"), indent=1, default=float)
# overlay image (side head)
r0, r1 = H - 760, H - 470                         # Z 470..760 mm, Y 0..320 mm from the nose
ov = np.zeros((r1 - r0, 320, 3), np.uint8) + 255
ov[mesh_side[r0:r1, :320]] = [235, 200, 150]
pm = ph_side[r0:r1, :320]; edge = pm & ~ndimage.binary_erosion(pm, iterations=2)
ov[edge] = [200, 30, 30]
sm, sp_ = stop_y(tm), stop_y(tp)
ov[:, sm] = [60, 60, 200]; ov[:, sp_] = [200, 30, 30]
Image.fromarray(ov).resize((640, 2 * (r1 - r0))).save(pre + "_side_overlay.png")
for k_, v in res["dev_pct"].items(): print(f"  {k_:42s} {v:+.1f} %" if v is not None else f"  {k_:42s} n/a")
print("RESULT:", res["result"])
