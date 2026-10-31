"""Ear module from an ear outline traced on the side reference photo.

Usage: python3 ear_from_trace.py ear_trace.json side_ref.png template.obj withers_mm out_dir
"""
import json, sys
from pathlib import Path
import numpy as np, trimesh
from scipy.interpolate import splprep, splev
sys.path.insert(0, str(Path(__file__).parent))
import align_measure as am

tr = json.load(open(sys.argv[1])); ref = sys.argv[2]; T = trimesh.load(sys.argv[3], process=False)
withers = float(sys.argv[4]); out = Path(sys.argv[5]); out.mkdir(parents=True, exist_ok=True)
mask = am.photo_mask(ref); res, _ = am.measure_mask(mask, 1.0)
s = withers / res["withers_height_mm"]                          # mm per photo pixel
cols = np.where(mask.any(0))[0]; nose_px = cols.min(); ground_px = res["landmarks_px"]["ground"]
ox, oy = tr["crop_origin_px"]; zf = tr["crop_zoom"]
P = np.array(tr["points_crop"], float)
px = ox + P[:, 0] / zf; py = oy + P[:, 1] / zf
y_mm = (px - nose_px) * s; z_mm = (ground_px - py) * s
# template frame: nose at Y = min Y of template; ground at Z = 0
Ty0 = T.vertices[:, 1].min()
yz = np.c_[y_mm + Ty0, z_mm]
# smooth closed spline through the traced points, 2 mm spacing
tck, u = splprep([yz[:, 0], yz[:, 1]], s=len(yz) * 4.0, per=True)
L = np.linalg.norm(np.diff(yz, axis=0, append=yz[:1]), axis=1).sum()
uu = np.linspace(0, 1, int(L / 2.0), endpoint=False)
outline = np.c_[splev(uu, tck)]
outline = np.array(outline).T if outline.shape[0] == 2 else outline
# attachment edge: spline section between the first and last attach point
ua = u[tr["attach_points"]]; att = np.where((uu >= ua.min()) & (uu <= ua.max()))[0]
# Project the attachment edge onto the +X side of the skull. The traced top edge sits on the head's side
# silhouette (the ear base is high on the skull), so a closest-point search collapses it onto the widest
# part of the head. Instead: drop each point to just under the skull top at that Y and cast a ray from +X.
ray = trimesh.ray.ray_triangle.RayMeshIntersector(T)
def skull_top(yv):
    loc, idx, _ = ray.intersects_location(np.c_[np.zeros(len(yv)), yv, np.full(len(yv), 3000.0)],
                                          np.tile([0, 0, -1.0], (len(yv), 1)), multiple_hits=False)
    zt = np.full(len(yv), np.nan); zt[idx] = loc[:, 2]; return zt
def side_hit(yv, zv):
    loc, idx, _ = ray.intersects_location(np.c_[np.full(len(yv), 400.0), yv, zv],
                                          np.tile([-1.0, 0, 0], (len(yv), 1)), multiple_hits=False)
    P = np.full((len(yv), 3), np.nan); P[idx] = loc; return P
ATTACH_BELOW_TOP_MM = 25.0     # breed standards (UKC): Lab ears set "somewhat low on the skull; slightly above eye level"
ya, za = outline[att, 0], outline[att, 1]
za = np.minimum(za, skull_top(ya) - ATTACH_BELOW_TOP_MM)
on = side_hit(ya, za)
miss = np.isnan(on[:, 0])
if miss.any():
    on[miss] = trimesh.proximity.closest_point(T, np.c_[np.full(miss.sum(), 300.0), ya[miss], za[miss]])[0]
np.save(out / "ear_outline_mm.npy", outline); np.save(out / "ear_attach_idx.npy", att); np.save(out / "ear_attach_on_template.npy", on)
area = 0.5 * abs(np.dot(outline[:, 0], np.roll(outline[:, 1], 1)) - np.dot(outline[:, 1], np.roll(outline[:, 0], 1)))
c = outline - outline.mean(0); _, _, vt = np.linalg.svd(c, full_matrices=False); ext = np.ptp(c @ vt.T, axis=0)
info = {"mm_per_px": round(s, 4), "ear_area_cm2_real": round(area / 100, 1), "ear_length_mm_real": round(float(ext[0]), 1),
        "ear_width_mm_real": round(float(ext[1]), 1),
        "attach_line_mm_real": round(float(np.linalg.norm(np.diff(on, axis=0), axis=1).sum()), 1),
        "attach_on_template_x_mm": [round(float(on[:, 0].min()), 1), round(float(on[:, 0].max()), 1)]}
json.dump(info, open(out / "ear_module.json", "w"), indent=1)
print(info)
plausible = (100 <= info["ear_length_mm_real"] <= 170 and 60 <= info["ear_width_mm_real"] <= 120
             and 50 <= info["attach_line_mm_real"] <= 110)          # real Lab ear base is about 6-9 cm
print("RESULT:", "PASS" if plausible else "CHECK")
