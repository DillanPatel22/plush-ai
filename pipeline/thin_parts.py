"""Check legs and tail are wide enough to turn and stuff at a given plush size.

Usage: python3 thin_parts.py template.obj measures.json nose_to_tail_mm min_width_mm out.json
Width = smallest width (minimum Feret diameter) of each cross-section, scaled to plush size.
"""
import json
import sys

import numpy as np
import trimesh
from shapely.geometry import Polygon

src, meas, size_mm, min_w, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
m = trimesh.load(src, process=False)
ms = json.load(open(meas))
nose_to_tail = ms["nose_to_chest_front_mm"] + ms["body_length_mm"]
k = size_mm / nose_to_tail


def min_feret(poly):
    hull = poly.convex_hull
    pts = np.array(hull.exterior.coords)
    best = np.inf
    for i in range(len(pts) - 1):
        e = pts[i + 1] - pts[i]; n = np.array([-e[1], e[0]]); nn = np.linalg.norm(n)
        if nn == 0: continue
        n /= nn; proj = pts @ n
        best = min(best, proj.max() - proj.min())
    return best


def sections(origin, normal, keep):
    sec = m.section(plane_origin=origin, plane_normal=normal)
    if sec is None: return []
    planar, T = sec.to_2D()
    res = []
    for poly in planar.polygons_full:
        c3 = trimesh.transform_points(np.c_[np.array(poly.centroid.coords), np.zeros(1)], T)[0]
        if keep(c3): res.append((c3, min_feret(poly)))
    return res

report = {"plush_nose_to_tail_mm": size_mm, "scale": round(k, 4), "min_width_mm": min_w, "legs": [], "tail": []}
H = ms["withers_height_mm"]
for z in np.linspace(0.04 * H, 0.35 * H, 8):
    for c, w in sections([0, 0, z], [0, 0, 1], lambda c: c[0] >= 0):          # +X legs only (symmetric)
        report["legs"].append({"z_mm": round(z, 1), "y_mm": round(float(c[1]), 1), "plush_width_mm": round(w * k, 1)})
y_but = ms["nose_to_chest_front_mm"] + ms["body_length_mm"]
ymax = m.vertices[:, 1].max()
tail_len = ymax - y_but
for y in np.linspace(y_but + 15, ymax - 0.12 * tail_len, 8):          # last 12 % is the rounded tip cap
    for c, w in sections([0, y, 0], [0, 1, 0], lambda c: c[2] > 0.45 * H):
        report["tail"].append({"y_mm": round(float(y), 1), "plush_width_mm": round(w * k, 1)})
thin = [s for s in report["legs"] + report["tail"] if s["plush_width_mm"] < min_w]
report["thin_sections"] = thin
report["result"] = "PASS" if not thin else "THIN"
json.dump(report, open(out, "w"), indent=1)
lw = [s["plush_width_mm"] for s in report["legs"]]; tw = [s["plush_width_mm"] for s in report["tail"]]
print(f"scale {k:.3f}; legs min {min(lw):.1f} mm, tail min {min(tw):.1f} mm (limit {min_w} mm) -> {report['result']}")
for s in thin: print("  thin:", s)
