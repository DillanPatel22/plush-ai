"""Flatten every piece of a seam layout (pieces.json from a seams_*.py script) at plush size.

Left pieces that mirror a right piece are not flattened again (the export mirrors them). Same method and checks
as flatten_pieces.py (v1): LSCM + ARAP, Tutte + SLIM if the outline folds; sliver triangles nudged first.

Usage: python3 flatten_layout.py pieces_dir measures.json plush_nose_to_tail_mm out_dir
"""
import json, sys
from pathlib import Path
import numpy as np, trimesh
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import Polygon
sys.path.insert(0, str(Path(__file__).parent))
from flatten import flatten, flatten_injective, distortion, fix_slivers

pdir, ms, size, out = Path(sys.argv[1]), json.load(open(sys.argv[2])), float(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
k = size / (ms["nose_to_chest_front_mm"] + ms["body_length_mm"])
layout = json.load(open(pdir / "pieces.json"))


def outline_ok(uv, bnd):
    P2 = uv[bnd]; seg = np.linalg.norm(np.diff(P2, axis=0, append=P2[:1]), axis=1).mean()
    sg = max(1.0, 1.5 / seg)
    Ps = np.c_[gaussian_filter1d(P2[:, 0], sg, mode="wrap"), gaussian_filter1d(P2[:, 1], sg, mode="wrap")]
    Pg = Polygon(Ps)
    if Pg.is_valid: return True
    Pc = Pg.buffer(0); sa = 0.5 * abs(np.dot(Ps[:, 0], np.roll(Ps[:, 1], -1)) - np.dot(Ps[:, 1], np.roll(Ps[:, 0], -1)))
    return Pc.geom_type == "Polygon" and abs(Pc.area - sa) / sa < 0.01


report = {"scale": round(k, 4), "plush_nose_to_tail_mm": size, "layout": layout.get("layout"), "pieces": {}}
for pid, info in layout["pieces"].items():
    if info.get("mirror_of") or not info.get("faces"): continue
    p = trimesh.load(pdir / f"{pid}.obj", process=False)
    V, F = np.asarray(p.vertices, float), np.asarray(p.faces, np.int64)
    Vk, slivers = fix_slivers(V * k, F)
    uv, bnd = flatten(Vk, F, iters=150); method = "lscm+arap"
    if not np.isfinite(uv).all() or not outline_ok(uv, bnd):
        uv, bnd = flatten_injective(Vk, F, iters=300); method = "tutte+slim"
    ar, lr, a3 = distortion(Vk, F, uv)
    L3 = np.linalg.norm(Vk[np.roll(bnd, -1)] - Vk[bnd], axis=1).sum(); L2 = np.linalg.norm(np.roll(uv[bnd], -1, 0) - uv[bnd], axis=1).sum()
    rec = {"name": info["name"], "faces": int(len(F)), "area_3d_cm2": round(float(a3.sum()) / 100, 1),
           "area_dev_p95": round(float(np.percentile(np.abs(ar - 1), 95)), 3), "len_dev_p95": round(float(np.percentile(np.abs(lr - 1), 95)), 3),
           "flipped": int((ar <= 0).sum()), "method": method, "sliver_fixes": int(slivers), "outline_valid": bool(outline_ok(uv, bnd)),
           "boundary_len_ratio": round(float(L2 / L3), 3)}
    report["pieces"][pid] = rec
    np.savez(out / f"{pid}.npz", V3=Vk, F=F, uv=uv, bnd=bnd, area_ratio=ar)
    print(pid, info["name"], {k_: rec[k_] for k_ in ["area_dev_p95", "boundary_len_ratio", "method", "outline_valid", "flipped"]})
json.dump(report, open(out / "flatten_report.json", "w"), indent=1)
bad = [p for p, r in report["pieces"].items() if r["flipped"] or not r["outline_valid"] or abs(r["boundary_len_ratio"] - 1) > 0.08]
print("RESULT:", "PASS" if not bad else f"FAIL {bad}")
