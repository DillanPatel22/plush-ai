"""Eye and nose placement on the template from the side and front reference photos (points read by eye,
stored in work/<breed>/ref/face_trace.json). Usage: face_points.py work_dir template.obj out.json k"""
import json, sys
import numpy as np, trimesh
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import align_measure as am
W, tpl, out, k = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
tr = json.load(open(f"{W}/ref/face_trace.json"))
T = trimesh.load(tpl, process=False)
side = am.photo_mask(f"{W}/{tr['side_image']}"); rs, _ = am.measure_mask(side, 1.0)
s_side = tr["withers_mm"] / rs["withers_height_mm"]; nose_px = np.where(side.any(0))[0].min(); ground = rs["landmarks_px"]["ground"]
front = am.photo_mask(f"{W}/{tr['front_image']}"); rows = np.where(front.any(1))[0]
s_front = float(T.vertices[:, 2].max()) / (rows.max() - rows.min())
y0 = T.vertices[:, 1].min()
ex = abs(tr["eye_front_px"][1][0] - tr["eye_front_px"][0][0]) / 2 * s_front
p = np.array([[ex, y0 + (tr["eye_side_px"][0] - nose_px) * s_side, (ground - tr["eye_side_px"][1]) * s_side]])
cp, dist, _ = trimesh.proximity.closest_point(T, p)
res = {"eye_R": cp[0].round(1).tolist(), "surface_dist_mm": round(float(dist[0]), 1),
       "eye_diam_mm_plush": round(tr["eye_width_front_px"] * s_front * k, 1),
       "nose_width_mm_plush": round(tr["nose_width_front_px"] * s_front * k, 1),
       "eye_spacing_mm_plush": round(2 * ex * k, 1), "source": "points read off the reference images; confirm on the first sample"}
json.dump(res, open(out, "w"), indent=1); print(res)
