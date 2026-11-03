"""Airbrush / colour guide for every pattern piece, and the coat's colour zones (swatches).

Each flattened piece is drawn in its cutting orientation (pile down) and filled with the coat colours transferred
from the photo-textured 3D model (colour_map.py) - this is what the airbrush artist reproduces on the white stock
fur. The colours of all pieces are grouped into a few zones (weighted k-means) so the workshop gets a short list of
paint colours with where each one goes.

Usage: python3 colour_guide.py pieces_dir flat_dir seam_mesh.obj seam_mesh_colours.npy nap.json ear.json k out_prefix
Writes out_prefix_pieces.png (pieces at one scale, coloured), out_prefix_zones.json.
"""
import json, math, sys
from pathlib import Path
import numpy as np, trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

pdir, fdir, smesh, scol, napj, earj, k, pre = sys.argv[1:9]
k = float(k)
lay = json.load(open(Path(pdir) / "pieces.json"))["pieces"]; nap = json.load(open(napj))["pieces"]
ear = json.load(open(earj)).get("ear_rgb", [180, 140, 100])
S = trimesh.load(smesh, process=False); SC = np.load(scol).astype(float); tree = cKDTree(S.vertices)
PX = 3.0                                                     # pixels per plush mm
PAD = np.array([58, 48, 44])                                  # paw pads: a separate pad fabric (black/brown on a yellow Lab)
pieces = []
for pid, info in lay.items():
    if info.get("mirror_of") or not info.get("faces"): continue
    d = np.load(Path(fdir) / f"{pid}.npz"); V3, F, uv = d["V3"], d["F"], d["uv"].copy()
    r = nap[pid]
    if r["flip_u"]: uv[:, 0] *= -1
    a = r["rotation_rad"]; R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]); uv = uv @ R.T
    vc = SC[tree.query(V3 / k)[1]]
    fc = vc[F].mean(1)
    e1, e2 = uv[F[:, 1]] - uv[F[:, 0]], uv[F[:, 2]] - uv[F[:, 0]]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    n = 2 if any(i.get("mirror_of") == pid for i in lay.values()) else 1
    pad = "sole" in info["name"]
    if pad: fc = np.tile(PAD, (len(F), 1)).astype(float)       # paw pads: pad colour, not coat (not in the zones)
    pieces.append(dict(pid=pid, name=info["name"], uv=uv, F=F, fc=fc, area=area, n=n, pad=pad))
pieces.append(dict(pid="EARS", name="ear outer pieces (x2)", uv=None, F=None, fc=np.array([ear], float), area=np.array([1.0]), n=2, pad=True))

# colour zones: weighted k-means on face colours (CIE-ish: work in sqrt RGB to separate darks a little more)
coat = [p for p in pieces if p["uv"] is not None and not p["pad"]]
allc = np.vstack([p["fc"] for p in coat]); allw = np.concatenate([p["area"] * p["n"] for p in coat])
X = np.sqrt(allc / 255.0)
rng = np.random.default_rng(0); K = 4
cent = X[rng.choice(len(X), K, replace=False, p=allw / allw.sum())]
for _ in range(40):
    lab = np.argmin(((X[:, None] - cent[None]) ** 2).sum(2), 1)
    cent = np.array([np.average(X[lab == j], axis=0, weights=allw[lab == j]) if (lab == j).any() else cent[j] for j in range(K)])
order = np.argsort(-cent.mean(1)); cent = cent[order]
lab = np.argmin(((X[:, None] - cent[None]) ** 2).sum(2), 1)
rgb = np.clip(np.rint((cent ** 2) * 255), 0, 255).astype(int)
zones = []
off = 0
per_piece = {}
for p in coat:
    l_ = lab[off:off + len(p["fc"])]; off += len(p["fc"])
    per_piece[p["pid"]] = [float((p["area"][l_ == j]).sum() / p["area"].sum()) for j in range(K)]
for j in range(K):
    share = float(allw[lab == j].sum() / allw.sum())
    top = sorted(per_piece, key=lambda q: -per_piece[q][j])[:3]
    zones.append({"zone": j + 1, "hex": "#%02x%02x%02x" % tuple(rgb[j]), "rgb": rgb[j].tolist(), "area_share": round(share, 3),
                  "mostly_on": [f"{q} {lay[q]['name']}" for q in top if per_piece[q][j] > 0.05]})
json.dump({"zones": zones, "ear_rgb": ear, "pad_rgb": PAD.tolist(), "note": "colours from the photo-textured 3D model; baked paw/underside shadow lifted; paw pads separate"},
          open(pre + "_zones.json", "w"), indent=1)

# draw all pieces at one scale, coloured, with pile arrows
pad = 14; boxes = []
for p in pieces:
    if p["uv"] is None: continue
    mn, mx = p["uv"].min(0), p["uv"].max(0); boxes.append((p, mn, mx))
boxes.sort(key=lambda t: -(t[2][1] - t[1][1]))
W = 2400; x = y = row = 0; place = []
for p, mn, mx in boxes:
    w, h = (mx - mn) * PX + 2 * pad + [0, 26]
    if x + w > W: x = 0; y += row; row = 0
    place.append((p, mn, mx, x, y)); x += w; row = max(row, h)
H = int(y + row + 10)
img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
for p, mn, mx, x0, y0 in place:
    P = (p["uv"] - mn) * PX; P[:, 1] = (mx[1] - mn[1]) * PX - P[:, 1]; P += [x0 + pad, y0 + pad + 24]
    for t, c in zip(p["F"], p["fc"]):
        d.polygon([tuple(P[i]) for i in t], fill=tuple(int(v) for v in c))
    d.text((x0 + pad, y0 + 2), p["pid"] + (" x2" if p["n"] == 2 else ""), fill=(30, 30, 30), font=f)
    cx, cy = P.mean(0); d.line([(cx, cy - 18), (cx, cy + 18)], fill=(200, 60, 40), width=4)
    d.polygon([(cx, cy + 26), (cx - 8, cy + 12), (cx + 8, cy + 12)], fill=(200, 60, 40))
img.save(pre + "_pieces.png")
for z in zones: print(z)
print("RESULT:", "PASS" if len(zones) == K and abs(sum(z["area_share"] for z in zones) - 1) < 0.01 else "FAIL")
