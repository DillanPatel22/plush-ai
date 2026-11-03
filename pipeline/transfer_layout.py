"""Carry a seam layout over to another breed's template (same vertex count and order: a reshape of the base mesh).

The layout's face labels live on a fixed subdivision of the template, so they transfer face for face; the seams are
then re-straightened on the new shape and the pieces written exactly as seams_standing_v2.py writes them. Same
pieces, same numbers, same seams - only the shape changes (spec: layouts are frozen per species and pose).

Breed modules (spec: swappable modules on the frozen layout), applied before the seams are straightened:
  --module pronounced_muzzle   flat-faced breeds (bulldogs, boxers, pugs, mastiffs): the muzzle side piece is split
      along the jowl crease into a muzzle top (nose side up to the stop) and a jowl (flews) piece, as bulldog
      plushes are built (jowls as their own lobes); the chin gusset becomes the chin / lower lip piece that sits
      proud for the undershot jaw. Split line from --jowl_line "y0,z0;y1,z1;..." (mm, Y from the nose tip).

Usage: python3 transfer_layout.py src_pieces_dir src_template.obj target_template.obj out_dir [--name "layout name"]
          [--module pronounced_muzzle --jowl_line "0,326;20,336;45,342;60,330;75,300"]
Acceptance: same face count as the source layout, every piece one disc (euler 1), symmetric.
"""
import json, sys
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, str(Path(__file__).parent))
from layout_tools import straighten, export_pieces, clean_labels

sdir, stpl, ttpl, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
name = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else None
S = trimesh.load(stpl, process=False); T = trimesh.load(ttpl, process=False)
assert len(S.vertices) == len(T.vertices) and (np.asarray(S.faces) == np.asarray(T.faces)).all(), "target is not a reshape of the source template"
m = trimesh.Trimesh(T.vertices, T.faces, process=False)
while len(m.faces) < 60000:                                     # the same subdivision the layout was labelled on
    v, f = trimesh.remesh.subdivide(m.vertices, m.faces); m = trimesh.Trimesh(v, f, process=False)
label = np.load(sdir / "face_labels.npy")
assert len(label) == len(m.faces), (len(label), len(m.faces))
src = json.load(open(sdir / "pieces.json"))
PIECES = {v["label"]: (pid, v["name"], v.get("mirror_of")) for pid, v in src["pieces"].items() if v.get("label") is not None}
if "--module" in sys.argv and sys.argv[sys.argv.index("--module") + 1] == "pronounced_muzzle":
    from scipy.sparse.csgraph import connected_components
    pts = np.array([[float(c) for c in q.split(",")] for q in sys.argv[sys.argv.index("--jowl_line") + 1].split(";")])
    C = m.triangles_center; yy = C[:, 1] - m.vertices[:, 1].min()
    zs = np.interp(yy, pts[:, 0], pts[:, 1])
    for side, (Lm, Lj) in ((1, (21, 27)), (-1, (22, 28))):
        j = (label == Lm) & (C[:, 2] < zs)
        label[j] = Lj
    # below the nose the head gusset gives way to the jowls, which meet each other down the philtrum (the nose,
    # sewn on afterwards, covers the point where gusset, muzzle tops and jowls meet)
    below = (label == 1) & (C[:, 2] < pts[0, 1]) & (yy < 40)
    label[below & (C[:, 0] >= 0)] = 27; label[below & (C[:, 0] < 0)] = 28
    # clean the new boundaries the same way the layout's own labels were cleaned (vote, symmetry, islands)
    label = clean_labels(m, label, only={1, 4, 21, 22, 27, 28}, swap={21: 22, 22: 21, 27: 28, 28: 27})
    PIECES[21] = ("P05", "muzzle top R (side of the nose up to the stop)", None)
    PIECES[22] = ("P06", "muzzle top L (side of the nose up to the stop)", "P05")
    PIECES[27] = ("P23", "jowl R (flews)", None); PIECES[28] = ("P24", "jowl L (flews)", "P23")
    PIECES[4] = ("P04", "chin / lower lip (undershot)", None)
    print("pronounced muzzle module: jowl faces", int((label == 27).sum()), "per side")
m = straighten(m, label)
report = export_pieces(m, label, PIECES, out, name or src["layout"], extra={"transferred_from": str(sdir)})
