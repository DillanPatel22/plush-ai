"""Acceptance test for pipeline/sew_stuff_sim.py against a result known in closed form.

A sphere of radius R is cut into one flat piece per triangle (each piece exactly its own triangle, so the pattern is
perfect) and sewn and stuffed. For this membrane (stretch stiffness kt + kc, no Poisson coupling) the balance of
pressure work and stretch energy gives a uniform stretch s with  s - 1 = a s^2,  a = P R / (2 kt (1 + kc)),
P = 2 kt * stuff / 40  (the s^2: pressure acts on the stretched surface), so  s = (1 - sqrt(1 - 4a)) / (2a).
The test runs the simulator on that sphere and checks the radius growth, that the shape stays a sphere, and that the
pattern compensation brings it back to the target size.

Usage: python3 tests/test_sew_stuff_sim.py      Prints PASS/FAIL with the numbers.
"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, trimesh

ROOT = Path(__file__).resolve().parents[1]
R, STUFF, KC = 40.0, 0.06, 0.02
tmp = Path(tempfile.mkdtemp())
flat, pcs, out = tmp / "flat", tmp / "pieces", tmp / "out"
flat.mkdir(); pcs.mkdir()
s = trimesh.creation.icosphere(subdivisions=3, radius=R)
V, F = s.vertices, s.faces
layout = {"layout": "test sphere", "pieces": {}}
for i, f in enumerate(F):
    P3 = V[f]; e1, e2 = P3[1] - P3[0], P3[2] - P3[0]
    u = e1 / np.linalg.norm(e1); n = np.cross(e1, e2); w = np.cross(n, u) / np.linalg.norm(np.cross(n, u))
    uv = np.c_[(P3 - P3[0]) @ u, (P3 - P3[0]) @ w]                      # the triangle laid flat, same edge lengths
    pid = f"T{i:04d}"
    np.savez(flat / f"{pid}.npz", V3=P3, F=np.array([[0, 1, 2]]), uv=uv, bnd=np.array([0, 1, 2]), area_ratio=np.ones(1))
    layout["pieces"][pid] = {"name": f"triangle {i}", "faces": 1}
json.dump(layout, open(pcs / "pieces.json", "w"))
s.export(pcs / "subdivided.obj")
json.dump({"nose_to_chest_front_mm": R, "body_length_mm": R, "front_leg_y_mm": [-R, -R / 2], "rear_leg_y_mm": [R / 2, R],
           "elbow_height_mm": -10 * R, "nose_y0": -R}, open(pcs / "measures.json", "w"))
cmd = [sys.executable, str(ROOT / "pipeline" / "sew_stuff_sim.py"), str(flat), str(pcs / "pieces.json"), str(pcs / "measures.json"),
       str(out), "--stuff", str(STUFF), "--kc", str(KC), "--bend", "0", "--iters", "1500", "--compensate", "2", "--damp", "1.0"]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode:
    print(r.stdout[-2000:], r.stderr[-2000:]); print("RESULT: FAIL (simulator crashed)"); sys.exit(1)
rep = json.load(open(out / "report.json"))
x = np.load(out / "sim_raw.npy"); c = x.mean(0); rad = np.linalg.norm(x - c, axis=1)
a = STUFF * R / 40 / (1 + KC)
expect = (1 - np.sqrt(1 - 4 * a)) / (2 * a) - 1
growth = rad.mean() / R - 1
roundness = rad.std() / rad.mean()
comp_size = rep["compensated"]["size_change_pct"]
print(f"radius growth {100 * growth:.2f} % (closed form {100 * expect:.2f} %), roundness (std/mean radius) {100 * roundness:.3f} %, "
      f"after compensation size change {comp_size} %, assembly open edges {rep['assembly']['open_edges']}")
ok = abs(growth - expect) <= 0.03 * expect and roundness < 0.005 and abs(comp_size) < 0.5 and rep["assembly"]["open_edges"] == 0
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
