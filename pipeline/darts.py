"""Darts for pieces that do not lie flat (domes: jowls, tail stubs, thighs) - spec: add darts where the flattening
distortion is too high, instead of easing.

For a flattened piece whose area distortion (p95) is above --target, up to --max darts are tried, one at a time:
  * the dart mouth is the boundary vertex where the flat piece is most stretched (smoothed along the edge)
  * the dart runs along the shortest path from there towards the point of the piece farthest from its edge
    (the top of the dome), --frac of the way
  * the piece is slit along that path (libigl cut_mesh; the apex stays joined) and flattened again; the slit
    opens into the dart wedge, sewn closed edge to edge
A dart is kept only if it cuts the distortion by at least 15 % of its value; nothing else about the piece changes.

Usage: python3 darts.py flat_dir [--target 0.08] [--max 2] [--frac 0.55] [--only P23,P17]
Rewrites flat_dir/<pid>.npz (adds 'darts': mouth copies and apex per dart; the original is kept as
nodart_<pid>.npz) and writes flat_dir/darts.json. Acceptance: every darted piece valid (no flips, outline
valid, boundary length ratio within 8 %) and better than before.
"""
import json, sys
from pathlib import Path
import numpy as np, igl, scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, str(Path(__file__).parent))
from flatten import flatten, distortion, flatten_injective, fix_slivers, outline_ok


def opt(n, d): return type(d)(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d


fdir = Path(sys.argv[1]); TARGET, MAXD, FRAC = opt("--target", 0.08), opt("--max", 2), opt("--frac", 0.55)
ONLY = opt("--only", "").split(",") if "--only" in sys.argv else None


def stats(V, F, uv, bnd):
    ar, lr, a3 = distortion(V, F, uv)
    L3 = np.linalg.norm(V[np.roll(bnd, -1)] - V[bnd], axis=1).sum(); L2 = np.linalg.norm(np.roll(uv[bnd], -1, 0) - uv[bnd], axis=1).sum()
    return {"area_dev_p95": round(float(np.percentile(np.abs(ar - 1), 95)), 3), "flipped": int((ar <= 0).sum()),
            "boundary_len_ratio": round(float(L2 / L3), 3), "outline_valid": bool(outline_ok(uv, bnd))}, ar


def flat(V, F):
    try:
        uv, bnd = flatten(V, F, iters=150)
    except RuntimeError:
        return flatten_injective(V, F, iters=300)
    if not np.isfinite(uv).all() or not outline_ok(uv, bnd): uv, bnd = flatten_injective(V, F, iters=300)
    return uv, bnd


def graph(V, F, bnd=None):
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); w = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    if bnd is not None:                                             # a dart never runs along the piece's edge
        onb = np.zeros(len(V), bool); onb[bnd] = True; w = np.where(onb[E[:, 0]] & onb[E[:, 1]], 1e6, w)
    G = sp.coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(len(V),) * 2).tocsr()
    return G


def one_dart(V, F, uv, bnd, ar, mode="dome"):
    """mode 'dome': from the most stretched point of the edge towards the top of the dome (jowls, cheeks);
    mode 'hotspot': from the nearest edge point into the most distorted area inside the piece (flank, thigh)."""
    vs = np.zeros(len(V)); cnt = np.zeros(len(V))
    for i in range(3): np.add.at(vs, F[:, i], ar); np.add.at(cnt, F[:, i], 1)
    vs /= np.maximum(cnt, 1)
    G = graph(V, F); Gi = graph(V, F, bnd)
    if mode == "dome":
        sb = gaussian_filter1d(vs[bnd], max(2.0, len(bnd) / 60), mode="wrap")
        b = bnd[int(np.argmax(sb))]
        db = dijkstra(G, indices=bnd, min_only=True); c = int(np.argmax(db)); frac = FRAC
    else:
        dev = np.abs(vs - 1)
        A = (G > 0).astype(float); A = A + A @ A                     # two-ring average
        dsm = (A @ dev) / np.maximum(np.asarray(A.sum(1)).ravel(), 1)
        onb = np.zeros(len(V), bool); onb[bnd] = True; dsm[onb] = 0
        c = int(np.argmax(dsm))
        d_c = dijkstra(Gi, indices=c); b = bnd[int(np.argmin(d_c[bnd]))]; frac = 1.0
    dist, pred = dijkstra(Gi, indices=b, return_predecessors=True)
    if not np.isfinite(dist[c]): return None
    path = [c]
    while path[-1] != b: path.append(pred[path[-1]])
    path = path[::-1]                                               # b ... c
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(V[path], axis=0), axis=1))]
    path = [p for p, l in zip(path, L) if l <= frac * L[-1] + 1e-9]
    if len(path) < 3: return None
    pe = set(tuple(sorted(e)) for e in zip(path[:-1], path[1:]))
    C = np.zeros(F.shape, bool)
    for fi, f in enumerate(F):
        for i in range(3):
            if tuple(sorted((f[i], f[(i + 1) % 3]))) in pe: C[fi, i] = True
    Vn, Fn, I = igl.cut_mesh(np.ascontiguousarray(V), np.ascontiguousarray(F, dtype=np.int64), C)
    mouth = [int(j) for j in np.where(I == b)[0]]
    if len(mouth) < 2: return None                                  # the slit did not open
    # the slit piece must still be one disc with one clean outline (libigl's flattening needs that)
    Ec = np.sort(np.vstack([Fn[:, [0, 1]], Fn[:, [1, 2]], Fn[:, [2, 0]]]), axis=1)
    u_, c_ = np.unique(Ec, axis=0, return_counts=True); bE = u_[c_ == 1]
    deg = np.bincount(bE.ravel(), minlength=len(Vn))
    if (deg > 2).any() or len(igl.boundary_loop(np.ascontiguousarray(Fn, dtype=np.int64))) != len(bE): return None
    return Vn, Fn, I, {"mode": mode, "mouth": mouth, "apex": int(path[-1]), "length_mm": round(float(L[len(path) - 1]), 1)}


report = {}
for f in sorted(fdir.glob("P*.npz")):
    pid = f.stem
    if ONLY and pid not in ONLY: continue
    src = fdir / f"nodart_{pid}.npz"
    d = dict(np.load(src if src.exists() else f))
    V, F, uv, bnd = d["V3"], d["F"].astype(np.int64), d["uv"], d["bnd"]
    st0, ar = stats(V, F, uv, bnd)
    if st0["area_dev_p95"] <= TARGET and abs(st0["boundary_len_ratio"] - 1) <= 0.06: continue
    best = (V, F, uv, bnd, st0, ar, []); darts = []
    for k in range(MAXD):
        V_, F_, uv_, bnd_, st_, ar_, darts_ = best; cand = []
        for mode in ("dome", "hotspot"):
            r = one_dart(V_, F_, uv_, bnd_, ar_, mode)
            if r is None: continue
            Vn, Fn, I, info = r
            Vn, _ = fix_slivers(Vn, Fn)
            uvn, bndn = flat(Vn, Fn)
            stn, arn = stats(Vn, Fn, uvn, bndn)
            info["opening_mm"] = round(float(np.linalg.norm(uvn[info["mouth"][0]] - uvn[info["mouth"][-1]])), 1)
            ok = stn["flipped"] == 0 and stn["outline_valid"] and stn["area_dev_p95"] <= 0.85 * st_["area_dev_p95"] and abs(stn["boundary_len_ratio"] - 1) <= 0.08 + max(0, abs(st_["boundary_len_ratio"] - 1) - 0.08)
            print(pid, f"dart {k + 1} ({mode}):", st_["area_dev_p95"], "->", stn["area_dev_p95"], "boundary", stn["boundary_len_ratio"], {q: info[q] for q in ("length_mm", "opening_mm")}, "ok" if ok else "no")
            if ok: cand.append((stn["area_dev_p95"], (Vn, Fn, uvn, bndn, stn, arn, darts_ + [info])))
        if not cand: break
        best = min(cand, key=lambda t: t[0])[1]
    V_, F_, uv_, bnd_, st_, ar_, darts_ = best
    report[pid] = {"before": st0, "after": st_, "darts": darts_}
    if darts_:
        if not src.exists(): np.savez(src, **d)
        np.savez(f, V3=V_, F=F_, uv=uv_, bnd=bnd_, area_ratio=ar_, darts=json.dumps(darts_))
json.dump(report, open(fdir / "darts.json", "w"), indent=1)
bad = [p for p, r in report.items() if r["darts"] and (r["after"]["flipped"] or not r["after"]["outline_valid"] or abs(r["after"]["boundary_len_ratio"] - 1) > 0.08)]
still = [p for p, r in report.items() if abs(r["after"]["boundary_len_ratio"] - 1) > 0.08 or r["after"]["area_dev_p95"] > 0.12]
print("darted:", {p: len(r["darts"]) for p, r in report.items()}, "| still above limits:", still)
print("RESULT:", "PASS" if not bad and not still else f"FAIL {bad + still}")
