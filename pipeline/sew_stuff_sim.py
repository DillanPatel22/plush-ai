"""Sew-and-stuff simulation: the shape the cut pieces actually make once sewn together and stuffed, compared
with the shape we designed (the target). Optionally compensates the pattern so the stuffed result lands on target.

Forward model (after Mori & Igarashi 2007 "Plushie" and Skouras et al. 2014 "Designing inflatable structures"):
  * each flattened piece (flat_dir/<pid>.npz: target 3D V3 at plush size + its 2D pattern uv) gives each of its
    triangles a flat rest shape; left pieces are mirrors of right pieces (pieces.json mirror_of)
  * pieces are sewn by joining vertices that sit at the same point of the target (seams and dart edges)
  * fabric = tension-field membrane: stiffness kt per unit width when stretched, kc = --kc * kt when compressed
    (it buckles into small wrinkles instead of pushing back), plus a small flat-rest bending stiffness
  * stuffing = internal pressure P doing work P*dV
  Equilibrium = minimum of membrane + bending - P*volume (L-BFGS), started from the target shape.
  Firmness --stuff is the stretch a 40 mm-radius sphere of this fabric would take (default 0.06, i.e. 6 %).
  kt, kc, bending and firmness are calibration values: they get measured on sewn samples (spec: simulator calibration).

Compensation (--compensate N): N rounds of the fixed-point pattern update used in inverse garment design: every
triangle edge of the pattern is scaled by target length / simulated length, each piece is re-fitted flat to those
lengths (ARAP local-global), then re-simulated. Seams, pieces and darts never change; only outlines move. What the
rounds cannot fix is the part of the target that stuffing cannot hold (concave areas, sharp creases) - those need a
seam, dart or needle-sculpting, and are reported as such.

Usage: python3 sew_stuff_sim.py flat_dir pieces.json measures.json out_dir [--stuff 0.06] [--kc 0.02]
       [--bend 1.0] [--compensate 0] [--tol 3.0] [--iters 4000]
Outputs: report.json, target.ply / sim.ply / sim_comp.ply (real-size frame, coloured by signed normal displacement:
red = bulges out, blue = sinks in), flat_comp/<pid>.npz (compensated pattern, same format as flatten_layout.py).
Acceptance: closed surface, solver converged; sewn result PASS when shape deviation p95 <= --tol mm (plush) and every
key width within 7 %; otherwise FAIL with the worst areas listed.
"""
import json, sys, time
from pathlib import Path
import numpy as np, trimesh, scipy.sparse as sp, igl
from scipy.optimize import minimize
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).parent))
from flatten import outline_ok


def opt(n, d): return type(d)(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d


fdir, pj, mj, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
STUFF, KC, BEND, NCOMP, TOL, ITERS = opt("--stuff", 0.06), opt("--kc", 0.02), opt("--bend", 1.0), opt("--compensate", 0), opt("--tol", 3.0), opt("--iters", 4000)
out.mkdir(parents=True, exist_ok=True)
layout, ms = json.load(open(pj)), json.load(open(mj))
KT = 1.0                                           # N/mm per unit width (everything else is relative to it)
P = 2 * KT * STUFF / 40.0                          # Laplace: a 40 mm sphere stretches by STUFF

# ---------------------------------------------------------------- assemble the sewn surface
pieces = []                                        # (pid, source pid, mirrored, V3, F, uv)
for pid, info in layout["pieces"].items():
    src = info.get("mirror_of") or pid
    f = fdir / f"{src}.npz"
    if not info.get("faces") or not f.exists(): continue
    d = np.load(f); V3, F, uv = d["V3"].astype(float), d["F"].astype(np.int64), d["uv"].astype(float)
    if info.get("mirror_of"): V3, uv, F = V3 * [-1, 1, 1], uv * [-1, 1], F[:, [0, 2, 1]]
    pieces.append((pid, src, bool(info.get("mirror_of")), V3, F, uv))
Vall = np.vstack([p[3] for p in pieces]); off = np.cumsum([0] + [len(p[3]) for p in pieces])
Fall = np.vstack([p[4] + o for p, o in zip(pieces, off[:-1])])
UVall = np.vstack([p[5] for p in pieces])
pidx = np.concatenate([np.full(len(p[4]), i) for i, p in enumerate(pieces)])
R_MERGE = 0.1                                      # mm; only outline vertices are joined (seams, dart edges)
onb = np.zeros(len(Vall), bool)
for p, o in zip(pieces, off[:-1]):
    Es = np.sort(np.vstack([p[4][:, [0, 1]], p[4][:, [1, 2]], p[4][:, [2, 0]]]), axis=1)
    u_, c_ = np.unique(Es, axis=0, return_counts=True); onb[u_[c_ == 1].ravel() + o] = True
# join each outline vertex to its mutual nearest outline vertex of every other piece (and dart duplicates inside a
# piece), so tightly packed points such as the tail tip do not collapse into one
vpc = np.concatenate([np.full(len(p[3]), i) for i, p in enumerate(pieces)])
B = [np.where(onb & (vpc == i))[0] for i in range(len(pieces))]
T = [cKDTree(Vall[b]) for b in B]
pairs = []
for i in range(len(pieces)):
    for j in range(i, len(pieces)):
        if i == j:
            q = T[i].query_pairs(1e-3, output_type="ndarray")
            if len(q): pairs.append(B[i][q])
            continue
        dij, nij = T[j].query(Vall[B[i]], distance_upper_bound=R_MERGE)
        ok_ = np.isfinite(dij)
        if not ok_.any(): continue
        a, b = np.where(ok_)[0], nij[ok_]
        _, nji = T[i].query(Vall[B[j][b]])
        mutual = nji == a
        pairs.append(np.c_[B[i][a[mutual]], B[j][b[mutual]]])
pairs = np.vstack(pairs) if pairs else np.zeros((0, 2), int)
G = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(Vall),) * 2)
ncomp, lab = sp.csgraph.connected_components(G, directed=False)
X0 = np.zeros((ncomp, 3)); np.add.at(X0, lab, Vall); X0 /= np.bincount(lab)[:, None]
F = lab[Fall]                                      # sewn faces on merged vertices
keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
n_degenerate = int((~keep).sum()); F, Fall = F[keep], Fall[keep]
pidx = pidx[keep]
N = len(X0)
mesh = trimesh.Trimesh(X0, F, process=False)
E_sorted = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
_, ecount = np.unique(E_sorted, axis=0, return_counts=True)
open_edges, nonmanifold = int((ecount == 1).sum()), int((ecount > 2).sum())
vol0 = float(np.einsum("ij,ij->i", X0[F[:, 0]], np.cross(X0[F[:, 1]], X0[F[:, 2]])).sum() / 6)
if vol0 < 0: F = F[:, [0, 2, 1]]; Fall = Fall[:, [0, 2, 1]]; vol0 = -vol0
print(f"assembled {len(pieces)} pieces: {N} vertices, {len(F)} triangles, open edges {open_edges}, non-manifold {nonmanifold}, "
      f"collapsed {n_degenerate}, volume {vol0 / 1e6:.2f} L")

# ---------------------------------------------------------------- energy
D1 = sp.csr_matrix((np.r_[np.ones(len(F)), -np.ones(len(F))], (np.r_[np.arange(len(F)), np.arange(len(F))], np.r_[F[:, 1], F[:, 0]])), shape=(len(F), N))
D2 = sp.csr_matrix((np.r_[np.ones(len(F)), -np.ones(len(F))], (np.r_[np.arange(len(F)), np.arange(len(F))], np.r_[F[:, 2], F[:, 0]])), shape=(len(F), N))
Eu = np.unique(E_sorted, axis=0)
Adj = sp.coo_matrix((np.ones(2 * len(Eu)), (np.r_[Eu[:, 0], Eu[:, 1]], np.r_[Eu[:, 1], Eu[:, 0]])), shape=(N, N)).tocsr()
deg = np.asarray(Adj.sum(1)).ravel()
Lap = sp.identity(N, format="csr") - sp.diags(1 / deg) @ Adj        # umbrella: Lx ~ (h^2/4) * mean-curvature normal
el = np.linalg.norm(X0[Eu[:, 0]] - X0[Eu[:, 1]], axis=1)
hsum = np.bincount(Eu[:, 0], el, N) + np.bincount(Eu[:, 1], el, N); hv = hsum / deg
WB = BEND * KT * 2 * np.sqrt(3) / hv ** 2           # flat-rest bending, bending length sqrt(kb/kt) = BEND mm
BtB = (Lap.T @ sp.diags(WB) @ Lap).tocsr()


def rest_from_uv(uv_all):
    """per-face inverse rest matrix and rest area from the pattern (uv of the un-merged face corners)."""
    a, b, c = uv_all[Fall[:, 0]], uv_all[Fall[:, 1]], uv_all[Fall[:, 2]]
    Dm = np.stack([b - a, c - a], axis=2)                     # (n,2,2) columns
    det = Dm[:, 0, 0] * Dm[:, 1, 1] - Dm[:, 0, 1] * Dm[:, 1, 0]
    A0 = 0.5 * np.abs(det)
    inv = np.stack([np.stack([Dm[:, 1, 1], -Dm[:, 0, 1]], 1), np.stack([-Dm[:, 1, 0], Dm[:, 0, 0]], 1)], 1) / det[:, None, None]
    return inv, A0


def energy(xf, DmI, A0, pressure=True):
    x = xf.reshape(N, 3)
    e1, e2 = D1 @ x, D2 @ x
    Ds = np.stack([e1, e2], axis=2)                            # (n,3,2)
    Fm = Ds @ DmI
    C = np.transpose(Fm, (0, 2, 1)) @ Fm
    w, Q = np.linalg.eigh(C)
    s = np.sqrt(np.maximum(w, 1e-12))
    st = s - 1
    kk = np.where(st > 0, KT + KC * KT, KC * KT)          # same split as the solver's two constraints
    W = 0.5 * (A0[:, None] * kk * st ** 2).sum()
    dWds = A0[:, None] * kk * st
    M = Q @ (np.eye(2)[None] * (dWds / s)[:, None, :]) @ np.transpose(Q, (0, 2, 1))
    dF = Fm @ M                                                 # dW/dF = F V diag(f'(s)/s) V^T
    dDs = dF @ np.transpose(DmI, (0, 2, 1))
    g = D1.T @ dDs[:, :, 0] + D2.T @ dDs[:, :, 1]
    Lx = BtB @ x
    W += 0.5 * float((x * Lx).sum()); g = g + Lx
    if pressure:
        x0, x1, x2 = x[F[:, 0]], x[F[:, 1]], x[F[:, 2]]
        vol = np.einsum("ij,ij->i", x0, np.cross(x1, x2)).sum() / 6
        W -= P * vol
        for i, (a, b) in enumerate(((x1, x2), (x2, x0), (x0, x1))):
            gi = np.cross(a, b) / 6
            g -= P * np.stack([np.bincount(F[:, i], gi[:, j], N) for j in range(3)], 1)
    return W, g.ravel(), s


GM = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])


class Info:
    success = False


def eig2(C):
    """closed-form eigen-decomposition of symmetric 2x2 matrices: (larger, smaller) eigenvalues, unit eigenvectors."""
    a, b, c = C[:, 0, 0], C[:, 0, 1], C[:, 1, 1]
    th = 0.5 * np.arctan2(2 * b, a - c); cs, sn = np.cos(th), np.sin(th)
    l1 = a * cs ** 2 + 2 * b * cs * sn + c * sn ** 2; l2 = a * sn ** 2 - 2 * b * cs * sn + c * cs ** 2
    return np.c_[l1, l2], np.c_[cs, sn], np.c_[-sn, cs]


def simulate(uv_all, x_start, tol=1e-4, m_aa=6):
    """Projective Dynamics (Bouaziz et al. 2014), quasi-static, with Anderson acceleration (Peng et al. 2018):
    local step projects every triangle onto its nearest allowed shape (tension-only: stretches clamped to <= 1, plus
    a soft full constraint of weight kc), global step solves one pre-factored sparse system; the stuffing pressure
    enters as the force P * dVolume/dx. An accelerated step is kept only if it lowers the energy."""
    DmI, A0 = rest_from_uv(uv_all)
    Gm = GM[None] @ DmI                                         # (n,3,2): F = [x0 x1 x2] @ Gm
    wt, wc = KT * A0, KC * KT * A0
    GG = Gm @ np.transpose(Gm, (0, 2, 1))
    rows = np.concatenate([F[:, a] for a in range(3) for b in range(3)]); cols = np.concatenate([F[:, b] for a in range(3) for b in range(3)])
    vals = np.concatenate([(wt + wc) * GG[:, a, b] for a in range(3) for b in range(3)])
    M = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    eps = 1e-6 * M.diagonal().mean()
    lu = splu((M + BtB + eps * sp.identity(N)).tocsc())

    def local(x):
        Fm = np.einsum("nja,njb->nab", x[F], Gm)
        lam, v1, v2 = eig2(np.transpose(Fm, (0, 2, 1)) @ Fm); s = np.sqrt(np.maximum(lam, 1e-12))
        P1, P2 = v1[:, :, None] * v1[:, None, :], v2[:, :, None] * v2[:, None, :]
        d1 = (wt * np.minimum(s[:, 0], 1) + wc) / s[:, 0]; d2 = (wt * np.minimum(s[:, 1], 1) + wc) / s[:, 1]
        T = Fm @ (d1[:, None, None] * P1 + d2[:, None, None] * P2)
        TG = T @ np.transpose(Gm, (0, 2, 1))
        rhs = np.stack([sum(np.bincount(F[:, j], TG[:, d, j], N) for j in range(3)) for d in range(3)], 1)
        x0, x1, x2 = x[F[:, 0]], x[F[:, 1]], x[F[:, 2]]
        fp = np.zeros((N, 3))
        for i, (a, b) in enumerate(((x1, x2), (x2, x0), (x0, x1))):
            gi = np.cross(a, b) / 6
            fp += np.stack([np.bincount(F[:, i], gi[:, j], N) for j in range(3)], 1)
        vol = np.einsum("ij,ij->i", x0, np.cross(x1, x2)).sum() / 6
        E = 0.5 * (wt * ((s - np.minimum(s, 1)) ** 2).T).sum() + 0.5 * (wc * ((s - 1) ** 2).T).sum() + 0.5 * float((x * (BtB @ x)).sum()) - P * vol
        return rhs + P * fp, E

    t = time.time(); info = Info(); it = 0; acc = rej = 0
    x = x_start.copy(); b_, E = local(x)
    Gh, Fh = [], []
    x_ref, E_ref, g_ref = x, E, None
    for it in range(ITERS):
        g = lu.solve(b_ + eps * x)                                  # plain PD step from x
        f = (g - x).ravel(); step = np.abs(f).max()
        if step < tol and it >= 200: x = g; info.success = True; break
        Gh.append(g.ravel()); Fh.append(f)
        if len(Gh) > m_aa + 1: Gh.pop(0); Fh.pop(0)
        if len(Gh) >= 2:
            dF = np.diff(np.array(Fh), axis=0).T; dG = np.diff(np.array(Gh), axis=0).T
            gam = np.linalg.lstsq(dF, f, rcond=None)[0]
            xa = (g.ravel() - dG @ gam).reshape(N, 3)
            ba, Ea = local(xa)
            if Ea <= E_ref + 1e-9 * abs(E_ref):
                x, b_, E = xa, ba, Ea; acc += 1
            else:                                                   # reject: take the plain step, restart the history
                x = g; b_, E = local(x); Gh, Fh = [g.ravel()], [f]; rej += 1
        else:
            x = g; b_, E = local(x)
        E_ref = E
        if it % 200 == 0: print(f"    it {it}: max step {step:.2e} mm, energy {E:.4f}, accelerated {acc}, rejected {rej}")
    _, gr, s = energy(x.ravel(), DmI, A0)
    gn = np.linalg.norm(gr.reshape(N, 3), axis=1)
    print(f"  solver: {it + 1} iterations ({acc} accelerated, {rej} rejected), {time.time() - t:.0f} s, last step {step:.1e} mm, "
          f"|residual force| p95 {np.percentile(gn, 95):.2e} (pressure force/vertex ~{P * np.median(hv) ** 2:.2e})")
    info.residual_p95 = float(np.percentile(gn, 95))
    return x, info, s, gn


# ---------------------------------------------------------------- metrics
def similarity(A, B):
    """best similarity transform taking A onto B: returns transformed A and the scale."""
    ma, mb = A.mean(0), B.mean(0); a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd(a.T @ b); dd = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ np.diag([1, 1, dd]) @ Vt; sc = (S * [1, 1, dd]).sum() / (a ** 2).sum()
    return sc * a @ Rm + mb, sc


REAL_LEN = ms["nose_to_chest_front_mm"] + ms["body_length_mm"]      # real mm; X0 is at plush size (x k)


def stations(k):
    eye_y = None
    for cand in (Path(mj).parent / "face_points.json",):
        if cand.exists(): eye_y = json.load(open(cand))["eye_R"][1]
    ny = ms.get("nose_y0", 0.0)
    if eye_y is None: eye_y = ny + 0.35 * ms["nose_to_chest_front_mm"]
    fl, rl, el = ms["front_leg_y_mm"], ms["rear_leg_y_mm"], ms["elbow_height_mm"]
    S = {"muzzle (between nose and eyes)": ("y", (ny + eye_y) / 2, "head"),
         "head at the eyes": ("y", eye_y, "head"),
         "skull behind the eyes": ("y", eye_y + 0.6 * (eye_y - ny), "head"),
         "chest at the front legs": ("y", np.mean(fl), "body"),
         "waist": ("y", (fl[1] + rl[0]) / 2, "body"),
         "hips at the hind legs": ("y", np.mean(rl), "body"),
         "front leg (half elbow height)": ("z", 0.5 * el, "front leg R"),
         "hind leg (half elbow height)": ("z", 0.5 * el, "hind leg R")}
    return {n: (ax, v * k, reg) for n, (ax, v, reg) in S.items()}


def region_masks():
    """parts of the target: head and tail by piece, legs below elbow height (front / hind, right / left), body = rest."""
    vreg = np.full(N, "body", dtype=object)
    nm = {p[0]: layout["pieces"][p[0]]["name"].lower() for p in pieces}
    head_p = [i for i, p in enumerate(pieces) if any(w in nm[p[0]] for w in ("head", "chin", "muzzle", "jowl"))]
    tail_p = [i for i, p in enumerate(pieces) if "tail" in nm[p[0]]]
    el = ms["elbow_height_mm"] * K; ymid = 0.5 * (ms["front_leg_y_mm"][1] + ms["rear_leg_y_mm"][0]) * K
    low = X0[:, 2] < el
    vreg[low & (X0[:, 1] < ymid) & (X0[:, 0] > 0)] = "front leg R"; vreg[low & (X0[:, 1] < ymid) & (X0[:, 0] < 0)] = "front leg L"
    vreg[low & (X0[:, 1] >= ymid) & (X0[:, 0] > 0)] = "hind leg R"; vreg[low & (X0[:, 1] >= ymid) & (X0[:, 0] < 0)] = "hind leg L"
    for i in head_p: vreg[np.unique(F[pidx == i])] = "head"
    for i in tail_p: vreg[np.unique(F[pidx == i])] = "tail"
    return vreg


def rigid(A, B):
    """best rotation + translation taking A onto B: aligned A, rotation angle (deg), shift of the centroid (mm)."""
    ma, mb = A.mean(0), B.mean(0); U, S, Vt = np.linalg.svd((A - ma).T @ (B - mb))
    dd = np.sign(np.linalg.det(U @ Vt)); Rm = U @ np.diag([1, 1, dd]) @ Vt
    ang = float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))))
    return (A - ma) @ Rm + mb, ang, float(np.linalg.norm(mb - ma))


def widths(x, st, k, sels):
    res = {}
    for n, (ax, v, reg) in st.items():
        sel = sels[n]
        if sel.sum() < 6: res[n] = None; continue
        res[n] = ({"width_mm": float(np.ptp(x[sel, 0])), "height_mm": float(np.ptp(x[sel, 2]))} if ax == "y"
                  else {"width_mm": float(np.ptp(x[sel, 0])), "depth_mm": float(np.ptp(x[sel, 1]))})
    return res


def silhouettes(xa, xb, n=480):
    """IoU of the orthographic outlines (side, front, top) of two shapes in the same frame."""
    out_ = {}
    for name, (i, j) in {"side": (1, 2), "front": (0, 2), "top": (0, 1)}.items():
        lo = np.minimum(xa[:, [i, j]].min(0), xb[:, [i, j]].min(0)); hi = np.maximum(xa[:, [i, j]].max(0), xb[:, [i, j]].max(0))
        sc = (n - 4) / (hi - lo).max()
        ims = []
        for x in (xa, xb):
            im = Image.new("1", (n, n), 0); dr = ImageDraw.Draw(im)
            P2 = (x[:, [i, j]] - lo) * sc + 2
            for f in F: dr.polygon([tuple(P2[f[0]]), tuple(P2[f[1]]), tuple(P2[f[2]])], fill=1)
            ims.append(np.asarray(im, bool))
        out_[name] = round(float((ims[0] & ims[1]).sum() / (ims[0] | ims[1]).sum()), 4)
    return out_


def vertex_normals(x):
    return trimesh.Trimesh(x, F, process=False).vertex_normals


def mean_curv_sign(x):
    Lx = -(Lap @ x)                                   # points toward the centre of curvature for convex areas... sign below
    n = vertex_normals(x)
    return np.einsum("ij,ij->i", Lx, n)               # > 0: concave (surface dips in), < 0: convex


def seam_lengths(uv_all):
    """2D length of every seam on each of its two sides (pieces joined along merged edges)."""
    rows = []
    for e_local in ((0, 1), (1, 2), (2, 0)):
        a, b = Fall[:, e_local[0]], Fall[:, e_local[1]]
        rows.append(np.c_[np.minimum(lab[a], lab[b]), np.maximum(lab[a], lab[b]), pidx, np.linalg.norm(uv_all[a] - uv_all[b], axis=1)])
    R_ = np.vstack(rows)
    key = R_[:, 0] * N + R_[:, 1]
    order = np.argsort(key); R_ = R_[order]; key = key[order]
    _, start, cnt = np.unique(key, return_index=True, return_counts=True)
    seams = {}
    for s0, c in zip(start, cnt):
        if c != 2: continue
        p1, p2 = int(R_[s0, 2]), int(R_[s0 + 1, 2])
        if p1 == p2: continue
        kk_ = tuple(sorted((pieces[p1][0], pieces[p2][0])))
        a_, b_ = (R_[s0, 3], R_[s0 + 1, 3]) if pieces[p1][0] == kk_[0] else (R_[s0 + 1, 3], R_[s0, 3])
        L = seams.setdefault(kk_, [0.0, 0.0]); L[0] += a_; L[1] += b_
    return {f"{a}|{b}": {"len_mm": [round(l[0], 1), round(l[1], 1)], "mismatch_pct": round(100 * abs(l[0] - l[1]) / max(l), 2)}
            for (a, b), l in seams.items() if max(l) > 10}


def evaluate(x, label):
    xs, sc = similarity(x, X0)
    d = xs - X0; n0 = vertex_normals(X0); dn = np.einsum("ij,ij->i", d, n0); dist = np.linalg.norm(d, axis=1)
    curv = mean_curv_sign(X0); conc = curv > np.percentile(curv, 90); conv = curv < np.percentile(curv, 10)
    # each part aligned on its own: separates "the head tilts" (pose) from "the muzzle got narrower" (shape)
    xl = xs.copy(); parts = {}
    for rg in ("head", "body", "tail", "front leg R", "front leg L", "hind leg R", "hind leg L"):
        m_ = VREG == rg
        if m_.sum() < 10: continue
        xl[m_], ang, sh = rigid(xs[m_], X0[m_])
        parts[rg] = {"tilt_deg": round(ang, 1), "shift_mm": round(sh, 1),
                     "shape_dev_p95_mm": round(float(np.percentile(np.linalg.norm(xl[m_] - X0[m_], axis=1), 95)), 2)}
    st = stations(K); wt, ws = widths(X0, st, K, SELS), widths(xl, st, K, SELS)
    wcmp = {}
    for nme in st:
        if wt[nme] is None or ws[nme] is None: continue
        wcmp[nme] = {q: {"target": round(wt[nme][q], 1), "sewn": round(ws[nme][q], 1), "change_pct": round(100 * (ws[nme][q] / wt[nme][q] - 1), 1)} for q in wt[nme]}
    per_piece = {}
    for i, p in enumerate(pieces):
        if p[2]: continue
        sel = np.unique(F[pidx == i]); sel = sel[X0[sel, 0] >= -0.5] if p[0] not in ("P01", "P04") else sel
        if not len(sel): continue
        dl = np.linalg.norm(xl[sel] - X0[sel], axis=1)
        per_piece[p[0]] = {"name": layout["pieces"][p[0]]["name"], "dev_p95_mm": round(float(np.percentile(dist[sel], 95)), 2),
                           "dev_p95_part_aligned_mm": round(float(np.percentile(dl, 95)), 2),
                           "bulge_out_max_mm": round(float(dn[sel].max()), 2), "sink_in_max_mm": round(float(-dn[sel].min()), 2)}
    dloc = np.linalg.norm(xl - X0, axis=1)
    rep = {"label": label, "size_change_pct": round(100 * (1 / sc - 1), 2),
           "volume_change_pct": round(100 * (abs(trimesh.Trimesh(x, F, process=False).volume) / vol0 - 1), 2),
           "shape_dev_mm": {"mean": round(float(dist.mean()), 2), "p95": round(float(np.percentile(dist, 95)), 2), "max": round(float(dist.max()), 2)},
           "shape_dev_parts_aligned_mm": {"mean": round(float(dloc.mean()), 2), "p95": round(float(np.percentile(dloc, 95)), 2)},
           "parts": parts,
           "concave_areas_fill_in_mm": round(float(dn[conc].mean()), 2), "convex_areas_move_mm": round(float(dn[conv].mean()), 2),
           "silhouette_iou": silhouettes(xs, X0), "key_widths": wcmp, "pieces": per_piece}
    return rep, xs, dn


def coloured(x, dn, path, lim=4.0):
    t = np.clip(dn / lim, -1, 1)
    c = np.c_[np.where(t > 0, 1, 1 + t), 1 - np.abs(t), np.where(t < 0, 1, 1 - t)]
    c = 0.82 * c + 0.1
    m = trimesh.Trimesh(x / K, F, vertex_colors=np.c_[(c * 255).astype(np.uint8), np.full(len(x), 255, np.uint8)], process=False)
    m.export(path)


# plush scale: flatten_layout.py used k = size / REAL_LEN; recover it from the stored target length
if (fdir / "flatten_report.json").exists():
    K = float(json.load(open(fdir / "flatten_report.json"))["scale"])
else:
    K = float(np.ptp(X0[:, 1]) / np.ptp(trimesh.load(Path(pj).parent / "subdivided.obj", process=False).vertices[:, 1]))
VREG = region_masks()
_st = stations(K)
SELS = {}
for _n, (_ax, _v, _reg) in _st.items():
    _sel = (np.abs(X0[:, 1] - _v) < 1.5) if _ax == "y" else (np.abs(X0[:, 2] - _v) < 1.5)
    SELS[_n] = _sel & (VREG == _reg) if _reg != "body" else _sel & (VREG == "body") & (X0[:, 2] > ms["elbow_height_mm"] * K)
print({r_: int((VREG == r_).sum()) for r_ in set(VREG)})
print(f"plush scale k = {K:.4f} (plush nose to tail {REAL_LEN * K:.0f} mm), stuffing {STUFF:.0%} sphere stretch -> P = {P * 1e3:.2f} kPa per (N/mm) fabric stiffness")

# ---------------------------------------------------------------- run
if "--gradcheck" in sys.argv:
    DmI, A0 = rest_from_uv(UVall); rng = np.random.default_rng(0)
    xt = X0 + rng.normal(0, 0.3, X0.shape); W0, g0, _ = energy(xt.ravel(), DmI, A0)
    for j in rng.choice(3 * N, 6, replace=False):
        e = np.zeros(3 * N); e[j] = 1e-4
        fd = (energy(xt.ravel() + e, DmI, A0)[0] - energy(xt.ravel() - e, DmI, A0)[0]) / 2e-4
        print("grad check", j, round(float(g0[j]), 6), round(float(fd), 6))
    sys.exit()
report = {"model": {"membrane": "tension-field (kt=1, kc=%.3g)" % KC, "bending_length_mm": BEND, "stuffing_sphere_stretch": STUFF,
                    "pressure_rel": P, "merge_radius_mm": R_MERGE},
          "assembly": {"pieces": [p[0] for p in pieces], "vertices": N, "triangles": int(len(F)), "open_edges": open_edges,
                       "non_manifold_edges": nonmanifold, "collapsed_triangles": n_degenerate, "target_volume_l": round(vol0 / 1e6, 3)},
          "seams_before": seam_lengths(UVall)}
uv_cur = UVall.copy()
x_init = np.load(sys.argv[sys.argv.index("--x0") + 1]) if "--x0" in sys.argv else X0
x, r, s, gn = simulate(uv_cur, x_init)
np.save(out / "sim_raw.npy", x)
rep, xs, dn = evaluate(x, "sewn as cut")
rep["converged"] = bool(r.success or r.residual_p95 < 0.05 * P * np.median(hv) ** 2)
rep["fabric_slack_share"] = round(float((s.min(1) < 0.98).mean()), 3)
rep["fabric_stretch"] = {"median_pct": round(float(100 * (np.median(s) - 1)), 2), "p95_pct": round(float(100 * (np.percentile(s, 95) - 1)), 2),
                         "compressed_share": round(float((s < 0.995).mean()), 3)}
report["as_cut"] = rep
coloured(X0, np.zeros(N), out / "target.ply"); coloured(xs, dn, out / "sim.ply")
np.save(out / "sim_x.npy", xs)
print("as cut:", {q: rep[q] for q in ("size_change_pct", "shape_dev_mm", "shape_dev_parts_aligned_mm", "concave_areas_fill_in_mm", "silhouette_iou")})
print("parts:", rep["parts"])

# ---------------------------------------------------------------- compensation
def arap_fit(uv, Fp, Lt, iters=12, exact_outline=True):
    """re-fit a flat piece so its triangle edges take the lengths Lt (m,3: edges 01,12,20); ARAP local-global.
    exact_outline: the outline (every seam) is first rebuilt with exactly its target edge lengths, keeping its turning
    angles and closing it with the smallest length changes, then held fixed while the inside is re-fitted; so both
    sides of each seam stay the same length."""
    n = len(uv)
    l01, l12, l20 = Lt[:, 0], Lt[:, 1], Lt[:, 2]
    cx = (l01 ** 2 + l20 ** 2 - l12 ** 2) / (2 * l01); cy = np.sqrt(np.maximum(l20 ** 2 - cx ** 2, (0.05 * l20) ** 2))
    Ref = np.stack([np.zeros((len(Fp), 2)), np.c_[l01, np.zeros(len(Fp))], np.c_[cx, cy]], 1)       # (m,3,2)
    e1_, e2_ = uv[Fp[:, 1]] - uv[Fp[:, 0]], uv[Fp[:, 2]] - uv[Fp[:, 0]]
    Ref[:, :, 1] *= np.sign(e1_[:, 0] * e2_[:, 1] - e1_[:, 1] * e2_[:, 0])[:, None]                   # keep orientation

    def cot(a, b, c):
        u, v = a - c, b - c; return (u * v).sum(1) / np.maximum(np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]), 1e-12)
    W = [np.clip(w, 1e-3, 1e3) for w in (cot(Ref[:, 1], Ref[:, 2], Ref[:, 0]), cot(Ref[:, 2], Ref[:, 0], Ref[:, 1]), cot(Ref[:, 0], Ref[:, 1], Ref[:, 2]))]
    E = [(1, 2), (2, 0), (0, 1)]                                   # edge opposite corner 0, 1, 2

    def solve(uv, fixed=None, Q=None):
        rows, cols, vals = [], [], []
        for w, (a, b) in zip(W, E):
            i_, j_ = Fp[:, a], Fp[:, b]
            rows += [i_, j_, i_, j_]; cols += [i_, j_, j_, i_]; vals += [w, w, -w, -w]
        Lm = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tolil()
        pins = [0] if fixed is None else list(fixed)
        for r_ in pins: Lm[r_, :] = 0; Lm[r_, r_] = 1
        lu = splu(Lm.tocsc())
        for _ in range(iters):
            Cv = np.zeros((len(Fp), 2, 2))
            for w, (a, b) in zip(W, E):
                dr, du = Ref[:, a] - Ref[:, b], uv[Fp[:, a]] - uv[Fp[:, b]]
                Cv += w[:, None, None] * du[:, :, None] * dr[:, None, :]
            U, _, Vt = np.linalg.svd(Cv); Rm = U @ Vt
            bad = np.linalg.det(Rm) < 0
            if bad.any(): U[bad, :, 1] *= -1; Rm[bad] = U[bad] @ Vt[bad]
            rhs = np.zeros((n, 2))
            for w, (a, b) in zip(W, E):
                v_ = w[:, None] * np.einsum("mij,mj->mi", Rm, Ref[:, a] - Ref[:, b])
                np.add.at(rhs, Fp[:, a], v_); np.add.at(rhs, Fp[:, b], -v_)
            if fixed is None: rhs[0] = uv[0]
            else: rhs[fixed] = Q
            uv = np.c_[lu.solve(rhs[:, 0]), lu.solve(rhs[:, 1])]
        return uv

    uv = solve(uv)
    if not exact_outline: return uv
    tgt = {}
    for k_, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
        for f_, (va, vb) in enumerate(zip(Fp[:, a], Fp[:, b])): tgt.setdefault((min(va, vb), max(va, vb)), []).append(Lt[f_, k_])
    bnd = igl.boundary_loop(np.ascontiguousarray(Fp, np.int64)); nb = len(bnd)
    Lb = np.array([np.mean(tgt[(min(bnd[q], bnd[(q + 1) % nb]), max(bnd[q], bnd[(q + 1) % nb]))]) for q in range(nb)])
    P_ = uv[bnd]; d_ = np.roll(P_, -1, axis=0) - P_; t_ = d_ / np.linalg.norm(d_, axis=1)[:, None]
    e_ = (Lb[:, None] * t_).sum(0); lam = np.linalg.solve((t_.T * Lb) @ t_, e_)
    Ln = Lb - Lb * (t_ @ lam)                                       # closes the outline with the least change
    Q = np.vstack([[0.0, 0.0], np.cumsum(Ln[:, None] * t_, axis=0)[:-1]]) + P_[0]
    uv0 = uv.copy(); uv0[bnd] = Q
    return solve(uv0, fixed=bnd, Q=Q)


history = []
DAMP = opt("--damp", 0.7)
if NCOMP:
    x_prev = x; best = None
    for it in range(NCOMP):
        # per-face-edge ratio target / simulated (after removing the overall size change, which the export rescales)
        xs_it, sc_it = similarity(x_prev, X0)
        new_uv = uv_cur.copy()
        EK = ((0, 1), (1, 2), (2, 0))
        Lr_all = np.stack([np.linalg.norm(uv_cur[Fall[:, a]] - uv_cur[Fall[:, b]], axis=1) for a, b in EK], 1)
        Lt_all = np.stack([np.linalg.norm(X0[F[:, a]] - X0[F[:, b]], axis=1) for a, b in EK], 1)
        Ls_all = np.stack([np.linalg.norm(x_prev[F[:, a]] - x_prev[F[:, b]], axis=1) for a, b in EK], 1)
        # an edge in tension (stretch s = Ls / Lr > 1) is cut shorter by s; an edge that is slack (s < 1, the fabric
        # ripples there) cannot be pulled out by cutting it longer, so its target is the plain target length
        desired = Lt_all / np.maximum(Ls_all / np.maximum(Lr_all, 1e-9), 1.0)
        Lnew_all = Lr_all * np.clip(desired / np.maximum(Lr_all, 1e-9), 0.8, 1.25) ** DAMP      # damped fixed-point step
        # both sides of a seam (and of a dart) get the same target length: the mean of what each side asks for
        keys = np.stack([np.minimum(F[:, a], F[:, b]) * N + np.maximum(F[:, a], F[:, b]) for a, b in EK], 1).ravel()
        _, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
        Lnew_all = (np.bincount(inv, Lnew_all.ravel()) / cnt)[inv].reshape(-1, 3)
        for i, p in enumerate(pieces):
            if p[2]: continue                                         # left pieces follow their right piece
            fi = np.where(pidx == i)[0]; Fp_g = Fall[fi]
            loc = np.unique(Fp_g); remap = -np.ones(len(UVall), int); remap[loc] = np.arange(len(loc))
            Fp = remap[Fp_g]; uv_p = uv_cur[loc]
            uv_new = arap_fit(uv_p, Fp, Lnew_all[fi])
            new_uv[loc] = uv_new
            for j, q in enumerate(pieces):                          # its mirror
                if q[2] and q[1] == p[0]:
                    qloc = np.unique(Fall[pidx == j]); new_uv[qloc] = uv_new * [-1, 1]
        uv_cur = new_uv
        x_prev, r, s, gn = simulate(uv_cur, x_prev)
        rep_c, xs_c, dn_c = evaluate(x_prev, f"compensated round {it + 1}")
        sl = seam_lengths(uv_cur); rep_c["seam_mismatch_max_pct"] = max((v["mismatch_pct"] for v in sl.values()), default=0.0)
        rep_c["fabric_slack_share"] = round(float((s.min(1) < 0.98).mean()), 3)
        oc = []
        for i, p in enumerate(pieces):
            if p[2]: continue
            bl = igl.boundary_loop(np.ascontiguousarray(p[4], np.int64)); b0 = off[i]
            u0, u1 = UVall[b0: b0 + len(p[3])][bl], uv_cur[b0: b0 + len(p[3])][bl]
            oc.append(np.linalg.norm(np.roll(u1, -1, 0) - u1, axis=1).sum() / np.linalg.norm(np.roll(u0, -1, 0) - u0, axis=1).sum() - 1)
        rep_c["outline_change_pct_range"] = [round(100 * min(oc), 1), round(100 * max(oc), 1)]
        history.append({q: rep_c[q] for q in ("size_change_pct", "shape_dev_mm", "shape_dev_parts_aligned_mm", "parts", "concave_areas_fill_in_mm", "silhouette_iou", "key_widths", "seam_mismatch_max_pct", "fabric_slack_share", "outline_change_pct_range")})
        np.save(out / f"sim_comp{it + 1}_raw.npy", x_prev)
        print(f"round {it + 1}:", history[-1])
        rep_c["converged"] = bool(r.success or r.residual_p95 < 0.05 * P * np.median(hv) ** 2); rep_c["round"] = it + 1
        cuttable = max(abs(q) for q in rep_c["outline_change_pct_range"]) <= 25
        score = rep_c["shape_dev_parts_aligned_mm"]["p95"] + abs(rep_c["size_change_pct"])     # mm of shape + % of size
        rep_c["score"] = round(score, 2)
        if cuttable and (best is None or score < best[0]["score"]):
            best = (rep_c, xs_c, dn_c, uv_cur.copy())
    if best is None: best = (rep_c, xs_c, dn_c, uv_cur.copy())
    rep_c, xs_c, dn_c, uv_cur = best                                  # keep the best cuttable round's pattern
    print("best round:", rep_c["round"])
    report["compensated"] = rep_c; report["compensation_history"] = history
    report["seams_after"] = seam_lengths(uv_cur)
    coloured(xs_c, dn_c, out / "sim_comp.ply")
    np.save(out / "sim_comp_x.npy", xs_c)
    fc = out / "flat_comp"; fc.mkdir(exist_ok=True)
    for i, p in enumerate(pieces):
        if p[2]: continue
        d = dict(np.load(fdir / f"{p[0]}.npz"))
        loc = np.unique(Fall[pidx == i]); base = off[i]
        uv_p = uv_cur[base: base + len(p[3])]
        d["uv"] = uv_p; np.savez(fc / f"{p[0]}.npz", **d)
        e1_, e2_ = uv_p[p[4][:, 1]] - uv_p[p[4][:, 0]], uv_p[p[4][:, 2]] - uv_p[p[4][:, 0]]
        sa = e1_[:, 0] * e2_[:, 1] - e1_[:, 1] * e2_[:, 0]
        bnd_p = igl.boundary_loop(np.ascontiguousarray(p[4], np.int64))
        Lo = np.linalg.norm(np.roll(UVall[base: base + len(p[3])][bnd_p], -1, 0) - UVall[base: base + len(p[3])][bnd_p], axis=1).sum()
        Ln_ = np.linalg.norm(np.roll(uv_p[bnd_p], -1, 0) - uv_p[bnd_p], axis=1).sum()
        report.setdefault("pattern_check", {})[p[0]] = {"flipped_triangles": int((np.sign(sa) != np.sign(np.median(sa))).sum()),
                                                        "outline_valid": bool(outline_ok(uv_p, bnd_p)), "outline_change_pct": round(100 * (Ln_ / Lo - 1), 1)}

# ---------------------------------------------------------------- verdict
final = report.get("compensated", report["as_cut"])
bad_w = [n for n, v in final["key_widths"].items() if any(abs(q["change_pct"]) > 7 for q in v.values())]
bad_cut = [q for q, v in report.get("pattern_check", {}).items() if not v["outline_valid"] or abs(v["outline_change_pct"]) > 25]
seam_bad = [s_ for s_, v in report.get("seams_after", report["seams_before"]).items() if v["mismatch_pct"] > 3]
tilted = [q for q, v in final["parts"].items() if v["tilt_deg"] > 5]
report["verdict"] = {"closed_surface": open_edges == 0, "shape_p95_ok": final["shape_dev_parts_aligned_mm"]["p95"] <= TOL, "parts_tilted_over_5deg": tilted, "widths_off_by_more_than_7pct": bad_w,
                     "seams_mismatch_over_3pct": seam_bad, "pieces_not_cuttable": bad_cut,
                     "worst_pieces": sorted(final["pieces"], key=lambda q: -final["pieces"][q]["dev_p95_mm"])[:4]}
json.dump(report, open(out / "report.json", "w"), indent=1)
ok = open_edges == 0 and final["converged"] and final["shape_dev_parts_aligned_mm"]["p95"] <= TOL and not bad_w and not tilted and not bad_cut
print("key widths (sewn vs target, %):", {n: [v[q]["change_pct"] for q in v] for n, v in final["key_widths"].items()})
print("worst pieces:", {q: final["pieces"][q]["dev_p95_mm"] for q in report["verdict"]["worst_pieces"]})
print("RESULT:", "PASS" if ok else f"FAIL (closed {open_edges == 0}, converged {final['converged']}, part shape p95 {final['shape_dev_parts_aligned_mm']['p95']} mm vs {TOL}, widths off {bad_w}, tilted {tilted})")
