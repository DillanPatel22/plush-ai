"""Shared seam-layout tools: seam straightening on the surface and piece export (used by seams_standing_v2.py
and transfer_layout.py, so a layout transferred to another breed is cleaned and written exactly the same way)."""
import json
import numpy as np, trimesh
from scipy.spatial import cKDTree


def clean_labels(m, label, only=None, swap=None):
    """Majority vote (one ring, then two rings) and island removal, as seams_standing_v2.py cleans its labels;
    only faces whose label is in `only` may change. With `swap` (label -> mirrored label) the result is made
    exactly mirror-symmetric. Returns the cleaned labels."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    label = label.copy(); C = m.triangles_center
    can = np.ones(len(label), bool) if only is None else np.isin(label, list(only))
    adj = m.face_adjacency
    Aj = sp.coo_matrix((np.ones(len(adj) * 2), (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])), shape=(len(C),) * 2).tocsr()
    for _ in range(12):
        new = label.copy()
        for i in np.where(can)[0]:
            nb = label[Aj.indices[Aj.indptr[i]:Aj.indptr[i + 1]]]
            vals, cnt = np.unique(nb, return_counts=True)
            if cnt.max() >= 2 and vals[np.argmax(cnt)] != label[i] and (only is None or vals[np.argmax(cnt)] in only): new[i] = vals[np.argmax(cnt)]
        label = new
    A2 = (Aj @ Aj).tocsr(); A2.setdiag(0); A2.eliminate_zeros()
    for _ in range(8):
        new = label.copy()
        for i in np.where(can)[0]:
            nb = label[A2.indices[A2.indptr[i]:A2.indptr[i + 1]]]
            vals, cnt = np.unique(nb, return_counts=True)
            if vals[np.argmax(cnt)] != label[i] and cnt.max() > 0.6 * len(nb) and (only is None or vals[np.argmax(cnt)] in only): new[i] = vals[np.argmax(cnt)]
        label = new
    if swap:
        mir = cKDTree(C).query(C * [-1, 1, 1])[1]; neg = C[:, 0] < 0
        sel = neg & can
        label[sel] = [swap.get(l, l) for l in label[mir[sel]]]
    for it in range(3):
        for L in np.unique(label[can]):
            idx = np.where(label == L)[0]
            n, lab = connected_components(Aj[idx][:, idx], directed=False)
            if n <= 1: continue
            big = np.argmax(np.bincount(lab))
            for k in np.where(lab != big)[0]:
                i = idx[k]; nb = label[Aj.indices[Aj.indptr[i]:Aj.indptr[i + 1]]]; nb = nb[nb != L]
                if len(nb): label[i] = np.bincount(nb).argmax()
    return label


def straighten(m, label):
    """Smooth every seam curve within the surface (junction points and sole rims stay put); exact symmetry kept."""
    orig = m.copy()
    VF = m.vertex_faces
    vlabels = [set(label[f] for f in row if f >= 0) for row in VF]
    nlab = np.array([len(s_) for s_ in vlabels])
    E = m.edges_unique
    def pair(i): return tuple(sorted(vlabels[i])) if nlab[i] == 2 else None
    nbrs = {}
    for a_, b_ in E:
        if nlab[a_] >= 2 and nlab[b_] >= 2:
            pa, pb = pair(a_), pair(b_)
            # boundary edge: both faces on the edge carry different labels
            nbrs.setdefault(a_, []).append(b_); nbrs.setdefault(b_, []).append(a_)
    # keep only true seam edges (the edge separates two different labels)
    ef = m.face_adjacency_edges; fa = m.face_adjacency
    seam_edge = set()
    for (u_, w_), (f1, f2) in zip(ef, fa):
        if label[f1] != label[f2]: seam_edge.add((min(u_, w_), max(u_, w_)))
    nb_seam = {}
    for u_, w_ in seam_edge:
        nb_seam.setdefault(u_, []).append(w_); nb_seam.setdefault(w_, []).append(u_)
    # sole rims sit on a sharp edge: projecting smoothed points there flips triangles, so they stay put
    movable = [v_ for v_, nb_ in nb_seam.items() if len(nb_) == 2 and nlab[v_] == 2 and m.vertices[v_, 2] > 4.0]
    print("three-way junctions:", sum(1 for v_ in nb_seam if nlab[v_] >= 3))
    Vs = m.vertices.copy()
    for it in range(30):
        newV = Vs.copy()
        for v_ in movable:
            a_, b_ = nb_seam[v_]
            newV[v_] = 0.5 * Vs[v_] + 0.25 * (Vs[a_] + Vs[b_])
        cp, _, _ = trimesh.proximity.closest_point(orig, newV[movable])
        newV[movable] = cp
        Vs = newV
    # keep the straightened seams exactly symmetric (left pieces are mirrors of right pieces)
    mv_idx = cKDTree(orig.vertices).query(orig.vertices * [-1, 1, 1])[1]
    Vs = 0.5 * (Vs + Vs[mv_idx] * [-1, 1, 1])
    Vs[np.abs(orig.vertices[:, 0]) < 1e-6, 0] = 0.0
    m = trimesh.Trimesh(Vs, m.faces, process=False)
    print("seam vertices smoothed:", len(movable))

    return m


def export_pieces(m, label, PIECES, out, layout_name, col=None, extra=None):
    """Write P*.obj per piece, face_labels.npy, subdivided.obj, labelled.ply and pieces.json; return the report."""
    names = {L: f"{pid} {nm}" for L, (pid, nm, _) in PIECES.items()}
    report = {}
    for L, (pid, nm, mirror_of) in PIECES.items():
        fi = np.where(label == L)[0]
        if not len(fi): report[pid] = {"name": nm, "faces": 0, "euler": None}; continue
        piece = m.submesh([fi], append=True)
        piece.export(out / f"{pid}.obj")
        report[pid] = {"name": nm, "label": int(L), "faces": int(len(fi)), "area_cm2_real": round(piece.area / 100, 1),
                       "euler": int(piece.euler_number), "mirror_of": mirror_of}
    np.save(out / "face_labels.npy", label)
    m.export(out / "subdivided.obj")
    # coloured preview mesh
    COL = col or {27: (190, 50, 140), 28: (190, 50, 140), 1: (230, 80, 60), 2: (70, 140, 220), 3: (70, 140, 220), 4: (240, 170, 40), 21: (250, 120, 170), 22: (250, 120, 170),
           5: (120, 190, 90), 6: (120, 190, 90), 9: (250, 220, 90), 26: (250, 220, 90), 10: (170, 100, 200), 11: (170, 100, 200), 14: (40, 120, 90), 15: (40, 120, 90),
           16: (150, 220, 200), 17: (150, 220, 200), 18: (200, 120, 60), 19: (200, 120, 60), 24: (240, 190, 150), 25: (240, 190, 150),
           30: (90, 90, 90), 31: (90, 90, 90), 32: (90, 90, 90), 33: (90, 90, 90), 12: (60, 190, 190), 13: (60, 190, 190)}
    fc = np.array([COL.get(int(l), (0, 0, 0)) for l in label], np.uint8)
    vis = m.copy(); vis.visual.face_colors = np.c_[fc, np.full(len(label), 255, np.uint8)]
    vis.export(out / "labelled.ply")
    json.dump({"layout": layout_name, "pieces": report, **(extra or {})}, open(out / "pieces.json", "w"), indent=1)
    for k, v in report.items(): print(k, v)
    bad = [k for k, v in report.items() if v["euler"] != 1]
    print("RESULT:", "PASS" if not bad else f"CHECK topology {bad} (euler != 1 means not a single disk)")

    return report
