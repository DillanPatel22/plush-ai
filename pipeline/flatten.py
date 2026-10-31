"""Shared flattening: LSCM initial guess, then ARAP to keep true lengths. Disk-topology triangle meshes only."""
import igl
import numpy as np


def flatten(V, F, iters=200):
    V = np.ascontiguousarray(V, dtype=np.float64); F = np.ascontiguousarray(F, dtype=np.int64)
    bnd = igl.boundary_loop(F)
    i0 = bnd[0]; i1 = bnd[np.argmax(np.linalg.norm(V[bnd] - V[i0], axis=1))]
    b = np.array([i0, i1], dtype=np.int64)
    bc = np.array([[0.0, 0.0], [np.linalg.norm(V[i1] - V[i0]), 0.0]])
    uv, _ = igl.lscm(V, F, b, bc)
    data = igl.ARAPData(); data.max_iter = iters
    pin = np.array([i0], dtype=np.int32)                 # ARAP needs one pinned vertex; scale comes from 3D lengths
    igl.arap_precomputation(V, F, 2, pin, data)
    uv = igl.arap_solve(np.ascontiguousarray(uv[pin]), data, np.ascontiguousarray(uv))
    return uv, bnd


def distortion(V, F, uv):
    """Per-face area ratio (flat / 3D) and per-edge length ratio."""
    def area3(P): return 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    def area2(P):
        a = P[:, 1] - P[:, 0]; b = P[:, 2] - P[:, 0]
        return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    a3 = area3(V[F]); a2 = area2(uv[F])
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    l3 = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1); l2 = np.linalg.norm(uv[E[:, 0]] - uv[E[:, 1]], axis=1)
    return a2 / np.maximum(a3, 1e-12), l2 / np.maximum(l3, 1e-12), a3


def flatten_injective(V, F, iters=60):
    """Tutte (boundary on a circle) start, then SLIM symmetric-Dirichlet with a free boundary: no flipped
    triangles, and far less prone to folding long appendages (legs) over the body than LSCM/ARAP."""
    V = np.asfortranarray(V, dtype=np.float64); F32 = np.asfortranarray(F, dtype=np.int32)
    Fi = np.ascontiguousarray(F, dtype=np.int64)
    bnd = igl.boundary_loop(Fi)
    bnd_uv = igl.map_vertices_to_circle(np.ascontiguousarray(V), bnd)
    uv0 = igl.harmonic(np.ascontiguousarray(V), Fi, bnd.astype(np.int64), bnd_uv, 1)
    # scale the start to the true area so SLIM converges faster
    def area2(P, F_):
        a = P[F_[:, 1]] - P[F_[:, 0]]; b = P[F_[:, 2]] - P[F_[:, 0]]
        return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()
    a3 = 0.5 * np.linalg.norm(np.cross(V[Fi[:, 1]] - V[Fi[:, 0]], V[Fi[:, 2]] - V[Fi[:, 0]]), axis=1).sum()
    uv0 = uv0 * np.sqrt(a3 / area2(uv0, Fi))
    data = igl.slim_precompute(V, F32, np.asfortranarray(uv0), igl.MappingEnergyType.SYMMETRIC_DIRICHLET,
                               np.zeros(0, np.int32), np.zeros((0, 2), order="F"), 0.0)
    uv = igl.slim_solve(data, iters)
    return np.asarray(uv), bnd


def prescribe_boundary_lengths(V, F, uv, bnd, iters=100):
    """Rebuild the flattened boundary so every boundary edge has its true 3D length (so both sides of every
    seam match), keeping the current boundary's turning angles; the curve is closed with the smallest
    length changes, then the interior is re-solved with ARAP and the boundary held fixed."""
    V = np.ascontiguousarray(V, float); F = np.ascontiguousarray(F, np.int64)
    n = len(bnd)
    P = uv[bnd]
    L = np.linalg.norm(V[np.roll(bnd, -1)] - V[bnd], axis=1)                  # 3D edge lengths
    d = np.roll(P, -1, axis=0) - P
    phi = np.arctan2(d[:, 1], d[:, 0])                                         # current edge directions
    t = np.c_[np.cos(phi), np.sin(phi)]                                        # unit directions
    e = (L[:, None] * t).sum(0)                                                # closure error
    Wd = L                                                                     # weight: longer edges absorb more
    TW = t.T * Wd                                                               # 2 x n
    lam = np.linalg.solve(TW @ t, e)
    Ln = L - Wd * (t @ lam)
    Q = np.vstack([[0.0, 0.0], np.cumsum(Ln[:, None] * t, axis=0)[:-1]]) + P[0]
    # interior: ARAP with the new boundary fixed, starting from the old map moved by the boundary change
    data = igl.ARAPData(); data.max_iter = iters
    b = np.ascontiguousarray(bnd.astype(np.int32))
    igl.arap_precomputation(V, F, 2, b, data)
    init = uv.copy(); init[bnd] = Q
    uv2 = igl.arap_solve(np.ascontiguousarray(Q), data, np.ascontiguousarray(init))
    closure_change = float(np.abs(Ln - L).max())
    return uv2, closure_change
