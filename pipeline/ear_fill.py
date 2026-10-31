"""Remove fused ears from a mirror-symmetric occupancy field and fill the skull beneath them.

Around a horizontal axis (x = 0, z = mean height of the traced ear outline) the head surface radius r(y, angle)
is read along rays. Where the outermost surface falls inside the traced side-view ear outline (grown by a
buffer), the ear is cleared and r is replaced by a harmonic fill from the surrounding skull radius. Harmonic
(not biharmonic) so the fill never bulges past its boundary; then a smooth ear-root cushion is added along
the top of the footprint (see below), never beyond the raw surface.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage
from shapely import contains_xy
from shapely.geometry import Polygon


def smoothstep(t):
    t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)


def ear_fill(field, xs, ys, zs, p, outline, buffer_mm=8.0, theta_max_deg=80.0, extra_mask=None, root_mm=12.0, debug=False):
    """extra_mask: optional (mask[ny, nz], cell, y0, z0) side-view ear mask from the 3D model's texture. The 3D ear
    can hang lower than the photo's ear, so the footprint is the union of both (the mask grown by buffer_mm)."""
    poly = Polygon(outline).buffer(buffer_mm)
    zax = float(outline[:, 1].mean())
    ylo, yhi = poly.bounds[0], poly.bounds[2]
    if extra_mask is not None:
        mk, cell, my0, mz0 = extra_mask
        mk = ndimage.binary_dilation(mk, iterations=max(1, int(round(buffer_mm / cell))))
        cols_ = np.where(mk.any(1))[0]
        ylo, yhi = min(ylo, my0 + cols_.min() * cell), max(yhi, my0 + (cols_.max() + 1) * cell)
    yg = np.arange(ylo - 3 * p, yhi + 3 * p, p)
    dth = np.deg2rad(1.5); th = np.arange(np.deg2rad(-85), np.deg2rad(85) + 1e-9, dth)
    rho = np.arange(0, 220, p / 2)
    Yg, Tg, Rg = np.meshgrid(yg, th, rho, indexing="ij")
    idx = [(Rg * np.cos(Tg) - xs[0]) / p, (Yg - ys[0]) / p, (zax + Rg * np.sin(Tg) - zs[0]) / p]
    val = ndimage.map_coordinates(field, [i.ravel() for i in idx], order=1, mode="constant", cval=0.0).reshape(Yg.shape)
    ins = val > 0.5
    ok = ins[:, :, 0]                                              # axis inside the head on this row
    first_out = np.argmax(~ins, axis=2)
    last_in = ins.shape[2] - 1 - np.argmax(ins[:, :, ::-1], axis=2)
    r_first = rho[first_out] - p / 4; r_last = rho[last_in] + p / 4
    zo = zax + r_last * np.sin(th)[None, :]
    YY = np.repeat(yg[:, None], len(th), 1)
    in_ear = contains_xy(poly, YY, zo)
    if extra_mask is not None:
        iy = np.floor((YY - my0) / cell).astype(int); iz = np.floor((zo - mz0) / cell).astype(int)
        okm = (iy >= 0) & (iy < mk.shape[0]) & (iz >= 0) & (iz < mk.shape[1])
        in_mask = np.zeros_like(in_ear); in_mask[okm] = mk[iy[okm], iz[okm]]
        in_ear |= in_mask
    foot = in_ear & (np.abs(th)[None, :] <= np.deg2rad(theta_max_deg)) & ok
    ny, nt = foot.shape; n = ny * nt; rbar = float(np.median(r_first[ok]))
    hy, ht = p, rbar * dth
    I = np.arange(n).reshape(ny, nt)
    rows, cols, vals = [], [], []
    for (a_, b_), h in [((I[1:, :], I[:-1, :]), hy), ((I[:, 1:], I[:, :-1]), ht)]:
        rows += [a_.ravel(), b_.ravel()]; cols += [b_.ravel(), a_.ravel()]; vals += [np.full(a_.size, 1.0 / h ** 2)] * 2
    W = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    L = (sp.diags(np.asarray(W.sum(1)).ravel()) - W).tocsr()
    U = foot.ravel(); K = ~U
    r = r_first.ravel().astype(float).copy()
    if U.any():
        r[U] = spla.spsolve(L[U][:, U].tocsc(), -(L[U][:, K] @ r[K]))
    r_fill = r.reshape(ny, nt)
    # Ear-root cushion: a Lab's broad, flat skull top is largely the ear roots sitting on its top corners.
    # Keep a smooth root (root_mm thick) along the top of the ear footprint so the head keeps that outline and
    # the sewn ear has a base to sit on; it fades out towards the crown, the ear's lower part and its ends.
    if root_mm > 0 and foot.any():
        z_crown = zax + r_first[:, -1] * np.sin(th[-1])                    # crown height on each row
        zf = zax + r_fill * np.sin(th)[None, :]
        w_h = smoothstep((zf - (z_crown[:, None] - 42.0)) / 20.0)           # top ~25 mm of the side of the skull
        w_t = smoothstep((np.deg2rad(theta_max_deg) - th[None, :]) / np.deg2rad(18.0))
        ya_ = yg[foot.any(1)]; w_y = smoothstep((yg - ya_.min() - 8) / 20.0) * smoothstep((ya_.max() - 8 - yg) / 20.0)
        w = ndimage.gaussian_filter(np.where(foot, w_h * w_t * w_y[:, None], 0.0), 1.5)
        r_fill = r_fill + root_mm * w
    r_fill = np.minimum(r_fill, r_last)                              # never add material outside the raw surface
    X3, Y3, Z3 = np.meshgrid(xs, ys, zs, indexing="ij")
    box = (Y3 >= yg[0]) & (Y3 <= yg[-1]) & (np.abs(Z3 - zax) < 230) & (np.abs(X3) > 1e-9)
    ax_, ay_, az_ = np.abs(X3[box]), Y3[box], Z3[box] - zax
    t_ = np.arctan2(az_, ax_); rr = np.hypot(ax_, az_)
    iy_ = np.clip(np.rint((ay_ - yg[0]) / p).astype(int), 0, ny - 1); it_ = np.rint((t_ - th[0]) / dth).astype(int)
    inr = (it_ >= 0) & (it_ < nt); it_ = np.clip(it_, 0, nt - 1)
    sel = inr & foot[iy_, it_] & (rr < r_last[iy_, it_] + 10.0)
    newv = (rr < r_fill[iy_, it_]).astype(field.dtype)
    sub = field[box]; sub[sel] = newv[sel]; field[box] = sub
    info = {"ear_footprint_cells": int(foot.sum()), "ear_axis_z_mm": round(zax, 1),
            "ear_mean_thickness_removed_mm": round(float((r_last - r_fill)[foot].mean()), 1) if foot.any() else 0.0,
            "ear_max_thickness_removed_mm": round(float((r_last - r_fill)[foot].max()), 1) if foot.any() else 0.0}
    if debug:
        info["_maps"] = dict(yg=yg, th=th, r_first=r_first, r_last=r_last, r_fill=r_fill, foot=foot)
    return field, info
