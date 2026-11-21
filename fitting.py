"""Pure-numpy piecewise polynomial fitting (no GUI imports)."""
import concurrent.futures
import itertools
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import Polynomial


def chunk_bounds(n_points, n_chunks, overlap_pct=0.0):
    """Half-open (start, stop) ranges of the array_split partition, optionally overlapped."""
    n_points, n_chunks = int(n_points), int(n_chunks)
    if n_chunks < 1 or n_chunks > n_points:
        raise ValueError(f"n_chunks must be in [1, {n_points}], got {n_chunks}")
    base, rem = divmod(n_points, n_chunks)
    bounds, start = [], 0
    for i in range(n_chunks):
        length = base + (1 if i < rem else 0)
        stop = start + length
        lo, hi = start, stop
        ov = int(length * overlap_pct / 100) if overlap_pct > 0 else 0
        if ov and n_chunks > 1:
            if i == 0:
                hi += ov
            elif i == n_chunks - 1:
                lo -= ov
            else:
                lo -= ov // 2
                hi += ov - ov // 2
        bounds.append((max(lo, 0), min(hi, n_points)))
        start = stop
    return bounds


def _fit_poly(xs, ys, order):
    order = min(int(order), len(xs) - 1)
    if order <= 0 or np.ptp(xs) == 0:
        return Polynomial([float(np.mean(ys))])
    return Polynomial.fit(xs, ys, order)


def _poly_coeffs(poly):
    """Power-basis coefficients in original x, highest degree first."""
    return poly.convert().coef[::-1].copy()


@dataclass
class ChunkFit:
    start: int
    stop: int
    x_min: float
    x_max: float
    coeffs: np.ndarray
    rmse: float
    poly: Polynomial = field(repr=False, default=None)
    base_start: int = 0
    base_stop: int = 0
    base_x_min: float = 0.0
    base_x_max: float = 0.0

    def predict(self, xq):
        return self.poly(np.asarray(xq, dtype=float))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def _make_chunk_fits(x, y, n_chunks, order, overlap_pct, fitter):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length")
    base = chunk_bounds(len(x), n_chunks, 0)
    ext = chunk_bounds(len(x), n_chunks, overlap_pct)
    fits = []
    for (bs, be), (s, e) in zip(base, ext):
        xs, ys = x[s:e], y[s:e]
        poly = fitter(xs, ys, order)
        fits.append(ChunkFit(
            start=s, stop=e, x_min=float(xs[0]), x_max=float(xs[-1]),
            coeffs=_poly_coeffs(poly), rmse=rmse(ys, poly(xs)), poly=poly,
            base_start=bs, base_stop=be,
            base_x_min=float(x[bs]), base_x_max=float(x[be - 1]),
        ))
    return fits


def fit_chunks(x, y, n_chunks, order, overlap_pct=0.0):
    """Least-squares polynomial fit of degree `order` on each (overlapped) chunk."""
    return _make_chunk_fits(x, y, n_chunks, order, overlap_pct, _fit_poly)


def _seams(fits, blend):
    """Per-seam blend interval [a_i, b_i]; a == b means a hard switch at that x."""
    a, b = [], []
    for i in range(len(fits) - 1):
        left, right = fits[i], fits[i + 1]
        knot = right.base_x_min
        lo, hi = right.x_min, left.x_max
        if blend:
            c_left = 0.5 * (left.base_x_min + left.base_x_max)
            c_right = 0.5 * (right.base_x_min + right.base_x_max)
            lo, hi = max(lo, c_left), min(hi, c_right)
        if not blend or not lo < hi:
            lo = hi = knot
        a.append(lo)
        b.append(hi)
    return np.asarray(a, dtype=float), np.asarray(b, dtype=float)


def evaluate_piecewise(fits, xq, blend=True):
    """Evaluate the piecewise model; with blend, cross-fade (smoothstep) in overlaps."""
    xq = np.asarray(xq, dtype=float)
    flat = np.atleast_1d(xq).ravel()
    a, b = _seams(fits, blend)
    k = np.searchsorted(a, flat, side="right")  # index of the chunk owning x
    out = np.empty_like(flat)
    for j, fit in enumerate(fits):
        sel = k == j
        if sel.any():
            out[sel] = fit.predict(flat[sel])
    for i in range(len(a)):
        if b[i] <= a[i]:
            continue
        sel = (k == i + 1) & (flat <= b[i])
        if sel.any():
            t = (flat[sel] - a[i]) / (b[i] - a[i])
            w = t * t * (3.0 - 2.0 * t)
            out[sel] = (1.0 - w) * fits[i].predict(flat[sel]) + w * fits[i + 1].predict(flat[sel])
    return out.reshape(xq.shape) if xq.ndim else out[0]


def _map_cell(x, y, order, n_chunks, overlap):
    fits = fit_chunks(x, y, int(n_chunks), int(order), float(overlap))
    return rmse(y, evaluate_piecewise(fits, x, blend=False))


def error_map(x, y, orders, chunk_counts, overlap_pcts, workers=None):
    """2-D un-normalized RMSE map over the two varying parameters."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    seqs = [list(orders), list(chunk_counts), list(overlap_pcts)]
    varying = [i for i, s in enumerate(seqs) if len(s) > 1]
    if len(varying) != 2 or any(len(s) == 0 for s in seqs):
        raise ValueError("exactly two of orders/chunk_counts/overlap_pcts must have length > 1")
    ax0, ax1 = (seqs[i] for i in varying)
    cells = list(itertools.product(range(len(ax0)), range(len(ax1))))

    def params(i, j):
        p = [s[0] for s in seqs]
        p[varying[0]], p[varying[1]] = ax0[i], ax1[j]
        return p

    out = np.empty((len(ax0), len(ax1)))
    if workers == 1:
        for i, j in cells:
            out[i, j] = _map_cell(x, y, *params(i, j))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_map_cell, x, y, *params(i, j)): (i, j) for i, j in cells}
            for fut in concurrent.futures.as_completed(futures):
                out[futures[fut]] = fut.result()
    return out


def extrapolate(x, y, fit_fraction, order):
    """Fit the first `fit_fraction` of points; return (coeffs, prediction over all x)."""
    if not 0 < fit_fraction <= 1:
        raise ValueError("fit_fraction must be in (0, 1]")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < order + 1:
        raise ValueError("not enough points for the requested order")
    n_fit = min(n, max(int(order) + 1, int(round(fit_fraction * n))))
    poly = _fit_poly(x[:n_fit], y[:n_fit], order)
    return _poly_coeffs(poly), poly(x)
