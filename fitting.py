"""Pure-numpy piecewise polynomial fitting core (no GUI imports)."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import Polynomial


def _as_xy(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    return x, y


def _base_edges(n_points, n_chunks):
    if n_chunks < 1 or n_chunks > n_points:
        raise ValueError("n_chunks must satisfy 1 <= n_chunks <= n_points")
    lengths = [len(p) for p in np.array_split(np.arange(n_points), n_chunks)]
    stops = np.cumsum(lengths)
    starts = stops - np.asarray(lengths)
    return list(zip(starts.tolist(), stops.tolist()))


def chunk_bounds(n_points, n_chunks, overlap_pct=0.0):
    base = _base_edges(n_points, n_chunks)
    last = len(base) - 1
    bounds = []
    for i, (start, stop) in enumerate(base):
        ov = int((stop - start) * overlap_pct / 100)
        if i == 0:
            stop += ov
        if i == last:
            start -= ov
        if 0 < i < last:
            start -= ov // 2
            stop += ov - ov // 2
        bounds.append((max(0, start), min(n_points, stop)))
    return bounds


def _fit_poly(xs, ys, order):
    if order < 0:
        raise ValueError("order must be >= 0")
    deg = min(order, len(xs) - 1)
    return Polynomial.fit(xs, ys, deg)


@dataclass(eq=False)
class ChunkFit:
    start: int
    stop: int
    x_min: float
    x_max: float
    coeffs: np.ndarray
    rmse: float
    base_start: int
    base_stop: int
    base_x_lo: float
    base_x_hi: float
    _poly: Polynomial = field(repr=False)

    def predict(self, xq):
        return self._poly(np.asarray(xq, dtype=float))


def fit_chunks(x, y, n_chunks, order, overlap_pct=0.0):
    x, y = _as_xy(x, y)
    bounds = chunk_bounds(len(x), n_chunks, overlap_pct)
    base = _base_edges(len(x), n_chunks)
    fits = []
    for (start, stop), (b0, b1) in zip(bounds, base):
        xs, ys = x[start:stop], y[start:stop]
        poly = _fit_poly(xs, ys, order)
        fits.append(
            ChunkFit(
                start=start,
                stop=stop,
                x_min=float(xs.min()),
                x_max=float(xs.max()),
                coeffs=poly.convert().coef[::-1],
                rmse=rmse(ys, poly(xs)),
                base_start=b0,
                base_stop=b1,
                base_x_lo=float(x[b0]),
                base_x_hi=float(x[b1 - 1]),
                _poly=poly,
            )
        )
    return fits


def evaluate_piecewise(fits, xq, blend=True):
    xq = np.atleast_1d(np.asarray(xq, dtype=float))
    lows = np.array([f.base_x_lo for f in fits])
    idx = np.clip(np.searchsorted(lows, xq, side="right") - 1, 0, len(fits) - 1)
    out = np.empty_like(xq)
    for i, fit in enumerate(fits):
        mask = idx == i
        if mask.any():
            out[mask] = fit.predict(xq[mask])
    if blend:
        for left, right in zip(fits[:-1], fits[1:]):
            lo, hi = right.x_min, left.x_max
            if hi < lo:
                continue
            mask = (xq >= lo) & (xq <= hi)
            if not mask.any():
                continue
            width = hi - lo
            if width > 0:
                w = (xq[mask] - lo) / width
            else:
                w = np.full(mask.sum(), 0.5)
            out[mask] = (1 - w) * left.predict(xq[mask]) + w * right.predict(xq[mask])
    return out


def rmse(y_true, y_pred):
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def r2(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    ss_res = np.sum((yt - np.asarray(y_pred, dtype=float)) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else float("nan")
    return float(1 - ss_res / ss_tot)


def _as_list(values):
    if np.ndim(values) == 0:
        return [values]
    return list(values)


def _error_cell(args):
    x, y, order, n_chunks, overlap = args
    fits = fit_chunks(x, y, n_chunks, order, overlap)
    return rmse(y, evaluate_piecewise(fits, x, blend=False))


def error_map(x, y, orders, chunk_counts, overlap_pcts, workers=None):
    x, y = _as_xy(x, y)
    seqs = [_as_list(orders), _as_list(chunk_counts), _as_list(overlap_pcts)]
    varying = [i for i, s in enumerate(seqs) if len(s) > 1]
    if len(varying) != 2:
        raise ValueError("exactly two of orders, chunk_counts, overlap_pcts must have length > 1")
    a, b = varying
    tasks = []
    for va in seqs[a]:
        for vb in seqs[b]:
            vals = [s[0] for s in seqs]
            vals[a] = va
            vals[b] = vb
            tasks.append((x, y, vals[0], vals[1], vals[2]))
    if workers == 1:
        results = [_error_cell(t) for t in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_error_cell, tasks))
    return np.asarray(results, dtype=float).reshape(len(seqs[a]), len(seqs[b]))


def extrapolate(x, y, fit_fraction, order):
    if not 0 < fit_fraction <= 1:
        raise ValueError("fit_fraction must satisfy 0 < fit_fraction <= 1")
    x, y = _as_xy(x, y)
    n = len(x)
    m = min(n, max(order + 1, int(np.ceil(fit_fraction * n))))
    if m < order + 1:
        raise ValueError("need at least order + 1 points")
    poly = Polynomial.fit(x[:m], y[:m], order)
    return poly.convert().coef[::-1], poly(x)


try:
    from _phase2 import *  # noqa
except ImportError:
    pass
