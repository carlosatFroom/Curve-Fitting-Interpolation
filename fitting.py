"""Pure-numpy piecewise polynomial fitting (no GUI imports)."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import product

import numpy as np
from numpy.polynomial import Polynomial

__all__ = [
    "chunk_bounds", "ChunkFit", "fit_chunks", "evaluate_piecewise", "rmse", "r2",
    "error_map", "extrapolate", "fit_spline", "SplineFit", "adaptive_chunks",
    "robust_fit_chunks", "select_order",
]


# ---------------------------------------------------------------- metrics

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------- chunking

def _base_bounds(n_points, n_chunks):
    sizes = np.full(n_chunks, n_points // n_chunks)
    sizes[: n_points % n_chunks] += 1  # same partition as np.array_split
    stops = np.cumsum(sizes)
    return [(int(b - s), int(b)) for s, b in zip(sizes, stops)]


def chunk_bounds(n_points: int, n_chunks: int, overlap_pct: float = 0.0):
    if n_chunks < 1 or n_chunks > n_points:
        raise ValueError(f"n_chunks must be in [1, {n_points}], got {n_chunks}")
    base = _base_bounds(n_points, n_chunks)
    if overlap_pct <= 0 or n_chunks == 1:
        return base
    out = []
    for i, (start, stop) in enumerate(base):
        ov = int((stop - start) * overlap_pct / 100)
        if i == 0:
            stop += ov
        elif i == n_chunks - 1:
            start -= ov
        else:
            start -= ov // 2
            stop += ov - ov // 2
        out.append((max(0, start), min(n_points, stop)))
    return out


# ---------------------------------------------------------------- chunk fits

@dataclass
class ChunkFit:
    start: int
    stop: int
    x_min: float
    x_max: float
    coeffs: np.ndarray
    rmse: float
    poly: Polynomial = field(repr=False)
    base_start: int = 0
    base_stop: int = 0
    base_x_min: float = 0.0
    base_x_max: float = 0.0

    @property
    def order(self) -> int:
        return len(self.coeffs) - 1

    def predict(self, xq) -> np.ndarray:
        return self.poly(np.asarray(xq, dtype=float))


def _poly_fit(x, y, order, w=None):
    """Stable fit in a scaled domain; order reduced if there are too few distinct x."""
    deg = max(0, min(order, len(x) - 1, len(np.unique(x)) - 1))
    if deg == 0:
        if w is None:
            c = float(np.mean(y))
        else:
            c = float(np.sum(w * y) / np.sum(w)) if np.sum(w) > 0 else float(np.mean(y))
        return Polynomial([c])
    return Polynomial.fit(x, y, deg, w=w)


def _coeffs(poly):
    return poly.convert().coef[::-1].copy()


def _make_fit(x, y, start, stop, base, poly):
    xs, ys = x[start:stop], y[start:stop]
    return ChunkFit(
        start=start, stop=stop,
        x_min=float(xs.min()), x_max=float(xs.max()),
        coeffs=_coeffs(poly), rmse=rmse(ys, poly(xs)), poly=poly,
        base_start=base[0], base_stop=base[1],
        base_x_min=float(x[base[0]]), base_x_max=float(x[base[1] - 1]),
    )


def _as_xy(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    return x, y


def fit_chunks(x, y, n_chunks: int, order: int, overlap_pct: float = 0.0):
    x, y = _as_xy(x, y)
    base = chunk_bounds(len(x), n_chunks, 0.0)
    bounds = chunk_bounds(len(x), n_chunks, overlap_pct)
    fits = []
    for (s, e), b in zip(bounds, base):
        fits.append(_make_fit(x, y, s, e, b, _poly_fit(x[s:e], y[s:e], order)))
    return fits


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def evaluate_piecewise(fits, xq, blend: bool = True) -> np.ndarray:
    xq = np.asarray(xq, dtype=float)
    shape = xq.shape
    xq = xq.ravel()
    if len(fits) == 1:
        return fits[0].predict(xq).reshape(shape)
    knots = np.array([f.base_x_min for f in fits[1:]])
    idx = np.searchsorted(knots, xq, side="right")
    out = np.empty_like(xq)
    for i, f in enumerate(fits):
        m = idx == i
        if m.any():
            out[m] = f.predict(xq[m])
    if blend:
        for left, right in zip(fits[:-1], fits[1:]):
            a, b = right.x_min, left.x_max
            if b <= a:
                continue
            m = (xq >= a) & (xq <= b)
            if m.any():
                w = _smoothstep((xq[m] - a) / (b - a))
                out[m] = (1.0 - w) * left.predict(xq[m]) + w * right.predict(xq[m])
    return out.reshape(shape)


# ---------------------------------------------------------------- error map

def _map_cell(x, y, order, n_chunks, overlap):
    if n_chunks > len(x):
        return np.nan
    fits = fit_chunks(x, y, int(n_chunks), int(order), float(overlap))
    return rmse(y, evaluate_piecewise(fits, x, blend=False))


def error_map(x, y, orders, chunk_counts, overlap_pcts, workers=None) -> np.ndarray:
    x, y = _as_xy(x, y)
    seqs = [list(orders), list(chunk_counts), list(overlap_pcts)]
    varying = [i for i, s in enumerate(seqs) if len(s) > 1]
    if len(varying) != 2 or any(len(s) == 0 for s in seqs):
        raise ValueError("exactly two of orders/chunk_counts/overlap_pcts must have length > 1, the third length 1")
    a, b = varying
    cells = list(product(range(len(seqs[a])), range(len(seqs[b]))))

    def run(ij):
        args = [s[0] for s in seqs]
        args[a], args[b] = seqs[a][ij[0]], seqs[b][ij[1]]
        return _map_cell(x, y, *args)

    if workers == 1:
        vals = [run(c) for c in cells]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            vals = list(ex.map(run, cells))
    return np.array(vals, dtype=float).reshape(len(seqs[a]), len(seqs[b]))


# ---------------------------------------------------------------- extrapolation

def extrapolate(x, y, fit_fraction: float, order: int):
    x, y = _as_xy(x, y)
    if not 0 < fit_fraction <= 1:
        raise ValueError("fit_fraction must be in (0, 1]")
    n_fit = min(len(x), max(order + 1, int(round(fit_fraction * len(x)))))
    poly = _poly_fit(x[:n_fit], y[:n_fit], order)
    return _coeffs(poly), poly(x)


# ---------------------------------------------------------------- phase 2: constrained spline
# (section owned by teammate "spline")


# ---------------------------------------------------------------- phase 2: adaptive / robust / order selection
# (section owned by teammate "robust")
