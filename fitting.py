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

_leg = np.polynomial.legendre


def _leg_basis(t, order, deriv=0):
    """Row per point, column per Legendre P_k (k<=order): its deriv-th d/dt."""
    t = np.asarray(t, dtype=float)
    out = np.zeros((t.size, order + 1))
    for k in range(deriv, order + 1):
        e = np.zeros(k + 1)
        e[k] = 1.0
        out[:, k] = _leg.legval(t, _leg.legder(e, deriv))
    return out


class SplineFit:
    """Piecewise polynomials on base chunks; piece i uses t = (x - centers[i]) / halfwidths[i]."""

    def __init__(self, knots, centers, halfwidths, coefs):
        self.knots = np.asarray(knots, dtype=float)
        self._centers = np.asarray(centers, dtype=float)
        self._halfwidths = np.asarray(halfwidths, dtype=float)
        self._coefs = np.asarray(coefs, dtype=float)  # (n_pieces, order + 1)
        self.order = self._coefs.shape[1] - 1

    def _eval(self, xq, m):
        xq_arr = np.asarray(xq, dtype=float)
        idx = np.searchsorted(self.knots, xq_arr.ravel(), side="right")
        t = (xq_arr.ravel() - self._centers[idx]) / self._halfwidths[idx]
        basis = _leg_basis(t, self.order, m)
        vals = np.sum(basis * self._coefs[idx], axis=1) * self._halfwidths[idx] ** (-m)
        return float(vals[0]) if xq_arr.ndim == 0 else vals.reshape(xq_arr.shape)

    def predict(self, xq):
        return self._eval(xq, 0)

    def derivative(self, xq, m: int = 1):
        if m < 0:
            raise ValueError("m must be >= 0")
        return self._eval(xq, m)


def fit_spline(x, y, n_chunks: int, order: int, continuity: int = 2) -> SplineFit:
    x, y = _as_xy(x, y)
    if not -1 <= continuity < order:
        raise ValueError("continuity must satisfy -1 <= continuity < order")
    bounds = chunk_bounds(len(x), n_chunks, 0.0)
    n_pieces, p = len(bounds), order + 1

    centers = np.empty(n_pieces)
    halfwidths = np.empty(n_pieces)
    A = np.zeros((len(x), n_pieces * p))
    for i, (s, e) in enumerate(bounds):
        lo, hi = x[s], x[e - 1]
        centers[i] = 0.5 * (lo + hi)
        halfwidths[i] = 0.5 * (hi - lo) if hi > lo else 1.0
        t = (x[s:e] - centers[i]) / halfwidths[i]
        A[s:e, i * p:(i + 1) * p] = _leg_basis(t, order)

    rows = []
    for k in range(n_pieces - 1):
        xk = x[bounds[k + 1][0]]
        ta = (xk - centers[k]) / halfwidths[k]
        tb = (xk - centers[k + 1]) / halfwidths[k + 1]
        for m in range(continuity + 1):
            r = np.zeros(n_pieces * p)
            r[k * p:(k + 1) * p] = _leg_basis([ta], order, m)[0] * halfwidths[k] ** (-m)
            r[(k + 1) * p:(k + 2) * p] = -_leg_basis([tb], order, m)[0] * halfwidths[k + 1] ** (-m)
            rows.append(r / np.linalg.norm(r))

    if rows:
        _, sv, vt = np.linalg.svd(np.vstack(rows))
        rank = int(np.sum(sv > sv[0] * 1e-10))
        null = vt[rank:].T
    else:
        null = np.eye(n_pieces * p)

    AN = A @ null
    scale = np.linalg.norm(AN, axis=0)
    scale[scale == 0] = 1.0
    z, *_ = np.linalg.lstsq(AN / scale, y, rcond=None)
    coefs = (null @ (z / scale)).reshape(n_pieces, p)

    knots = [x[s] for s, _ in bounds[1:]]
    return SplineFit(knots, centers, halfwidths, coefs)


# ---------------------------------------------------------------- phase 2: adaptive / robust / order selection
# (section owned by teammate "robust")


def _slice_rmse(x, y, a, b, order):
    xs, ys = x[a:b], y[a:b]
    return rmse(ys, _poly_fit(xs, ys, order)(xs))


def _greedy_partition(x, y, order, tol):
    """Longest-first partition; chunks shorter than `enforce` are exempt from tol."""
    n = len(x)
    enforce = 2 * (order + 1)

    def ok(a, b):
        return _slice_rmse(x, y, a, b, order) <= tol

    bounds, start = [], 0
    while start < n:
        if n - start < enforce:
            stop = n  # exempt tail
        elif not ok(start, start + enforce):
            stop = start + enforce - 1  # exempt short chunk
        else:
            lo, hi, step = start + enforce, None, 1  # lo is known to fit
            while lo < n:
                cand = min(n, lo + step)
                if ok(start, cand):
                    lo, step = cand, step * 2
                else:
                    hi = cand
                    break
            while hi is not None and hi - lo > 1:
                mid = (lo + hi) // 2
                if ok(start, mid):
                    lo = mid
                else:
                    hi = mid
            stop = lo
        bounds.append((start, stop))
        start = stop
    return bounds


def adaptive_chunks(x, y, order: int, tol: float, max_chunks: int):
    x, y = _as_xy(x, y)
    n = len(x)
    if n == 0:
        raise ValueError("need at least one point")
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if tol < 0:
        raise ValueError(f"tol must be >= 0, got {tol}")
    if max_chunks < 1:
        raise ValueError(f"max_chunks must be >= 1, got {max_chunks}")

    bounds = _greedy_partition(x, y, order, tol)
    if len(bounds) <= max_chunks:
        return bounds

    # Budget exceeded: repeatedly bisect the worst-RMSE chunk (best effort vs tol).
    enforce = 2 * (order + 1)
    parts = [(0, n, _slice_rmse(x, y, 0, n, order))]
    while len(parts) < max_chunks:
        cand = [i for i, (a, b, e) in enumerate(parts) if e > tol and b - a >= enforce]
        if not cand:
            break
        i = max(cand, key=lambda j: parts[j][2])
        a, b, _ = parts[i]
        m = (a + b) // 2
        parts[i:i + 1] = [(a, m, _slice_rmse(x, y, a, m, order)),
                          (m, b, _slice_rmse(x, y, m, b, order))]
    return [(a, b) for a, b, _ in parts]


def _robust_poly(xs, ys, order, n_iter=50, c=4.685):
    """Tukey-bisquare IRLS with MAD scale. Weights are unsquared (Polynomial.fit convention)."""
    poly = _poly_fit(xs, ys, order)
    deg = poly.degree()
    for _ in range(n_iter):
        r = ys - poly(xs)
        scale = 1.4826 * np.median(np.abs(r - np.median(r)))
        if scale == 0:
            scale = float(np.mean(np.abs(r)))
            if scale == 0:
                break  # exact fit already
        u = r / (c * scale)
        w = np.where(np.abs(u) < 1.0, 1.0 - u * u, 0.0)
        if np.count_nonzero(w) <= deg:
            break  # too few inliers left to determine the polynomial
        new = _poly_fit(xs, ys, order, w=w)
        converged = np.max(np.abs(new(xs) - poly(xs))) <= 1e-8 * scale
        poly = new
        if converged:
            break
    return poly


def robust_fit_chunks(x, y, n_chunks: int, order: int, overlap_pct: float = 0.0):
    x, y = _as_xy(x, y)
    base = chunk_bounds(len(x), n_chunks, 0.0)
    bounds = chunk_bounds(len(x), n_chunks, overlap_pct)
    return [
        _make_fit(x, y, s, e, b, _robust_poly(x[s:e], y[s:e], order))
        for (s, e), b in zip(bounds, base)
    ]


def select_order(x, y, n_chunks: int, max_order: int, method: str = "cv") -> int:
    x, y = _as_xy(x, y)
    if max_order < 0:
        raise ValueError(f"max_order must be >= 0, got {max_order}")
    if method not in ("cv", "aic"):
        raise ValueError(f"method must be 'cv' or 'aic', got {method!r}")
    n = len(x)

    if method == "cv":
        k = min(5, n)
        if k < 2:
            raise ValueError("cross-validation needs at least 2 points")
        fold = np.arange(n) % k  # interleaved: every fold spans the whole x range
        scores = []
        for order in range(max_order + 1):
            errs = []
            for f in range(k):
                train = fold != f
                test = ~train
                fits = fit_chunks(x[train], y[train], min(n_chunks, int(train.sum())), order)
                errs.append(rmse(y[test], evaluate_piecewise(fits, x[test], blend=False)))
            scores.append(np.mean(errs))
        return int(np.argmin(scores))

    scores = []
    var_floor = 1e-12 * max(float(np.var(y)), 1e-300)  # guards SSE == 0
    for order in range(max_order + 1):
        fits = fit_chunks(x, y, n_chunks, order)
        sse = float(np.sum((y - evaluate_piecewise(fits, x, blend=False)) ** 2))
        k_params = n_chunks * (order + 1)
        scores.append(n * np.log(max(sse / n, var_floor)) + 2 * k_params)
    return int(np.argmin(scores))
