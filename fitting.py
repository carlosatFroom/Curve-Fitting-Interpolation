"""Pure numerical core for piecewise polynomial curve fitting.

No GUI or plotting imports: this module must stay importable without PyQt,
pyqtgraph or matplotlib.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import Chebyshev
from numpy.polynomial import chebyshev as cheb
from numpy.polynomial import Polynomial


def chunk_bounds(
    n_points: int, n_chunks: int, overlap_pct: float = 0.0
) -> list[tuple[int, int]]:
    """Half-open (start, stop) index ranges, one per chunk, optionally overlapping."""
    if n_chunks < 1 or n_chunks > n_points:
        raise ValueError(f"n_chunks must be in [1, {n_points}], got {n_chunks}")
    if overlap_pct < 0:
        raise ValueError("overlap_pct must be >= 0")

    bounds = []
    for idx, part in enumerate(np.array_split(np.arange(n_points), n_chunks)):
        base_start, base_stop = int(part[0]), int(part[-1]) + 1
        ov = int(len(part) * overlap_pct / 100)
        start, stop = base_start, base_stop
        if idx == 0:
            stop += ov
        elif idx == n_chunks - 1:
            start -= ov
        else:
            start -= ov // 2
            stop += ov - ov // 2
        bounds.append((max(0, start), min(n_points, stop)))
    return bounds


@dataclass
class ChunkFit:
    start: int
    stop: int
    x_min: float
    x_max: float
    coeffs: np.ndarray
    rmse: float
    poly: Polynomial | None = field(default=None, repr=False)
    base_start: int = 0
    base_stop: int = 0
    base_x_min: float | None = None
    base_x_max: float | None = None

    def __post_init__(self):
        if self.base_x_min is None:
            self.base_x_min = self.x_min
        if self.base_x_max is None:
            self.base_x_max = self.x_max

    def predict(self, xq) -> np.ndarray:
        xq = np.asarray(xq, dtype=float)
        if self.poly is None:
            return np.polyval(self.coeffs, xq)
        return self.poly(xq)


def _fit_slice(x, y, start, stop, order) -> ChunkFit:
    xs, ys = x[start:stop], y[start:stop]
    degree = min(order, len(xs) - 1)
    poly = Polynomial.fit(xs, ys, degree)
    return ChunkFit(
        start=start,
        stop=stop,
        x_min=float(xs[0]),
        x_max=float(xs[-1]),
        coeffs=poly.convert().coef[::-1],
        rmse=rmse(ys, poly(xs)),
        poly=poly,
    )


def fit_chunks(
    x, y, n_chunks: int, order: int, overlap_pct: float = 0.0
) -> list[ChunkFit]:
    """Fit one polynomial per chunk; a chunk too small for `order` gets len-1."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    base = chunk_bounds(len(x), n_chunks, 0.0)
    fits = []
    for (start, stop), (base_start, base_stop) in zip(
        chunk_bounds(len(x), n_chunks, overlap_pct), base
    ):
        fit = _fit_slice(x, y, start, stop, order)
        fit.base_start, fit.base_stop = base_start, base_stop
        fit.base_x_min = float(x[base_start])
        fit.base_x_max = float(x[base_stop - 1])
        fits.append(fit)
    return fits


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _base_assignment(fits: list[ChunkFit], xq: np.ndarray) -> np.ndarray:
    """Index of the chunk whose base range contains each query point."""
    base_starts = np.array([f.base_x_min for f in fits])
    idx = np.searchsorted(base_starts, xq, side="right") - 1
    return np.clip(idx, 0, len(fits) - 1)


def _blend_weights(fits: list[ChunkFit], xq: np.ndarray) -> list[np.ndarray]:
    """Partition-of-unity weights: smoothstep cross-fade across each seam.

    Seam j joins chunks j and j+1 over their x-overlap [a, b]. Without overlap
    (b <= a) the seam is a hard switch at a.
    """
    ramps = []
    for left, right in zip(fits, fits[1:]):
        a, b = right.x_min, left.x_max
        if b > a:
            ramps.append(_smoothstep((xq - a) / (b - a)))
        else:
            ramps.append((xq >= a).astype(float))

    weights = []
    last = len(fits) - 1
    for j in range(len(fits)):
        w = np.ones_like(xq)
        if j > 0:
            w = w * ramps[j - 1]
        if j < last:
            w = w * (1.0 - ramps[j])
        weights.append(w)
    return weights


def evaluate_piecewise(fits: list[ChunkFit], xq, blend: bool = True) -> np.ndarray:
    """Evaluate the piecewise model at `xq`.

    blend=False: each point uses the chunk whose base range contains it.
    blend=True: seams are cross-faded (smoothstep) across the overlap of adjacent
    chunks. Without overlap the seam is a hard switch.
    """
    xq = np.atleast_1d(np.asarray(xq, dtype=float))
    if not blend:
        assignment = _base_assignment(fits, xq)
        out = np.empty_like(xq)
        for k, fit in enumerate(fits):
            mask = assignment == k
            if mask.any():
                out[mask] = fit.predict(xq[mask])
        return out

    weights = _blend_weights(fits, xq)
    num = np.zeros_like(xq)
    den = np.zeros_like(xq)
    for fit, w in zip(fits, weights):
        mask = w > 0
        if mask.any():
            num[mask] += w[mask] * fit.predict(xq[mask])
            den[mask] += w[mask]
    fallback = evaluate_piecewise(fits, xq, blend=False)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), fallback)


def rmse(y_true, y_pred) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    ss_res = np.sum((y_true - np.asarray(y_pred, dtype=float)) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def error_map(
    x, y, orders, chunk_counts, overlap_pcts, workers: int | None = None
) -> np.ndarray:
    """Un-normalised RMSE grid over exactly two varying sequences.

    Axis 0 is the first varying argument (orders, chunk_counts, overlap_pcts in
    that order); axis 1 the second.
    """
    axes = {
        "order": list(orders),
        "chunks": list(chunk_counts),
        "overlap": list(overlap_pcts),
    }
    if any(len(v) == 0 for v in axes.values()):
        raise ValueError("every sequence needs at least one value")
    varying = [name for name, vals in axes.items() if len(vals) > 1]
    if len(varying) != 2:
        raise ValueError("exactly two of orders/chunk_counts/overlap_pcts must vary")
    fixed = {name: vals[0] for name, vals in axes.items() if len(vals) == 1}
    first, second = varying

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def cell(pair):
        params = {**fixed, first: pair[0], second: pair[1]}
        fits = fit_chunks(x, y, params["chunks"], params["order"], params["overlap"])
        return rmse(y, evaluate_piecewise(fits, x, blend=False))

    grid = [(a, b) for a in axes[first] for b in axes[second]]
    if workers == 1:
        values = [cell(p) for p in grid]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            values = list(pool.map(cell, grid))
    return np.array(values).reshape(len(axes[first]), len(axes[second]))


def extrapolate(x, y, fit_fraction: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit one polynomial to the first `fit_fraction` of points; predict all x."""
    if not 0 < fit_fraction <= 1:
        raise ValueError("fit_fraction must be in (0, 1]")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if order + 1 > len(x):
        raise ValueError("not enough points for the requested order")
    m = min(len(x), max(order + 1, int(np.ceil(fit_fraction * len(x)))))
    poly = Polynomial.fit(x[:m], y[:m], order)
    return poly.convert().coef[::-1], poly(x)


# ---- PHASE 2 ----

"""Constrained piecewise-polynomial splines and adaptive chunking (pure numpy)."""



from numpy.polynomial import chebyshev as cheb
from numpy.polynomial import Chebyshev


def _base_bounds(n_points: int, n_chunks: int) -> list[tuple[int, int]]:
    parts = np.array_split(np.arange(n_points), n_chunks)
    return [(int(p[0]), int(p[-1]) + 1) for p in parts]


def _end_derivative_row(order: int, m: int, t: float) -> np.ndarray:
    """Row of m-th t-derivatives of the Chebyshev basis T_0..T_order at t."""
    row = np.zeros(order + 1)
    for j in range(order + 1):
        basis = np.zeros(order + 1)
        basis[j] = 1.0
        row[j] = cheb.chebval(t, cheb.chebder(basis, m))
    return row


def _check_xy(x, y) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape or x.size == 0:
        raise ValueError("x and y must be non-empty and the same length")
    if np.any(np.diff(x) < 0):
        raise ValueError("x must be non-decreasing")
    return x, y


@dataclass
class SplineFit:
    knots: np.ndarray        # (n_pieces - 1,) interior knots in original x
    centers: np.ndarray      # (n_pieces,) piece centres in original x
    halfs: np.ndarray        # (n_pieces,) piece half-widths in original x
    coeffs: np.ndarray       # (n_pieces, order + 1) Chebyshev coeffs in t = (x - centre) / half

    def _piece_index(self, xq: np.ndarray) -> np.ndarray:
        # A point exactly on a knot belongs to the right-hand piece.
        return np.searchsorted(self.knots, xq, side="right")

    def derivative(self, xq, m: int = 1) -> np.ndarray:
        """m-th derivative with respect to original x (m = 0 gives predictions)."""
        if int(m) != m or m < 0:
            raise ValueError("m must be a non-negative integer")
        m = int(m)
        arr = np.asarray(xq, dtype=float)
        flat = arr.ravel()
        idx = self._piece_index(flat)
        out = np.zeros(flat.shape)
        for i in np.unique(idx):
            mask = idx == i
            h = self.halfs[i]
            t = (flat[mask] - self.centers[i]) / h
            out[mask] = cheb.chebval(t, cheb.chebder(self.coeffs[i], m)) / h**m
        return out.reshape(arr.shape)

    def predict(self, xq) -> np.ndarray:
        return self.derivative(xq, 0)


def fit_spline(x, y, n_chunks: int, order: int, continuity: int = 2) -> SplineFit:
    """One degree-`order` polynomial per base chunk, fitted jointly with C^continuity joins."""
    x, y = _check_xy(x, y)
    if not -1 <= continuity < order:
        raise ValueError("continuity must satisfy -1 <= continuity < order")
    n = x.size
    if not 1 <= n_chunks <= n:
        raise ValueError("n_chunks must be between 1 and the number of points")

    q = order + 1
    n_pieces = n_chunks
    bounds = _base_bounds(n, n_chunks)
    knots = x[[s for s, _ in bounds[1:]]]
    lo = np.concatenate(([x[0]], knots))
    hi = np.concatenate((knots, [x[-1]]))
    centers = (lo + hi) / 2
    halfs = np.where((hi - lo) / 2 > 0, (hi - lo) / 2, 1.0)

    design = np.zeros((n, n_pieces * q))
    for i, (s, e) in enumerate(bounds):
        t = (x[s:e] - centers[i]) / halfs[i]
        design[s:e, i * q:(i + 1) * q] = cheb.chebvander(t, order)

    n_con = max(0, n_pieces - 1) * (continuity + 1)
    cons = np.zeros((n_con, n_pieces * q))
    row = 0
    left_rows = [_end_derivative_row(order, m, 1.0) for m in range(continuity + 1)]
    right_rows = [_end_derivative_row(order, m, -1.0) for m in range(continuity + 1)]
    for k in range(n_pieces - 1):
        for m in range(continuity + 1):
            cons[row, k * q:(k + 1) * q] = left_rows[m] / halfs[k] ** m
            cons[row, (k + 1) * q:(k + 2) * q] = -right_rows[m] / halfs[k + 1] ** m
            row += 1

    # Null-space elimination of the equality constraints: z = N w.
    if n_con:
        _, s_vals, vh = np.linalg.svd(cons, full_matrices=True)
        tol = s_vals.max() * max(cons.shape) * np.finfo(float).eps
        rank = int(np.sum(s_vals > tol))
        null = vh[rank:].T
    else:
        null = np.eye(n_pieces * q)

    w, *_ = np.linalg.lstsq(design @ null, y, rcond=None)
    coeffs = (null @ w).reshape(n_pieces, q)
    return SplineFit(knots=knots, centers=centers, halfs=halfs, coeffs=coeffs)


def _rmse(x: np.ndarray, y: np.ndarray, s: int, e: int, order: int) -> float:
    fit = Chebyshev.fit(x[s:e], y[s:e], order)
    return float(np.sqrt(np.mean((fit(x[s:e]) - y[s:e]) ** 2)))


def adaptive_chunks(x, y, order: int, tol: float, max_chunks: int) -> list[tuple[int, int]]:
    """Greedy minimal contiguous partition whose degree-`order` fits meet `tol` RMSE.

    Falls back to equal splits into `max_chunks` pieces when the greedy partition
    needs more chunks than allowed.
    """
    x, y = _check_xy(x, y)
    if order < 0 or tol < 0 or max_chunks < 1:
        raise ValueError("order >= 0, tol >= 0 and max_chunks >= 1 are required")
    n = x.size
    min_len = min(2 * (order + 1), n)

    bounds: list[tuple[int, int]] = []
    s = 0
    while s < n:
        base_end = min(s + min_len, n)
        if _rmse(x, y, s, base_end, order) > tol:
            e = base_end  # tol unreachable at minimal length: best effort, may exceed tol
        elif _rmse(x, y, s, n, order) <= tol:
            e = n
        else:
            good, bad = base_end, n  # good feasible, bad infeasible
            while bad - good > 1:
                mid = (good + bad) // 2
                if _rmse(x, y, s, mid, order) <= tol:
                    good = mid
                else:
                    bad = mid
            e = good
        bounds.append((s, e))
        s = e
        if len(bounds) > max_chunks:
            break

    if len(bounds) > max_chunks:
        return _base_bounds(n, min(max_chunks, n))
    return bounds




_MAD_TO_SIGMA = 1.4826
_TUKEY_C = 4.685
_MAX_IRLS_ITER = 50
_CV_FOLDS = 5
_METHODS = ("cv", "aic")


def _robust_poly(xs, ys, deg):
    poly = Polynomial.fit(xs, ys, deg)
    for _ in range(_MAX_IRLS_ITER):
        resid = ys - poly(xs)
        scale = _MAD_TO_SIGMA * np.median(np.abs(resid - np.median(resid)))
        if scale <= 0:
            break
        u = resid / (_TUKEY_C * scale)
        weights = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)
        if np.count_nonzero(weights) <= deg:
            break
        new = Polynomial.fit(xs, ys, deg, w=np.sqrt(weights))
        converged = np.max(np.abs(new.coef - poly.coef)) <= 1e-10 * (1.0 + np.max(np.abs(poly.coef)))
        poly = new
        if converged:
            break
    return poly


def robust_fit_chunks(x, y, n_chunks, order, overlap_pct=0.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fits = []
    for start, stop in chunk_bounds(len(x), n_chunks, overlap_pct):
        xs, ys = x[start:stop], y[start:stop]
        deg = min(order, len(xs) - 1)
        poly = _robust_poly(xs, ys, deg)
        resid = ys - poly(xs)
        fits.append(ChunkFit(
            start=start,
            stop=stop,
            x_min=float(xs.min()),
            x_max=float(xs.max()),
            coeffs=_power_coeffs(poly),
            rmse=float(np.sqrt(np.mean(resid**2))),
            poly=poly,
        ))
    return fits


def _power_coeffs(poly):
    off, scl = poly.mapparms()
    lin = Polynomial([off, scl])
    total = Polynomial([0.0])
    for k, c in enumerate(poly.coef):
        total = total + c * lin**k
    return total.coef[::-1]


def _cv_scores(x, y, n_chunks, max_order):
    bounds = chunk_bounds(len(x), n_chunks, 0.0)
    folds = np.arange(len(x)) % _CV_FOLDS
    scores = {}
    for order in range(max_order + 1):
        sse = 0.0
        count = 0
        feasible = True
        for start, stop in bounds:
            for fold in range(_CV_FOLDS):
                in_chunk = np.zeros(len(x), dtype=bool)
                in_chunk[start:stop] = True
                train = in_chunk & (folds != fold)
                test = in_chunk & (folds == fold)
                if not test.any():
                    continue
                if np.count_nonzero(train) <= order:
                    feasible = False
                    break
                poly = Polynomial.fit(x[train], y[train], order)
                sse += float(np.sum((y[test] - poly(x[test])) ** 2))
                count += int(np.count_nonzero(test))
            if not feasible:
                break
        if feasible and count:
            scores[order] = np.sqrt(sse / count)
    return scores


def _aic_scores(x, y, n_chunks, max_order):
    n = len(x)
    scores = {}
    for order in range(max_order + 1):
        fits = fit_chunks(x, y, n_chunks, order)
        rss = sum(f.rmse**2 * (f.stop - f.start) for f in fits)
        params = sum(len(f.coeffs) for f in fits)
        scores[order] = n * np.log(max(rss, 1e-300) / n) + 2 * params
    return scores


def select_order(x, y, n_chunks, max_order, method="cv"):
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {_METHODS}")
    if max_order < 0:
        raise ValueError("max_order must be >= 0")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if method == "cv":
        scores = _cv_scores(x, y, n_chunks, max_order)
    else:
        scores = _aic_scores(x, y, n_chunks, max_order)
    if not scores:
        return 0
    return int(min(scores, key=scores.get))
