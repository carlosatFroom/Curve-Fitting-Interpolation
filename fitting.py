"""Pure-numpy piecewise polynomial curve fitting.

Importing this module must NOT import PyQt5, pyqtgraph or matplotlib. It depends
only on numpy (and the standard library). scipy is allowed but not required.

Public API:
    chunk_bounds, ChunkFit, fit_chunks, evaluate_piecewise, rmse, r2,
    error_map, extrapolate,
    fit_spline, SplineFit, adaptive_chunks, robust_fit_chunks, select_order
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import Polynomial

__all__ = [
    "chunk_bounds",
    "ChunkFit",
    "fit_chunks",
    "evaluate_piecewise",
    "rmse",
    "r2",
    "error_map",
    "extrapolate",
    "fit_spline",
    "SplineFit",
    "adaptive_chunks",
    "robust_fit_chunks",
    "select_order",
]


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def chunk_bounds(n_points: int, n_chunks: int, overlap_pct: float = 0.0):
    """Half-open ``(start, stop)`` index ranges, one per chunk.

    Base chunks are the partition produced by
    ``np.array_split(np.arange(n_points), n_chunks)``. With ``overlap_pct > 0``,
    ``ov = int(base_len * overlap_pct / 100)`` for each chunk; the first chunk
    extends ``stop`` by ``ov``, the last extends ``start`` back by ``ov``, and
    middle chunks extend ``start`` back by ``ov // 2`` and ``stop`` forward by
    ``ov - ov // 2``. All ranges are clipped to ``[0, n_points]``.
    """
    if n_chunks < 1 or n_chunks > n_points:
        raise ValueError(
            f"n_chunks must satisfy 1 <= n_chunks <= n_points ({n_points}); got {n_chunks}"
        )

    base = np.array_split(np.arange(n_points), n_chunks)
    bounds = []
    for i, block in enumerate(base):
        s = int(block[0])
        e = int(block[-1]) + 1
        base_len = e - s
        ov = int(base_len * overlap_pct / 100.0)
        start, stop = s, e
        if i == 0:
            stop += ov
        elif i == n_chunks - 1:
            start -= ov
        else:
            start -= ov // 2
            stop += ov - ov // 2
        start = max(0, start)
        stop = min(n_points, stop)
        bounds.append((start, stop))
    return bounds


# --------------------------------------------------------------------------- #
# Single-chunk fit
# --------------------------------------------------------------------------- #
def _fit_poly(x, y, deg, w=None):
    """Fit a numpy Polynomial in a scaled domain (numerically stable)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    deg = int(max(0, min(deg, len(x) - 1)))
    if w is None:
        return Polynomial.fit(x, y, deg)
    return Polynomial.fit(x, y, deg, w=np.asarray(w, dtype=float))


@dataclass
class ChunkFit:
    start: int
    stop: int
    x_min: float
    x_max: float
    coeffs: np.ndarray  # power-basis in ORIGINAL x, highest degree first
    rmse: float
    poly: Polynomial = field(repr=False, default=None)
    # Base (non-overlapping) range, used for piecewise evaluation / blending.
    base_start: int = 0
    base_stop: int = 0
    base_x_min: float = 0.0
    base_x_max: float = 0.0

    def predict(self, xq) -> np.ndarray:
        """Evaluate the fitted polynomial, numerically stable for high orders."""
        return self.poly(np.asarray(xq, dtype=float))


def _make_chunkfit(x, y, s, e, order, bs, be, w=None):
    xs = np.asarray(x[s:e], dtype=float)
    ys = np.asarray(y[s:e], dtype=float)
    deg = order if (e - s) > order else (e - s) - 1
    poly = _fit_poly(xs, ys, deg, w=w)
    # display coefficients in original x, highest degree first
    coeffs = poly.convert().coef[::-1].copy()
    resid = poly(xs) - ys
    chunk_rmse = float(np.sqrt(np.mean(resid ** 2))) if len(xs) else 0.0
    return ChunkFit(
        start=int(s),
        stop=int(e),
        x_min=float(xs.min()),
        x_max=float(xs.max()),
        coeffs=coeffs,
        rmse=chunk_rmse,
        poly=poly,
        base_start=int(bs),
        base_stop=int(be),
        base_x_min=float(x[bs]),
        base_x_max=float(x[be - 1]),
    )


def fit_chunks(x, y, n_chunks: int, order: int, overlap_pct: float = 0.0):
    """Fit one polynomial per (possibly overlapping) chunk.

    If a chunk has ``<= order`` points, its order is reduced to ``len - 1``
    instead of failing.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    bounds = chunk_bounds(n, n_chunks, overlap_pct)
    base = chunk_bounds(n, n_chunks, 0.0)
    fits = []
    for (s, e), (bs, be) in zip(bounds, base):
        fits.append(_make_chunkfit(x, y, s, e, order, bs, be))
    return fits


# --------------------------------------------------------------------------- #
# Piecewise evaluation
# --------------------------------------------------------------------------- #
def evaluate_piecewise(fits, xq, blend: bool = True) -> np.ndarray:
    """Evaluate the piecewise model at ``xq``.

    ``blend=False``: each point uses the chunk whose base range contains it.
    ``blend=True``: inside the overlap of two adjacent chunks the result is a
    linear cross-fade of the two predictions, so the curve has no seam jumps.
    """
    xq_arr = np.asarray(xq, dtype=float)
    scalar = xq_arr.ndim == 0
    xq_arr = np.atleast_1d(xq_arr)

    if not fits:
        raise ValueError("fits must be non-empty")

    n_chunks = len(fits)
    if n_chunks == 1:
        home = np.zeros(xq_arr.shape, dtype=int)
    else:
        knots = np.array([f.base_x_min for f in fits[1:]], dtype=float)
        # a point exactly at a knot belongs to the right-hand piece
        home = np.searchsorted(knots, xq_arr, side="right")

    pred = np.empty(xq_arr.shape, dtype=float)
    for idx, f in enumerate(fits):
        mask = home == idx
        if mask.any():
            pred[mask] = f.predict(xq_arr[mask])

    if blend and n_chunks >= 2:
        for i in range(n_chunks - 1):
            a, b = fits[i], fits[i + 1]
            lo, hi = b.x_min, a.x_max  # overlap interval [lo, hi]
            if hi <= lo:
                continue
            m = (xq_arr >= lo) & (xq_arr <= hi)
            if not m.any():
                continue
            t = (xq_arr[m] - lo) / (hi - lo)
            pred[m] = (1.0 - t) * a.predict(xq_arr[m]) + t * b.predict(xq_arr[m])

    return float(pred[0]) if scalar else pred


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------- #
# Error map
# --------------------------------------------------------------------------- #
def _map_cell(x, y, order, n_chunks, overlap):
    fits = fit_chunks(x, y, n_chunks, order, overlap)
    pred = evaluate_piecewise(fits, x, blend=False)
    return rmse(y, pred)


def error_map(x, y, orders, chunk_counts, overlap_pcts, workers=None) -> np.ndarray:
    """Un-normalized RMSE map over two varying parameters.

    Exactly two of the three sequences have length > 1. Axis 0 is the first
    varying argument in the order (orders, chunk_counts, overlap_pcts); axis 1
    the second. Uses ``concurrent.futures`` unless ``workers == 1``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    orders = list(orders)
    chunk_counts = list(chunk_counts)
    overlap_pcts = list(overlap_pcts)

    seqs = [("orders", orders), ("chunks", chunk_counts), ("overlap", overlap_pcts)]
    varying = [i for i, (_, s) in enumerate(seqs) if len(s) > 1]
    if len(varying) != 2:
        raise ValueError(
            "exactly two of (orders, chunk_counts, overlap_pcts) must have length > 1"
        )
    ax0, ax1 = varying
    seq0, seq1 = seqs[ax0][1], seqs[ax1][1]

    out = np.zeros((len(seq0), len(seq1)), dtype=float)

    # Build the full parameter list for each cell.
    def params_for(v0, v1):
        p = {0: orders[0], 1: chunk_counts[0], 2: overlap_pcts[0]}
        p[ax0] = v0
        p[ax1] = v1
        return p[0], p[1], p[2]

    cells = [(i, j) for i in range(len(seq0)) for j in range(len(seq1))]

    if workers == 1:
        for i, j in cells:
            order, nch, ov = params_for(seq0[i], seq1[j])
            out[i, j] = _map_cell(x, y, order, nch, ov)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for i, j in cells:
                order, nch, ov = params_for(seq0[i], seq1[j])
                futs[ex.submit(_map_cell, x, y, order, nch, ov)] = (i, j)
            for fut, (i, j) in futs.items():
                out[i, j] = fut.result()
    return out


# --------------------------------------------------------------------------- #
# Extrapolation
# --------------------------------------------------------------------------- #
def extrapolate(x, y, fit_fraction: float, order: int):
    """Fit one polynomial to the first ``fit_fraction`` of the points.

    Returns ``(coeffs, y_pred_over_all_x)`` where ``coeffs`` is the power-basis
    in original x, highest degree first.
    """
    if not (0.0 < fit_fraction <= 1.0):
        raise ValueError("fit_fraction must satisfy 0 < fit_fraction <= 1")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    n_fit = int(fit_fraction * n)
    n_fit = max(n_fit, order + 1)
    n_fit = min(n_fit, n)
    if n_fit < order + 1:
        raise ValueError("not enough points to fit the requested order")
    poly = _fit_poly(x[:n_fit], y[:n_fit], order)
    coeffs = poly.convert().coef[::-1].copy()
    y_pred = poly(x)
    return coeffs, y_pred


# --------------------------------------------------------------------------- #
# Phase 2: constrained spline
# --------------------------------------------------------------------------- #
def _falling_factorial(k, m):
    r = 1
    for i in range(m):
        r *= (k - i)
    return r


@dataclass
class SplineFit:
    knots: np.ndarray  # internal knot x-values
    order: int
    continuity: int
    _centers: np.ndarray = field(repr=False, default=None)
    _scales: np.ndarray = field(repr=False, default=None)
    _coeffs: np.ndarray = field(repr=False, default=None)  # (n_pieces, order+1) ascending in u

    def _piece_index(self, xq):
        if len(self.knots) == 0:
            return np.zeros(xq.shape, dtype=int)
        return np.searchsorted(self.knots, xq, side="right")

    def _eval(self, xq, m=0):
        xq_arr = np.asarray(xq, dtype=float)
        scalar = xq_arr.ndim == 0
        xq_arr = np.atleast_1d(xq_arr)
        idx = self._piece_index(xq_arr)
        out = np.zeros(xq_arr.shape, dtype=float)
        order = self.order
        for p in range(self._coeffs.shape[0]):
            mask = idx == p
            if not mask.any():
                continue
            c = self._centers[p]
            s = self._scales[p]
            u = (xq_arr[mask] - c) / s
            coef = self._coeffs[p]
            val = np.zeros(u.shape, dtype=float)
            for k in range(m, order + 1):
                ff = _falling_factorial(k, m)
                val += coef[k] * ff * u ** (k - m)
            out[mask] = val / (s ** m)
        return float(out[0]) if scalar else out

    def predict(self, xq) -> np.ndarray:
        return self._eval(xq, m=0)

    def derivative(self, xq, m: int = 1) -> np.ndarray:
        if m < 0:
            raise ValueError("m must be >= 0")
        return self._eval(xq, m=m)


def fit_spline(x, y, n_chunks: int, order: int, continuity: int = 2) -> SplineFit:
    """Fit one degree-``order`` polynomial per base chunk, solved jointly as a
    single constrained least-squares problem.

    At every internal knot the value and the first ``continuity`` derivatives of
    adjacent pieces are forced equal (``continuity`` = -1 means no constraint).
    """
    if not (-1 <= continuity < order):
        raise ValueError("continuity must satisfy -1 <= continuity < order")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    base = chunk_bounds(n, n_chunks, 0.0)
    n_pieces = len(base)
    deg = order
    ncoef = deg + 1

    # per-piece local scaling for conditioning
    centers = np.zeros(n_pieces)
    scales = np.zeros(n_pieces)
    for p, (s, e) in enumerate(base):
        a, b = x[s], x[e - 1]
        centers[p] = 0.5 * (a + b)
        half = 0.5 * (b - a)
        scales[p] = half if half > 0 else 1.0

    N = n_pieces * ncoef

    # design matrix: each point contributes to its piece block only
    A = np.zeros((n, N))
    for p, (s, e) in enumerate(base):
        u = (x[s:e] - centers[p]) / scales[p]
        V = np.vander(u, ncoef, increasing=True)  # columns u^0..u^deg
        A[s:e, p * ncoef:(p + 1) * ncoef] = V

    # internal knots: x at start of next base chunk (point at knot -> right piece)
    knot_x = np.array([x[base[p + 1][0]] for p in range(n_pieces - 1)], dtype=float)

    # equality constraints C w = 0
    rows = []
    for p in range(n_pieces - 1):
        X = knot_x[p]
        for m in range(continuity + 1):
            row = np.zeros(N)
            for (piece, sign) in ((p, 1.0), (p + 1, -1.0)):
                c = centers[piece]
                sc = scales[piece]
                u = (X - c) / sc
                for k in range(m, ncoef):
                    ff = _falling_factorial(k, m)
                    row[piece * ncoef + k] += sign * ff * (u ** (k - m)) / (sc ** m)
            rows.append(row)
    C = np.array(rows) if rows else np.zeros((0, N))

    AtA = A.T @ A
    Aty = A.T @ y
    # small ridge for conditioning / underdetermined pieces (display-negligible)
    ridge = 1e-10 * (np.trace(AtA) / N + 1.0)
    AtA = AtA + ridge * np.eye(N)

    M = C.shape[0]
    if M == 0:
        w = np.linalg.solve(AtA, Aty)
    else:
        KKT = np.zeros((N + M, N + M))
        KKT[:N, :N] = AtA
        KKT[:N, N:] = C.T
        KKT[N:, :N] = C
        rhs = np.concatenate([Aty, np.zeros(M)])
        try:
            sol = np.linalg.solve(KKT, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(KKT, rhs, rcond=None)
        w = sol[:N]

    coeffs = w.reshape(n_pieces, ncoef)  # ascending in local u
    return SplineFit(
        knots=knot_x,
        order=order,
        continuity=continuity,
        _centers=centers,
        _scales=scales,
        _coeffs=coeffs,
    )


# --------------------------------------------------------------------------- #
# Phase 2: adaptive chunking
# --------------------------------------------------------------------------- #
def _chunk_rmse(x, y, s, e, order):
    deg = order if (e - s) > order else (e - s) - 1
    if deg < 0:
        return 0.0
    poly = _fit_poly(x[s:e], y[s:e], deg)
    return rmse(y[s:e], poly(x[s:e]))


def adaptive_chunks(x, y, order: int, tol: float, max_chunks: int):
    """Greedy left-to-right partition of ``[0, len(x))`` such that a degree-
    ``order`` fit on every chunk has RMSE <= ``tol`` (using as few chunks as
    reasonably possible, never more than ``max_chunks``)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n == 0:
        return []
    if max_chunks < 1:
        raise ValueError("max_chunks must be >= 1")

    bounds = []
    s = 0
    while s < n:
        # largest e in (s, n] with rmse(x[s:e]) <= tol; rmse grows with e.
        if _chunk_rmse(x, y, s, n, order) <= tol:
            e = n
        else:
            lo, hi = s + 1, n  # ok(lo) assumed, find last ok
            # exponential search for first failing end
            step = 1
            last_ok = s + 1
            e_probe = s + 1
            while e_probe <= n and _chunk_rmse(x, y, s, e_probe, order) <= tol:
                last_ok = e_probe
                step *= 2
                e_probe = s + step
            lo, hi = last_ok, min(n, e_probe)
            # binary search boundary in (lo, hi]
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if _chunk_rmse(x, y, s, mid, order) <= tol:
                    lo = mid
                else:
                    hi = mid
            e = lo
        e = max(e, s + 1)
        bounds.append((s, e))
        s = e

    if len(bounds) > max_chunks:
        # cannot meet tol within the budget -> fall back to equal partition
        bounds = chunk_bounds(n, max_chunks, 0.0)
    return bounds


# --------------------------------------------------------------------------- #
# Phase 2: robust fit (Huber IRLS)
# --------------------------------------------------------------------------- #
def _robust_poly(x, y, deg, n_iter=12, delta=1.345):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    deg = int(max(0, min(deg, len(x) - 1)))
    poly = _fit_poly(x, y, deg)
    for _ in range(n_iter):
        resid = y - poly(x)
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        scale = 1.4826 * mad
        if scale <= 1e-12:
            scale = np.std(resid)
        if scale <= 1e-12:
            break
        r = resid / scale
        wobj = np.where(np.abs(r) <= delta, 1.0, delta / np.maximum(np.abs(r), 1e-12))
        new_poly = _fit_poly(x, y, deg, w=np.sqrt(wobj))
        if np.allclose(new_poly.coef, poly.coef, rtol=1e-8, atol=1e-10):
            poly = new_poly
            break
        poly = new_poly
    return poly


def robust_fit_chunks(x, y, n_chunks: int, order: int, overlap_pct: float = 0.0):
    """Same contract as :func:`fit_chunks`, but resistant to outliers via Huber
    IRLS with a robust (MAD) scale estimate."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    bounds = chunk_bounds(n, n_chunks, overlap_pct)
    base = chunk_bounds(n, n_chunks, 0.0)
    fits = []
    for (s, e), (bs, be) in zip(bounds, base):
        xs = x[s:e]
        ys = y[s:e]
        deg = order if (e - s) > order else (e - s) - 1
        poly = _robust_poly(xs, ys, deg)
        coeffs = poly.convert().coef[::-1].copy()
        chunk_rmse = float(np.sqrt(np.mean((poly(xs) - ys) ** 2)))
        fits.append(
            ChunkFit(
                start=int(s),
                stop=int(e),
                x_min=float(xs.min()),
                x_max=float(xs.max()),
                coeffs=coeffs,
                rmse=chunk_rmse,
                poly=poly,
                base_start=int(bs),
                base_stop=int(be),
                base_x_min=float(x[bs]),
                base_x_max=float(x[be - 1]),
            )
        )
    return fits


# --------------------------------------------------------------------------- #
# Phase 2: order selection
# --------------------------------------------------------------------------- #
def select_order(x, y, n_chunks: int, max_order: int, method: str = "cv") -> int:
    """Pick the polynomial order in ``[0, max_order]`` for ``fit_chunks`` with
    ``n_chunks`` chunks, using k-fold CV (``method="cv"``) or AIC
    (``method="aic"``)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    if method == "cv":
        k = min(5, n)
        if k < 2:
            return 0
        idx = np.arange(n)
        best_order, best_score = 0, math.inf
        for order in range(max_order + 1):
            errs = []
            for fold in range(k):
                test = idx % k == fold
                train = ~test
                ntr = int(train.sum())
                if ntr < 2 or test.sum() == 0:
                    continue
                nc = min(n_chunks, ntr)
                fits = fit_chunks(x[train], y[train], nc, order)
                pred = evaluate_piecewise(fits, x[test], blend=False)
                errs.append(rmse(y[test], pred))
            score = float(np.mean(errs)) if errs else math.inf
            if score < best_score:
                best_score, best_order = score, order
        return best_order

    elif method == "aic":
        best_order, best_aic = 0, math.inf
        for order in range(max_order + 1):
            fits = fit_chunks(x, y, n_chunks, order)
            pred = evaluate_piecewise(fits, x, blend=False)
            rss = float(np.sum((y - pred) ** 2))
            kp = n_chunks * (order + 1)
            if rss <= 0:
                aic = -math.inf  # perfect fit
            else:
                aic = n * math.log(rss / n) + 2 * kp
            if aic < best_aic:
                best_aic, best_order = aic, order
        return best_order

    raise ValueError(f"unknown method: {method!r}")
