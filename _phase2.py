"""Phase 2 algorithms: constrained splines, adaptive chunking, robust fits, order selection.

Pure numpy. ``fitting`` is imported lazily inside functions because ``fitting``
imports this module at its bottom.
"""

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from numpy.polynomial.chebyshev import chebder, chebvander

__all__ = [
    "SplineFit",
    "fit_spline",
    "adaptive_chunks",
    "robust_fit_chunks",
    "select_order",
]


class SplineFit:
    """Piecewise Chebyshev polynomial with C^k joins at ``knots``."""

    def __init__(self, knots, domains, series):
        self._knots = np.asarray(knots, dtype=float)
        self._series = series  # one numpy Chebyshev per piece, domain = piece x-range
        self._domains = domains

    @property
    def knots(self) -> np.ndarray:
        return self._knots

    def _eval(self, xq, m):
        xq = np.asarray(xq, dtype=float)
        idx = np.searchsorted(self._knots, xq, side="right")  # knot belongs to right piece
        out = np.zeros(xq.shape, dtype=float)
        for p in np.unique(idx):
            mask = idx == p
            ser = self._series[p]
            if m:
                ser = ser.deriv(m)
            out[mask] = ser(xq[mask])
        return out

    def predict(self, xq) -> np.ndarray:
        return self._eval(xq, 0)

    def derivative(self, xq, m: int = 1) -> np.ndarray:
        if m < 0:
            raise ValueError("m must be >= 0")
        return self._eval(xq, m)


def _cheb_deriv_rows(order: int, max_m: int, t: float) -> np.ndarray:
    """Rows [m, k] = d^m T_k / dt^m evaluated at local coordinate t, for m in 0..max_m."""
    rows = np.zeros((max_m + 1, order + 1))
    eye = np.eye(order + 1)
    for m in range(max_m + 1):
        dmat = chebder(eye, m, axis=0) if m else eye  # column k = coeffs of T_k^(m)
        rows[m, :] = chebvander(np.array([t]), order - m)[0] @ dmat
    return rows


def fit_spline(x, y, n_chunks: int, order: int, continuity: int = 2) -> SplineFit:
    from fitting import chunk_bounds

    if not -1 <= continuity < order:
        raise ValueError(f"continuity must satisfy -1 <= continuity < order, got {continuity} (order={order})")
    if order < 0:
        raise ValueError("order must be >= 0")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    perm = np.argsort(x, kind="stable")
    x, y = x[perm], y[perm]
    n = len(x)
    bounds = chunk_bounds(n, n_chunks, 0.0)
    n_pieces = len(bounds)
    width = order + 1

    lefts = [x[0]] + [x[s] for s, _ in bounds[1:]]
    rights = [x[s] for s, _ in bounds[1:]] + [x[-1]]
    domains = []
    for a, b in zip(lefts, rights):
        if not b > a:
            raise ValueError("x must have strictly increasing piece widths (duplicate x values?)")
        domains.append((a, b))
    knots = np.array([x[s] for s, _ in bounds[1:]], dtype=float)

    # Design matrix: block-diagonal in pieces, each piece in local Chebyshev basis.
    A = np.zeros((n, n_pieces * width))
    for p, ((s, e), (a, b)) in enumerate(zip(bounds, domains)):
        t = (2.0 * x[s:e] - (a + b)) / (b - a)
        A[s:e, p * width:(p + 1) * width] = chebvander(t, order)

    # Continuity constraints: value and derivatives 0..continuity match at each knot.
    rows = []
    dpos = _cheb_deriv_rows(order, continuity, 1.0)
    dneg = _cheb_deriv_rows(order, continuity, -1.0)
    for i in range(n_pieces - 1):
        a_l, b_l = domains[i]
        a_r, b_r = domains[i + 1]
        for m in range(continuity + 1):
            row = np.zeros(n_pieces * width)
            row[i * width:(i + 1) * width] = dpos[m] * (2.0 / (b_l - a_l)) ** m
            row[(i + 1) * width:(i + 2) * width] = -dneg[m] * (2.0 / (b_r - a_r)) ** m
            nrm = np.linalg.norm(row)
            rows.append(row / nrm if nrm > 0 else row)

    if rows:
        C = np.vstack(rows)
        _, S, Vt = np.linalg.svd(C)
        rank = int(np.sum(S > S[0] * max(C.shape) * np.finfo(float).eps * 10)) if S.size else 0
        N = Vt[rank:].T  # orthonormal null-space basis of the constraints
    else:
        N = np.eye(n_pieces * width)

    w, *_ = np.linalg.lstsq(A @ N, y, rcond=None)
    z = N @ w

    series = [
        Chebyshev(z[p * width:(p + 1) * width], domain=list(domains[p]))
        for p in range(n_pieces)
    ]
    return SplineFit(knots, domains, series)


def _seg_rmse(x, y, s, e, order):
    deg = min(order, e - s - 1)
    p = Polynomial.fit(x[s:e], y[s:e], deg)
    return float(np.sqrt(np.mean((y[s:e] - p(x[s:e])) ** 2)))


def _greedy_partition(x, y, order, tol):
    n = len(x)
    bounds = []
    s = 0
    while s < n:
        if _seg_rmse(x, y, s, n, order) <= tol:
            e = n
        else:
            lo, hi = s + 1, n  # lo always acceptable (single point), hi not
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if _seg_rmse(x, y, s, mid, order) <= tol:
                    lo = mid
                else:
                    hi = mid
            e = lo
        bounds.append((s, e))
        s = e
    return bounds


def adaptive_chunks(x, y, order: int, tol: float, max_chunks: int) -> list[tuple[int, int]]:
    """Greedy partition: each chunk is extended while its degree-``order`` RMSE <= ``tol``.

    If that needs more than ``max_chunks`` chunks, the tolerance is relaxed (smallest
    relaxation found by bisection) so that the result respects ``max_chunks``.
    """
    if max_chunks < 1:
        raise ValueError("max_chunks must be >= 1")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0:
        raise ValueError("no points")
    bounds = _greedy_partition(x, y, order, tol)
    if len(bounds) <= max_chunks:
        return bounds

    lo = tol
    hi = _seg_rmse(x, y, 0, len(x), order)
    best = [(0, len(x))]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cand = _greedy_partition(x, y, order, mid)
        if len(cand) <= max_chunks:
            best = cand
            hi = mid
        else:
            lo = mid
    return best


def _huber_weights(r, scale, k=1.345):
    u = np.abs(r) / scale
    return np.where(u <= k, 1.0, k / np.maximum(u, 1e-300))


def _tukey_weights(r, scale, c=4.685):
    u = r / scale
    return np.where(np.abs(u) < c, (1.0 - (u / c) ** 2) ** 2, 0.0)


def _robust_poly(x, y, order):
    """Huber IRLS warm start, then Tukey bisquare refinement, MAD scale."""
    deg = min(order, len(x) - 1)
    w = np.ones(len(x))
    p = Polynomial.fit(x, y, deg)
    for weight_fn, iters in ((_huber_weights, 15), (_tukey_weights, 25)):
        for _ in range(iters):
            r = y - p(x)
            scale = 1.4826 * np.median(np.abs(r - np.median(r)))
            if scale <= 1e-300:
                break
            w = weight_fn(r, scale)
            if not np.any(w > 0):
                break
            p = Polynomial.fit(x, y, deg, w=w)
    return p, w


def robust_fit_chunks(x, y, n_chunks: int, order: int, overlap_pct: float = 0.0):
    from dataclasses import replace

    from fitting import fit_chunks

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    templates = fit_chunks(x, y, n_chunks, order, overlap_pct)
    fits = []
    for tpl in templates:
        xs, ys = x[tpl.start:tpl.stop], y[tpl.start:tpl.stop]
        p, _ = _robust_poly(xs, ys, min(order, len(xs) - 1))
        fits.append(
            replace(
                tpl,
                coeffs=p.convert().coef[::-1],
                rmse=float(np.sqrt(np.mean((ys - p(xs)) ** 2))),
                _poly=p,
            )
        )
    return fits


def _n_params(chunk_lengths, order):
    return sum(min(order, m - 1) + 1 for m in chunk_lengths)


def select_order(x, y, n_chunks: int, max_order: int, method: str = "cv") -> int:
    from fitting import chunk_bounds, evaluate_piecewise, fit_chunks

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if max_order < 0:
        raise ValueError("max_order must be >= 0")
    if method == "cv":
        k = 5
        if n < k:
            raise ValueError(f"need at least {k} points for cv")
        fold = np.arange(n) % k
        fold_mse = np.zeros((max_order + 1, k))
        for order in range(max_order + 1):
            for f in range(k):
                train, test = fold != f, fold == f
                nt = int(train.sum())
                fits = fit_chunks(x[train], y[train], min(n_chunks, nt), order)
                pred = evaluate_piecewise(fits, x[test], blend=False)
                fold_mse[order, f] = float(np.mean((y[test] - pred) ** 2))
        mean = fold_mse.mean(axis=1)
        best = int(np.argmin(mean))
        se_best = fold_mse[best].std(ddof=1) / np.sqrt(k)
        # one-standard-error rule: simplest order within one SE of the best CV error
        return int(np.flatnonzero(mean <= mean[best] + se_best)[0])
    elif method == "aic":
        scores = []
        nc = min(n_chunks, n)
        for order in range(max_order + 1):
            fits = fit_chunks(x, y, nc, order)
            pred = evaluate_piecewise(fits, x, blend=False)
            sse = max(float(np.sum((y - pred) ** 2)), 1e-300)
            lengths = [stop - start for start, stop in chunk_bounds(n, nc, 0.0)]
            k_params = _n_params(lengths, order)
            scores.append(n * np.log(sse / n) + 2 * k_params)
    else:
        raise ValueError(f"unknown method {method!r}; use 'cv' or 'aic'")
    return int(np.argmin(scores))
