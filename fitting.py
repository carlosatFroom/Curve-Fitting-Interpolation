"""Pure-numpy piecewise polynomial curve fitting.

Importing this module must NOT import PyQt5, pyqtgraph or matplotlib. It depends
only on numpy (scipy is permitted but not required / not used here).

Phase 1 (this file, top section):
    chunk_bounds, ChunkFit, fit_chunks, evaluate_piecewise, rmse, r2,
    error_map, extrapolate

Phase 2 (lower section):
    fit_spline / SplineFit, adaptive_chunks, robust_fit_chunks, select_order
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
# chunk bounds
# --------------------------------------------------------------------------- #
def chunk_bounds(n_points, n_chunks, overlap_pct=0.0):
    """Half-open (start, stop) index ranges, one per chunk.

    Base chunks are the partition produced by
    ``np.array_split(np.arange(n_points), n_chunks)``. With ``overlap_pct > 0``,
    let ``ov = int(base_len * overlap_pct / 100)`` for that chunk; the first
    chunk extends ``stop`` by ``ov``, the last extends ``start`` back by ``ov``,
    middle chunks extend ``start`` back by ``ov // 2`` and ``stop`` forward by
    ``ov - ov // 2``. Everything is clipped to ``[0, n_points]``.
    """
    n_points = int(n_points)
    n_chunks = int(n_chunks)
    if n_chunks < 1 or n_chunks > n_points:
        raise ValueError(
            f"n_chunks must satisfy 1 <= n_chunks <= n_points "
            f"(got n_chunks={n_chunks}, n_points={n_points})"
        )

    # Base partition lengths (as np.array_split produces them).
    base = np.array_split(np.arange(n_points), n_chunks)
    starts = []
    stops = []
    pos = 0
    for b in base:
        starts.append(pos)
        pos += len(b)
        stops.append(pos)

    bounds = []
    for i in range(n_chunks):
        start, stop = starts[i], stops[i]
        base_len = stop - start
        if overlap_pct > 0 and n_chunks > 1:
            ov = int(base_len * overlap_pct / 100)
            if i == 0:
                stop += ov
            elif i == n_chunks - 1:
                start -= ov
            else:
                start -= ov // 2
                stop += ov - ov // 2
        start = max(0, min(start, n_points))
        stop = max(0, min(stop, n_points))
        bounds.append((start, stop))
    return bounds


# --------------------------------------------------------------------------- #
# single-chunk fit
# --------------------------------------------------------------------------- #
@dataclass
class ChunkFit:
    """One polynomial fitted to one slice of the data.

    ``coeffs`` are power-basis coefficients in the ORIGINAL x, highest degree
    first (``np.polyfit`` convention), provided for display only. ``predict``
    evaluates the fit through a scaled domain for numerical stability.
    """

    start: int
    stop: int
    x_min: float
    x_max: float
    coeffs: np.ndarray
    rmse: float
    # base (non-overlapped) x-range, used for point ownership in piecewise eval
    base_x_min: float = 0.0
    base_x_max: float = 0.0
    # scaled-domain polynomial actually used for evaluation (not part of the
    # public contract, but how predict stays stable at high order)
    _poly: Polynomial = field(default=None, repr=False, compare=False)

    def predict(self, xq):
        xq = np.asarray(xq, dtype=float)
        if self._poly is not None:
            return self._poly(xq)
        # Fallback to the display coefficients if no scaled polynomial stored.
        return np.polyval(self.coeffs, xq)


def _fit_one(xs, ys, order):
    """Fit a single scaled polynomial; return (Polynomial, coeffs_desc, rmse)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    eff_order = min(int(order), n - 1)
    eff_order = max(eff_order, 0)
    # Polynomial.fit scales/shifts x into [-1, 1] internally -> stable.
    poly = Polynomial.fit(xs, ys, eff_order)
    # Power-basis coefficients in original x, highest degree first.
    coeffs = poly.convert().coef[::-1].copy()
    y_hat = poly(xs)
    res = float(np.sqrt(np.mean((y_hat - ys) ** 2)))
    return poly, coeffs, res


def fit_chunks(x, y, n_chunks, order, overlap_pct=0.0):
    """Fit one polynomial per chunk (chunks may overlap).

    If a chunk has ``<= order`` points, that chunk's order is reduced to
    ``len - 1`` instead of failing.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    bounds = chunk_bounds(n, n_chunks, overlap_pct)
    base = chunk_bounds(n, n_chunks, 0.0)

    fits = []
    for (start, stop), (bstart, bstop) in zip(bounds, base):
        xs = x[start:stop]
        ys = y[start:stop]
        poly, coeffs, res = _fit_one(xs, ys, order)
        fits.append(
            ChunkFit(
                start=start,
                stop=stop,
                x_min=float(xs.min()),
                x_max=float(xs.max()),
                coeffs=coeffs,
                rmse=res,
                base_x_min=float(x[bstart]),
                base_x_max=float(x[bstop - 1]),
                _poly=poly,
            )
        )
    return fits


# --------------------------------------------------------------------------- #
# piecewise evaluation
# --------------------------------------------------------------------------- #
def evaluate_piecewise(fits, xq, blend=True):
    """Evaluate the piecewise model at ``xq``.

    With ``blend=False`` each point uses the chunk whose base range contains it.
    With ``blend=True``, inside the overlap of two adjacent chunks the result is
    a linear cross-fade of the two predictions, so there are no seam jumps.
    """
    if not fits:
        raise ValueError("fits must be a non-empty list of ChunkFit")
    xq = np.asarray(xq, dtype=float)
    scalar = xq.ndim == 0
    xq = np.atleast_1d(xq)
    n = len(fits)

    if n == 1:
        out = fits[0].predict(xq)
        return out[0] if scalar else out

    # Base ownership: chunk i owns [base_x_min_i, base_x_min_{i+1}); the first
    # chunk also owns everything below, the last everything above.
    left_edges = np.array([f.base_x_min for f in fits], dtype=float)
    idx = np.searchsorted(left_edges, xq, side="right") - 1
    idx = np.clip(idx, 0, n - 1)

    preds = np.vstack([f.predict(xq) for f in fits])  # (n, len(xq))
    out = preds[idx, np.arange(len(xq))].astype(float)

    if blend:
        # Cross-fade in the x-overlap of each adjacent pair of fitted slices.
        for i in range(n - 1):
            a, b = fits[i], fits[i + 1]
            lo = b.x_min  # left edge of the overlap (deep inside chunk i)
            hi = a.x_max  # right edge of the overlap (deep inside chunk i+1)
            if hi <= lo:
                continue  # no overlap -> hard seam, nothing to blend
            mask = (xq >= lo) & (xq <= hi)
            if not mask.any():
                continue
            w = (hi - xq[mask]) / (hi - lo)  # 1 at lo (chunk i), 0 at hi (i+1)
            out[mask] = w * preds[i][mask] + (1.0 - w) * preds[i + 1][mask]

    return out[0] if scalar else out


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------- #
# error map
# --------------------------------------------------------------------------- #
def _map_rmse(x, y, order, n_chunks, overlap):
    fits = fit_chunks(x, y, n_chunks, order, overlap)
    y_pred = evaluate_piecewise(fits, x, blend=False)
    return rmse(y, y_pred)


def error_map(x, y, orders, chunk_counts, overlap_pcts, workers=None):
    """2-D grid of un-normalized RMSE over the two varying parameter sequences.

    Exactly two of ``orders``, ``chunk_counts``, ``overlap_pcts`` have length
    > 1; the third has length 1. Axis 0 is the first varying argument in that
    order, axis 1 the second. Uses ``concurrent.futures`` unless ``workers == 1``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    orders = list(orders)
    chunk_counts = list(chunk_counts)
    overlap_pcts = list(overlap_pcts)

    seqs = [("orders", orders), ("chunks", chunk_counts), ("overlap", overlap_pcts)]
    varying = [name for name, seq in seqs if len(seq) > 1]
    if len(varying) != 2:
        raise ValueError(
            "exactly two of (orders, chunk_counts, overlap_pcts) must have "
            "length > 1"
        )
    axis0, axis1 = varying

    def val(name, i0, i1):
        sel = {axis0: i0, axis1: i1}
        if name == "orders":
            return orders[sel["orders"]] if "orders" in sel else orders[0]
        if name == "chunks":
            return chunk_counts[sel["chunks"]] if "chunks" in sel else chunk_counts[0]
        return overlap_pcts[sel["overlap"]] if "overlap" in sel else overlap_pcts[0]

    len0 = len(dict(seqs)[axis0])
    len1 = len(dict(seqs)[axis1])
    out = np.zeros((len0, len1), dtype=float)

    jobs = []  # (i, j, order, n_chunks, overlap)
    for i in range(len0):
        for j in range(len1):
            order = val("orders", i, j)
            n_chunks = val("chunks", i, j)
            overlap = val("overlap", i, j)
            jobs.append((i, j, order, n_chunks, overlap))

    if workers == 1:
        for i, j, order, n_chunks, overlap in jobs:
            out[i, j] = _map_rmse(x, y, order, n_chunks, overlap)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(_map_rmse, x, y, order, n_chunks, overlap): (i, j)
                for i, j, order, n_chunks, overlap in jobs
            }
            for fut in futs:
                i, j = futs[fut]
                out[i, j] = fut.result()

    return out


# --------------------------------------------------------------------------- #
# extrapolation
# --------------------------------------------------------------------------- #
def extrapolate(x, y, fit_fraction, order):
    """Fit one polynomial to the first ``fit_fraction`` of the points.

    Uses at least ``order + 1`` points. Returns ``(coeffs, y_pred_over_all_x)``
    where ``coeffs`` follows the ``np.polyfit`` convention (highest degree
    first) and the prediction spans every x.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if not (0.0 < fit_fraction <= 1.0):
        raise ValueError("fit_fraction must be in (0, 1]")
    k = int(math.floor(n * fit_fraction))
    k = max(k, int(order) + 1)
    k = min(k, n)
    poly, coeffs, _ = _fit_one(x[:k], y[:k], order)
    y_pred = poly(x)
    return coeffs, y_pred


# --------------------------------------------------------------------------- #
# Phase 2: constrained joint spline
# --------------------------------------------------------------------------- #
def _deriv_row(u, ncoef, m, h):
    """Row w with ``w . a`` = m-th x-derivative of ``sum a_k u**k`` at u.

    Here ``u = (x - c) / h`` so ``d^m/dx^m = h**-m d^m/du^m``.
    """
    w = np.zeros(ncoef)
    for k in range(m, ncoef):
        coef = 1.0
        for j in range(m):
            coef *= (k - j)  # falling factorial k*(k-1)*...*(k-m+1)
        w[k] = coef * (u ** (k - m)) / (h ** m)
    return w


@dataclass
class SplineFit:
    """Piecewise polynomial with enforced continuity at the knots.

    Each base chunk carries one degree-``order`` polynomial, stored in a local
    scaled variable ``u = (x - center) / half`` for numerical stability.
    """

    knots: np.ndarray
    _centers: np.ndarray = field(repr=False)
    _halfs: np.ndarray = field(repr=False)
    _coeffs: np.ndarray = field(repr=False)  # (n_pieces, order+1), ascending in u
    order: int = 0

    def _assign(self, xq):
        n_pieces = self._coeffs.shape[0]
        return np.clip(
            np.searchsorted(self.knots, xq, side="right"), 0, n_pieces - 1
        )

    def predict(self, xq):
        return self.derivative(xq, m=0)

    def derivative(self, xq, m=1):
        xq = np.asarray(xq, dtype=float)
        scalar = xq.ndim == 0
        xq = np.atleast_1d(xq)
        pieces = self._assign(xq)
        out = np.empty(len(xq), dtype=float)
        for i in range(self._coeffs.shape[0]):
            mask = pieces == i
            if not mask.any():
                continue
            u = (xq[mask] - self._centers[i]) / self._halfs[i]
            c = self._coeffs[i]
            if m > 0:
                c = np.polynomial.polynomial.polyder(c, m)
            vals = np.polynomial.polynomial.polyval(u, c)
            out[mask] = vals / (self._halfs[i] ** m)
        return out[0] if scalar else out


def fit_spline(x, y, n_chunks, order, continuity=2):
    """One degree-``order`` polynomial per base chunk, solved jointly.

    Minimises the total squared error over all points subject to the continuity
    constraints (value and first ``continuity`` derivatives equal at each knot),
    rather than fitting then patching. ``continuity`` = -1 means no constraint,
    0 = C0, 1 = C1, 2 = C2. Knot ``i`` is ``x[start of base chunk i+1]`` and a
    point at a knot belongs to the right-hand piece.
    """
    order = int(order)
    if not (-1 <= continuity < order):
        raise ValueError(
            f"continuity must satisfy -1 <= continuity < order "
            f"(got continuity={continuity}, order={order})"
        )
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    base = chunk_bounds(n, n_chunks, 0.0)
    ncoef = order + 1
    N = n_chunks * ncoef

    centers = np.empty(n_chunks)
    halfs = np.empty(n_chunks)
    for i, (s, e) in enumerate(base):
        xs = x[s:e]
        xlo, xhi = float(xs.min()), float(xs.max())
        centers[i] = 0.5 * (xlo + xhi)
        h = 0.5 * (xhi - xlo)
        halfs[i] = h if h > 0 else 1.0

    knots = np.array([x[base[i + 1][0]] for i in range(n_chunks - 1)], dtype=float)

    # Data block (block-diagonal: each point belongs to exactly one base chunk).
    A = np.zeros((n, N))
    for i, (s, e) in enumerate(base):
        u = (x[s:e] - centers[i]) / halfs[i]
        A[s:e, i * ncoef:(i + 1) * ncoef] = np.vander(u, ncoef, increasing=True)

    # Continuity constraints  C @ a = 0.
    rows = []
    if continuity >= 0:
        for i in range(n_chunks - 1):
            xk = float(x[base[i + 1][0]])
            ui = (xk - centers[i]) / halfs[i]
            ui1 = (xk - centers[i + 1]) / halfs[i + 1]
            for m in range(continuity + 1):
                row = np.zeros(N)
                row[i * ncoef:(i + 1) * ncoef] = _deriv_row(ui, ncoef, m, halfs[i])
                row[(i + 1) * ncoef:(i + 2) * ncoef] = -_deriv_row(
                    ui1, ncoef, m, halfs[i + 1]
                )
                rows.append(row)

    if rows:
        C = np.vstack(rows)
        # Null-space of C via SVD, then unconstrained least squares in it.
        _, s_vals, vh = np.linalg.svd(C)
        tol = max(C.shape) * np.finfo(float).eps * (s_vals[0] if s_vals.size else 0.0)
        r = int(np.sum(s_vals > tol))
        Z = vh[r:].T  # N x (N - r)
        z, *_ = np.linalg.lstsq(A @ Z, y, rcond=None)
        a = Z @ z
    else:
        a, *_ = np.linalg.lstsq(A, y, rcond=None)

    coeffs = a.reshape(n_chunks, ncoef)  # ascending in local u
    return SplineFit(
        knots=knots, _centers=centers, _halfs=halfs, _coeffs=coeffs, order=order
    )


# --------------------------------------------------------------------------- #
# Phase 2: adaptive chunking, robust fits, order selection
# --------------------------------------------------------------------------- #
def adaptive_chunks(x, y, order, tol, max_chunks):
    """Greedy contiguous partition of ``[0, len(x))`` with per-chunk RMSE <= tol.

    Each chunk grows as far right as it can while its degree-``order`` fit stays
    within ``tol``. Chunks shorter than ``2 * (order + 1)`` points are always
    accepted. If the partition would exceed ``max_chunks``, all remaining points
    go into the final chunk. The growth search is galloping + bisection.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("x must be non-empty")
    if max_chunks < 1:
        raise ValueError(f"max_chunks must be >= 1 (got {max_chunks})")

    min_len = 2 * (int(order) + 1)

    def ok(a, b):
        if b - a < min_len:
            return True
        return _fit_one(x[a:b], y[a:b], order)[2] <= tol

    def largest_ok(start):
        lo = start + 1  # a single point is always accepted
        step = 1
        while True:
            probe = lo + step
            if probe > n:
                if lo < n and ok(start, n):
                    return n
                hi = n
                break
            if ok(start, probe):
                lo = probe
                step *= 2
            else:
                hi = probe
                break
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ok(start, mid):
                lo = mid
            else:
                hi = mid
        return lo

    chunks = []
    start = 0
    while start < n:
        if len(chunks) + 1 == max_chunks:
            chunks.append((start, n))
            break
        stop = largest_ok(start)
        chunks.append((start, stop))
        start = stop
    return chunks


def robust_fit_chunks(x, y, n_chunks, order, overlap_pct=0.0):
    """Like ``fit_chunks`` but each chunk uses Huber-weighted IRLS.

    Resistant to outliers: with ~10% gross outliers the fit still tracks the
    clean signal. Returns fully-populated ``ChunkFit`` objects compatible with
    ``evaluate_piecewise``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    bounds = chunk_bounds(n, n_chunks, overlap_pct)
    base = chunk_bounds(n, n_chunks, 0.0)

    c = 1.345
    fits = []
    for (start, stop), (bstart, bstop) in zip(bounds, base):
        xs = x[start:stop]
        ys = y[start:stop]
        eff_order = min(int(order), len(xs) - 1)
        eff_order = max(eff_order, 0)
        poly = Polynomial.fit(xs, ys, eff_order)
        for _ in range(10):
            r = ys - poly(xs)
            s = 1.4826 * np.median(np.abs(r - np.median(r)))
            if s == 0:
                break
            w = np.ones_like(r)
            mask = np.abs(r) > c * s
            w[mask] = c * s / np.abs(r[mask])
            poly = Polynomial.fit(xs, ys, eff_order, w=np.sqrt(w))

        fits.append(
            ChunkFit(
                start=start,
                stop=stop,
                x_min=float(xs.min()),
                x_max=float(xs.max()),
                coeffs=poly.convert().coef[::-1].copy(),
                rmse=float(np.sqrt(np.mean((poly(xs) - ys) ** 2))),
                base_x_min=float(x[bstart]),
                base_x_max=float(x[bstop - 1]),
                _poly=poly,
            )
        )
    return fits


def select_order(x, y, n_chunks, max_order, method="cv"):
    """Pick the polynomial order in ``[0, max_order]`` for ``fit_chunks``.

    ``method="cv"``: 5-fold (or fewer) random-split cross-validation on held-out
    RMSE; folds whose training set has fewer than ``n_chunks`` points are
    skipped. ``method="aic"``: Akaike information criterion on the full-data RSS
    with ``k = n_chunks * (order + 1)`` parameters. Ties go to the lower order.
    """
    if method not in ("cv", "aic"):
        raise ValueError(f"unknown method {method!r} (use 'cv' or 'aic')")
    if max_order < 0:
        raise ValueError(f"max_order must be >= 0 (got {max_order})")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    best_order = 0
    best_score = np.inf

    if method == "cv":
        k = min(5, n)
        perm = np.random.default_rng(0).permutation(n)
        folds = [np.sort(f) for f in np.array_split(perm, k)]
        splits = []
        for test_idx in folds:
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False
            if train_mask.sum() < n_chunks:
                continue
            splits.append((np.flatnonzero(train_mask), test_idx))
        if not splits:
            raise ValueError("no CV fold leaves enough training points for n_chunks")

        for order in range(max_order + 1):
            errs = []
            for train_idx, test_idx in splits:
                fits = fit_chunks(x[train_idx], y[train_idx], n_chunks, order)
                y_pred = evaluate_piecewise(fits, x[test_idx], blend=False)
                errs.append(rmse(y[test_idx], y_pred))
            score = float(np.mean(errs))
            if score < best_score:
                best_score = score
                best_order = order
    else:
        for order in range(max_order + 1):
            fits = fit_chunks(x, y, n_chunks, order)
            resid = y - evaluate_piecewise(fits, x, blend=False)
            rss = max(float(np.sum(resid ** 2)), 1e-300)
            k_params = n_chunks * (order + 1)
            score = n * np.log(rss / n) + 2 * k_params
            if score < best_score:
                best_score = score
                best_order = order

    return int(best_order)
