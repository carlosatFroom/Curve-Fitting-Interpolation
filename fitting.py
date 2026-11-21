"""Pure-numpy curve fitting: chunked polynomial fits, piecewise blending,
error maps, constrained splines, robust fits and order selection.

Importing this module must NOT import PyQt5, pyqtgraph or matplotlib.
"""
from __future__ import annotations

import itertools
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

Polynomial = np.polynomial.Polynomial

__all__ = [
    "chunk_bounds",
    "ChunkFit",
    "fit_chunks",
    "evaluate_piecewise",
    "rmse",
    "r2",
    "error_map",
    "extrapolate",
    "SplineFit",
    "fit_spline",
    "adaptive_chunks",
    "robust_fit_chunks",
    "select_order",
]


# --------------------------------------------------------------------------
# Foundation (written by lead; shared contract for all other functions)
# --------------------------------------------------------------------------
def rmse(y_true, y_pred) -> float:
    """sqrt(mean((y_true - y_pred)**2))."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination. Returns 1.0 if y_true is constant and the
    prediction matches it, else -inf-ish handled by the usual formula."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def chunk_bounds(n_points: int, n_chunks: int, overlap_pct: float = 0.0):
    """Half-open (start, stop) index ranges, one per chunk.

    Base chunks are the partition produced by
    ``np.array_split(np.arange(n_points), n_chunks)``. With ``overlap_pct > 0``,
    for each chunk ``ov = int(base_len * overlap_pct / 100)``; the first chunk
    extends ``stop`` by ``ov``, the last extends ``start`` back by ``ov``, and
    middle chunks extend ``start`` back by ``ov // 2`` and ``stop`` forward by
    ``ov - ov // 2``. Everything is clipped to ``[0, n_points]``.
    """
    if n_chunks < 1 or n_chunks > n_points:
        raise ValueError(
            f"n_chunks must satisfy 1 <= n_chunks <= n_points ({n_points}), got {n_chunks}"
        )
    sizes = [len(c) for c in np.array_split(np.arange(n_points), n_chunks)]
    bounds = []
    start = 0
    last = n_chunks - 1
    for i, base_len in enumerate(sizes):
        base_start = start
        base_stop = start + base_len
        start = base_stop  # advance for next iteration (base partition)

        s, e = base_start, base_stop
        if overlap_pct > 0 and n_chunks > 1:
            ov = int(base_len * overlap_pct / 100)
            if i == 0:
                e += ov
            elif i == last:
                s -= ov
            else:
                s -= ov // 2
                e += ov - ov // 2
        s = max(0, min(s, n_points))
        e = max(0, min(e, n_points))
        bounds.append((int(s), int(e)))
    return bounds


def _fit_poly(x, y, order: int):
    """Fit a degree-``order`` polynomial in a scaled domain for numerical
    stability. Returns ``(poly, coeffs)`` where ``poly`` is a
    ``numpy.polynomial.Polynomial`` mapped onto ``[x.min, x.max]`` and
    ``coeffs`` are power-basis coefficients in the ORIGINAL x, highest degree
    first (np.polyfit convention), for display only.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = int(order)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax == xmin:
        # Degenerate domain: fall back to the mean constant.
        mean = float(np.mean(y))
        poly = Polynomial([mean])
        coeffs = np.array([mean], dtype=float)
        return poly, coeffs
    poly = Polynomial.fit(x, y, order)
    # Convert to original-x power basis (window == domain => identity scaling).
    p_orig = poly.convert(domain=poly.domain, window=poly.domain)
    coeffs = np.asarray(p_orig.coef, dtype=float)[::-1]  # highest degree first
    return poly, coeffs


@dataclass
class ChunkFit:
    """Fit of a single chunk.

    ``coeffs`` are power-basis coefficients in the ORIGINAL x, highest degree
    first (for display). ``predict`` evaluates in a scaled domain and is stable
    for high orders.
    """

    start: int
    stop: int
    x_min: float
    x_max: float
    coeffs: np.ndarray
    rmse: float
    # Internal: base (non-overlapped) range of this chunk, used for piecewise
    # assignment and blending. Defaults mirror the fitted slice.
    base_start: int = -1
    base_stop: int = -1
    base_x_min: float = field(default=np.nan, compare=False)
    base_x_max: float = field(default=np.nan, compare=False)
    _poly: "Polynomial" = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.base_start == -1:
            self.base_start = self.start
        if self.base_stop == -1:
            self.base_stop = self.stop
        if np.isnan(self.base_x_min):
            self.base_x_min = self.x_min
        if np.isnan(self.base_x_max):
            self.base_x_max = self.x_max

    def predict(self, xq) -> np.ndarray:
        xq = np.asarray(xq, dtype=float)
        if self._poly is None:
            return np.polyval(self.coeffs, xq)
        return np.asarray(self._poly(xq), dtype=float)


def _make_chunk_fit(x, y, start, stop, order, base_start=None, base_stop=None) -> "ChunkFit":
    """Build a ChunkFit for slice x[start:stop], reducing order if too few
    points (order -> len-1). Shared helper used by fit_chunks and friends."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xs = x[start:stop]
    ys = y[start:stop]
    use_order = min(int(order), len(xs) - 1)
    if use_order < 0:
        use_order = 0
    poly, coeffs = _fit_poly(xs, ys, use_order)
    pred = np.asarray(poly(xs), dtype=float)
    if base_start is None:
        base_start = start
    if base_stop is None:
        base_stop = stop
    return ChunkFit(
        start=int(start),
        stop=int(stop),
        x_min=float(np.min(xs)),
        x_max=float(np.max(xs)),
        coeffs=coeffs,
        rmse=rmse(ys, pred),
        base_start=int(base_start),
        base_stop=int(base_stop),
        base_x_min=float(x[base_start]),
        base_x_max=float(x[base_stop - 1]),
        _poly=poly,
    )


# --------------------------------------------------------------------------
# Phase-1 core (chunked fit, piecewise evaluation, extrapolation)
# --------------------------------------------------------------------------
def fit_chunks(x, y, n_chunks: int, order: int, overlap_pct: float = 0.0):
    """Fit one polynomial per chunk. Chunks are ``chunk_bounds(n, n_chunks,
    overlap_pct)``; each chunk carries its base (non-overlapped) range too. If a
    chunk has ``<= order`` points its order is reduced to ``len - 1``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    base = chunk_bounds(n, n_chunks, 0.0)
    fitted = chunk_bounds(n, n_chunks, overlap_pct)
    fits = []
    for (bs, be), (s, e) in zip(base, fitted):
        fits.append(_make_chunk_fit(x, y, s, e, order, base_start=bs, base_stop=be))
    return fits


def evaluate_piecewise(fits, xq, blend: bool = True) -> np.ndarray:
    """Evaluate the piecewise model at ``xq``.

    With ``blend=False`` each point uses the chunk whose base x-range contains
    it. With ``blend=True`` points inside the overlap of two adjacent chunks are
    a linear cross-fade of the two predictions, so the curve has no seams.
    """
    xq = np.asarray(xq, dtype=float)
    n = len(fits)
    if n == 0:
        return np.full(xq.shape, np.nan)
    if n == 1:
        return fits[0].predict(xq)

    # Assign each query point to a base chunk via midpoint knots.
    knots = np.array(
        [(fits[i].base_x_max + fits[i + 1].base_x_min) / 2.0 for i in range(n - 1)],
        dtype=float,
    )
    idx = np.clip(np.searchsorted(knots, xq, side="right"), 0, n - 1)

    out = np.empty(xq.shape, dtype=float)
    for i, fit in enumerate(fits):
        mask = idx == i
        if np.any(mask):
            out[mask] = fit.predict(xq[mask])
    if not blend:
        return out

    # Cross-fade inside each adjacent overlap region (fitted-range overlap).
    for i in range(n - 1):
        a, b = fits[i], fits[i + 1]
        lo = b.x_min  # start of right chunk's fitted range
        hi = a.x_max  # end of left chunk's fitted range
        if hi <= lo:
            continue  # no overlap between these two chunks
        m = (xq >= lo) & (xq <= hi)
        if not np.any(m):
            continue
        xm = xq[m]
        wb = (xm - lo) / (hi - lo)  # 0 at lo (use a), 1 at hi (use b)
        out[m] = (1.0 - wb) * a.predict(xm) + wb * b.predict(xm)
    return out


def extrapolate(x, y, fit_fraction: float, order: int):
    """Fit one polynomial to the first ``fit_fraction`` of the points (at least
    ``order + 1``), return ``(coeffs, y_pred_over_all_x)``. ``coeffs`` are
    power-basis, highest degree first (np.polyfit convention)."""
    if not (0.0 < fit_fraction <= 1.0):
        raise ValueError(f"fit_fraction must be in (0, 1], got {fit_fraction}")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    n_fit = int(fit_fraction * n)
    n_fit = max(n_fit, int(order) + 1)
    n_fit = min(n_fit, n)
    poly, coeffs = _fit_poly(x[:n_fit], y[:n_fit], order)
    y_pred = np.asarray(poly(x), dtype=float)
    return coeffs, y_pred


# --------------------------------------------------------------------------
# Error map (parallel RMSE grid) — teammate: ErrMap
# --------------------------------------------------------------------------
def _cell(args):
    """Un-normalized RMSE for one (x, y, order, n_chunks, overlap_pct) cell.

    Never raises: a failing cell yields NaN so one bad combination cannot
    crash the whole map.
    """
    x, y, order, n_chunks, overlap_pct = args
    try:
        fits = fit_chunks(x, y, n_chunks, order, overlap_pct)
        return rmse(y, evaluate_piecewise(fits, x, blend=False))
    except Exception:
        return np.nan


def error_map(x, y, orders, chunk_counts, overlap_pcts, workers: int | None = None) -> np.ndarray:
    """2-D RMSE map over the two sequences that vary (length > 1).

    Axis 0 is the first varying sequence in (orders, chunk_counts,
    overlap_pcts) order; axis 1 is the second. The remaining sequence must
    have length 1 and supplies its single value.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sequences = [list(orders), list(chunk_counts), list(overlap_pcts)]
    varying = [i for i, seq in enumerate(sequences) if len(seq) > 1]
    if len(varying) != 2 or any(len(seq) == 0 for seq in sequences):
        raise ValueError("exactly two of orders, chunk_counts, overlap_pcts must have length > 1")
    fixed_by_index = {i: seq[0] for i, seq in enumerate(sequences) if i not in varying}
    axis_a, axis_b = varying
    shape = (len(sequences[axis_a]), len(sequences[axis_b]))

    tasks = []
    for va, vb in itertools.product(sequences[axis_a], sequences[axis_b]):
        values = dict(fixed_by_index)
        values[axis_a] = va
        values[axis_b] = vb
        tasks.append((x, y, values[0], values[1], values[2]))

    if workers == 1:
        results = [_cell(t) for t in tasks]
    else:
        n_workers = workers if workers is not None else min(os.cpu_count() or 1, len(tasks))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_cell, tasks))
    return np.asarray(results, dtype=float).reshape(shape)


# --------------------------------------------------------------------------
# Robust fits & order selection — teammate: Robust
# --------------------------------------------------------------------------
HUBER_K = 1.345
MAD_SCALE = 1.4826
IRLS_ITERS = 10
CV_FOLDS = 5


def _robust_scale(r: np.ndarray) -> float:
    return MAD_SCALE * float(np.median(np.abs(r - np.median(r))))


def _huber_weights(u: np.ndarray) -> np.ndarray:
    au = np.abs(u)
    return np.where(au <= 1.0, 1.0, 1.0 / np.maximum(au, 1e-12))


def _irls_fit(xs: np.ndarray, ys: np.ndarray, order: int):
    w = np.ones_like(ys)
    poly = Polynomial.fit(xs, ys, order, w=w)
    for _ in range(IRLS_ITERS):
        r = ys - poly(xs)
        s = _robust_scale(r)
        if s == 0.0:
            break
        new_w = _huber_weights(r / (HUBER_K * s))
        if np.allclose(new_w, w, rtol=1e-6, atol=1e-9):
            break
        w = new_w
        poly = Polynomial.fit(xs, ys, order, w=w)
    return poly


def robust_fit_chunks(x, y, n_chunks: int, order: int, overlap_pct: float = 0.0) -> list[ChunkFit]:
    """Like ``fit_chunks`` but each chunk is fitted by Huber IRLS, so gross
    outliers have bounded influence."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    base = chunk_bounds(n, n_chunks, 0.0)
    fitted = chunk_bounds(n, n_chunks, overlap_pct)
    fits = []
    for (bs, be), (s, e) in zip(base, fitted):
        xs = x[s:e]
        ys = y[s:e]
        use_order = max(0, min(int(order), len(xs) - 1))
        if np.ptp(xs) == 0:
            poly = Polynomial([float(np.mean(ys))])
        else:
            poly = _irls_fit(xs, ys, use_order)
        p_orig = poly.convert(domain=poly.domain, window=poly.domain)
        coeffs = np.asarray(p_orig.coef, dtype=float)[::-1]
        fits.append(
            ChunkFit(
                start=int(s),
                stop=int(e),
                x_min=float(np.min(xs)),
                x_max=float(np.max(xs)),
                coeffs=coeffs,
                rmse=rmse(ys, poly(xs)),
                base_start=int(bs),
                base_stop=int(be),
                base_x_min=float(x[bs]),
                base_x_max=float(x[be - 1]),
                _poly=poly,
            )
        )
    return fits


def _cv_rmse(x, y, n_chunks: int, order: int, k: int) -> float | None:
    idx = np.arange(len(x))
    errs = []
    for f in range(k):
        test = idx % k == f
        train = ~test
        n_train = int(train.sum())
        if n_train == 0 or min(len(c) for c in np.array_split(np.arange(n_train), n_chunks)) <= order:
            return None
        fits = fit_chunks(x[train], y[train], n_chunks, order)
        yhat = evaluate_piecewise(fits, x[test], blend=False)
        errs.append(rmse(y[test], yhat))
    return float(np.mean(errs))


def _aic(x, y, n_chunks: int, order: int) -> float:
    n = len(x)
    fits = fit_chunks(x, y, n_chunks, order)
    yhat = evaluate_piecewise(fits, x, blend=False)
    sse = float(np.sum((y - yhat) ** 2))
    k_params = n_chunks * (order + 1)
    return n * np.log(sse / n + 1e-12) + 2 * k_params


def select_order(x, y, n_chunks: int, max_order: int, method: str = "cv") -> int:
    """Return the polynomial order in ``[0, max_order]`` that best suits
    ``fit_chunks(x, y, n_chunks, order)``, by k-fold CV rmse or AIC."""
    if method not in ("cv", "aic"):
        raise ValueError(f"unknown method {method!r}; expected 'cv' or 'aic'")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if method == "cv":
        k = min(CV_FOLDS, n)
        scores = {}
        for order in range(max_order + 1):
            score = _cv_rmse(x, y, n_chunks, order, k)
            if score is not None:
                scores[order] = score
        if not scores:
            return 0
        return min(scores, key=scores.get)

    scores = {order: _aic(x, y, n_chunks, order) for order in range(max_order + 1)}
    return min(scores, key=scores.get)


# --------------------------------------------------------------------------
# Constrained splines & adaptive chunking — teammate: Spline
# --------------------------------------------------------------------------
def _falling(n: np.ndarray, d: int) -> np.ndarray:
    """n! / (n - d)! elementwise, zero where n < d."""
    out = np.ones(np.shape(n), dtype=float)
    for t in range(d):
        out = out * (n - t)
    return np.where(n >= d, out, 0.0)


class SplineFit:
    """Piecewise polynomial in u, with coef[j, k] the u**k coefficient of piece j."""

    def __init__(self, knots, coef, xmid, xhalf, order, continuity):
        self.knots = np.asarray(knots, dtype=float)
        self.coef = np.asarray(coef, dtype=float)
        self.xmid = float(xmid)
        self.xhalf = float(xhalf)
        self.order = int(order)
        self.continuity = int(continuity)
        self.n_chunks = self.coef.shape[0]

    def _piece_and_u(self, xq):
        xq = np.asarray(xq, dtype=float)
        piece = np.clip(
            np.searchsorted(self.knots, xq, side="right"), 0, self.n_chunks - 1
        )
        u = (xq - self.xmid) / self.xhalf
        return piece, u

    @staticmethod
    def _eval(coef, piece, u):
        powers = u[..., None] ** np.arange(coef.shape[1])
        return np.sum(coef[piece] * powers, axis=-1)

    def predict(self, xq) -> np.ndarray:
        piece, u = self._piece_and_u(xq)
        return self._eval(self.coef, piece, u)

    def derivative(self, xq, m: int = 1) -> np.ndarray:
        """m-th derivative with respect to the ORIGINAL x."""
        if m < 0:
            raise ValueError(f"m must be >= 0, got {m}")
        if m == 0:
            return self.predict(xq)
        p = self.order
        dcoef = np.zeros_like(self.coef)
        if m <= p:
            k = np.arange(p + 1 - m)
            dcoef[:, : p + 1 - m] = self.coef[:, m:] * _falling(k + m, m)
        piece, u = self._piece_and_u(xq)
        return self._eval(dcoef, piece, u) * (1.0 / self.xhalf) ** m


def fit_spline(x, y, n_chunks: int, order: int, continuity: int = 2) -> SplineFit:
    """Jointly fit one degree-``order`` polynomial per base chunk, minimising
    total squared error subject to C^continuity joins at the interior knots."""
    if not -1 <= continuity < order:
        raise ValueError(
            f"continuity must satisfy -1 <= continuity < order ({order}), got {continuity}"
        )
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    base = chunk_bounds(n, n_chunks, 0.0)
    P = order + 1
    N = n_chunks * P

    xmin, xmax = float(x.min()), float(x.max())
    xmid = (xmin + xmax) / 2.0
    xhalf = (xmax - xmin) / 2.0 or 1.0
    u = (x - xmid) / xhalf

    piece_of = np.empty(n, dtype=int)
    for j, (s, e) in enumerate(base):
        piece_of[s:e] = j
    A = np.zeros((n, N))
    cols = piece_of[:, None] * P + np.arange(P)
    A[np.arange(n)[:, None], cols] = u[:, None] ** np.arange(P)

    k = np.arange(P)
    rows = []
    for i in range(1, n_chunks):
        u0 = (x[base[i][0]] - xmid) / xhalf
        for d in range(continuity + 1):
            vals = _falling(k, d) * u0 ** np.maximum(k - d, 0)
            row = np.zeros(N)
            row[(i - 1) * P : i * P] = vals
            row[i * P : (i + 1) * P] = -vals
            rows.append(row)
    C = np.array(rows, dtype=float).reshape(-1, N)

    if C.shape[0] == 0:
        Z = np.eye(N)
    else:
        _, s, vh = np.linalg.svd(C)
        tol = max(C.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
        rank = int(np.sum(s > tol))
        Z = vh[rank:].T

    z = np.linalg.lstsq(A @ Z, y, rcond=1e-12)[0]
    coef = (Z @ z).reshape(n_chunks, P)
    knots = np.array([x[base[i][0]] for i in range(1, n_chunks)], dtype=float)
    return SplineFit(knots, coef, xmid, xhalf, order, continuity)


def _chunk_rmse(xs, ys, order: int) -> float:
    m = len(xs)
    if m == 0:
        return 0.0
    if xs.max() == xs.min():
        return rmse(ys, np.full(m, ys.mean()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        poly = Polynomial.fit(xs, ys, min(order, m - 1))
    return rmse(ys, poly(xs))


def adaptive_chunks(x, y, order: int, tol: float, max_chunks: int) -> list[tuple[int, int]]:
    """Greedy contiguous partition: extend each chunk while its degree-``order``
    fit rmse stays <= tol, capped at ``max_chunks`` (remainder goes last)."""
    if max_chunks < 1:
        raise ValueError(f"max_chunks must be >= 1, got {max_chunks}")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < n:
        if len(chunks) == max_chunks - 1:
            chunks.append((start, n))
            break
        end = start + 1
        while end < n and _chunk_rmse(x[start : end + 1], y[start : end + 1], order) <= tol:
            end += 1
        chunks.append((start, end))
        start = end
    return chunks
