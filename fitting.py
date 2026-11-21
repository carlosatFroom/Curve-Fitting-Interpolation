"""Pure-numpy piecewise polynomial fitting used by the curve-fitting app.

No GUI imports here: this module must stay importable without PyQt5,
pyqtgraph or matplotlib.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import itertools
import os

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial import legendre as L

__all__ = [
    "chunk_bounds", "ChunkFit", "fit_chunks", "evaluate_piecewise", "rmse", "r2",
    "error_map", "extrapolate", "SplineFit", "fit_spline", "adaptive_chunks",
    "robust_fit_chunks", "select_order",
]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------

def _base_bounds(n_points: int, n_chunks: int) -> list[tuple[int, int]]:
    # Same partition as np.array_split(np.arange(n_points), n_chunks).
    q, r = divmod(n_points, n_chunks)
    bounds, start = [], 0
    for i in range(n_chunks):
        stop = start + q + (1 if i < r else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def chunk_bounds(n_points: int, n_chunks: int, overlap_pct: float = 0.0) -> list[tuple[int, int]]:
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


# ---------------------------------------------------------------------------
# per-chunk polynomial fits
# ---------------------------------------------------------------------------

def _fit_poly(xs, ys, order: int, w=None) -> Polynomial:
    """Least-squares polynomial in a scaled domain; degrades order when data is short."""
    deg = max(0, min(order, len(xs) - 1))
    if len(xs) == 0:
        return Polynomial([0.0])
    if xs.min() == xs.max():
        # All x identical: only a constant is identifiable.
        if w is None:
            c = float(np.mean(ys))
        else:
            ww = np.asarray(w, dtype=float) ** 2
            c = float(np.sum(ww * ys) / np.sum(ww)) if ww.sum() > 0 else float(np.mean(ys))
        return Polynomial([c], domain=[xs[0] - 1, xs[0] + 1])
    return Polynomial.fit(xs, ys, deg, w=w)


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

    @property
    def order(self) -> int:
        return len(self.coeffs) - 1

    def predict(self, xq) -> np.ndarray:
        return np.asarray(self.poly(np.asarray(xq, dtype=float)), dtype=float)


def _make_chunkfit(x, y, start, stop, base, poly) -> ChunkFit:
    xs, ys = x[start:stop], y[start:stop]
    coeffs = poly.convert().coef[::-1].copy()  # original x, highest degree first
    return ChunkFit(
        start=start, stop=stop, x_min=float(xs.min()), x_max=float(xs.max()),
        coeffs=coeffs, rmse=rmse(ys, poly(xs)), poly=poly,
        base_start=base[0], base_stop=base[1], base_x_min=float(x[base[0]]),
    )


def _as_xy(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    return x, y


def fit_chunks(x, y, n_chunks: int, order: int, overlap_pct: float = 0.0) -> list[ChunkFit]:
    x, y = _as_xy(x, y)
    if order < 0:
        raise ValueError("order must be >= 0")
    bounds = chunk_bounds(len(x), n_chunks, overlap_pct)
    base = _base_bounds(len(x), n_chunks)
    return [
        _make_chunkfit(x, y, s, e, b, _fit_poly(x[s:e], y[s:e], order))
        for (s, e), b in zip(bounds, base)
    ]


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def evaluate_piecewise(fits: list[ChunkFit], xq, blend: bool = True) -> np.ndarray:
    if not fits:
        raise ValueError("fits is empty")
    xq = np.asarray(xq, dtype=float)
    shape = xq.shape
    xq = xq.ravel()
    knots = np.array([f.base_x_min for f in fits[1:]])
    idx = np.searchsorted(knots, xq, side="right")  # a point at a knot -> right piece
    out = np.empty_like(xq)
    for i, f in enumerate(fits):
        m = idx == i
        if m.any():
            out[m] = f.predict(xq[m])
    if blend:
        for i in range(len(fits) - 1):
            left, right = fits[i], fits[i + 1]
            knot = right.base_x_min
            lo = min(right.x_min, knot)
            hi = max(left.x_max, knot)
            if hi <= lo:
                continue
            m = (xq >= lo) & (xq <= hi)
            if not m.any():
                continue
            w = _smoothstep((xq[m] - lo) / (hi - lo))
            out[m] = (1.0 - w) * left.predict(xq[m]) + w * right.predict(xq[m])
    return out.reshape(shape)


# ---------------------------------------------------------------------------
# error map
# ---------------------------------------------------------------------------

def _piecewise_rmse(x, y, n_chunks, order, overlap_pct) -> float:
    if n_chunks > len(x):
        return float("nan")
    fits = fit_chunks(x, y, n_chunks, order, overlap_pct)
    return rmse(y, evaluate_piecewise(fits, x, blend=False))


def error_map(x, y, orders, chunk_counts, overlap_pcts, workers: int | None = None) -> np.ndarray:
    x, y = _as_xy(x, y)
    seqs = [list(orders), list(chunk_counts), list(overlap_pcts)]
    varying = [i for i, s in enumerate(seqs) if len(s) > 1]
    if len(varying) != 2 or any(len(s) == 0 for s in seqs):
        raise ValueError("exactly two of orders/chunk_counts/overlap_pcts must have length > 1, "
                         "the third length 1")
    a, b = varying
    shape = (len(seqs[a]), len(seqs[b]))
    cells = list(itertools.product(range(shape[0]), range(shape[1])))

    def job(cell):
        args = [s[0] for s in seqs]
        args[a] = seqs[a][cell[0]]
        args[b] = seqs[b][cell[1]]
        order, n_chunks, ov = args
        return _piecewise_rmse(x, y, int(n_chunks), int(order), float(ov))

    if workers == 1:
        values = [job(c) for c in cells]
    else:
        n_workers = workers or min(8, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            values = list(ex.map(job, cells))
    return np.array(values, dtype=float).reshape(shape)


# ---------------------------------------------------------------------------
# extrapolation
# ---------------------------------------------------------------------------

def extrapolate(x, y, fit_fraction: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = _as_xy(x, y)
    if not 0 < fit_fraction <= 1:
        raise ValueError("fit_fraction must be in (0, 1]")
    if order < 0 or len(x) < order + 1:
        raise ValueError("need at least order + 1 points")
    n_fit = min(len(x), max(order + 1, int(round(fit_fraction * len(x)))))
    poly = _fit_poly(x[:n_fit], y[:n_fit], order)
    return poly.convert().coef[::-1].copy(), np.asarray(poly(x), dtype=float)


# ---------------------------------------------------------------------------
# phase 2: jointly-solved constrained spline
# ---------------------------------------------------------------------------

@dataclass
class SplineFit:
    knots: np.ndarray
    order: int
    continuity: int
    # per piece: Legendre coefficients in local variable t = (x - center) / half
    leg_coeffs: np.ndarray = field(repr=False)
    centers: np.ndarray = field(repr=False)
    halves: np.ndarray = field(repr=False)

    def _piece_index(self, xq):
        return np.searchsorted(self.knots, xq, side="right")

    def derivative(self, xq, m: int = 1) -> np.ndarray:
        if m < 0:
            raise ValueError("m must be >= 0")
        xq = np.asarray(xq, dtype=float)
        shape = xq.shape
        xq = xq.ravel()
        idx = self._piece_index(xq)
        out = np.zeros_like(xq)
        for i in np.unique(idx):
            sel = idx == i
            c = self.leg_coeffs[i]
            if m > 0:
                c = L.legder(c, m) if m <= self.order else np.zeros(1)
            t = (xq[sel] - self.centers[i]) / self.halves[i]
            out[sel] = L.legval(t, c) / self.halves[i] ** m
        return out.reshape(shape)

    def predict(self, xq) -> np.ndarray:
        return self.derivative(xq, 0)


def _leg_basis_derivs(t, order, m):
    """Row vector of d^m/dt^m P_k(t), k = 0..order, at scalar t."""
    row = np.empty(order + 1)
    for k in range(order + 1):
        e = np.zeros(k + 1)
        e[k] = 1.0
        row[k] = L.legval(t, L.legder(e, m)) if m <= k else 0.0
    return row


def fit_spline(x, y, n_chunks: int, order: int, continuity: int = 2) -> SplineFit:
    x, y = _as_xy(x, y)
    if not -1 <= continuity < order:
        raise ValueError("need -1 <= continuity < order")
    bounds = chunk_bounds(len(x), n_chunks, 0)
    knots = np.array([x[s] for s, _ in bounds[1:]], dtype=float)
    edges = np.concatenate([[x[0]], knots, [x[-1]]])
    centers = (edges[:-1] + edges[1:]) / 2
    halves = (edges[1:] - edges[:-1]) / 2
    halves[halves <= 0] = 1.0
    p = order + 1
    n_unk = n_chunks * p

    # Design matrix: each point belongs to one piece (knot -> right piece).
    idx = np.searchsorted(knots, x, side="right")
    A = np.zeros((len(x), n_unk))
    for i in range(n_chunks):
        sel = idx == i
        t = (x[sel] - centers[i]) / halves[i]
        A[np.ix_(sel, np.arange(i * p, (i + 1) * p))] = L.legvander(t, order)

    # Continuity constraints: C c = 0 (derivatives w.r.t. x, rescaled per row).
    rows = []
    for i, k in enumerate(knots):
        tl = (k - centers[i]) / halves[i]
        tr = (k - centers[i + 1]) / halves[i + 1]
        for m in range(continuity + 1):
            row = np.zeros(n_unk)
            row[i * p:(i + 1) * p] = _leg_basis_derivs(tl, order, m) / halves[i] ** m
            row[(i + 1) * p:(i + 2) * p] = -_leg_basis_derivs(tr, order, m) / halves[i + 1] ** m
            scale = np.abs(row).max()
            rows.append(row / scale if scale > 0 else row)

    if rows:
        C = np.array(rows)
        # Null-space parametrisation: c = N z exactly satisfies C c = 0.
        _, s, vt = np.linalg.svd(C)
        rank = int(np.sum(s > s[0] * max(C.shape) * np.finfo(float).eps))
        N = vt[rank:].T
        z, *_ = np.linalg.lstsq(A @ N, y, rcond=None)
        c = N @ z
    else:
        c, *_ = np.linalg.lstsq(A, y, rcond=None)

    return SplineFit(knots=knots, order=order, continuity=continuity,
                     leg_coeffs=c.reshape(n_chunks, p), centers=centers, halves=halves)


# ---------------------------------------------------------------------------
# phase 2: adaptive chunking
# ---------------------------------------------------------------------------

def _segment_rmse(x, y, s, e, order) -> float:
    xs, ys = x[s:e], y[s:e]
    return rmse(ys, _fit_poly(xs, ys, order)(xs))


def adaptive_chunks(x, y, order: int, tol: float, max_chunks: int) -> list[tuple[int, int]]:
    x, y = _as_xy(x, y)
    n = len(x)
    if n == 0:
        return []
    if max_chunks < 1:
        raise ValueError("max_chunks must be >= 1")

    def ok(s, e):
        return _segment_rmse(x, y, s, e, order) <= tol

    # Greedy: from each start take the longest feasible segment (exponential +
    # binary search). A segment of order + 1 points is interpolated exactly.
    min_len = order + 1
    chunks, s = [], 0
    while s < n:
        lo = min(n, s + min_len)  # always accepted
        if lo == n:
            chunks.append((s, n))
            break
        step, hi = 1, None
        while lo < n:
            cand = min(n, lo + step)
            if not ok(s, cand):
                hi = cand
                break
            lo = cand
            step *= 2
        if hi is None:  # reached the end while still feasible
            chunks.append((s, n))
            break
        # lo feasible, hi infeasible -> bisect
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ok(s, mid):
                lo = mid
            else:
                hi = mid
        chunks.append((s, lo))
        s = lo

    # Too many chunks: merge the adjacent pair that hurts least.
    while len(chunks) > max_chunks:
        costs = [_segment_rmse(x, y, chunks[i][0], chunks[i + 1][1], order)
                 for i in range(len(chunks) - 1)]
        i = int(np.argmin(costs))
        chunks[i:i + 2] = [(chunks[i][0], chunks[i + 1][1])]
    return chunks


# ---------------------------------------------------------------------------
# phase 2: robust fitting (IRLS: Huber then Tukey bisquare, MAD scale)
# ---------------------------------------------------------------------------

def _mad_scale(r) -> float:
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def _robust_poly(xs, ys, order, iters=30) -> Polynomial:
    poly = _fit_poly(xs, ys, order)
    deg = max(0, min(order, len(xs) - 1))
    if len(xs) <= deg + 1:
        return poly
    for c, kind in ((1.345, "huber"), (4.685, "tukey")):
        for _ in range(iters):
            r = ys - poly(xs)
            scale = _mad_scale(r)
            if scale <= 1e-12 * (np.abs(ys).max() + 1e-300):
                return poly
            u = r / (c * scale)
            if kind == "huber":
                wts = np.minimum(1.0, 1.0 / np.maximum(np.abs(u), 1e-12))
            else:
                wts = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
            if np.count_nonzero(wts) <= deg:
                return poly
            new = _fit_poly(xs, ys, order, w=np.sqrt(wts))
            delta = np.max(np.abs(new(xs) - poly(xs)))
            poly = new
            if delta <= 1e-10 * (scale + 1e-300):
                break
    return poly


def robust_fit_chunks(x, y, n_chunks: int, order: int, overlap_pct: float = 0.0) -> list[ChunkFit]:
    x, y = _as_xy(x, y)
    if order < 0:
        raise ValueError("order must be >= 0")
    bounds = chunk_bounds(len(x), n_chunks, overlap_pct)
    base = _base_bounds(len(x), n_chunks)
    return [
        _make_chunkfit(x, y, s, e, b, _robust_poly(x[s:e], y[s:e], order))
        for (s, e), b in zip(bounds, base)
    ]


# ---------------------------------------------------------------------------
# phase 2: model order selection
# ---------------------------------------------------------------------------

def select_order(x, y, n_chunks: int, max_order: int, method: str = "cv") -> int:
    x, y = _as_xy(x, y)
    n = len(x)
    if max_order < 0:
        raise ValueError("max_order must be >= 0")
    orders = range(max_order + 1)
    if method == "aic":
        scores = []
        for o in orders:
            fits = fit_chunks(x, y, n_chunks, o)
            rss = float(np.sum((y - evaluate_piecewise(fits, x, blend=False)) ** 2))
            k = sum(f.order + 1 for f in fits)
            scores.append(n * np.log(max(rss / n, 1e-300)) + 2 * k)
    elif method == "cv":
        k_folds = min(5, n)
        folds = np.arange(n) % k_folds  # interleaved folds keep every chunk covered
        scores = []
        for o in orders:
            sse = 0.0
            for f in range(k_folds):
                tr, te = folds != f, folds == f
                nc = min(n_chunks, int(tr.sum()))
                fits = fit_chunks(x[tr], y[tr], nc, o)
                sse += float(np.sum((y[te] - evaluate_piecewise(fits, x[te], blend=False)) ** 2))
            scores.append(sse)
    else:
        raise ValueError("method must be 'cv' or 'aic'")
    scores = np.asarray(scores)
    best = np.nanmin(scores)
    # Smallest order within a hair of the best score (prefer simpler models on ties).
    tol = 1e-9 * abs(best) + 1e-12
    return int(np.flatnonzero(scores <= best + tol)[0])
