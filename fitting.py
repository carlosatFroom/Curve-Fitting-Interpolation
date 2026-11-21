"""Pure-numpy piecewise polynomial fitting. No GUI imports."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial import chebyshev as cheb


# ----------------------------------------------------------------- helpers

def _as_xy(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    return x, y


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)  # C2 quintic


def rmse(y_true, y_pred) -> float:
    d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(d ** 2)))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    ss_res = float(np.sum((y_true - np.asarray(y_pred, dtype=float)) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


# ------------------------------------------------------------------ chunks

def chunk_bounds(n_points: int, n_chunks: int, overlap_pct: float = 0.0):
    if n_chunks < 1 or n_chunks > n_points:
        raise ValueError("need 1 <= n_chunks <= n_points")
    base = [(int(a[0]), int(a[-1]) + 1) for a in np.array_split(np.arange(n_points), n_chunks)]
    if overlap_pct <= 0:
        return base
    out = []
    for i, (a, b) in enumerate(base):
        ov = int((b - a) * overlap_pct / 100)
        if i == 0:
            b += ov
        elif i == n_chunks - 1:
            a -= ov
        else:
            a -= ov // 2
            b += ov - ov // 2
        out.append((max(a, 0), min(b, n_points)))
    return out


@dataclass
class ChunkFit:
    start: int
    stop: int
    x_min: float
    x_max: float
    coeffs: np.ndarray  # original-x power basis, highest degree first (display only)
    rmse: float
    poly: Optional[Polynomial] = None  # scaled-domain polynomial used by predict
    base_x_start: Optional[float] = None  # x of first point of the base (non-overlapped) chunk

    def predict(self, xq) -> np.ndarray:
        xq = np.asarray(xq, dtype=float)
        if self.poly is not None:
            return self.poly(xq)
        return np.polyval(self.coeffs, xq)


def _fit_poly(x, y, order, weights=None) -> Polynomial:
    """Scaled-domain least squares; order reduced when too few points / degenerate x."""
    order = max(0, min(int(order), len(x) - 1))
    if len(x) == 0:
        raise ValueError("cannot fit an empty chunk")
    if np.ptp(x) == 0:
        order = 0
    w = None if weights is None else np.sqrt(weights)
    return Polynomial.fit(x, y, order, w=w)


def _make_chunk(x, y, start, stop, poly, base_x_start):
    xs, ys = x[start:stop], y[start:stop]
    coeffs = poly.convert().coef[::-1].copy()
    return ChunkFit(start, stop, float(xs.min()), float(xs.max()), coeffs,
                    rmse(ys, poly(xs)), poly, float(base_x_start))


def _build(x, y, n_chunks, order, overlap_pct, fitter):
    x, y = _as_xy(x, y)
    if order < 0:
        raise ValueError("order must be >= 0")
    base = chunk_bounds(len(x), n_chunks, 0.0)
    fits = []
    for (a, b), (s, e) in zip(base, chunk_bounds(len(x), n_chunks, overlap_pct)):
        fits.append(_make_chunk(x, y, s, e, fitter(x[s:e], y[s:e], order), x[a]))
    return fits


def fit_chunks(x, y, n_chunks: int, order: int, overlap_pct: float = 0.0):
    return _build(x, y, n_chunks, order, overlap_pct, _fit_poly)


def _seams(fits, blend):
    """Per-seam (lo, hi): weight of next chunk goes 0->1 over [lo, hi]; lo == hi is a step."""
    lo = np.empty(len(fits) - 1)
    hi = np.empty(len(fits) - 1)
    for i in range(len(fits) - 1):
        a, b = fits[i], fits[i + 1]
        boundary = b.base_x_start if b.base_x_start is not None else b.x_min
        l, h = b.x_min, a.x_max
        if not blend or h <= l:
            l = h = boundary
        lo[i], hi[i] = l, h
    lo = np.maximum.accumulate(lo) if len(lo) else lo
    hi = np.maximum(np.maximum.accumulate(hi), lo) if len(hi) else hi
    return lo, hi


def evaluate_piecewise(fits, xq, blend: bool = True) -> np.ndarray:
    xq = np.asarray(xq, dtype=float)
    flat = xq.ravel()
    out = np.empty(flat.shape)
    if len(fits) == 1:
        return fits[0].predict(xq)
    lo, hi = _seams(fits, blend)
    k = np.searchsorted(hi, flat, side="right")  # chunk fully active
    t = np.zeros(flat.shape)
    mixed = np.zeros(flat.shape, dtype=bool)
    inside = k < len(lo)
    kk = np.minimum(k, len(lo) - 1)
    mixed = inside & (flat > lo[kk])
    t[mixed] = _smoothstep((flat[mixed] - lo[kk[mixed]]) / (hi[kk[mixed]] - lo[kk[mixed]]))
    for j in range(len(fits)):
        m = k == j
        if not m.any():
            continue
        sel = m & ~mixed
        if sel.any():
            out[sel] = fits[j].predict(flat[sel])
        mm = m & mixed
        if mm.any():
            out[mm] = (1 - t[mm]) * fits[j].predict(flat[mm]) + t[mm] * fits[j + 1].predict(flat[mm])
    return out.reshape(xq.shape)


# --------------------------------------------------------------- error map

def _cell_rmse(x, y, order, n_chunks, overlap_pct):
    n = len(x)
    base = chunk_bounds(n, n_chunks, 0.0)
    ss = 0.0
    for (a, b), (s, e) in zip(base, chunk_bounds(n, n_chunks, overlap_pct)):
        p = _fit_poly(x[s:e], y[s:e], order)
        ss += float(np.sum((y[a:b] - p(x[a:b])) ** 2))
    return float(np.sqrt(ss / n))


def _error_cells(x, y, cells, workers=None, should_cancel: Optional[Callable[[], bool]] = None):
    x, y = _as_xy(x, y)

    def run(cell):
        if should_cancel is not None and should_cancel():
            raise CancelledError()
        return _cell_rmse(x, y, *cell)

    if workers == 1 or len(cells) < 2:
        return [run(c) for c in cells]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, cells))


def error_map(x, y, orders, chunk_counts, overlap_pcts, workers: Optional[int] = None,
              should_cancel: Optional[Callable[[], bool]] = None) -> np.ndarray:
    seqs = [list(orders), list(chunk_counts), list(overlap_pcts)]
    varying = [i for i, s in enumerate(seqs) if len(s) > 1]
    if len(varying) != 2 or any(len(s) == 0 for s in seqs):
        raise ValueError("exactly two of the three sequences must have length > 1 (the third length 1)")
    ia, ib = varying
    cells = []
    for va in seqs[ia]:
        for vb in seqs[ib]:
            c = [seqs[0][0], seqs[1][0], seqs[2][0]]
            c[ia], c[ib] = va, vb
            cells.append((int(c[0]), int(c[1]), float(c[2])))
    vals = _error_cells(x, y, cells, workers, should_cancel)
    return np.array(vals).reshape(len(seqs[ia]), len(seqs[ib]))


# ----------------------------------------------------------- extrapolation

def _extrapolation_count(n: int, fit_fraction: float, order: int) -> int:
    if not 0 < fit_fraction <= 1:
        raise ValueError("fit_fraction must be in (0, 1]")
    return min(n, max(order + 1, int(np.floor(fit_fraction * n + 1e-9))))


def extrapolate(x, y, fit_fraction: float, order: int):
    x, y = _as_xy(x, y)
    m = _extrapolation_count(len(x), fit_fraction, order)
    poly = _fit_poly(x[:m], y[:m], order)
    return poly.convert().coef[::-1].copy(), poly(x)


# ------------------------------------------------------------ phase 2: spline

class SplineFit:
    """Piecewise polynomial; each piece is a Chebyshev series in a local scaled variable."""

    def __init__(self, knots, centers, halfwidths, coefs, order, continuity):
        self.knots = np.asarray(knots, dtype=float)
        self.centers = centers
        self.halfwidths = halfwidths
        self.coefs = coefs  # (n_pieces, order + 1)
        self.order = order
        self.continuity = continuity

    def _eval(self, xq, m):
        xq = np.asarray(xq, dtype=float)
        flat = xq.ravel()
        idx = np.searchsorted(self.knots, flat, side="right")
        out = np.zeros(flat.shape)
        for i in np.unique(idx):
            sel = idx == i
            c = self.coefs[i]
            if m:
                if m > self.order:
                    continue
                c = cheb.chebder(c, m)
            out[sel] = cheb.chebval((flat[sel] - self.centers[i]) / self.halfwidths[i], c) / self.halfwidths[i] ** m
        return out.reshape(xq.shape)

    def predict(self, xq) -> np.ndarray:
        return self._eval(xq, 0)

    def derivative(self, xq, m: int = 1) -> np.ndarray:
        if m < 0:
            raise ValueError("m must be >= 0")
        return self._eval(xq, int(m))


def fit_spline(x, y, n_chunks: int, order: int, continuity: int = 2) -> SplineFit:
    x, y = _as_xy(x, y)
    if not -1 <= continuity < order:
        raise ValueError("need -1 <= continuity < order")
    perm = np.argsort(x, kind="stable")
    x, y = x[perm], y[perm]
    n, d1, P = len(x), order + 1, n_chunks
    base = chunk_bounds(n, n_chunks, 0.0)
    knots = np.array([x[a] for a, _ in base[1:]])
    edges = np.concatenate(([x[0]], knots, [x[-1]]))
    centers = (edges[:-1] + edges[1:]) / 2
    halfw = (edges[1:] - edges[:-1]) / 2
    halfw[halfw == 0] = 1.0

    A = np.zeros((n, P * d1))
    for i, (a, b) in enumerate(base):
        A[a:b, i * d1:(i + 1) * d1] = cheb.chebvander((x[a:b] - centers[i]) / halfw[i], order)

    if continuity >= 0 and P > 1:
        rows = []
        eye = np.eye(d1)
        for i, kn in enumerate(knots):
            for m in range(continuity + 1):
                D = cheb.chebder(eye, m, axis=0) if m else eye
                left = cheb.chebval((kn - centers[i]) / halfw[i], D) / halfw[i] ** m
                right = cheb.chebval((kn - centers[i + 1]) / halfw[i + 1], D) / halfw[i + 1] ** m
                row = np.zeros(P * d1)
                row[i * d1:(i + 1) * d1] = left
                row[(i + 1) * d1:(i + 2) * d1] = -right
                rows.append(row / np.linalg.norm(row))
        C = np.array(rows)
        _, s, vt = np.linalg.svd(C, full_matrices=True)
        rank = int(np.sum(s > s.max() * max(C.shape) * np.finfo(float).eps)) if len(s) else 0
        N = vt[rank:].T
        z = np.linalg.lstsq(A @ N, y, rcond=None)[0]
        sol = N @ z
    else:
        sol = np.linalg.lstsq(A, y, rcond=None)[0]
    return SplineFit(knots, centers, halfw, sol.reshape(P, d1), order, continuity)


# ------------------------------------------------------- phase 2: adaptive

def _seg_rmse(x, y, a, b, order):
    p = _fit_poly(x[a:b], y[a:b], order)
    return rmse(y[a:b], p(x[a:b]))


def _greedy_partition(x, y, order, tol, max_chunks):
    n = len(x)
    min_len = 2 * (order + 1)
    parts, a = [], 0
    while a < n:
        if len(parts) == max_chunks:
            return None
        lo = min(min_len, n - a)  # always accepted
        if lo < n - a:
            good, step = lo, max(lo, 1)
            bad = None
            while True:  # grow
                cand = min(good + step, n - a)
                if _seg_rmse(x, y, a, a + cand, order) <= tol:
                    good = cand
                    if cand == n - a:
                        break
                    step *= 2
                else:
                    bad = cand
                    break
            while bad is not None and bad - good > 1:  # bisect
                mid = (good + bad) // 2
                if _seg_rmse(x, y, a, a + mid, order) <= tol:
                    good = mid
                else:
                    bad = mid
            lo = good
        parts.append((a, a + lo))
        a += lo
    return parts


def adaptive_chunks(x, y, order: int, tol: float, max_chunks: int):
    x, y = _as_xy(x, y)
    n = len(x)
    if n == 0:
        return []
    if max_chunks < 1:
        raise ValueError("max_chunks must be >= 1")
    parts = _greedy_partition(x, y, order, tol, max_chunks)
    if parts is not None:
        return parts
    lo, hi = tol, max(tol, _seg_rmse(x, y, 0, n, order))  # hi always yields one chunk
    best = [(0, n)]
    for _ in range(40):
        mid = (lo + hi) / 2
        p = _greedy_partition(x, y, order, mid, max_chunks)
        if p is None:
            lo = mid
        else:
            best, hi = p, mid
    return best


# --------------------------------------------------------- phase 2: robust

_HUBER_C, _TUKEY_C = 1.345, 4.685


def _robust_poly(x, y, order, iters=40):
    order = max(0, min(int(order), len(x) - 1))
    poly = _fit_poly(x, y, order)
    if len(x) <= order + 1 or np.ptp(x) == 0:
        return poly
    floor = 1e-12 * (np.abs(y).max() + 1e-300)
    for it in range(iters):
        res = y - poly(x)
        scale = max(1.4826 * np.median(np.abs(res - np.median(res))), floor)
        u = res / scale
        if it < iters // 2:
            w = np.where(np.abs(u) <= _HUBER_C, 1.0, _HUBER_C / np.maximum(np.abs(u), 1e-300))
        else:
            w = np.where(np.abs(u) < _TUKEY_C, (1 - (u / _TUKEY_C) ** 2) ** 2, 0.0)
        if np.count_nonzero(w > 1e-8) <= order:
            break
        new = _fit_poly(x, y, order, weights=w)
        done = np.allclose(new.coef, poly.coef, rtol=1e-9, atol=1e-12 * (1 + np.abs(poly.coef).max()))
        poly = new
        if done and it >= iters // 2:
            break
    return poly


def robust_fit_chunks(x, y, n_chunks: int, order: int, overlap_pct: float = 0.0):
    return _build(x, y, n_chunks, order, overlap_pct, _robust_poly)


# ------------------------------------------------------ phase 2: order choice

def select_order(x, y, n_chunks: int, max_order: int, method: str = "cv") -> int:
    x, y = _as_xy(x, y)
    if method not in ("cv", "aic"):
        raise ValueError("method must be 'cv' or 'aic'")
    if max_order < 0:
        raise ValueError("max_order must be >= 0")
    n = len(x)
    orders = range(max_order + 1)
    if method == "cv":
        k = min(5, n)
        if k < 2:
            return 0
        folds = np.arange(n) % k  # interleaved folds
        scores = []
        for o in orders:
            sse = 0.0
            for f in range(k):
                tr, te = folds != f, folds == f
                nc = min(n_chunks, int(tr.sum()))
                fits = fit_chunks(x[tr], y[tr], nc, o)
                sse += float(np.sum((y[te] - evaluate_piecewise(fits, x[te], blend=False)) ** 2))
            scores.append(sse / n)
        scores = np.array(scores)
        slack = 0.01 * scores.min() + 1e-12 * (np.ptp(y) ** 2 + 1e-300)
    else:
        mse_floor = max(1e-24 * np.ptp(y) ** 2, 1e-300)
        aic, aicc = [], []
        for o in orders:
            fits = fit_chunks(x, y, n_chunks, o)
            sse = sum(float(np.sum((y[f.start:f.stop] - f.predict(x[f.start:f.stop])) ** 2)) for f in fits)
            kp = sum(len(f.coeffs) for f in fits)
            a = n * np.log(max(sse / n, mse_floor)) + 2 * kp
            aic.append(a)
            aicc.append(a + 2 * kp * (kp + 1) / (n - kp - 1) if n - kp - 1 > 0 else np.inf)
        scores = np.array(aicc if np.isfinite(aicc).any() else aic)
        slack = 1e-9 * (1 + np.abs(scores[np.isfinite(scores)]).max())
    return int(np.flatnonzero(scores <= scores.min() + slack)[0])
