"""Pure-numpy piecewise polynomial fitting (no GUI imports)."""
import concurrent.futures
import itertools
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import Polynomial, legendre


def chunk_bounds(n_points, n_chunks, overlap_pct=0.0):
    """Half-open (start, stop) ranges of the array_split partition, optionally overlapped."""
    n_points, n_chunks = int(n_points), int(n_chunks)
    if n_chunks < 1 or n_chunks > n_points:
        raise ValueError(f"n_chunks must be in [1, {n_points}], got {n_chunks}")
    base, rem = divmod(n_points, n_chunks)
    bounds, start = [], 0
    for i in range(n_chunks):
        length = base + (1 if i < rem else 0)
        stop = start + length
        lo, hi = start, stop
        ov = int(length * overlap_pct / 100) if overlap_pct > 0 else 0
        if ov and n_chunks > 1:
            if i == 0:
                hi += ov
            elif i == n_chunks - 1:
                lo -= ov
            else:
                lo -= ov // 2
                hi += ov - ov // 2
        bounds.append((max(lo, 0), min(hi, n_points)))
        start = stop
    return bounds


def _fit_poly(xs, ys, order):
    order = min(int(order), len(xs) - 1)
    if order <= 0 or np.ptp(xs) == 0:
        return Polynomial([float(np.mean(ys))])
    return Polynomial.fit(xs, ys, order)


def _poly_coeffs(poly):
    """Power-basis coefficients in original x, highest degree first."""
    return poly.convert().coef[::-1].copy()


@dataclass
class ChunkFit:
    start: int
    stop: int
    x_min: float
    x_max: float
    coeffs: np.ndarray
    rmse: float
    poly: Polynomial = field(repr=False, default=None)
    base_start: int = 0
    base_stop: int = 0
    base_x_min: float = 0.0
    base_x_max: float = 0.0

    def predict(self, xq):
        return self.poly(np.asarray(xq, dtype=float))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def _make_chunk_fits(x, y, n_chunks, order, overlap_pct, fitter):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length")
    base = chunk_bounds(len(x), n_chunks, 0)
    ext = chunk_bounds(len(x), n_chunks, overlap_pct)
    fits = []
    for (bs, be), (s, e) in zip(base, ext):
        xs, ys = x[s:e], y[s:e]
        poly = fitter(xs, ys, order)
        fits.append(ChunkFit(
            start=s, stop=e, x_min=float(xs[0]), x_max=float(xs[-1]),
            coeffs=_poly_coeffs(poly), rmse=rmse(ys, poly(xs)), poly=poly,
            base_start=bs, base_stop=be,
            base_x_min=float(x[bs]), base_x_max=float(x[be - 1]),
        ))
    return fits


def fit_chunks(x, y, n_chunks, order, overlap_pct=0.0):
    """Least-squares polynomial fit of degree `order` on each (overlapped) chunk."""
    return _make_chunk_fits(x, y, n_chunks, order, overlap_pct, _fit_poly)


def _seams(fits, blend):
    """Per-seam blend interval [a_i, b_i]; a == b means a hard switch at that x."""
    a, b = [], []
    for i in range(len(fits) - 1):
        left, right = fits[i], fits[i + 1]
        knot = right.base_x_min
        lo, hi = right.x_min, left.x_max
        if blend:
            c_left = 0.5 * (left.base_x_min + left.base_x_max)
            c_right = 0.5 * (right.base_x_min + right.base_x_max)
            lo, hi = max(lo, c_left), min(hi, c_right)
        if not blend or not lo < hi:
            lo = hi = knot
        a.append(lo)
        b.append(hi)
    return np.asarray(a, dtype=float), np.asarray(b, dtype=float)


def evaluate_piecewise(fits, xq, blend=True):
    """Evaluate the piecewise model; with blend, cross-fade (smoothstep) in overlaps."""
    xq = np.asarray(xq, dtype=float)
    flat = np.atleast_1d(xq).ravel()
    a, b = _seams(fits, blend)
    k = np.searchsorted(a, flat, side="right")  # index of the chunk owning x
    out = np.empty_like(flat)
    for j, fit in enumerate(fits):
        sel = k == j
        if sel.any():
            out[sel] = fit.predict(flat[sel])
    for i in range(len(a)):
        if b[i] <= a[i]:
            continue
        sel = (k == i + 1) & (flat <= b[i])
        if sel.any():
            t = (flat[sel] - a[i]) / (b[i] - a[i])
            w = t * t * (3.0 - 2.0 * t)
            out[sel] = (1.0 - w) * fits[i].predict(flat[sel]) + w * fits[i + 1].predict(flat[sel])
    return out.reshape(xq.shape) if xq.ndim else out[0]


def _map_cell(x, y, order, n_chunks, overlap):
    fits = fit_chunks(x, y, int(n_chunks), int(order), float(overlap))
    return rmse(y, evaluate_piecewise(fits, x, blend=False))


def error_map(x, y, orders, chunk_counts, overlap_pcts, workers=None):
    """2-D un-normalized RMSE map over the two varying parameters."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    seqs = [list(orders), list(chunk_counts), list(overlap_pcts)]
    varying = [i for i, s in enumerate(seqs) if len(s) > 1]
    if len(varying) != 2 or any(len(s) == 0 for s in seqs):
        raise ValueError("exactly two of orders/chunk_counts/overlap_pcts must have length > 1")
    ax0, ax1 = (seqs[i] for i in varying)
    cells = list(itertools.product(range(len(ax0)), range(len(ax1))))

    def params(i, j):
        p = [s[0] for s in seqs]
        p[varying[0]], p[varying[1]] = ax0[i], ax1[j]
        return p

    out = np.empty((len(ax0), len(ax1)))
    if workers == 1:
        for i, j in cells:
            out[i, j] = _map_cell(x, y, *params(i, j))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_map_cell, x, y, *params(i, j)): (i, j) for i, j in cells}
            for fut in concurrent.futures.as_completed(futures):
                out[futures[fut]] = fut.result()
    return out


def extrapolate(x, y, fit_fraction, order):
    """Fit the first `fit_fraction` of points; return (coeffs, prediction over all x)."""
    if not 0 < fit_fraction <= 1:
        raise ValueError("fit_fraction must be in (0, 1]")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < order + 1:
        raise ValueError("not enough points for the requested order")
    n_fit = min(n, max(int(order) + 1, int(round(fit_fraction * n))))
    poly = _fit_poly(x[:n_fit], y[:n_fit], order)
    return _poly_coeffs(poly), poly(x)


def adaptive_chunks(x, y, order, tol, max_chunks):
    """Greedy contiguous partition where each degree-`order` chunk has RMSE <= tol.

    Chunks never shorter than 2*(order+1) (a shorter remainder is folded into the
    previous chunk). If more than `max_chunks` chunks are needed, the tail is merged
    into the last allowed chunk, which may then exceed tol.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length")
    if int(max_chunks) < 1:
        raise ValueError("max_chunks must be >= 1")
    if tol < 0:
        raise ValueError("tol must be >= 0")
    order, n = int(order), len(x)
    min_len = 2 * (order + 1)

    def ok(s, e):
        xs, ys = x[s:e], y[s:e]
        return rmse(ys, _fit_poly(xs, ys, order)(xs)) <= tol

    parts, s = [], 0
    while s < n:
        if n - s <= min_len or ok(s, n):
            parts.append((s, n))
            break
        lo, hi = min(s + min_len, n), n  # lo assumed fine, hi known bad
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ok(s, mid):
                lo = mid
            else:
                hi = mid
        e = lo
        if 0 < n - e < min_len:
            e = n - min_len if n - min_len >= s + min_len else n
        parts.append((s, e))
        s = e
    if len(parts) > max_chunks:
        parts = parts[:int(max_chunks) - 1] + [(parts[int(max_chunks) - 1][0], n)]
    return parts


def _robust_poly(xs, ys, order):
    """Tukey-bisquare IRLS polynomial fit (scale = 1.4826 * MAD of residuals)."""
    order = min(int(order), len(xs) - 1)
    if order <= 0 or np.ptp(xs) == 0:
        return Polynomial([float(np.median(ys))])
    c = 4.685
    poly = Polynomial.fit(xs, ys, order)
    for _ in range(30):
        r = ys - poly(xs)
        scale = 1.4826 * np.median(np.abs(r - np.median(r)))
        if scale == 0:
            break
        u = r / (c * scale)
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        if np.count_nonzero(w) <= order:
            break
        new = Polynomial.fit(xs, ys, order, w=np.sqrt(w))
        converged = np.allclose(new.coef, poly.coef, rtol=1e-10, atol=1e-12)
        poly = new
        if converged:
            break
    return poly


def robust_fit_chunks(x, y, n_chunks, order, overlap_pct=0.0):
    """Like fit_chunks, but each chunk uses a Tukey-bisquare IRLS fit."""
    return _make_chunk_fits(x, y, n_chunks, order, overlap_pct, _robust_poly)


def select_order(x, y, n_chunks, max_order, method="cv"):
    """Pick the order in [0, max_order] by 5-fold CV ("cv") or AIC ("aic")."""
    if method not in ("cv", "aic"):
        raise ValueError(f"unknown method {method!r}; use 'cv' or 'aic'")
    if int(max_order) < 0:
        raise ValueError("max_order must be >= 0")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, n_chunks = len(x), int(n_chunks)
    scores = []
    if method == "aic":
        for order in range(int(max_order) + 1):
            fits = fit_chunks(x, y, n_chunks, order)
            rss = float(np.sum((y - evaluate_piecewise(fits, x, blend=False)) ** 2))
            k_params = n_chunks * (order + 1)
            scores.append(n * np.log(rss / n + 1e-300) + 2 * k_params)
    else:
        k = 5
        if n < k:
            raise ValueError(f"need at least {k} points for cv, got {n}")
        idx = np.arange(n)
        for order in range(int(max_order) + 1):
            sse = 0.0
            for f in range(k):
                test = idx % k == f
                train = ~test
                if train.sum() < n_chunks:
                    raise ValueError("training folds have fewer points than n_chunks")
                fits = fit_chunks(x[train], y[train], n_chunks, order)
                pred = evaluate_piecewise(fits, x[test], blend=False)
                sse += float(np.sum((y[test] - pred) ** 2))
            scores.append(sse / n)
    return int(np.argmin(scores))


@dataclass
class SplineFit:
    knots: np.ndarray
    order: int
    continuity: int
    centers: np.ndarray = field(repr=False)
    half_widths: np.ndarray = field(repr=False)
    coefs: np.ndarray = field(repr=False)  # (n_pieces, order + 1) Legendre coefs in local t

    def piece_index(self, xq):
        return np.searchsorted(self.knots, xq, side="right")

    def _eval_piece(self, j, xq, m=0):
        c = legendre.legder(self.coefs[j], m) if m else self.coefs[j]
        t = (xq - self.centers[j]) / self.half_widths[j]
        return legendre.legval(t, c) / self.half_widths[j] ** m

    def derivative(self, xq, m=1):
        if m < 0:
            raise ValueError("m must be >= 0")
        xq = np.asarray(xq, dtype=float)
        flat = np.atleast_1d(xq).ravel()
        idx = self.piece_index(flat)
        out = np.empty_like(flat)
        for j in np.unique(idx):
            sel = idx == j
            out[sel] = self._eval_piece(j, flat[sel], m)
        return out.reshape(xq.shape) if xq.ndim else out[0]

    def predict(self, xq):
        return self.derivative(xq, 0)


def _leg_derivative_rows(t, deg, m):
    """Row of d^m/dt^m of each Legendre basis function at scalar t."""
    eye = np.eye(deg + 1)
    return np.array([legendre.legval(t, legendre.legder(eye[k], m)) if m else legendre.legval(t, eye[k])
                     for k in range(deg + 1)])


def fit_spline(x, y, n_chunks, order, continuity=2):
    """Jointly constrained least-squares piecewise polynomial (C^continuity at knots)."""
    order, continuity = int(order), int(continuity)
    if not -1 <= continuity < order:
        raise ValueError("continuity must satisfy -1 <= continuity < order")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    base = chunk_bounds(len(x), n_chunks, 0)
    knots = np.array([x[s] for s, _ in base[1:]], dtype=float)
    edges = np.concatenate(([x[0]], knots, [x[-1]]))
    centers = 0.5 * (edges[:-1] + edges[1:])
    half = 0.5 * (edges[1:] - edges[:-1])
    half[half <= 0] = 1.0
    p = order + 1
    n_pieces = len(base)

    design = np.zeros((len(x), n_pieces * p))
    for j, (s, e) in enumerate(base):
        t = (x[s:e] - centers[j]) / half[j]
        design[s:e, j * p:(j + 1) * p] = legendre.legvander(t, order)

    rows = []
    for i, k in enumerate(knots):
        h_ref = 0.5 * (half[i] + half[i + 1])
        tl = (k - centers[i]) / half[i]
        tr = (k - centers[i + 1]) / half[i + 1]
        for m in range(continuity + 1):
            row = np.zeros(n_pieces * p)
            row[i * p:(i + 1) * p] = _leg_derivative_rows(tl, order, m) * (h_ref / half[i]) ** m
            row[(i + 1) * p:(i + 2) * p] = -_leg_derivative_rows(tr, order, m) * (h_ref / half[i + 1]) ** m
            rows.append(row)

    if rows:
        cons = np.array(rows)
        q, _ = np.linalg.qr(cons.T, mode="complete")
        null = q[:, cons.shape[0]:]
        z = np.linalg.lstsq(design @ null, y, rcond=None)[0]
        sol = null @ z
    else:
        sol = np.linalg.lstsq(design, y, rcond=None)[0]
    return SplineFit(knots=knots, order=order, continuity=continuity, centers=centers,
                     half_widths=half, coefs=sol.reshape(n_pieces, p))
