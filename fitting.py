"""Pure curve-fitting algorithms: chunked polynomial fits, blending, error maps.

Importing this module must not pull in any GUI or plotting library.
"""

from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Polynomial


@dataclass(eq=False)
class ChunkFit:
    start: int
    stop: int
    x_min: float
    x_max: float
    coeffs: np.ndarray
    rmse: float
    poly: Polynomial
    base_lo: float | None = None
    base_hi: float | None = None

    def __post_init__(self) -> None:
        if self.base_lo is None:
            self.base_lo = self.x_min
        if self.base_hi is None:
            self.base_hi = self.x_max

    def predict(self, xq) -> np.ndarray:
        return self.poly(np.asarray(xq, dtype=float))


def chunk_bounds(
    n_points: int, n_chunks: int, overlap_pct: float = 0.0
) -> list[tuple[int, int]]:
    if n_chunks < 1 or n_chunks > n_points:
        raise ValueError(f"need 1 <= n_chunks <= n_points, got {n_chunks} for {n_points}")
    if overlap_pct < 0:
        raise ValueError(f"overlap_pct must be >= 0, got {overlap_pct}")

    last = n_chunks - 1
    bounds = []
    for i, part in enumerate(np.array_split(np.arange(n_points), n_chunks)):
        start, stop = int(part[0]), int(part[-1]) + 1
        ov = int(len(part) * overlap_pct / 100)
        if i == 0:
            stop += ov
        if i == last:
            start -= ov
        if 0 < i < last:
            start -= ov // 2
            stop += ov - ov // 2
        bounds.append((max(start, 0), min(stop, n_points)))
    return bounds


def _fit_poly(x: np.ndarray, y: np.ndarray, order: int):
    """Fit a polynomial of degree min(order, len-1) in a scaled domain.

    Returns (poly, coeffs in original x, highest degree first and padded to
    order + 1, rmse on the given points).
    """
    deg = min(order, len(x) - 1)
    lo, hi = float(x.min()), float(x.max())
    domain = [lo, hi] if hi > lo else [lo - 1.0, hi + 1.0]
    poly = Polynomial.fit(x, y, deg, domain=domain)

    coeffs = poly.convert().coef[::-1]
    coeffs = np.concatenate([np.zeros(order + 1 - len(coeffs)), coeffs])

    return poly, coeffs, rmse(y, poly(x))


def fit_chunks(
    x, y, n_chunks: int, order: int, overlap_pct: float = 0.0
) -> list[ChunkFit]:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    if np.any(np.diff(x) < 0):
        raise ValueError("x must be sorted in ascending order")
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")

    base = chunk_bounds(len(x), n_chunks, 0.0)
    fits = []
    for (start, stop), (base_start, base_stop) in zip(
        chunk_bounds(len(x), n_chunks, overlap_pct), base
    ):
        xs, ys = x[start:stop], y[start:stop]
        poly, coeffs, err = _fit_poly(xs, ys, order)
        fits.append(
            ChunkFit(
                start=start,
                stop=stop,
                x_min=float(xs.min()),
                x_max=float(xs.max()),
                coeffs=coeffs,
                rmse=err,
                poly=poly,
                base_lo=float(x[base_start]),
                base_hi=float(x[base_stop - 1]),
            )
        )
    return fits


def evaluate_piecewise(fits: list[ChunkFit], xq, blend: bool = True) -> np.ndarray:
    if not fits:
        raise ValueError("no chunk fits to evaluate")
    xq_arr = np.asarray(xq, dtype=float)
    flat = xq_arr.ravel()
    out = np.empty_like(flat)

    # Each point belongs to the chunk whose base range contains it; split
    # points sit midway between adjacent base ranges.
    lo = np.array([f.base_lo for f in fits])
    hi = np.array([f.base_hi for f in fits])
    edges = (hi[:-1] + lo[1:]) / 2
    idx = np.searchsorted(edges, flat, side="right")
    for i, fit in enumerate(fits):
        mask = idx == i
        if mask.any():
            out[mask] = fit.predict(flat[mask])

    if blend:
        # Cross-fade across the overlap of adjacent fitted slices. The blend
        # weight is 0 at the start of the overlap and 1 at its end, so the
        # curve stays continuous at both seams.
        for left, right in zip(fits[:-1], fits[1:]):
            a, b = right.x_min, left.x_max
            if b <= a:
                continue
            mask = (flat >= a) & (flat <= b)
            if mask.any():
                w = (flat[mask] - a) / (b - a)
                pts = flat[mask]
                out[mask] = (1 - w) * left.predict(pts) + w * right.predict(pts)

    return out.reshape(xq_arr.shape)


def rmse(y_true, y_pred) -> float:
    d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(d * d)))


def r2(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def error_map(
    x,
    y,
    orders,
    chunk_counts,
    overlap_pcts,
    workers: int | None = None,
) -> np.ndarray:
    values = [list(orders), list(chunk_counts), list(overlap_pcts)]
    varying = [i for i, v in enumerate(values) if len(v) > 1]
    fixed = [i for i, v in enumerate(values) if len(v) == 1]
    if len(varying) != 2 or len(fixed) != 1:
        raise ValueError("exactly two of orders, chunk_counts, overlap_pcts must have length > 1")
    ax0, ax1 = varying
    fixed_i = fixed[0]
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()

    def cell(i: int, j: int) -> float:
        args = [None, None, None]
        args[fixed_i] = values[fixed_i][0]
        args[ax0] = values[ax0][i]
        args[ax1] = values[ax1][j]
        order, n_chunks, ov = args
        fits = fit_chunks(x_arr, y_arr, int(n_chunks), int(order), float(ov))
        return rmse(y_arr, evaluate_piecewise(fits, x_arr, blend=False))

    grid = [(i, j) for i in range(len(values[ax0])) for j in range(len(values[ax1]))]
    if workers == 1:
        results = [cell(i, j) for i, j in grid]
    else:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda ij: cell(*ij), grid))

    return np.array(results).reshape(len(values[ax0]), len(values[ax1]))


def extrapolate(
    x, y, fit_fraction: float, order: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < fit_fraction <= 1:
        raise ValueError(f"fit_fraction must be in (0, 1], got {fit_fraction}")
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(x)
    m = max(order + 1, int(n * fit_fraction))
    if m > n:
        raise ValueError(f"need at least {order + 1} points, have {n}")
    poly, coeffs, _ = _fit_poly(x[:m], y[:m], order)
    return coeffs, poly(x)
