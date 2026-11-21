"""Tests for the Phase-1 foundation/core in fitting.py (lead-owned)."""
import subprocess
import sys

import numpy as np
import pytest

from fitting import (
    chunk_bounds,
    ChunkFit,
    fit_chunks,
    evaluate_piecewise,
    extrapolate,
    rmse,
    r2,
)


# ------------------------------------------------------- import is GUI-free
def test_import_fitting_pulls_no_gui():
    """`import fitting` must not drag in PyQt5/pyqtgraph/matplotlib, so it works
    on a machine without them installed."""
    code = (
        "import sys, fitting; "
        "bad=[m for m in ('PyQt5','pyqtgraph','matplotlib') if m in sys.modules]; "
        "print(bad); assert not bad, bad"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------- chunk_bounds
def test_chunk_bounds_partition_tiles_exactly():
    n = 100
    b = chunk_bounds(n, 7, 0.0)
    assert b[0][0] == 0 and b[-1][1] == n
    # contiguous, no gaps, matches np.array_split sizes
    sizes = [len(c) for c in np.array_split(np.arange(n), 7)]
    assert [e - s for s, e in b] == sizes
    for (s0, e0), (s1, e1) in zip(b, b[1:]):
        assert e0 == s1


def test_chunk_bounds_overlap_math():
    # 100 pts, 4 chunks (base 25 each), 20% -> ov=5
    b = chunk_bounds(100, 4, 20)
    assert b == [(0, 30), (23, 53), (48, 78), (70, 100)]


def test_chunk_bounds_clip_and_single():
    assert chunk_bounds(10, 1, 50) == [(0, 10)]
    for s, e in chunk_bounds(50, 5, 80):
        assert 0 <= s <= e <= 50


def test_chunk_bounds_validation():
    with pytest.raises(ValueError):
        chunk_bounds(5, 0)
    with pytest.raises(ValueError):
        chunk_bounds(5, 6)


# ------------------------------------------------------------------- rmse / r2
def test_rmse_r2_basic():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert rmse(y, y) == 0.0
    assert r2(y, y) == 1.0
    assert rmse(y, y + 2.0) == pytest.approx(2.0)
    # r2 of mean predictor is 0
    assert r2(y, np.full_like(y, y.mean())) == pytest.approx(0.0)


# ------------------------------------------------------------------ fit_chunks
def test_fit_chunks_fits_well_and_stable_highorder():
    x = np.linspace(0, 10, 500)
    y = np.sin(x) + 0.05 * x
    fits = fit_chunks(x, y, 5, 8, 0.0)
    assert len(fits) == 5
    yp = evaluate_piecewise(fits, x, blend=False)
    assert rmse(y, yp) < 1e-2
    # coeffs are highest-degree-first and predict is stable (no overflow)
    for f in fits:
        assert np.all(np.isfinite(f.predict(x)))
        assert isinstance(f, ChunkFit)


def test_fit_chunks_order_reduction_small_chunk():
    x = np.linspace(0, 1, 6)
    y = x ** 2
    fits = fit_chunks(x, y, 6, 8)  # 1 point per chunk -> order collapses to 0
    for f in fits:
        assert len(f.coeffs) == 1


def test_fit_chunks_rmse_is_real_rmse():
    x = np.linspace(0, 10, 200)
    y = 3 * x + 1
    fits = fit_chunks(x, y, 1, 1)
    # perfect linear fit
    assert fits[0].rmse == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------- evaluate_piecewise
def test_evaluate_piecewise_blend_no_seam():
    x = np.linspace(0, 10, 800)
    y = np.sin(x)
    fits = fit_chunks(x, y, 6, 3, overlap_pct=25)
    yb = evaluate_piecewise(fits, x, blend=True)
    yf = evaluate_piecewise(fits, x, blend=False)
    # blended fit is smooth: consecutive differences never spike far beyond the
    # underlying signal's own step
    signal_step = np.max(np.abs(np.diff(y)))
    assert np.max(np.abs(np.diff(yb))) < 5 * signal_step
    # both reconstruct the signal well
    assert rmse(y, yb) < 1e-2
    assert rmse(y, yf) < 1e-2


def test_evaluate_piecewise_single_chunk():
    x = np.linspace(0, 5, 100)
    y = x ** 2
    fits = fit_chunks(x, y, 1, 2)
    yp = evaluate_piecewise(fits, x)
    assert rmse(y, yp) == pytest.approx(0.0, abs=1e-6)


# -------------------------------------------------------------------- extrapolate
def test_extrapolate_shape_and_fit():
    x = np.linspace(0, 10, 300)
    y = 2 * x ** 2 - 3 * x + 1
    coeffs, ypred = extrapolate(x, y, 0.5, 2)
    assert ypred.shape == x.shape
    assert len(coeffs) == 3
    # a quadratic fit on the first half extrapolates the whole quadratic exactly
    assert rmse(y, ypred) < 1e-6


def test_extrapolate_min_points_and_validation():
    x = np.linspace(0, 1, 50)
    y = np.cos(x)
    # tiny fraction still uses at least order+1 points
    coeffs, ypred = extrapolate(x, y, 0.01, 3)
    assert len(coeffs) == 4
    assert np.all(np.isfinite(ypred))
    with pytest.raises(ValueError):
        extrapolate(x, y, 0.0, 2)
    with pytest.raises(ValueError):
        extrapolate(x, y, 1.5, 2)
