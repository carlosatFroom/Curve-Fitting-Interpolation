import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import fitting
from fitting import (
    ChunkFit,
    SplineFit,
    adaptive_chunks,
    chunk_bounds,
    error_map,
    evaluate_piecewise,
    extrapolate,
    fit_chunks,
    fit_spline,
    r2,
    rmse,
    robust_fit_chunks,
    select_order,
)

REPO = Path(__file__).resolve().parent.parent


def test_no_gui_imports():
    assert "PyQt5" not in sys.modules
    assert "pyqtgraph" not in sys.modules
    assert "matplotlib" not in sys.modules


# --------------------------------------------------------------------------- #
# chunk_bounds
# --------------------------------------------------------------------------- #
def test_chunk_bounds_partition_no_overlap():
    n = 103
    bounds = chunk_bounds(n, 7, 0.0)
    base = np.array_split(np.arange(n), 7)
    assert len(bounds) == 7
    # contiguous, covers everything, matches array_split
    assert bounds[0][0] == 0
    assert bounds[-1][1] == n
    for (s, e), block in zip(bounds, base):
        assert s == block[0]
        assert e == block[-1] + 1
    for i in range(len(bounds) - 1):
        assert bounds[i][1] == bounds[i + 1][0]


def test_chunk_bounds_validation():
    with pytest.raises(ValueError):
        chunk_bounds(10, 0)
    with pytest.raises(ValueError):
        chunk_bounds(10, 11)
    # boundary ok
    chunk_bounds(10, 10)
    chunk_bounds(10, 1)


def test_chunk_bounds_overlap_rule():
    n = 100
    nc = 4
    ov_pct = 20.0
    bounds = chunk_bounds(n, nc, ov_pct)
    base = chunk_bounds(n, nc, 0.0)
    for i, ((s, e), (bs, be)) in enumerate(zip(bounds, base)):
        base_len = be - bs
        ov = int(base_len * ov_pct / 100.0)
        if i == 0:
            assert s == bs
            assert e == min(n, be + ov)
        elif i == nc - 1:
            assert s == max(0, bs - ov)
            assert e == be
        else:
            assert s == max(0, bs - ov // 2)
            assert e == min(n, be + (ov - ov // 2))


def test_chunk_bounds_clip():
    bounds = chunk_bounds(50, 3, 500.0)  # huge overlap -> clip to [0, n]
    for s, e in bounds:
        assert 0 <= s <= e <= 50


# --------------------------------------------------------------------------- #
# ChunkFit / fit_chunks
# --------------------------------------------------------------------------- #
def test_fit_chunks_recovers_polynomial():
    x = np.linspace(-3, 3, 200)
    y = 2 * x ** 2 - x + 1
    fits = fit_chunks(x, y, 4, 2)
    assert all(isinstance(f, ChunkFit) for f in fits)
    pred = evaluate_piecewise(fits, x, blend=False)
    assert rmse(y, pred) < 1e-6


def test_chunkfit_predict_high_order_stable():
    # unscaled x with high order is ill-conditioned for raw polyfit; predict must
    # stay accurate because it fits in a scaled domain.
    x = np.linspace(1000, 1010, 300)
    y = np.sin((x - 1000) * 0.9) + 0.01 * (x - 1000) ** 2
    fits = fit_chunks(x, y, 1, 9)
    pred = fits[0].predict(x)
    assert rmse(y, pred) < 1e-2
    assert np.all(np.isfinite(fits[0].coeffs))


def test_fit_chunks_order_reduction():
    x = np.linspace(0, 1, 5)
    y = x.copy()
    # 5 chunks on 5 points -> 1 point each; order 3 must reduce, not crash
    fits = fit_chunks(x, y, 5, 3)
    assert len(fits) == 5
    for f in fits:
        assert np.all(np.isfinite(f.coeffs))


def test_coeffs_convention_highest_first():
    x = np.linspace(-2, 2, 100)
    y = 3 * x ** 2 + 2 * x + 1
    fits = fit_chunks(x, y, 1, 2)
    c = fits[0].coeffs
    assert c[0] == pytest.approx(3.0, abs=1e-6)
    assert c[1] == pytest.approx(2.0, abs=1e-6)
    assert c[2] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# evaluate_piecewise / blending
# --------------------------------------------------------------------------- #
def test_blend_is_continuous_at_seams():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 500)
    y = np.sin(x) + 0.05 * rng.standard_normal(500)
    fits = fit_chunks(x, y, 5, 3, overlap_pct=20.0)
    xq = np.linspace(x.min(), x.max(), 4000)
    blended = evaluate_piecewise(fits, xq, blend=True)
    hard = evaluate_piecewise(fits, xq, blend=False)
    # blended curve has smaller maximum jump between consecutive samples
    assert np.max(np.abs(np.diff(blended))) <= np.max(np.abs(np.diff(hard))) + 1e-9
    assert np.all(np.isfinite(blended))


def test_blend_false_uses_base_chunk():
    x = np.linspace(0, 10, 100)
    y = x ** 2
    fits = fit_chunks(x, y, 4, 2, overlap_pct=10.0)
    pred = evaluate_piecewise(fits, x, blend=False)
    assert rmse(y, pred) < 1e-6


def test_evaluate_scalar():
    x = np.linspace(0, 5, 50)
    y = 2 * x
    fits = fit_chunks(x, y, 2, 1)
    val = evaluate_piecewise(fits, 2.5, blend=False)
    assert np.isscalar(val) or np.ndim(val) == 0
    assert val == pytest.approx(5.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_rmse_and_r2():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert rmse(y, y) == 0.0
    assert r2(y, y) == pytest.approx(1.0)
    yp = y + 1.0
    assert rmse(y, yp) == pytest.approx(1.0)
    # constant truth -> r2 defined as 0
    assert r2(np.ones(4), np.ones(4) + 0.5) == 0.0


# --------------------------------------------------------------------------- #
# error_map
# --------------------------------------------------------------------------- #
def test_error_map_shape_and_axes():
    x = np.linspace(0, 10, 300)
    y = np.sin(x)
    orders = [1, 2, 3]
    chunks = [1, 2, 3, 4]
    emap = error_map(x, y, orders, chunks, [0.0])
    assert emap.shape == (3, 4)
    # cross-check a single cell against a direct computation
    fits = fit_chunks(x, y, chunks[2], orders[1])
    expected = rmse(y, evaluate_piecewise(fits, x, blend=False))
    assert emap[1, 2] == pytest.approx(expected, rel=1e-9)


def test_error_map_requires_two_varying():
    x = np.linspace(0, 1, 50)
    y = x
    with pytest.raises(ValueError):
        error_map(x, y, [1, 2], [3], [0.0])  # only one varying


def test_error_map_workers_consistent():
    x = np.linspace(0, 10, 200)
    y = np.cos(x)
    a = error_map(x, y, [1, 2, 3], [2, 3], [0.0], workers=1)
    b = error_map(x, y, [1, 2, 3], [2, 3], [0.0], workers=4)
    assert np.allclose(a, b)


def test_error_map_fast():
    import time

    rng = np.random.default_rng(1)
    x = np.linspace(0, 20, 1000)
    y = np.sin(x) + 0.1 * rng.standard_normal(1000)
    t0 = time.perf_counter()
    emap = error_map(x, y, list(range(1, 11)), list(range(1, 11)), [0.0])
    dt = time.perf_counter() - t0
    assert emap.shape == (10, 10)
    assert dt < 2.0


# --------------------------------------------------------------------------- #
# extrapolate
# --------------------------------------------------------------------------- #
def test_extrapolate():
    x = np.linspace(0, 10, 500)
    y = 1.5 * x + 0.5
    coeffs, y_pred = extrapolate(x, y, 0.3, 1)
    assert y_pred.shape == x.shape
    assert coeffs[0] == pytest.approx(1.5, abs=1e-6)
    assert rmse(y, y_pred) < 1e-6  # linear extrapolation is exact here


def test_extrapolate_validation():
    x = np.linspace(0, 1, 20)
    y = x
    with pytest.raises(ValueError):
        extrapolate(x, y, 0.0, 1)
    with pytest.raises(ValueError):
        extrapolate(x, y, 1.5, 1)


# --------------------------------------------------------------------------- #
# Phase 2: spline
# --------------------------------------------------------------------------- #
def test_fit_spline_continuity():
    x = np.linspace(0, 10, 400)
    y = np.sin(x)
    sp = fit_spline(x, y, 6, 4, continuity=2)
    assert isinstance(sp, SplineFit)
    for X in sp.knots:
        left = X - 1e-6
        right = X + 1e-6
        # value, 1st and 2nd derivatives continuous across the knot
        assert sp.predict(left) == pytest.approx(sp.predict(right), abs=1e-3)
        assert sp.derivative(left, 1) == pytest.approx(sp.derivative(right, 1), abs=1e-2)
        assert sp.derivative(left, 2) == pytest.approx(sp.derivative(right, 2), abs=1e-1)
    assert rmse(y, sp.predict(x)) < 1e-2


def test_fit_spline_validation():
    x = np.linspace(0, 1, 50)
    y = x
    with pytest.raises(ValueError):
        fit_spline(x, y, 3, 2, continuity=2)  # continuity must be < order
    with pytest.raises(ValueError):
        fit_spline(x, y, 3, 3, continuity=-2)


def test_fit_spline_stable_high_order_many_chunks():
    x = np.linspace(0, 50, 1000)
    y = np.sin(x) + 0.001 * x ** 2
    sp = fit_spline(x, y, 50, 10, continuity=2)
    pred = sp.predict(x)
    assert np.all(np.isfinite(pred))
    assert rmse(y, pred) < 1e-1


def test_spline_is_joint_not_patch():
    # a constrained joint fit should beat independent-per-chunk at matching a
    # smooth signal in continuity while staying accurate
    x = np.linspace(0, 10, 300)
    y = np.sin(x)
    sp = fit_spline(x, y, 5, 3, continuity=2)
    # derivative continuity should be far tighter than naive piecewise fit
    jumps = [abs(sp.derivative(k - 1e-6, 2) - sp.derivative(k + 1e-6, 2)) for k in sp.knots]
    assert max(jumps) < 1e-1


# --------------------------------------------------------------------------- #
# Phase 2: adaptive chunking
# --------------------------------------------------------------------------- #
def test_adaptive_chunks_partition_and_tol():
    x = np.linspace(0, 20, 600)
    y = np.sin(x)
    parts = adaptive_chunks(x, y, order=3, tol=0.05, max_chunks=50)
    # contiguous partition of [0, n)
    assert parts[0][0] == 0
    assert parts[-1][1] == len(x)
    for a, b in zip(parts, parts[1:]):
        assert a[1] == b[0]
    assert len(parts) <= 50
    # every sufficiently large chunk meets tol
    for s, e in parts:
        if (e - s) >= 2 * (3 + 1):
            poly = fitting._fit_poly(x[s:e], y[s:e], 3)
            assert rmse(y[s:e], poly(x[s:e])) <= 0.05 + 1e-9


def test_adaptive_chunks_respects_max():
    x = np.linspace(0, 100, 1000)
    y = np.sin(x)  # wiggly -> needs many chunks
    parts = adaptive_chunks(x, y, order=1, tol=1e-6, max_chunks=5)
    assert len(parts) <= 5
    assert parts[0][0] == 0
    assert parts[-1][1] == len(x)


# --------------------------------------------------------------------------- #
# Phase 2: robust fit
# --------------------------------------------------------------------------- #
def test_robust_fit_tracks_signal_with_outliers():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 10, 500)
    clean = 0.5 * x + np.sin(x)
    y = clean.copy()
    # 10% gross outliers
    idx = rng.choice(len(x), size=50, replace=False)
    y[idx] += rng.choice([-1, 1], size=50) * 20.0
    robust = robust_fit_chunks(x, y, 5, 3)
    ordinary = fit_chunks(x, y, 5, 3)
    r_pred = evaluate_piecewise(robust, x, blend=False)
    o_pred = evaluate_piecewise(ordinary, x, blend=False)
    # robust tracks the clean signal much better
    assert rmse(clean, r_pred) < rmse(clean, o_pred)
    assert rmse(clean, r_pred) < 1.0


# --------------------------------------------------------------------------- #
# Phase 2: order selection
# --------------------------------------------------------------------------- #
def test_select_order_cv_recovers_degree():
    rng = np.random.default_rng(4)
    x = np.linspace(-3, 3, 400)
    y = 1 + 2 * x - 0.5 * x ** 2 + 0.05 * rng.standard_normal(400)
    chosen = select_order(x, y, n_chunks=1, max_order=8, method="cv")
    assert 2 <= chosen <= 4  # should not grossly overfit


def test_select_order_aic():
    rng = np.random.default_rng(5)
    x = np.linspace(-3, 3, 400)
    y = x ** 3 - x + 0.05 * rng.standard_normal(400)
    chosen = select_order(x, y, n_chunks=1, max_order=8, method="aic")
    assert chosen >= 3


def test_select_order_bad_method():
    x = np.linspace(0, 1, 50)
    y = x
    with pytest.raises(ValueError):
        select_order(x, y, 1, 4, method="nope")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli(tmp_path):
    csv = tmp_path / "data.csv"
    x = np.linspace(0, 10, 300)
    y = np.sin(x)
    np.savetxt(csv, np.column_stack([x, y]), delimiter=",")
    out = tmp_path / "out.json"
    res = subprocess.run(
        [sys.executable, str(REPO / "fitting_cli.py"), str(csv),
         "--orders", "1-5", "--chunks", "1-5", "--json", str(out)],
        cwd=str(REPO), capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    data = json.loads(out.read_text())
    assert data["orders"] == [1, 2, 3, 4, 5]
    assert data["chunks"] == [1, 2, 3, 4, 5]
    assert len(data["error_map"]) == 5
    assert len(data["error_map"][0]) == 5
    assert "seconds" in data
    assert set(data["best"]) == {"order", "chunks", "rmse"}
    # best is the minimum cell
    emap = np.array(data["error_map"])
    i, j = np.unravel_index(np.argmin(emap), emap.shape)
    assert data["best"]["order"] == data["orders"][i]
    assert data["best"]["chunks"] == data["chunks"][j]
