import time

import numpy as np
import pytest

from fitting import (
    chunk_bounds,
    error_map,
    evaluate_piecewise,
    extrapolate,
    fit_chunks,
    r2,
    rmse,
)


def test_chunk_bounds_unequal_chunks_no_overlap():
    assert chunk_bounds(10, 3) == [(0, 4), (4, 7), (7, 10)]


def test_chunk_bounds_overlap():
    assert chunk_bounds(10, 3, overlap_pct=50) == [(0, 6), (4, 8), (6, 10)]


def test_chunk_bounds_clipped_to_range():
    bounds = chunk_bounds(10, 2, overlap_pct=100)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 10
    assert all(0 <= s < e <= 10 for s, e in bounds)


@pytest.mark.parametrize("n_chunks", [0, -1, 11])
def test_chunk_bounds_invalid_chunks_raises(n_chunks):
    with pytest.raises(ValueError):
        chunk_bounds(10, n_chunks)


def test_coeffs_match_polyfit_low_order():
    x = np.linspace(0, 10, 50)
    y = 0.5 * x**3 - 2 * x**2 + x - 4
    (fit,) = fit_chunks(x, y, n_chunks=1, order=3)
    expected = np.polyfit(x, y, 3)
    assert np.allclose(fit.coeffs, expected, atol=1e-6)
    assert np.allclose(fit.predict(x), np.polyval(expected, x), atol=1e-6)


def test_high_order_stability():
    x = np.linspace(1000.0, 1010.0, 400)
    y = 1e-3 * (x - 1005.0) ** 10 + 3 * (x - 1005.0) ** 3 - 2
    (fit,) = fit_chunks(x, y, n_chunks=1, order=10)
    assert np.max(np.abs(fit.predict(x) - y)) < 1e-6


def test_small_chunk_order_reduction():
    x = np.arange(10, dtype=float)
    y = x**2
    fits = fit_chunks(x, y, n_chunks=5, order=5)
    assert len(fits) == 5
    for fit in fits:
        assert len(fit.coeffs) == 2  # 2 points per chunk -> linear
        assert fit.rmse < 1e-9


def test_blend_is_continuous_across_seams():
    x = np.linspace(0, 2 * np.pi, 2000)
    y = np.sin(x)
    fits = fit_chunks(x, y, n_chunks=4, order=5, overlap_pct=20)
    grid = np.linspace(0, 2 * np.pi, 4001)
    out = evaluate_piecewise(fits, grid, blend=True)
    step = grid[1] - grid[0]
    assert np.max(np.abs(np.diff(out))) < 3 * step
    assert np.max(np.abs(out - np.sin(grid))) < 1e-3


def test_no_overlap_blend_equals_plain():
    x = np.linspace(0, 1, 100)
    y = np.cos(3 * x)
    fits = fit_chunks(x, y, n_chunks=4, order=4)
    assert np.allclose(evaluate_piecewise(fits, x, blend=True), evaluate_piecewise(fits, x, blend=False))


def test_rmse_and_r2():
    assert rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, y) == pytest.approx(1.0)
    assert r2(y, np.full(4, y.mean())) == pytest.approx(0.0)


def test_error_map_shape_and_axis_order():
    x = np.linspace(0, 1, 200)
    y = x**2
    m = error_map(x, y, orders=[1, 2, 3], chunk_counts=[1, 2], overlap_pcts=[0])
    assert m.shape == (3, 2)
    m = error_map(x, y, orders=[2], chunk_counts=[1, 2, 4], overlap_pcts=[0, 10])
    assert m.shape == (3, 2)
    m = error_map(x, y, orders=[1, 2], chunk_counts=[2], overlap_pcts=[0, 10, 20])
    assert m.shape == (2, 3)
    m = error_map(x, y, orders=[1, 2], chunk_counts=[1, 2, 4], overlap_pcts=[0])
    assert m.shape == (2, 3)
    assert m[1, 0] == pytest.approx(0.0, abs=1e-12)  # order 2 fits x^2 exactly with 1 chunk


def test_error_map_unnormalized_rmse_value():
    x = np.linspace(0, 1, 100)
    y = np.sin(4 * x)
    m = error_map(x, y, orders=[3, 4], chunk_counts=[1, 2], overlap_pcts=[0], workers=1)
    fits = fit_chunks(x, y, 2, 3)
    expected = rmse(y, evaluate_piecewise(fits, x, blend=False))
    assert m[0, 1] == pytest.approx(expected)


def test_error_map_serial_and_threaded_agree():
    x = np.linspace(0, 1, 300)
    y = np.exp(x)
    serial = error_map(x, y, [1, 2, 3], [1, 2, 3], [0], workers=1)
    threaded = error_map(x, y, [1, 2, 3], [1, 2, 3], [0], workers=4)
    assert np.allclose(serial, threaded)


def test_error_map_requires_exactly_two_varying():
    x = np.linspace(0, 1, 50)
    with pytest.raises(ValueError):
        error_map(x, x, [1, 2], [1, 2], [0, 10])
    with pytest.raises(ValueError):
        error_map(x, x, [1], [1], [0])


def test_error_map_timing():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 1000)
    y = np.sin(x) + 0.1 * rng.standard_normal(x.size)
    start = time.perf_counter()
    m = error_map(x, y, orders=list(range(1, 11)), chunk_counts=list(range(1, 11)), overlap_pcts=[0])
    elapsed = time.perf_counter() - start
    assert m.shape == (10, 10)
    assert elapsed < 2.0


def test_extrapolate_exact_line():
    x = np.linspace(0, 10, 101)
    y = 2 * x + 1
    coeffs, y_pred = extrapolate(x, y, fit_fraction=0.5, order=1)
    assert np.allclose(coeffs, [2.0, 1.0])
    assert y_pred.shape == x.shape
    assert np.allclose(y_pred, y)


def test_extrapolate_invalid_fraction():
    x = np.linspace(0, 1, 10)
    with pytest.raises(ValueError):
        extrapolate(x, x, fit_fraction=0.0, order=1)
    with pytest.raises(ValueError):
        extrapolate(x, x, fit_fraction=1.5, order=1)
