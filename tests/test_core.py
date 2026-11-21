import time

import numpy as np
import pytest

import fitting
from fitting import (
    chunk_bounds,
    error_map,
    evaluate_piecewise,
    extrapolate,
    fit_chunks,
    r2,
    rmse,
)


def test_chunk_bounds_equal_partition_no_overlap():
    assert chunk_bounds(10, 2) == [(0, 5), (5, 10)]


def test_chunk_bounds_unequal_partition_no_overlap():
    assert chunk_bounds(10, 3) == [(0, 4), (4, 7), (7, 10)]


def test_chunk_bounds_overlap_formula():
    # base chunks (0,4),(4,7),(7,10): base lens 4,3,3; overlap 50%
    # ov = 2, 1, 1. first: stop+2; middle: start-0, stop+1; last: start-1
    assert chunk_bounds(10, 3, 50) == [(0, 6), (4, 8), (6, 10)]


def test_chunk_bounds_middle_overlap_split_and_clipping():
    # 5 chunks of 10 points, 40% -> ov=4 for each base chunk of len 2
    bounds = chunk_bounds(10, 5, 100)
    assert bounds[0] == (0, 4)  # stop += 2
    assert bounds[1] == (1, 5)  # start -= 1, stop += 1
    assert bounds[-1] == (6, 10)  # start -= 2
    assert all(0 <= a < b <= 10 for a, b in bounds)


@pytest.mark.parametrize("n_chunks", [0, 11])
def test_chunk_bounds_rejects_bad_chunk_count(n_chunks):
    with pytest.raises(ValueError):
        chunk_bounds(10, n_chunks)


def test_chunk_fit_matches_np_polyfit_for_low_order():
    x = np.linspace(0, 10, 50)
    y = 2 * x**2 - 3 * x + 1
    fit = fit_chunks(x, y, 1, 2)[0]
    np.testing.assert_allclose(fit.coeffs, [2, -3, 1], atol=1e-8)
    np.testing.assert_allclose(fit.coeffs, np.polyfit(x, y, 2), atol=1e-8)
    assert fit.rmse < 1e-10
    np.testing.assert_allclose(
        fit.predict([0.5, 7.0]), [2 * 0.25 - 1.5 + 1, 98 - 21 + 1], atol=1e-9
    )


def test_high_order_stable_on_offset_domain():
    x = np.linspace(1000.0, 1010.0, 400)
    y = np.sin((x - 1000.0) / 3.0)
    fit = fit_chunks(x, y, 1, 10)[0]
    assert fit.rmse < 1e-6
    assert np.all(np.isfinite(fit.predict(x)))


def test_chunk_too_small_reduces_order():
    x = np.arange(12, dtype=float)
    y = x**2
    fits = fit_chunks(x, y, 4, 5)  # chunks of 3 points, order 5 requested
    for fit in fits:
        assert len(fit.coeffs) == 3  # degree 2 = len - 1
        assert fit.rmse < 1e-9


def test_fit_chunks_stores_fitted_and_base_ranges():
    x = np.arange(10, dtype=float)
    fits = fit_chunks(x, x, 3, 1, overlap_pct=50)
    assert [(f.start, f.stop) for f in fits] == [(0, 6), (4, 8), (6, 10)]
    assert [(f.base_start, f.base_stop) for f in fits] == [(0, 4), (4, 7), (7, 10)]
    assert fits[1].x_min == 4.0 and fits[1].x_max == 7.0
    assert fits[1].base_x_min == 4.0 and fits[1].base_x_max == 6.0


def test_evaluate_piecewise_no_blend_uses_base_range():
    x = np.arange(10, dtype=float)
    y = np.where(x < 5, 0.0, 100.0)
    fits = fit_chunks(x, y, 2, 0, overlap_pct=50)  # order-0 fits on extended slices
    out = evaluate_piecewise(fits, x, blend=False)
    np.testing.assert_allclose(out[:5], fits[0].predict(x[:5]))
    np.testing.assert_allclose(out[5:], fits[1].predict(x[5:]))
    # a point inside chunk 0's fitted slice but in chunk 1's base range uses chunk 1
    assert out[5] == pytest.approx(fits[1].predict(5.0))


def test_evaluate_piecewise_blend_is_convex_in_overlap_and_matches_edges():
    x = np.linspace(0, 1, 201)
    y = np.sin(2 * np.pi * x)
    fits = fit_chunks(x, y, 2, 3, overlap_pct=40)
    left, right = fits
    a, b = right.x_min, left.x_max
    xq = np.linspace(a, b, 51)
    out = evaluate_piecewise(fits, xq, blend=True)
    lo = np.minimum(left.predict(xq), right.predict(xq))
    hi = np.maximum(left.predict(xq), right.predict(xq))
    assert np.all(out >= lo - 1e-12) and np.all(out <= hi + 1e-12)
    np.testing.assert_allclose(out[0], left.predict(a), atol=1e-12)
    np.testing.assert_allclose(out[-1], right.predict(b), atol=1e-12)


def test_evaluate_piecewise_blend_has_no_jump_at_seam():
    x = np.linspace(0, 1, 1001)
    y = np.sin(2 * np.pi * x) + 0.3 * np.cos(7 * x)
    fits = fit_chunks(x, y, 4, 2, overlap_pct=30)
    dense = np.linspace(0, 1, 20001)
    out = evaluate_piecewise(fits, dense, blend=True)
    step = np.abs(np.diff(out))
    assert step.max() < 5 * np.median(step) + 1e-3


def test_rmse_and_r2():
    assert rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))
    assert r2([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert r2([1, 2, 3], [2, 2, 2]) == pytest.approx(0.0)


def test_error_map_orientation_and_unnormalised_rmse():
    x = np.linspace(0, 1, 300)
    y = np.sin(3 * x)
    orders = [1, 2, 3]
    chunks = [1, 2]
    grid = error_map(x, y, orders, chunks, [0], workers=1)
    assert grid.shape == (3, 2)
    fits = fit_chunks(x, y, 2, 3, 0.0)
    expected = rmse(y, evaluate_piecewise(fits, x, blend=False))
    assert grid[2, 1] == pytest.approx(expected)
    assert grid[0, 0] > grid[2, 0]  # higher order fits better


def test_error_map_workers_match_and_shapes_on_other_axes():
    x = np.linspace(0, 1, 200)
    y = np.cos(5 * x)
    serial = error_map(x, y, [1, 2], [1, 2, 3], [0], workers=1)
    threaded = error_map(x, y, [1, 2], [1, 2, 3], [0], workers=None)
    np.testing.assert_allclose(serial, threaded)
    by_overlap = error_map(x, y, [1, 2], [2], [0, 10, 20], workers=1)
    assert by_overlap.shape == (2, 3)  # axis 0 orders, axis 1 overlaps


def test_error_map_requires_exactly_two_varying():
    x = np.linspace(0, 1, 50)
    with pytest.raises(ValueError):
        error_map(x, x, [1, 2], [1, 2], [0, 10])
    with pytest.raises(ValueError):
        error_map(x, x, [1], [1], [0])


def test_error_map_fast_enough_for_full_grid():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 1000)
    y = np.sin(x) + 0.1 * rng.standard_normal(1000)
    start = time.perf_counter()
    grid = error_map(x, y, range(1, 11), range(1, 11), [0], workers=None)
    elapsed = time.perf_counter() - start
    assert grid.shape == (10, 10)
    assert elapsed < 2.0


def test_extrapolate_exact_on_linear_data():
    x = np.arange(100, dtype=float)
    y = 3 * x + 2
    coeffs, pred = extrapolate(x, y, 0.2, 1)
    np.testing.assert_allclose(coeffs, [3, 2], atol=1e-8)
    assert pred.shape == x.shape
    np.testing.assert_allclose(pred, y, atol=1e-8)


def test_extrapolate_rejects_bad_fraction_and_too_few_points():
    x = np.arange(5, dtype=float)
    with pytest.raises(ValueError):
        extrapolate(x, x, 0.0, 1)
    with pytest.raises(ValueError):
        extrapolate(x, x, 1.5, 1)
    with pytest.raises(ValueError):
        extrapolate(x, x, 1.0, 10)


def test_import_pulls_no_gui_packages():
    import subprocess
    import sys

    code = (
        "import sys, fitting; "
        "bad = [m for m in ('PyQt5', 'pyqtgraph', 'matplotlib', 'pandas') if m in sys.modules]; "
        "assert not bad, bad"
    )
    root = fitting.__file__.rsplit("/", 1)[0]
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)
