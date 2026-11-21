import time

import numpy as np
import pytest

import fitting


def sine_data(n=1000):
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 4.0 * np.pi, n)
    y = np.sin(x) + 0.001 * rng.standard_normal(n)
    return x, y


# --------------------------------------------------------------------------- #
# chunk_bounds
# --------------------------------------------------------------------------- #
def test_chunk_bounds_no_overlap_matches_array_split():
    assert fitting.chunk_bounds(10, 3, 0) == [(0, 4), (4, 7), (7, 10)]


def test_chunk_bounds_rejects_bad_chunk_counts():
    with pytest.raises(ValueError):
        fitting.chunk_bounds(10, 0)
    with pytest.raises(ValueError):
        fitting.chunk_bounds(10, 11)


def test_chunk_bounds_overlap_widens_middle_chunks():
    n = 1000
    bounds = fitting.chunk_bounds(n, 4, 10.0)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == n
    base = fitting.chunk_bounds(n, 4, 0.0)
    for (s, e), (bs, be) in zip(bounds, base):
        assert 0 <= s <= bs and be <= e <= n
    # middle chunks widen on both sides
    for (s, e), (bs, be) in zip(bounds[1:-1], base[1:-1]):
        assert s < bs and e > be


# --------------------------------------------------------------------------- #
# fit_chunks / evaluate_piecewise
# --------------------------------------------------------------------------- #
def test_piecewise_sine_fit_is_accurate():
    x, y = sine_data()
    fits = fitting.fit_chunks(x, y, n_chunks=8, order=5)
    pred = fitting.evaluate_piecewise(fits, x, blend=False)
    assert fitting.rmse(y, pred) < 0.01


def test_blend_does_not_increase_seam_jumps():
    x, y = sine_data()
    fits = fitting.fit_chunks(x, y, n_chunks=8, order=5, overlap_pct=20.0)
    grid = np.linspace(x[0], x[-1], 20000)
    hard = fitting.evaluate_piecewise(fits, grid, blend=False)
    soft = fitting.evaluate_piecewise(fits, grid, blend=True)
    jump_hard = np.max(np.abs(np.diff(hard)))
    jump_soft = np.max(np.abs(np.diff(soft)))
    assert jump_soft <= jump_hard + 1e-12


def test_small_chunk_reduces_order_instead_of_raising():
    x = np.arange(6.0)
    y = x ** 2
    fits = fitting.fit_chunks(x, y, n_chunks=3, order=10)
    assert len(fits) == 3
    pred = fitting.evaluate_piecewise(fits, x, blend=False)
    assert fitting.rmse(y, pred) < 1e-8


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_rmse_hand_computed():
    # errors: 1, -1, 2 -> squares 1, 1, 4 -> mean 2 -> sqrt 2
    assert fitting.rmse([0.0, 0.0, 0.0], [1.0, -1.0, 2.0]) == pytest.approx(np.sqrt(2.0))


def test_r2_hand_computed():
    # y = [1, 2, 3], mean 2, ss_tot = 2; pred = [1, 2, 4], ss_res = 1 -> r2 = 0.5
    assert fitting.r2([1.0, 2.0, 3.0], [1.0, 2.0, 4.0]) == pytest.approx(0.5)


def test_r2_perfect_fit_is_one():
    y = np.array([0.3, -1.2, 4.0, 2.5])
    assert fitting.r2(y, y) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# error_map
# --------------------------------------------------------------------------- #
def test_error_map_shape_and_values():
    x, y = sine_data()
    orders = [1, 2, 3]
    chunks = [1, 2, 4, 8]
    grid = fitting.error_map(x, y, orders, chunks, [0])
    assert grid.shape == (len(orders), len(chunks))
    assert np.all(np.isfinite(grid))
    assert np.all(grid >= 0)


def test_error_map_full_grid_is_fast():
    x, y = sine_data()
    start = time.perf_counter()
    grid = fitting.error_map(x, y, range(1, 11), range(1, 11), [0])
    elapsed = time.perf_counter() - start
    assert grid.shape == (10, 10)
    assert elapsed < 2.0


# --------------------------------------------------------------------------- #
# extrapolate
# --------------------------------------------------------------------------- #
def test_extrapolate_shapes():
    x = np.linspace(0.0, 10.0, 200)
    y = 2.0 * x ** 2 - x + 3.0
    coeffs, y_pred = fitting.extrapolate(x, y, fit_fraction=0.5, order=2)
    assert len(y_pred) == len(x)
    assert len(coeffs) == 3


def test_extrapolate_recovers_low_degree_polynomial():
    x = np.linspace(0.0, 10.0, 400)
    y = 0.5 * x ** 3 - 2.0 * x + 1.0
    coeffs, y_pred = fitting.extrapolate(x, y, fit_fraction=0.5, order=3)
    assert np.allclose(y_pred, y, rtol=1e-6, atol=1e-6)
    assert np.allclose(coeffs, [0.5, 0.0, -2.0, 1.0], atol=1e-6)
