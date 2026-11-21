import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import fitting
from fitting import (
    chunk_bounds, error_map, evaluate_piecewise, extrapolate, fit_chunks, r2, rmse,
)

ROOT = Path(__file__).resolve().parent.parent


def expected_base(n, k):
    return [(int(c[0]), int(c[-1]) + 1) for c in np.array_split(np.arange(n), k)]


# ---------------------------------------------------------------- chunk_bounds

@pytest.mark.parametrize("n", [10, 11, 17, 100])
def test_partition_matches_array_split(n):
    for k in range(1, min(n, 7) + 1):
        assert chunk_bounds(n, k, 0.0) == expected_base(n, k)


def test_unequal_partition_example():
    assert chunk_bounds(10, 3) == [(0, 4), (4, 7), (7, 10)]


def test_overlap_first_middle_last_even():
    # base (0,25),(25,50),(50,75),(75,100); ov = 2 -> first +2, middle -1/+1, last -2
    got = chunk_bounds(100, 4, 10.0)
    assert got == [(0, 27), (24, 51), (49, 76), (73, 100)]


def test_overlap_uses_each_chunks_own_base_length():
    # base (0,4),(4,7),(7,10): ov = 2, 1, 1
    got = chunk_bounds(10, 3, 50.0)
    assert got == [(0, 6), (4, 8), (6, 10)]


def test_overlap_zero_equals_base():
    assert chunk_bounds(50, 5, 0.0) == expected_base(50, 5)


def test_bounds_always_clipped_and_cover_points():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 300))
        k = int(rng.integers(1, n + 1))
        pct = float(rng.uniform(0, 100))
        bounds = chunk_bounds(n, k, pct)
        assert len(bounds) == k
        for s, e in bounds:
            assert 0 <= s < e <= n
        assert bounds[0][0] == 0 and bounds[-1][1] == n


@pytest.mark.parametrize("k", [0, -1, 11])
def test_chunk_bounds_rejects_bad_counts(k):
    with pytest.raises(ValueError):
        chunk_bounds(10, k, 0.0)


# ---------------------------------------------------------------- ChunkFit / fit_chunks

def test_exact_recovery_of_cubic():
    x = np.linspace(0, 10, 101)
    y = 2 * x**3 - x + 1
    fits = fit_chunks(x, y, n_chunks=2, order=3)
    for f in fits:
        np.testing.assert_allclose(f.coeffs, [2, 0, -1, 1], atol=1e-8)
        assert f.rmse < 1e-9
        np.testing.assert_allclose(f.predict(x[f.start:f.stop]), y[f.start:f.stop], atol=1e-7)


def test_fit_slice_metadata():
    x = np.linspace(0, 1, 40)
    fits = fit_chunks(x, x, n_chunks=4, order=1, overlap_pct=0)
    assert [(f.start, f.stop) for f in fits] == expected_base(40, 4)
    assert fits[1].x_min == pytest.approx(x[10]) and fits[1].x_max == pytest.approx(x[19])


def test_order_reduced_when_chunk_too_small():
    x = np.arange(10, dtype=float)
    y = 3 * x + 2
    fits = fit_chunks(x, y, n_chunks=5, order=3)  # chunks of 2 points -> linear
    for f in fits:
        assert len(f.coeffs) == 2
        np.testing.assert_allclose(f.coeffs, [3, 2], atol=1e-9)


def test_order_reduced_for_three_points():
    x = np.arange(9, dtype=float)
    fits = fit_chunks(x, x**2, n_chunks=3, order=5)  # chunks of 3 -> quadratic
    for f in fits:
        assert len(f.coeffs) == 3
        assert f.rmse < 1e-9


def test_coeffs_match_polyfit_low_order():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 60)
    y = 1.5 - 2 * x + 0.7 * x**2 + rng.normal(0, 0.01, x.size)
    (fit,) = fit_chunks(x, y, n_chunks=1, order=2)
    np.testing.assert_allclose(fit.coeffs, np.polyfit(x, y, 2), rtol=1e-6, atol=1e-8)


def test_predict_stable_at_order_10_far_from_origin():
    x = np.linspace(1000, 1001, 200)
    c = np.array([0.3, -1.2, 0.8, 2.0, -0.5, 0.1, 0.9, -0.2, 0.4, 1.1, 5.0])
    y = np.polyval(c, x - 1000)
    (fit,) = fit_chunks(x, y, n_chunks=1, order=10)
    assert np.all(np.isfinite(fit.predict(x)))
    np.testing.assert_allclose(fit.predict(x), y, atol=1e-6)
    assert fit.rmse < 1e-6


def test_predict_sine_order_10_far_from_origin():
    x = np.linspace(1000, 1001, 300)
    y = np.sin(2 * np.pi * (x - 1000))
    (fit,) = fit_chunks(x, y, n_chunks=1, order=10)
    assert np.max(np.abs(fit.predict(x) - y)) < 1e-4


def test_rmse_field_matches_predict():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 5, 80)
    y = 0.5 * x + rng.normal(0, 0.2, x.size)
    fits = fit_chunks(x, y, n_chunks=2, order=1)
    for f in fits:
        ys = y[f.start:f.stop]
        assert f.rmse == pytest.approx(rmse(ys, f.predict(x[f.start:f.stop])), rel=1e-9)


# ---------------------------------------------------------------- evaluate_piecewise

def test_blend_false_uses_base_chunk():
    x = np.linspace(0, 10, 90)
    y = np.sin(x)
    fits = fit_chunks(x, y, n_chunks=3, order=2, overlap_pct=30)
    got = evaluate_piecewise(fits, x, blend=False)
    for i, (s, e) in enumerate(expected_base(90, 3)):
        np.testing.assert_allclose(got[s:e], fits[i].predict(x[s:e]), atol=1e-12)


def test_blend_true_no_jumps_at_seams():
    x = np.linspace(0, 10, 1001)
    y = np.sin(x)
    fits = fit_chunks(x, y, n_chunks=4, order=3, overlap_pct=20)
    dense = np.linspace(0, 10, 20001)
    pred = evaluate_piecewise(fits, dense, blend=True)
    dx = dense[1] - dense[0]
    max_step = np.max(np.abs(np.diff(pred)))
    assert max_step < 2e-3  # |sin'| <= 1 -> true step ~ dx = 5e-4
    assert max_step < 4 * dx


def test_blend_true_without_overlap_equals_blend_false():
    x = np.linspace(0, 10, 100)
    fits = fit_chunks(x, np.cos(x), n_chunks=4, order=3, overlap_pct=0)
    np.testing.assert_allclose(evaluate_piecewise(fits, x, True), evaluate_piecewise(fits, x, False))


def test_blend_true_actually_differs_in_overlap():
    x = np.linspace(0, 10, 200)
    fits = fit_chunks(x, np.sin(x) + 0.05 * x, n_chunks=2, order=1, overlap_pct=40)
    a = evaluate_piecewise(fits, x, True)
    b = evaluate_piecewise(fits, x, False)
    assert np.any(np.abs(a - b) > 1e-6)


# ---------------------------------------------------------------- metrics

def test_rmse_known_value():
    assert rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))


def test_r2_known_values():
    y = [1.0, 2.0, 3.0]
    assert r2(y, y) == pytest.approx(1.0)
    assert r2(y, [2.0, 2.0, 2.0]) == pytest.approx(0.0)  # predicting the mean
    assert r2(y, [1.0, 2.0, 4.0]) == pytest.approx(0.5)  # ss_res 1 / ss_tot 2


def test_r2_constant_target():
    assert r2([2.0, 2.0], [2.0, 2.0]) == 1.0
    assert r2([2.0, 2.0], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------- error_map

X = np.linspace(0, 10, 200)
Y = np.sin(X) + 0.1 * np.cos(3 * X)


def manual_cell(x, y, order, n_chunks, overlap):
    """Independent reference: np.polyfit per overlapped chunk, base chunk owns each point."""
    n = len(x)
    base = expected_base(n, n_chunks)
    bounds = chunk_bounds(n, n_chunks, overlap)
    pred = np.empty(n)
    for (bs, be), (s, e) in zip(base, bounds):
        coef = np.polyfit(x[s:e], y[s:e], order)
        pred[bs:be] = np.polyval(coef, x[bs:be])
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def test_error_map_orders_by_chunks_shape_and_values():
    orders, chunks = [1, 2, 3], [1, 2, 4, 5]
    em = error_map(X, Y, orders, chunks, [0.0])
    assert em.shape == (3, 4)
    for i, o in enumerate(orders):
        for j, k in enumerate(chunks):
            assert em[i, j] == pytest.approx(manual_cell(X, Y, o, k, 0.0), rel=1e-6, abs=1e-12)


def test_error_map_orders_by_overlap_shape_and_values():
    orders, overlaps = [1, 2, 3], [0.0, 10.0]
    em = error_map(X, Y, orders, [4], overlaps)
    assert em.shape == (3, 2)
    for i, o in enumerate(orders):
        for j, ov in enumerate(overlaps):
            assert em[i, j] == pytest.approx(manual_cell(X, Y, o, 4, ov), rel=1e-6, abs=1e-12)


def test_error_map_chunks_by_overlap_shape_and_values():
    chunks, overlaps = [1, 2, 4, 5], [0.0, 10.0, 25.0]
    em = error_map(X, Y, [2], chunks, overlaps)
    assert em.shape == (4, 3)
    for i, k in enumerate(chunks):
        for j, ov in enumerate(overlaps):
            assert em[i, j] == pytest.approx(manual_cell(X, Y, 2, k, ov), rel=1e-6, abs=1e-12)


def test_error_map_is_un_normalized():
    # scaling y scales the RMSE linearly (no normalisation by range or variance)
    em1 = error_map(X, Y, [2, 3], [2, 4], [0.0])
    em2 = error_map(X, 10 * Y, [2, 3], [2, 4], [0.0])
    np.testing.assert_allclose(em2, 10 * em1, rtol=1e-9)


def test_error_map_workers_equal():
    args = (X, Y, [1, 2, 3], [1, 2, 3, 4], [0.0])
    serial = error_map(*args, workers=1)
    np.testing.assert_array_equal(serial, error_map(*args))
    np.testing.assert_array_equal(serial, error_map(*args, workers=2))


@pytest.mark.parametrize("orders, chunks, overlaps", [
    ([1, 2], [1, 2], [0.0, 10.0]),  # three varying
    ([1], [1], [0.0]),              # none varying
    ([1, 2], [3], [0.0]),           # only one varying
])
def test_error_map_requires_exactly_two_varying(orders, chunks, overlaps):
    with pytest.raises(ValueError):
        error_map(X, Y, orders, chunks, overlaps)


def test_error_map_timing_1000_points():
    x = np.linspace(0, 1, 1000)
    y = np.sin(7 * x) + x**2
    start = time.perf_counter()
    em = error_map(x, y, list(range(1, 11)), list(range(1, 11)), [0.0])
    elapsed = time.perf_counter() - start
    assert em.shape == (10, 10)
    assert np.all(np.isfinite(em))
    assert elapsed < 2.0, f"error_map took {elapsed:.2f}s"


# ---------------------------------------------------------------- extrapolate

def test_extrapolate_fits_only_first_fraction():
    x = np.linspace(0, 10, 100)
    y = np.where(x < 5, 2 * x + 1, 100 - 5 * x)
    coeffs, pred = extrapolate(x, y, fit_fraction=0.5, order=1)
    np.testing.assert_allclose(coeffs, [2, 1], atol=1e-9)
    assert pred.shape == x.shape
    np.testing.assert_allclose(pred, 2 * x + 1, atol=1e-6)


def test_extrapolate_uses_at_least_order_plus_one_points():
    x = np.linspace(0, 10, 20)
    cubic = np.array([1.0, -2.0, 0.5, 3.0])
    y = np.polyval(cubic, x)
    coeffs, pred = extrapolate(x, y, fit_fraction=0.01, order=3)  # 0.2 pts -> 4 pts
    assert len(coeffs) == 4
    np.testing.assert_allclose(coeffs, cubic, atol=1e-8)
    np.testing.assert_allclose(pred, y, atol=1e-8)


def test_extrapolate_returns_predictions_over_all_x():
    x = np.linspace(-1, 1, 37)
    coeffs, pred = extrapolate(x, x**2, fit_fraction=1.0, order=2)
    assert len(coeffs) == 3 and pred.shape == (37,)
    np.testing.assert_allclose(pred, x**2, atol=1e-9)


@pytest.mark.parametrize("frac", [0.0, -0.5, 1.5])
def test_extrapolate_rejects_bad_fraction(frac):
    with pytest.raises(ValueError):
        extrapolate(X, Y, frac, 2)


# ---------------------------------------------------------------- import hygiene

def test_import_does_not_pull_gui_or_plot_modules():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import fitting; "
        "bad = [m for m in sys.modules if m.split('.')[0] in ('PyQt5', 'pyqtgraph', 'matplotlib')]; "
        "print(bad)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(ROOT)], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_public_names_present():
    for name in ("chunk_bounds", "ChunkFit", "fit_chunks", "evaluate_piecewise",
                 "rmse", "r2", "error_map", "extrapolate"):
        assert hasattr(fitting, name)
