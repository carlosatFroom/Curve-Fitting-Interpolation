import os
import subprocess
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import fitting  # noqa: E402
from fitting import (chunk_bounds, error_map, evaluate_piecewise, extrapolate,  # noqa: E402
                     fit_chunks, r2, rmse)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_chunk_bounds_partition_matches_array_split():
    for n, k in [(10, 3), (17, 4), (1000, 7), (5, 5)]:
        expected = []
        start = 0
        for part in np.array_split(np.arange(n), k):
            expected.append((start, start + len(part)))
            start += len(part)
        assert chunk_bounds(n, k, 0) == expected


def test_chunk_bounds_overlap_exact_values():
    assert chunk_bounds(10, 3, 50) == [(0, 6), (4, 8), (6, 10)]


def test_chunk_bounds_clipped_to_range():
    bounds = chunk_bounds(10, 2, 100)
    assert bounds == [(0, 10), (0, 10)]


def test_chunk_bounds_rejects_bad_chunk_counts():
    with pytest.raises(ValueError):
        chunk_bounds(10, 0)
    with pytest.raises(ValueError):
        chunk_bounds(10, 11)


def test_fit_chunks_recovers_cubic_exactly():
    x = np.linspace(0, 1, 40)
    y = 2 * x**3 - x + 0.5
    fits = fit_chunks(x, y, 4, 3)
    assert len(fits) == 4
    for f in fits:
        assert f.rmse < 1e-10
        np.testing.assert_allclose(f.coeffs, [2, 0, -1, 0.5], atol=1e-8)
        np.testing.assert_allclose(f.predict(x[f.start:f.stop]), y[f.start:f.stop], atol=1e-9)


def test_fit_chunks_coeffs_match_polyfit_low_order():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 300)
    y = np.sin(x) + 0.05 * rng.standard_normal(x.size)
    for f in fit_chunks(x, y, 5, 2):
        expected = np.polyfit(x[f.start:f.stop], y[f.start:f.stop], 2)
        np.testing.assert_allclose(f.coeffs, expected, rtol=1e-6, atol=1e-8)


def test_fit_chunks_reduces_order_for_small_chunks():
    x = np.arange(6, dtype=float)
    y = 3 * x + 1
    fits = fit_chunks(x, y, 3, 5)  # each chunk has 2 points -> order 1
    for f in fits:
        assert len(f.coeffs) == 2
        np.testing.assert_allclose(f.predict(x[f.start:f.stop]), y[f.start:f.stop], atol=1e-10)


def test_fit_chunks_high_order_stable_far_from_origin():
    rng = np.random.default_rng(1)
    x = np.linspace(1000, 1001, 200)
    t = x - 1000
    coef = rng.uniform(-1, 1, 11)
    y = np.polynomial.polynomial.polyval(t, coef)
    (f,) = fit_chunks(x, y, 1, 10)
    assert np.max(np.abs(f.predict(x) - y)) < 1e-8


def test_evaluate_piecewise_no_blend_uses_base_chunk_predictions():
    x = np.linspace(0, 5, 101)
    y = np.cos(x) + 0.1 * x
    fits = fit_chunks(x, y, 3, 2, overlap_pct=20)
    out = evaluate_piecewise(fits, x, blend=False)
    for (bs, be), f in zip(chunk_bounds(len(x), 3, 0), fits):
        np.testing.assert_allclose(out[bs:be], f.predict(x[bs:be]), atol=1e-12)


def test_evaluate_piecewise_blend_has_no_seam_jump():
    x = np.linspace(0, 10, 500)
    y = np.sin(x) + 0.2 * x
    fits = fit_chunks(x, y, 4, 3, overlap_pct=20)
    xq = np.linspace(0, 10, 20001)
    hard = np.abs(np.diff(evaluate_piecewise(fits, xq, blend=False)))
    soft = np.abs(np.diff(evaluate_piecewise(fits, xq, blend=True)))
    assert hard.max() > 10 * soft.max()
    assert soft.max() < 0.01


def test_rmse_and_r2_known_values():
    assert rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, y) == pytest.approx(1.0)
    assert r2(y, np.full(4, y.mean())) == pytest.approx(0.0)


@pytest.fixture
def data():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 4, 300)
    y = np.sin(2 * x) + 0.1 * rng.standard_normal(x.size)
    return x, y


def _direct(x, y, order, n_chunks, overlap):
    return rmse(y, evaluate_piecewise(fit_chunks(x, y, n_chunks, order, overlap), x, blend=False))


def test_error_map_orders_by_chunks(data):
    x, y = data
    orders, chunks = [1, 2, 3], [1, 2, 4, 5]
    em = error_map(x, y, orders, chunks, [0])
    assert em.shape == (3, 4)
    for i, o in enumerate(orders):
        for j, c in enumerate(chunks):
            assert em[i, j] == pytest.approx(_direct(x, y, o, c, 0))


def test_error_map_orders_by_overlap(data):
    x, y = data
    orders, overlaps = [1, 2, 3], [0, 10, 25]
    em = error_map(x, y, orders, [3], overlaps)
    assert em.shape == (3, 3)
    for i, o in enumerate(orders):
        for j, ov in enumerate(overlaps):
            assert em[i, j] == pytest.approx(_direct(x, y, o, 3, ov))


def test_error_map_chunks_by_overlap(data):
    x, y = data
    chunks, overlaps = [1, 2, 5], [0, 5, 20]
    em = error_map(x, y, [2], chunks, overlaps)
    assert em.shape == (3, 3)
    for i, c in enumerate(chunks):
        for j, ov in enumerate(overlaps):
            assert em[i, j] == pytest.approx(_direct(x, y, 2, c, ov))


def test_error_map_threaded_equals_serial(data):
    x, y = data
    serial = error_map(x, y, range(1, 6), range(1, 6), [0], workers=1)
    threaded = error_map(x, y, range(1, 6), range(1, 6), [0], workers=4)
    np.testing.assert_allclose(serial, threaded)


def test_error_map_rejects_wrong_number_of_varying_sequences(data):
    x, y = data
    with pytest.raises(ValueError):
        error_map(x, y, [1, 2], [1, 2], [0, 5])


def test_error_map_speed_orders_by_chunks():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 1, 1000)
    y = np.sin(7 * x) + 0.1 * rng.standard_normal(x.size)
    start = time.perf_counter()
    em = error_map(x, y, range(1, 11), range(1, 11), [0])
    assert em.shape == (10, 10)
    assert time.perf_counter() - start < 2.0


def test_extrapolate_exact_polynomial():
    x = np.linspace(0, 10, 100)
    y = 2 * x**2 - 3 * x + 1
    coeffs, pred = extrapolate(x, y, 0.3, 2)
    np.testing.assert_allclose(coeffs, [2, -3, 1], atol=1e-8)
    np.testing.assert_allclose(pred, y, atol=1e-8)


def test_extrapolate_uses_at_least_order_plus_one_points():
    x = np.linspace(0, 10, 100)
    y = x**3 - 2 * x + 1
    coeffs, pred = extrapolate(x, y, 0.01, 3)  # 1 requested point -> 4 used
    np.testing.assert_allclose(coeffs, [1, 0, -2, 1], atol=1e-6)
    np.testing.assert_allclose(pred, y, atol=1e-6)


def test_extrapolate_rejects_bad_fraction():
    x = np.linspace(0, 1, 20)
    with pytest.raises(ValueError):
        extrapolate(x, x, 0.0, 1)


def test_import_fitting_does_not_load_gui_modules():
    code = (
        "import sys, fitting; "
        "bad = [m for m in ('PyQt5', 'pyqtgraph', 'matplotlib') if m in sys.modules]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
