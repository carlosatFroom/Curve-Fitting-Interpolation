import itertools
import os
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import fitting  # noqa: E402
from fitting import (  # noqa: E402
    chunk_bounds,
    error_map,
    evaluate_piecewise,
    extrapolate,
    fit_chunks,
    r2,
    rmse,
)


def _data(n=1000):
    x = np.linspace(0.0, 10.0, n)
    y = np.sin(x) + 0.1 * x
    return x, y


def test_chunk_bounds_unequal_chunks_no_overlap():
    # array_split(arange(10), 3) -> lengths 4, 3, 3
    assert chunk_bounds(10, 3) == [(0, 4), (4, 7), (7, 10)]


def test_chunk_bounds_overlap_formula():
    # base lengths 34, 33, 33; ov = int(len * 10 / 100) = 3, 3, 3
    b = chunk_bounds(100, 3, 10)
    assert b[0] == (0, 34 + 3)  # first: stop extended by ov
    assert b[1] == (34 - 1, 67 + 2)  # middle: start -= ov//2, stop += ov - ov//2
    assert b[2] == (67 - 3, 100)  # last: start extended back by ov


def test_chunk_bounds_clipped_to_range():
    assert chunk_bounds(10, 2, 100) == [(0, 10), (0, 10)]
    assert chunk_bounds(10, 1, 50) == [(0, 10)]


@pytest.mark.parametrize("n_chunks", [0, -1, 11])
def test_chunk_bounds_invalid_chunks_raise(n_chunks):
    with pytest.raises(ValueError):
        chunk_bounds(10, n_chunks)


def test_exact_recovery_of_polynomial():
    x = np.linspace(-2, 3, 60)
    coeffs_true = np.array([0.5, -1.0, 2.0, 3.0])  # highest degree first
    y = np.polyval(coeffs_true, x)
    (fit,) = fit_chunks(x, y, 1, 3)
    np.testing.assert_allclose(fit.predict(x), y, atol=1e-9)
    np.testing.assert_allclose(fit.coeffs, coeffs_true, atol=1e-9)
    assert fit.rmse < 1e-9


def test_high_order_stability_on_offset_x():
    # Raw x around 2000 makes plain power-basis fits ill-conditioned.
    x = np.linspace(1990.0, 2020.0, 300)
    y = np.cos(x / 5.0)
    (fit,) = fit_chunks(x, y, 1, 12)
    assert np.all(np.isfinite(fit.predict(x)))
    assert np.max(np.abs(fit.predict(x) - y)) < 1e-6


def test_small_chunk_order_reduction():
    x, y = _data(20)
    fits = fit_chunks(x, y, 10, 5)  # chunks of 2 points each
    for f in fits:
        assert len(f.coeffs) == 6  # padded to requested order + 1
        assert np.all(f.coeffs[:4] == 0)  # reduced to degree 1
        assert f.rmse < 1e-12


def test_single_point_chunk_does_not_fail():
    x = np.arange(5.0)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fits = fit_chunks(x, y, 5, 3)
    p = evaluate_piecewise(fits, x, blend=False)
    np.testing.assert_allclose(p, y, atol=1e-9)


def test_blend_is_continuous_at_seams():
    x, y = _data(2000)
    fits = fit_chunks(x, y, 4, 6, overlap_pct=20)
    xq = np.linspace(0.0, 10.0, 20001)
    p = evaluate_piecewise(fits, xq, blend=True)
    # Smooth function, fine grid: no jumps anywhere, including seams.
    assert np.max(np.abs(np.diff(p))) < 5e-3

    for left, right in zip(fits[:-1], fits[1:]):
        a, b = right.x_min, left.x_max
        if b > a:
            at_a = evaluate_piecewise([left, right], np.array([a]), blend=True)[0]
            at_b = evaluate_piecewise([left, right], np.array([b]), blend=True)[0]
            assert at_a == pytest.approx(left.predict(np.array([a]))[0])
            assert at_b == pytest.approx(right.predict(np.array([b]))[0])


def test_blend_false_uses_base_ranges():
    x, y = _data(100)
    fits = fit_chunks(x, y, 4, 3, overlap_pct=30)
    p = evaluate_piecewise(fits, x, blend=False)
    for i, f in enumerate(fits):
        lo = f.base_lo
        hi = f.base_hi
        m = (x >= lo) & (x <= hi)
        np.testing.assert_allclose(p[m], f.predict(x[m]))


def test_evaluate_preserves_shape():
    x, y = _data(100)
    fits = fit_chunks(x, y, 2, 3)
    xq = x.reshape(10, 10)
    assert evaluate_piecewise(fits, xq).shape == (10, 10)


def test_rmse_and_r2():
    yt = np.array([1.0, 2.0, 3.0, 4.0])
    assert rmse(yt, yt) == 0.0
    assert rmse(yt, yt + 2.0) == pytest.approx(2.0)
    assert r2(yt, yt) == pytest.approx(1.0)
    assert r2(yt, np.full_like(yt, yt.mean())) == pytest.approx(0.0)


def _brute(x, y, order, n_chunks, ov):
    fits = fit_chunks(x, y, n_chunks, order, ov)
    return rmse(y, evaluate_piecewise(fits, x, blend=False))


@pytest.mark.parametrize(
    "pairing",
    ["orders_chunks", "orders_overlap", "chunks_overlap"],
)
def test_error_map_matches_brute_force(pairing):
    x, y = _data(200)
    orders = [1, 2, 3, 4]
    chunks = [1, 2, 3, 5]
    overlaps = [0.0, 10.0, 25.0]
    if pairing == "orders_chunks":
        m = error_map(x, y, orders, chunks, [10.0], workers=1)
        expected = np.array([[_brute(x, y, o, c, 10.0) for c in chunks] for o in orders])
    elif pairing == "orders_overlap":
        m = error_map(x, y, orders, [3], overlaps, workers=1)
        expected = np.array([[_brute(x, y, o, 3, ov) for ov in overlaps] for o in orders])
    else:
        m = error_map(x, y, [2], chunks, overlaps, workers=1)
        expected = np.array([[_brute(x, y, 2, c, ov) for ov in overlaps] for c in chunks])
    assert m.shape == expected.shape
    np.testing.assert_allclose(m, expected, rtol=1e-10, atol=1e-12)


def test_error_map_threaded_matches_serial():
    x, y = _data(200)
    serial = error_map(x, y, range(1, 5), range(1, 5), [5.0], workers=1)
    threaded = error_map(x, y, range(1, 5), range(1, 5), [5.0], workers=4)
    np.testing.assert_allclose(serial, threaded)


def test_error_map_requires_exactly_two_varying():
    x, y = _data(50)
    with pytest.raises(ValueError):
        error_map(x, y, [1, 2], [1, 2], [0.0, 5.0])
    with pytest.raises(ValueError):
        error_map(x, y, [1, 2], [3], [0.0])


def test_error_map_timing():
    x, y = _data(1000)
    start = time.perf_counter()
    m = error_map(x, y, range(1, 11), range(1, 11), [0.0])
    elapsed = time.perf_counter() - start
    assert m.shape == (10, 10)
    assert elapsed < 2.0


def test_extrapolate_fits_prefix_and_predicts_all():
    x = np.linspace(0, 10, 101)
    y = 2.0 * x**2 - x + 3.0
    coeffs, pred = extrapolate(x, y, 0.5, 2)
    assert pred.shape == x.shape
    np.testing.assert_allclose(coeffs, [2.0, -1.0, 3.0], atol=1e-8)
    np.testing.assert_allclose(pred, y, atol=1e-7)


def test_extrapolate_minimum_points_and_validation():
    x = np.linspace(0, 1, 20)
    y = x * 4.0
    coeffs, pred = extrapolate(x, y, 0.001, 3)  # forced to order + 1 points
    assert len(coeffs) == 4
    with pytest.raises(ValueError):
        extrapolate(x, y, 0.0, 1)
    with pytest.raises(ValueError):
        extrapolate(x, y, 1.5, 1)


def test_import_does_not_load_gui_or_plotting():
    code = (
        "import sys, fitting; "
        "bad = [m for m in ('PyQt5', 'pyqtgraph', 'matplotlib') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT)
    assert result.returncode == 0
