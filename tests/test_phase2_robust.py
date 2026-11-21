import numpy as np
import pytest

import fitting


def _sine(n, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    return x, np.sin(2 * np.pi * x) + rng.normal(0, noise, n)


def _assert_partition(bounds, n):
    assert bounds[0][0] == 0 and bounds[-1][1] == n
    for (a, b), (c, _) in zip(bounds, bounds[1:]):
        assert a < b and b == c


def test_adaptive_partition_is_contiguous_and_covers():
    x, y = _sine(500)
    bounds = fitting.adaptive_chunks(x, y, order=2, tol=0.01, max_chunks=50)
    _assert_partition(bounds, 500)


def test_adaptive_tol_satisfied_for_long_chunks():
    x, y = _sine(600, noise=0.02)
    order, tol = 3, 0.03
    bounds = fitting.adaptive_chunks(x, y, order=order, tol=tol, max_chunks=100)
    _assert_partition(bounds, 600)
    for a, b in bounds:
        if b - a >= 2 * (order + 1):
            assert fitting.rmse(y[a:b], fitting._poly_fit(x[a:b], y[a:b], order)(x[a:b])) <= tol


def test_adaptive_respects_max_chunks():
    x, y = _sine(800, noise=0.05)
    bounds = fitting.adaptive_chunks(x, y, order=1, tol=1e-6, max_chunks=7)
    assert len(bounds) <= 7
    _assert_partition(bounds, 800)


def test_adaptive_three_linear_segments_gives_three_chunks():
    x = np.linspace(0.0, 1.0, 600)
    knots_x = [0.0, 0.33, 0.66, 1.0]
    knots_y = [0.0, 1.0, 0.2, 1.0]
    y = np.interp(x, knots_x, knots_y)
    bounds = fitting.adaptive_chunks(x, y, order=1, tol=1e-6, max_chunks=20)
    _assert_partition(bounds, 600)
    assert len(bounds) == 3


def test_adaptive_huge_tol_gives_one_chunk():
    x, y = _sine(300)
    assert fitting.adaptive_chunks(x, y, order=3, tol=1e9, max_chunks=10) == [(0, 300)]


@pytest.mark.parametrize("kwargs", [
    dict(order=1, tol=-1.0, max_chunks=5),
    dict(order=1, tol=0.1, max_chunks=0),
    dict(order=-1, tol=0.1, max_chunks=5),
])
def test_adaptive_validates_arguments(kwargs):
    x, y = _sine(100)
    with pytest.raises(ValueError):
        fitting.adaptive_chunks(x, y, **kwargs)


def _gross_outliers(n=400, frac=0.10, seed=1):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    clean = np.sin(2 * np.pi * x) + 0.5 * x
    y = clean + rng.normal(0, 0.01, n)
    idx = rng.choice(n, int(frac * n), replace=False)
    y[idx] += rng.choice([-1.0, 1.0], idx.size) * rng.uniform(5, 10, idx.size)
    return x, y, clean


def test_robust_fit_tracks_clean_signal_under_outliers():
    x, y, clean = _gross_outliers()
    robust = fitting.robust_fit_chunks(x, y, n_chunks=2, order=3)
    plain = fitting.fit_chunks(x, y, n_chunks=2, order=3)
    err_robust = fitting.rmse(clean, fitting.evaluate_piecewise(robust, x, blend=False))
    err_plain = fitting.rmse(clean, fitting.evaluate_piecewise(plain, x, blend=False))
    assert err_robust < 0.05
    assert err_robust < 0.2 * err_plain


def test_robust_fit_fields_match_fit_chunks():
    x, y, _ = _gross_outliers()
    robust = fitting.robust_fit_chunks(x, y, n_chunks=3, order=2, overlap_pct=10)
    plain = fitting.fit_chunks(x, y, n_chunks=3, order=2, overlap_pct=10)
    assert len(robust) == len(plain)
    for r, p in zip(robust, plain):
        assert (r.start, r.stop) == (p.start, p.stop)
        assert (r.base_start, r.base_stop) == (p.base_start, p.base_stop)
        assert r.x_min == p.x_min and r.x_max == p.x_max
        assert r.base_x_min == p.base_x_min and r.base_x_max == p.base_x_max
        assert r.coeffs.shape == p.coeffs.shape
        assert r.rmse == pytest.approx(
            fitting.rmse(y[r.start:r.stop], r.predict(x[r.start:r.stop])))


def test_robust_fit_clean_data_matches_least_squares():
    x, y = _sine(300, noise=0.01)
    robust = fitting.robust_fit_chunks(x, y, n_chunks=1, order=5)
    plain = fitting.fit_chunks(x, y, n_chunks=1, order=5)
    assert robust[0].rmse == pytest.approx(plain[0].rmse, rel=0.1)


def _noisy_cubic(n=600, seed=3):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n)
    y = 0.5 * x**3 - x + 0.3 + rng.normal(0, 0.05, n)
    return x, y


@pytest.mark.parametrize("method", ["cv", "aic"])
def test_select_order_finds_cubic(method):
    x, y = _noisy_cubic()
    order = fitting.select_order(x, y, n_chunks=2, max_order=8, method=method)
    assert 0 <= order <= 8
    assert order == 3


def test_select_order_rejects_bad_arguments():
    x, y = _noisy_cubic(50)
    with pytest.raises(ValueError):
        fitting.select_order(x, y, n_chunks=1, max_order=3, method="bogus")
    with pytest.raises(ValueError):
        fitting.select_order(x, y, n_chunks=1, max_order=-1)
