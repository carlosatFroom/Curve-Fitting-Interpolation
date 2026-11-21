import numpy as np
import pytest

from fitting import adaptive_chunks, fit_spline


def _knot_jump(fit, m):
    eps = 1e-9
    left = fit.derivative(fit.knots - eps, m)
    right = fit.derivative(fit.knots, m)
    return np.abs(left - right)


@pytest.mark.parametrize("continuity", [0, 1, 2])
def test_continuity_at_knots(continuity):
    x = np.linspace(0.0, 10.0, 400)
    y = np.sin(x) + 0.1 * np.cos(3 * x)
    fit = fit_spline(x, y, n_chunks=6, order=4, continuity=continuity)
    for m in range(continuity + 1):
        assert np.all(_knot_jump(fit, m) < 1e-6), f"discontinuity in derivative {m}"


def test_derivative_matches_finite_difference():
    x = np.linspace(0.0, 2 * np.pi, 300)
    y = np.sin(x)
    fit = fit_spline(x, y, n_chunks=4, order=6, continuity=2)
    xq = np.linspace(0.3, 2 * np.pi - 0.3, 50)
    h = 1e-5
    fd = (fit.predict(xq + h) - fit.predict(xq - h)) / (2 * h)
    assert np.allclose(fit.derivative(xq, 1), fd, atol=1e-5)
    fd2 = (fit.derivative(xq + h, 1) - fit.derivative(xq - h, 1)) / (2 * h)
    assert np.allclose(fit.derivative(xq, 2), fd2, atol=1e-3)


def test_global_cubic_recovered_exactly():
    x = np.linspace(-2.0, 3.0, 200)
    y = 0.5 * x**3 - 1.2 * x**2 + 0.7 * x - 4.0
    fit = fit_spline(x, y, n_chunks=5, order=3, continuity=2)
    assert np.allclose(fit.predict(x), y, atol=1e-8)
    xq = np.linspace(-2.0, 3.0, 37)
    assert np.allclose(fit.derivative(xq, 1), 1.5 * xq**2 - 2.4 * xq + 0.7, atol=1e-7)


def test_global_polynomial_order_5_recovered():
    x = np.linspace(-1.0, 1.0, 300)
    coeffs = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 0.25])
    y = np.polynomial.polynomial.polyval(x, coeffs)
    fit = fit_spline(x, y, n_chunks=3, order=5, continuity=4)
    assert np.allclose(fit.predict(x), y, atol=1e-8)


def test_order_10_fifty_chunks_stable():
    x = np.linspace(-1.0, 1.0, 1000)
    coeffs = np.random.default_rng(0).normal(size=11)
    y = np.polynomial.polynomial.polyval(x, coeffs)
    fit = fit_spline(x, y, n_chunks=50, order=10, continuity=2)
    pred = fit.predict(x)
    assert np.all(np.isfinite(pred))
    assert np.sqrt(np.mean((pred - y) ** 2)) < 1e-8
    assert np.all(_knot_jump(fit, 2) < 1e-5)


def test_order_10_fifty_chunks_on_smooth_signal():
    x = np.linspace(0.0, 10.0, 1000)
    y = np.sin(x)
    fit = fit_spline(x, y, n_chunks=50, order=10, continuity=2)
    assert np.sqrt(np.mean((fit.predict(x) - y) ** 2)) < 1e-6


def test_knots_are_chunk_starts():
    x = np.arange(10.0)
    fit = fit_spline(x, x**2, n_chunks=3, order=2, continuity=1)
    # array_split(arange(10), 3) -> [0..3], [4..6], [7..9]
    assert np.allclose(fit.knots, [4.0, 7.0])


@pytest.mark.parametrize("continuity", [-2, 3, 4])
def test_invalid_continuity_raises(continuity):
    x = np.linspace(0, 1, 50)
    with pytest.raises(ValueError):
        fit_spline(x, x, n_chunks=2, order=3, continuity=continuity)


def test_no_constraint_continuity_allowed():
    x = np.linspace(0, 1, 60)
    y = np.abs(x - 0.5)
    fit = fit_spline(x, y, n_chunks=3, order=2, continuity=-1)
    assert np.all(np.isfinite(fit.predict(x)))


def _assert_partition(bounds, n):
    assert bounds[0][0] == 0 and bounds[-1][1] == n
    for (s, e), (s2, _) in zip(bounds, bounds[1:]):
        assert s < e and e == s2
    assert all(s < e for s, e in bounds)


def test_adaptive_chunks_partition_and_tolerance():
    x = np.linspace(0.0, 10.0, 800)
    y = np.sin(x) * np.exp(-x / 5)
    order, tol = 3, 1e-3
    bounds = adaptive_chunks(x, y, order=order, tol=tol, max_chunks=200)
    _assert_partition(bounds, x.size)
    assert len(bounds) <= 200
    for s, e in bounds:
        if e - s >= 2 * (order + 1):
            p = np.polynomial.Chebyshev.fit(x[s:e], y[s:e], order)
            rmse = np.sqrt(np.mean((p(x[s:e]) - y[s:e]) ** 2))
            assert rmse <= tol * 1.0001


def test_adaptive_chunks_uses_few_chunks_for_polynomial():
    x = np.linspace(0.0, 1.0, 500)
    y = 2 * x**2 - x
    bounds = adaptive_chunks(x, y, order=2, tol=1e-9, max_chunks=10)
    _assert_partition(bounds, x.size)
    assert len(bounds) == 1


def test_adaptive_chunks_falls_back_within_max_chunks():
    x = np.linspace(0.0, 10.0, 300)
    y = np.sin(5 * x) + np.sign(np.sin(3 * x))
    bounds = adaptive_chunks(x, y, order=1, tol=1e-12, max_chunks=4)
    _assert_partition(bounds, x.size)
    assert len(bounds) <= 4
