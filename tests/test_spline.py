import numpy as np
import pytest

from fitting import Polynomial, chunk_bounds, rmse
from fitting import SplineFit, adaptive_chunks, fit_spline

X = np.linspace(0, 10, 400)
Y = np.sin(X)


def test_fit_spline_reproduces_smooth_function():
    sp = fit_spline(X, Y, n_chunks=8, order=5, continuity=2)
    assert isinstance(sp, SplineFit)
    assert rmse(Y, sp.predict(X)) < 1e-4


def test_continuity_at_knots():
    sp = fit_spline(X, Y, n_chunks=6, order=5, continuity=2)
    eps = 1e-9
    for k in sp.knots:
        for m in (0, 1, 2):
            left = sp.derivative(np.array([k - eps]), m)[0]
            right = sp.derivative(np.array([k + eps]), m)[0]
            assert abs(left - right) < 1e-6, (k, m, left, right)


@pytest.mark.parametrize("continuity", [5, 6])
def test_continuity_ge_order_raises(continuity):
    with pytest.raises(ValueError):
        fit_spline(X, Y, n_chunks=4, order=5, continuity=continuity)


def test_continuity_below_minus_one_raises():
    with pytest.raises(ValueError):
        fit_spline(X, Y, n_chunks=4, order=5, continuity=-2)


def test_knots_at_expected_x_values():
    sp = fit_spline(X, Y, n_chunks=4, order=5, continuity=2)
    assert len(sp.knots) == 3
    expected = [X[s] for s, _ in chunk_bounds(len(X), 4, 0.0)[1:]]
    assert np.allclose(sp.knots, expected)
    assert np.allclose(sp.knots, [X[100], X[200], X[300]])


def test_single_chunk_has_no_knots():
    sp = fit_spline(X, Y, n_chunks=1, order=5, continuity=2)
    assert sp.knots.size == 0
    baseline = Polynomial.fit(X, Y, 5)(X)
    assert np.allclose(sp.predict(X), baseline, atol=1e-6)


def test_derivative_zero_equals_predict():
    sp = fit_spline(X, Y, n_chunks=5, order=5, continuity=2)
    xq = np.linspace(0, 10, 37)
    assert np.allclose(sp.derivative(xq, 0), sp.predict(xq))


def test_first_derivative_approximates_cos():
    sp = fit_spline(X, Y, n_chunks=8, order=5, continuity=2)
    xq = np.linspace(0.5, 9.5, 50)
    assert rmse(np.cos(xq), sp.derivative(xq, 1)) < 1e-2


def test_continuity_minus_one_has_no_constraints():
    sp = fit_spline(X, Y, n_chunks=4, order=5, continuity=-1)
    assert np.all(np.isfinite(sp.predict(X)))
    assert rmse(Y, sp.predict(X)) < 1e-3


def test_stability_order10_50_chunks():
    x = np.linspace(0, 10, 300)
    sp = fit_spline(x, np.sin(x), n_chunks=50, order=10, continuity=2)
    assert np.all(np.isfinite(sp.predict(x)))
    assert np.all(np.isfinite(sp.derivative(x, 2)))


def test_adaptive_contiguous_cover():
    ch = adaptive_chunks(X, Y, order=3, tol=1e-3, max_chunks=1000)
    assert ch[0][0] == 0
    assert ch[-1][1] == len(X)
    for (a, b), (c, d) in zip(ch, ch[1:]):
        assert b == c
        assert a < b
    assert ch[-1][0] < ch[-1][1]


def test_adaptive_small_tol_uses_more_chunks():
    small = adaptive_chunks(X, Y, order=3, tol=1e-4, max_chunks=1000)
    large = adaptive_chunks(X, Y, order=3, tol=1e-1, max_chunks=1000)
    assert len(small) > len(large)


def test_adaptive_respects_max_chunks_and_covers():
    ch = adaptive_chunks(X, Y, order=3, tol=1e-9, max_chunks=5)
    assert len(ch) <= 5
    assert ch[0][0] == 0 and ch[-1][1] == len(X)
    for (_, b), (c, _) in zip(ch, ch[1:]):
        assert b == c


def test_adaptive_chunks_meet_tol_when_large_enough():
    order, tol = 3, 1e-3
    ch = adaptive_chunks(X, Y, order=order, tol=tol, max_chunks=1000)
    for a, b in ch:
        if b - a >= 2 * (order + 1):
            p = Polynomial.fit(X[a:b], Y[a:b], order)
            assert rmse(Y[a:b], p(X[a:b])) <= tol


def test_adaptive_empty_input():
    assert adaptive_chunks(np.array([]), np.array([]), 3, 1e-3, 5) == []
