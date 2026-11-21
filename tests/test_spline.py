import numpy as np
import pytest

import fitting


def _jumps(s, m):
    return max(abs(s._eval_piece(i, k, m) - s._eval_piece(i + 1, k, m)) for i, k in enumerate(s.knots))


@pytest.mark.parametrize("continuity", [0, 1, 2])
def test_spline_continuity_at_knots(continuity):
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 500)
    y = np.sin(x) + 0.1 * rng.standard_normal(x.size)
    s = fitting.fit_spline(x, y, 8, 4, continuity)
    for m in range(continuity + 1):
        assert _jumps(s, m) < 1e-8
    assert _jumps(s, continuity + 1) > 1e-6


def test_spline_knots_and_right_piece():
    x = np.arange(10.0)
    s = fitting.fit_spline(x, x ** 2, 3, 2, 1)
    np.testing.assert_array_equal(s.knots, [4.0, 7.0])
    assert s.piece_index(4.0) == 1


def test_spline_unconstrained_matches_fit_chunks():
    rng = np.random.default_rng(2)
    x = np.linspace(-3, 3, 400)
    y = np.cos(2 * x) + 0.05 * rng.standard_normal(x.size)
    s = fitting.fit_spline(x, y, 5, 3, -1)
    ref = fitting.evaluate_piecewise(fitting.fit_chunks(x, y, 5, 3), x, blend=False)
    np.testing.assert_allclose(s.predict(x), ref, atol=1e-9)


def test_spline_is_joint_least_squares():
    """Constrained fit cannot beat the unconstrained one, and C0 beats C2."""
    rng = np.random.default_rng(3)
    x = np.linspace(0, 1, 300)
    y = np.abs(x - 0.4) + 0.01 * rng.standard_normal(x.size)
    errs = [fitting.rmse(y, fitting.fit_spline(x, y, 6, 3, c).predict(x)) for c in (-1, 0, 1, 2)]
    assert errs == sorted(errs)


def test_spline_reproduces_global_polynomial_and_derivatives():
    x = np.linspace(-1, 2, 300)
    y = x ** 3 - x
    s = fitting.fit_spline(x, y, 6, 3, 2)
    np.testing.assert_allclose(s.predict(x), y, atol=1e-10)
    np.testing.assert_allclose(s.derivative(x, 1), 3 * x ** 2 - 1, atol=1e-9)
    np.testing.assert_allclose(s.derivative(x, 2), 6 * x, atol=1e-8)
    np.testing.assert_allclose(s.derivative(x, 0), y, atol=1e-10)


def test_spline_stable_high_order_many_chunks():
    rng = np.random.default_rng(4)
    x = np.linspace(1000, 1001, 1000)
    y = np.sin(20 * x) + 0.01 * rng.standard_normal(x.size)
    s = fitting.fit_spline(x, y, 50, 10, 2)
    assert fitting.rmse(y, s.predict(x)) < 0.02
    assert _jumps(s, 0) < 1e-9
    assert np.all(np.isfinite(s.derivative(x, 2)))


@pytest.mark.parametrize("continuity", [-2, 3, 5])
def test_spline_rejects_bad_continuity(continuity):
    x = np.linspace(0, 1, 50)
    with pytest.raises(ValueError):
        fitting.fit_spline(x, x, 2, 3, continuity)
