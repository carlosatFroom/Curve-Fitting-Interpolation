import numpy as np
import pytest

from fitting import chunk_bounds, fit_chunks, fit_spline, rmse


def _knot_jump(fit, k, m):
    """Difference of m-th derivative just left and right of knot k (x = knot)."""
    xk = fit.knots[k]
    eps = 1e-9 * max(1.0, abs(xk))
    left = fit.derivative(np.array([xk - eps]), m)[0]
    right = fit.derivative(np.array([xk]), m)[0]
    return abs(left - right), max(1.0, abs(right))


def test_global_cubic_recovered_exactly():
    x = np.linspace(-3, 5, 200)
    y = 2 * x**3 - x**2 + 0.5 * x - 4
    fit = fit_spline(x, y, n_chunks=4, order=3, continuity=2)
    np.testing.assert_allclose(fit.predict(x), y, atol=1e-7)


def test_continuity_at_knots():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 300)
    y = np.sin(x) + 0.05 * rng.standard_normal(x.size)
    fit = fit_spline(x, y, n_chunks=5, order=6, continuity=2)
    assert len(fit.knots) == 4
    for k in range(len(fit.knots)):
        for m in range(3):
            jump, scale = _knot_jump(fit, k, m)
            assert jump <= 1e-6 * scale, (k, m, jump)


def test_continuity_minus_one_matches_independent_chunk_fits():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 100, 400)
    y = np.cos(x / 7) + 0.1 * rng.standard_normal(x.size)
    fit = fit_spline(x, y, n_chunks=4, order=4, continuity=-1)
    expected_sse = 0.0
    for cf, (s, e) in zip(fit_chunks(x, y, 4, 4), chunk_bounds(len(x), 4, 0.0)):
        expected_sse += np.sum((y[s:e] - cf.predict(x[s:e])) ** 2)
    sse = np.sum((y - fit.predict(x)) ** 2)
    assert sse == pytest.approx(expected_sse, rel=1e-8)


def test_constrained_sse_is_monotone_in_continuity():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 20, 300)
    y = np.sin(x) + 0.2 * rng.standard_normal(x.size)
    sses = []
    for c in (-1, 0, 1, 2):
        fit = fit_spline(x, y, n_chunks=6, order=5, continuity=c)
        sses.append(np.sum((y - fit.predict(x)) ** 2))
    for looser, tighter in zip(sses, sses[1:]):
        assert tighter >= looser - 1e-9


@pytest.mark.parametrize("continuity", [-2, 3, 4])
def test_bad_continuity_raises(continuity):
    x = np.linspace(0, 1, 50)
    with pytest.raises(ValueError):
        fit_spline(x, x, n_chunks=3, order=3, continuity=continuity)


@pytest.mark.parametrize("n_chunks", [0, 51])
def test_bad_n_chunks_raises(n_chunks):
    x = np.linspace(0, 1, 50)
    with pytest.raises(ValueError):
        fit_spline(x, x, n_chunks=n_chunks, order=2)


def test_stability_order10_50chunks():
    x = np.linspace(0, 1000, 1000)
    y = np.sin(x)
    fit = fit_spline(x, y, n_chunks=50, order=10, continuity=2)
    pred = fit.predict(x)
    assert np.all(np.isfinite(pred))
    # unconstrained per-chunk optimum is ~0.09; C2 constrained optimum is ~0.26
    assert rmse(y, pred) < 0.3


def test_scalar_and_array_queries_agree():
    x = np.linspace(0, 10, 100)
    y = x**2
    fit = fit_spline(x, y, n_chunks=3, order=3, continuity=2)
    assert fit.predict(4.2) == pytest.approx(fit.predict(np.array([4.2]))[0])
    grid = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert fit.predict(grid).shape == (2, 2)


def test_extrapolation_uses_end_pieces():
    x = np.linspace(0, 10, 100)
    y = 3 * x - 1  # global line: end pieces should continue it
    fit = fit_spline(x, y, n_chunks=4, order=2, continuity=1)
    np.testing.assert_allclose(fit.predict(np.array([-5.0, 15.0])), [-16.0, 44.0], atol=1e-6)


@pytest.mark.parametrize("m", [1, 2])
def test_derivative_matches_finite_differences(m):
    x = np.linspace(0, 6, 240)
    y = np.sin(x)
    fit = fit_spline(x, y, n_chunks=4, order=7, continuity=2)
    xq = np.linspace(0.3, 5.7, 25)
    h = 1e-5
    if m == 1:
        fd = (fit.predict(xq + h) - fit.predict(xq - h)) / (2 * h)
    else:
        fd = (fit.predict(xq + h) - 2 * fit.predict(xq) + fit.predict(xq - h)) / h**2
    np.testing.assert_allclose(fit.derivative(xq, m), fd, atol=1e-4)
    np.testing.assert_allclose(fit.derivative(xq, 0), fit.predict(xq))


def test_negative_derivative_order_raises():
    x = np.linspace(0, 1, 20)
    fit = fit_spline(x, x, n_chunks=2, order=2, continuity=1)
    with pytest.raises(ValueError):
        fit.derivative(0.5, -1)
