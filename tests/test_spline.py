"""Tests for the constrained joint spline (fit_spline / SplineFit)."""

import numpy as np
import pytest

import fitting


def _data(n=800, noise=0.0, seed=0):
    rng = np.random.RandomState(seed)
    x = np.linspace(0.0, 10.0, n)
    y = np.sin(x) + noise * rng.randn(n)
    return x, y


def test_validation():
    x, y = _data()
    with pytest.raises(ValueError):
        fitting.fit_spline(x, y, 4, order=3, continuity=3)  # continuity == order
    with pytest.raises(ValueError):
        fitting.fit_spline(x, y, 4, order=3, continuity=-2)  # < -1
    with pytest.raises(ValueError):
        fitting.fit_spline(x, y, 4, order=0, continuity=0)  # continuity >= order


def test_knots():
    x, y = _data(n=100)
    n_chunks = 5
    sf = fitting.fit_spline(x, y, n_chunks, order=3, continuity=2)
    base = fitting.chunk_bounds(len(x), n_chunks, 0.0)
    expected = np.array([x[base[i + 1][0]] for i in range(n_chunks - 1)])
    assert sf.knots.shape == (n_chunks - 1,)
    np.testing.assert_allclose(sf.knots, expected)


@pytest.mark.parametrize("continuity", [0, 1, 2])
def test_continuity_enforced(continuity):
    x, y = _data(noise=0.02)
    sf = fitting.fit_spline(x, y, 6, order=4, continuity=continuity)
    eps = 1e-6
    for k in sf.knots:
        for m in range(continuity + 1):
            left = float(sf.derivative(k - eps, m))
            right = float(sf.derivative(k + eps, m))
            assert abs(left - right) < 1e-3, (continuity, m, k)


def test_fits_signal_well():
    x, y = _data(noise=0.01)
    sf = fitting.fit_spline(x, y, 8, order=4, continuity=2)
    assert fitting.rmse(y, sf.predict(x)) < 0.05


def test_high_order_many_chunks_stable():
    x, y = _data(n=1000, noise=0.01)
    sf = fitting.fit_spline(x, y, 50, order=10, continuity=2)
    pred = sf.predict(x)
    assert np.all(np.isfinite(pred))
    assert fitting.rmse(y, pred) < 0.1


def test_derivative_matches_known():
    x = np.linspace(0.0, 10.0, 1000)
    y = np.sin(x)
    sf = fitting.fit_spline(x, y, 8, order=6, continuity=2)
    xt = np.linspace(0.5, 9.5, 40)
    d1 = sf.derivative(xt, 1)
    assert np.max(np.abs(d1 - np.cos(xt))) < 1e-2


def test_no_constraint_matches_independent_fit():
    # continuity == -1 should just be independent per-chunk least squares.
    x, y = _data(noise=0.01)
    sf = fitting.fit_spline(x, y, 5, order=3, continuity=-1)
    assert sf.knots.shape == (4,)
    assert fitting.rmse(y, sf.predict(x)) < 0.05
