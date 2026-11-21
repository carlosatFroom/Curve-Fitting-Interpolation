import numpy as np
import pytest

import fitting
from fitting import adaptive_chunks, robust_fit_chunks, select_order


def _sine_with_outliers(n=400, frac=0.10, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 4 * np.pi, n)
    clean = np.sin(x)
    y = clean.copy()
    n_out = int(frac * n)
    idx = rng.choice(n, size=n_out, replace=False)
    signs = rng.choice([-1.0, 1.0], size=n_out)
    y[idx] += signs * rng.uniform(2.0, 4.0, size=n_out)
    return x, y, clean


def test_adaptive_chunks_partition_and_tolerance():
    n = 500
    x = np.linspace(0.0, 10.0, n)
    y = np.sin(x) + 0.5 * np.cos(2 * x)
    order, tol, max_chunks = 3, 1e-2, 50

    chunks = adaptive_chunks(x, y, order, tol, max_chunks)

    assert 1 <= len(chunks) <= max_chunks
    assert chunks[0][0] == 0
    assert chunks[-1][1] == n
    for (a, b), (c, _) in zip(chunks, chunks[1:]):
        assert b == c
    for a, b in chunks:
        assert a < b
        if b - a >= 2 * (order + 1):
            _, _, err = fitting._fit_one(x[a:b], y[a:b], order)
            assert err <= tol


def test_adaptive_chunks_respects_max_chunks_and_covers():
    n = 300
    x = np.linspace(0.0, 10.0, n)
    y = np.sin(3 * x)
    chunks = adaptive_chunks(x, y, 2, 1e-6, 4)

    assert len(chunks) <= 4
    assert chunks[0][0] == 0 and chunks[-1][1] == n
    assert all(b == c for (_, b), (c, _) in zip(chunks, chunks[1:]))


def test_robust_fit_beats_plain_fit_on_outliers():
    x, y, clean = _sine_with_outliers()
    n_chunks, order = 4, 3

    plain = fitting.fit_chunks(x, y, n_chunks, order)
    robust = robust_fit_chunks(x, y, n_chunks, order)

    assert len(robust) == n_chunks
    assert all(isinstance(f, fitting.ChunkFit) for f in robust)

    err_plain = fitting.rmse(clean, fitting.evaluate_piecewise(plain, x, blend=False))
    err_robust = fitting.rmse(clean, fitting.evaluate_piecewise(robust, x, blend=False))
    assert err_robust < 0.5 * err_plain


@pytest.mark.parametrize("method", ["cv", "aic"])
def test_select_order_low_degree_polynomial(method):
    rng = np.random.default_rng(1)
    x = np.linspace(-1.0, 1.0, 200)
    y = 1.0 + 2.0 * x - 3.0 * x ** 2 + 0.01 * rng.standard_normal(len(x))

    best = select_order(x, y, n_chunks=1, max_order=6, method=method)

    assert isinstance(best, int)
    assert 0 <= best <= 6
    assert best <= 3


def test_select_order_unknown_method():
    x = np.linspace(0, 1, 50)
    with pytest.raises(ValueError):
        select_order(x, x, 1, 3, method="bogus")
