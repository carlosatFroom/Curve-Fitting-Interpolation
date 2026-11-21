import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fitting import ChunkFit, evaluate_piecewise, fit_chunks, rmse  # noqa: E402
from fitting import robust_fit_chunks, select_order  # noqa: E402


def _corrupt(y, frac, rng, scale=(5.0, 10.0)):
    yc = y.copy()
    n_bad = int(len(y) * frac)
    idx = rng.choice(len(y), n_bad, replace=False)
    yc[idx] += rng.choice([-1.0, 1.0], n_bad) * rng.uniform(*scale, n_bad)
    return yc


def test_robust_beats_plain_fit_on_outliers():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 200)
    clean = np.sin(x) + 0.5 * x
    corrupted = _corrupt(clean, 0.10, rng)

    robust = robust_fit_chunks(x, corrupted, n_chunks=4, order=3)
    plain = fit_chunks(x, corrupted, n_chunks=4, order=3)
    err_robust = rmse(clean, evaluate_piecewise(robust, x, blend=False))
    err_plain = rmse(clean, evaluate_piecewise(plain, x, blend=False))
    print(f"\noutlier rmse vs clean: robust={err_robust:.4f} plain={err_plain:.4f}")
    assert err_robust < 0.5 * err_plain


def test_robust_fit_contract():
    x = np.linspace(0, 1, 60)
    y = 2 + 3 * x - x**2
    fits = robust_fit_chunks(x, y, n_chunks=3, order=2, overlap_pct=10)
    assert len(fits) == 3
    assert all(isinstance(f, ChunkFit) for f in fits)
    for f in fits:
        assert len(f.coeffs) == 3
        assert f.base_start < f.base_stop
        assert f.base_x_min == x[f.base_start]
        assert f.base_x_max == x[f.base_stop - 1]
    # clean quadratic: highest-first coeffs in original x
    np.testing.assert_allclose(fits[0].coeffs, [-1, 3, 2], atol=1e-6)
    np.testing.assert_allclose(fits[1].predict(x[30:31]), y[30:31], atol=1e-6)


def test_robust_order_reduction_for_tiny_chunk():
    x = np.arange(5, dtype=float)
    y = x * 2.0
    fits = robust_fit_chunks(x, y, n_chunks=5, order=3)
    assert len(fits) == 5
    assert all(len(f.coeffs) == 1 for f in fits)  # 1 point -> order 0
    assert np.all(np.isfinite(fits[2].predict([2.0])))


def test_select_order_recovers_degree():
    rng = np.random.default_rng(1)
    x = np.linspace(-1, 1, 300)
    y = 1 - 2 * x + 4 * x**3 + rng.normal(0, 0.02, x.size)
    d = 3
    cv = select_order(x, y, n_chunks=2, max_order=8, method="cv")
    aic = select_order(x, y, n_chunks=2, max_order=8, method="aic")
    print(f"\nselected orders: cv={cv} aic={aic} (true d={d})")
    for o in (cv, aic):
        assert isinstance(o, int)
        assert 0 <= o <= 8
        assert d <= o <= d + 2


def test_select_order_unknown_method():
    x = np.linspace(0, 1, 50)
    with pytest.raises(ValueError):
        select_order(x, x, n_chunks=2, max_order=3, method="bogus")
