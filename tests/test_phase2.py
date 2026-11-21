import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fitting import (  # noqa: E402
    adaptive_chunks,
    fit_chunks,
    evaluate_piecewise,
    rmse,
    robust_fit_chunks,
    select_order,
)


def _piecewise_signal(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n)
    y = np.where(x < 3, np.sin(x), np.where(x < 7, 0.5 * x - 1.0, 2.0 + np.cos(2 * x)))
    return x, y + rng.normal(0, 0.01, n)


def test_adaptive_chunks_contiguous_partition():
    x, y = _piecewise_signal()
    parts = adaptive_chunks(x, y, order=3, tol=0.05, max_chunks=50)
    assert parts[0][0] == 0 and parts[-1][1] == len(x)
    for (_, stop), (start, _) in zip(parts, parts[1:]):
        assert stop == start
    assert all(s < e for s, e in parts)


def test_adaptive_chunks_respects_tol_on_piecewise_signal():
    x, y = _piecewise_signal()
    tol = 0.05
    parts = adaptive_chunks(x, y, order=3, tol=tol, max_chunks=50)
    assert len(parts) > 1
    min_len = 2 * (3 + 1)
    for s, e in parts:
        if e - s >= min_len:
            coeffs = np.polyfit(x[s:e], y[s:e], 3)
            assert rmse(y[s:e], np.polyval(coeffs, x[s:e])) <= tol + 1e-9


def test_adaptive_chunks_respects_max_chunks():
    x, y = _piecewise_signal()
    parts = adaptive_chunks(x, y, order=1, tol=1e-6, max_chunks=4)
    assert len(parts) <= 4
    assert parts[0][0] == 0 and parts[-1][1] == len(x)


def test_adaptive_chunks_single_chunk_for_exact_polynomial():
    x = np.linspace(-2, 2, 200)
    y = 1.0 - 2.0 * x + 0.5 * x ** 3
    assert adaptive_chunks(x, y, order=3, tol=1e-6, max_chunks=10) == [(0, 200)]


def test_adaptive_chunks_validates_args():
    x = np.linspace(0, 1, 50)
    with pytest.raises(ValueError):
        adaptive_chunks(x, x, 2, 0.1, 0)
    with pytest.raises(ValueError):
        adaptive_chunks(x, x, 2, -0.1, 5)


def test_robust_fit_tracks_clean_signal_with_outliers():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 4, 400)
    clean = np.sin(x) + 0.5 * x
    y = clean + rng.normal(0, 0.02, x.size)
    bad = rng.choice(x.size, size=x.size // 10, replace=False)
    y[bad] += rng.choice([-1.0, 1.0], size=bad.size) * rng.uniform(3, 6, bad.size)

    robust = robust_fit_chunks(x, y, n_chunks=1, order=5)
    plain = fit_chunks(x, y, n_chunks=1, order=5)
    err_robust = rmse(clean, evaluate_piecewise(robust, x, blend=False))
    err_plain = rmse(clean, evaluate_piecewise(plain, x, blend=False))
    assert err_robust < 0.05
    assert err_robust < 0.2 * err_plain


@pytest.mark.parametrize("method", ["cv", "aic"])
def test_select_order_recovers_cubic(method):
    rng = np.random.default_rng(2)
    x = np.linspace(-1, 1, 400)
    y = 0.5 - 1.5 * x + 2.0 * x ** 2 - 0.8 * x ** 3 + rng.normal(0, 0.2, x.size)
    assert select_order(x, y, n_chunks=1, max_order=8, method=method) == 3


def test_select_order_bad_method_raises():
    x = np.linspace(0, 1, 50)
    with pytest.raises(ValueError):
        select_order(x, x, n_chunks=1, max_order=3, method="bogus")
