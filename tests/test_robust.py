import numpy as np
import pytest

from fitting import fit_chunks
from fitting import robust_fit_chunks, select_order


def _chunk_predict(fits, x):
    out = np.empty_like(x)
    for f in fits:
        out[f.start:f.stop] = f.predict(x[f.start:f.stop])
    return out


def _cubic(x):
    return 1.5 * x**3 - 2.0 * x**2 + 0.5 * x + 1.0


def test_robust_tracks_clean_signal_with_outliers():
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 400)
    clean = _cubic(x)
    y = clean + rng.normal(0.0, 0.01, x.size)
    idx = rng.choice(x.size, size=x.size // 10, replace=False)
    y[idx] += rng.choice([-1.0, 1.0], size=idx.size) * rng.uniform(3.0, 6.0, idx.size)

    robust = _chunk_predict(robust_fit_chunks(x, y, 2, 3), x)
    plain = _chunk_predict(fit_chunks(x, y, 2, 3), x)

    robust_err = np.max(np.abs(robust - clean))
    plain_err = np.max(np.abs(plain - clean))
    assert robust_err < 0.05
    assert plain_err > 5 * robust_err


def test_robust_no_outliers_close_to_plain_fit():
    rng = np.random.default_rng(1)
    x = np.linspace(0.0, 1.0, 300)
    y = _cubic(x) + rng.normal(0.0, 0.01, x.size)
    robust = _chunk_predict(robust_fit_chunks(x, y, 3, 3), x)
    plain = _chunk_predict(fit_chunks(x, y, 3, 3), x)
    assert np.max(np.abs(robust - plain)) < 0.02


def test_robust_small_chunk_reduces_order():
    x = np.arange(6, dtype=float)
    y = 2.0 * x + 1.0
    fits = robust_fit_chunks(x, y, 3, 5)
    assert len(fits) == 3
    for f in fits:
        assert f.rmse < 1e-8
        assert len(f.coeffs) <= (f.stop - f.start)


@pytest.mark.parametrize("method", ["cv", "aic"])
def test_select_order_picks_true_order_on_noisy_cubic(method):
    rng = np.random.default_rng(2)
    x = np.linspace(-1.0, 1.0, 400)
    y = _cubic(x) + rng.normal(0.0, 0.05, x.size)
    assert select_order(x, y, 1, 8, method=method) == 3


@pytest.mark.parametrize("method", ["cv", "aic"])
def test_select_order_returns_valid_range(method):
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 10.0, 60)
    y = np.sin(x) + rng.normal(0.0, 0.1, x.size)
    for n_chunks in (1, 4, 20):
        order = select_order(x, y, n_chunks, 6, method=method)
        assert isinstance(order, int)
        assert 0 <= order <= 6


def test_select_order_tiny_chunks_still_returns_valid():
    x = np.arange(8, dtype=float)
    y = x**2
    order = select_order(x, y, 8, 4, method="cv")
    assert 0 <= order <= 4


def test_select_order_unknown_method_raises():
    x = np.linspace(0.0, 1.0, 50)
    with pytest.raises(ValueError):
        select_order(x, x, 2, 3, method="bic")
