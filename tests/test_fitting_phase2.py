import numpy as np
import pytest

from fitting import chunk_bounds, evaluate_piecewise, fit_chunks, rmse
from _phase2 import SplineFit, adaptive_chunks, fit_spline, robust_fit_chunks, select_order


def _sine(n=600, lo=0.0, hi=10.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(lo, hi, n)
    clean = np.sin(x)
    y = clean + noise * rng.standard_normal(n)
    return x, y, clean


# ---------- fit_spline: continuity, validation, derivatives ----------

@pytest.mark.parametrize("continuity", [0, 1, 2])
def test_spline_joins_are_continuous_to_requested_order(continuity):
    x, y, _ = _sine(noise=0.05)
    sp = fit_spline(x, y, n_chunks=4, order=5, continuity=continuity)
    eps = 1e-9
    for k in sp.knots:
        for m in range(continuity + 1):
            if m == 0:
                left = sp.predict(np.array([k - eps]))[0]
                right = sp.predict(np.array([k]))[0]
            else:
                left = sp.derivative(np.array([k - eps]), m)[0]
                right = sp.derivative(np.array([k]), m)[0]
            assert abs(left - right) < 1e-5, (k, m, left, right)


def test_spline_free_when_continuity_minus_one_has_jumps():
    x, y, _ = _sine(noise=0.05)
    sp = fit_spline(x, y, n_chunks=4, order=5, continuity=-1)
    jumps = [abs(sp.predict(np.array([k - 1e-9]))[0] - sp.predict(np.array([k]))[0]) for k in sp.knots]
    assert max(jumps) > 1e-6


def test_spline_knots_and_type():
    x, y, _ = _sine()
    sp = fit_spline(x, y, n_chunks=5, order=4, continuity=2)
    assert isinstance(sp, SplineFit)
    assert isinstance(sp.knots, np.ndarray)
    assert len(sp.knots) == 4
    assert np.allclose(sp.knots, [x[s] for s, _ in chunk_bounds(len(x), 5, 0.0)[1:]])


@pytest.mark.parametrize("continuity", [-2, 5, 6])
def test_spline_continuity_out_of_range_raises(continuity):
    x, y, _ = _sine()
    with pytest.raises(ValueError):
        fit_spline(x, y, n_chunks=3, order=5, continuity=continuity)


def test_spline_continuity_equal_to_order_raises():
    x, y, _ = _sine()
    with pytest.raises(ValueError):
        fit_spline(x, y, n_chunks=3, order=3, continuity=3)


def test_spline_continuity_order_minus_one_is_allowed():
    x, y, _ = _sine()
    fit_spline(x, y, n_chunks=3, order=3, continuity=2)  # -1 <= 2 < 3, must not raise


def test_spline_derivative_matches_finite_difference():
    x, y, _ = _sine(noise=0.0)
    sp = fit_spline(x, y, n_chunks=4, order=8, continuity=2)
    xq = np.linspace(0.3, 9.7, 200)
    knot_dist = np.min(np.abs(xq[:, None] - sp.knots[None, :]), axis=1)
    xq = xq[knot_dist > 0.05]
    h = 1e-5
    fd1 = (sp.predict(xq + h) - sp.predict(xq - h)) / (2 * h)
    fd2 = (sp.predict(xq + h) - 2 * sp.predict(xq) + sp.predict(xq - h)) / h**2
    assert np.max(np.abs(sp.derivative(xq, 1) - fd1)) < 1e-5
    assert np.max(np.abs(sp.derivative(xq, 2) - fd2)) < 1e-2


def test_spline_derivative_zero_order_is_predict_and_negative_m_raises():
    x, y, _ = _sine()
    sp = fit_spline(x, y, n_chunks=3, order=4, continuity=2)
    xq = np.linspace(0, 10, 50)
    assert np.allclose(sp.derivative(xq, 0), sp.predict(xq))
    with pytest.raises(ValueError):
        sp.derivative(xq, -1)


def test_spline_tracks_clean_sine():
    x, y, clean = _sine(noise=0.0)
    sp = fit_spline(x, y, n_chunks=4, order=8, continuity=2)
    assert rmse(clean, sp.predict(x)) < 1e-3


# ---------- fit_spline: stability at order 10 / 50 chunks ----------

def test_spline_order10_50_chunks_stable():
    x, y, clean = _sine(n=2000, lo=1000.0, hi=1010.0, noise=0.01, seed=1)
    sp = fit_spline(x, y, n_chunks=50, order=10, continuity=2)
    pred = sp.predict(x)
    assert np.all(np.isfinite(pred))
    assert rmse(clean, pred) < 0.05


def test_spline_order50_stable():
    x, y, clean = _sine(n=1500, lo=-5.0, hi=5.0, noise=0.0, seed=2)
    sp = fit_spline(x, y, n_chunks=5, order=50, continuity=2)
    pred = sp.predict(x)
    assert np.all(np.isfinite(pred))
    assert rmse(clean, pred) < 1e-6


# ---------- adaptive_chunks ----------

@pytest.mark.parametrize("order,tol,max_chunks", [(3, 0.05, 50), (5, 0.01, 50), (2, 0.2, 10)])
def test_adaptive_chunks_partition_tol_and_budget(order, tol, max_chunks):
    x, y, _ = _sine(n=800, noise=0.01, seed=3)
    parts = adaptive_chunks(x, y, order=order, tol=tol, max_chunks=max_chunks)
    assert 1 <= len(parts) <= max_chunks
    assert parts[0][0] == 0 and parts[-1][1] == len(x)
    for (a, b), (c, _d) in zip(parts, parts[1:]):
        assert b == c
    for s, e in parts:
        assert 0 <= s < e <= len(x)
        if e - s >= 2 * (order + 1):
            fit = fit_chunks(x[s:e], y[s:e], 1, order)[0]
            assert fit.rmse <= tol + 1e-12


def test_adaptive_chunks_respects_max_chunks_when_tol_is_hard():
    x, y, _ = _sine(n=800, noise=0.2, seed=4)
    parts = adaptive_chunks(x, y, order=1, tol=1e-6, max_chunks=4)
    assert len(parts) <= 4
    assert parts[0][0] == 0 and parts[-1][1] == len(x)


def test_adaptive_chunks_loose_tol_gives_one_chunk():
    x, y, _ = _sine(n=300, noise=0.0)
    parts = adaptive_chunks(x, y, order=3, tol=10.0, max_chunks=5)
    assert parts == [(0, len(x))]


# ---------- robust_fit_chunks ----------

def test_robust_fit_tracks_clean_sine_better_than_fit_chunks():
    rng = np.random.default_rng(5)
    x = np.linspace(0, 4 * np.pi, 400)
    clean = np.sin(x)
    y = clean + 0.02 * rng.standard_normal(len(x))
    idx = rng.choice(len(x), size=len(x) // 10, replace=False)
    y[idx] += rng.choice([-1.0, 1.0], size=len(idx)) * 5.0

    plain = evaluate_piecewise(fit_chunks(x, y, 3, 5), x, blend=False)
    robust = evaluate_piecewise(robust_fit_chunks(x, y, 3, 5), x, blend=False)
    assert rmse(clean, robust) < 0.5 * rmse(clean, plain)
    assert rmse(clean, robust) < 0.05


def test_robust_fit_returns_chunkfit_list_like_fit_chunks():
    x, y, _ = _sine(n=200, noise=0.01, seed=6)
    plain = fit_chunks(x, y, 4, 3, overlap_pct=10)
    robust = robust_fit_chunks(x, y, 4, 3, overlap_pct=10)
    assert len(robust) == len(plain)
    for p, r in zip(plain, robust):
        assert (r.start, r.stop) == (p.start, p.stop)
        assert r.x_min == p.x_min and r.x_max == p.x_max
        assert np.isfinite(r.rmse)
        assert np.all(np.isfinite(r.predict(np.linspace(r.x_min, r.x_max, 5))))


# ---------- select_order ----------

@pytest.mark.parametrize("method", ["cv", "aic"])
def test_select_order_does_not_underfit_a_cubic(method):
    # Underfitting a cubic is clearly penalised. Extra terms beyond the true
    # degree cost only ~noise^2 each, so CV/AIC may pick a few orders above 3.
    # CV on one chunk is high-variance, so check several seeds, not one.
    x = np.linspace(-1.0, 1.0, 300)
    picks = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        y = 0.5 - 2.0 * x + 3.0 * x**2 + 1.0 * x**3 + 0.1 * rng.standard_normal(len(x))
        order = select_order(x, y, n_chunks=1, max_order=8, method=method)
        assert isinstance(order, int)
        assert 0 <= order <= 8
        picks.append(order)
    assert min(picks) >= 3, picks
    assert float(np.median(picks)) <= 5, picks


def test_select_order_rejects_unknown_method():
    x, y, _ = _sine()
    with pytest.raises(ValueError):
        select_order(x, y, n_chunks=2, max_order=3, method="bogus")
