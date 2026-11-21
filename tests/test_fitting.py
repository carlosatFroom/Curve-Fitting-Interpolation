import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import fitting as F  # noqa: E402


def sample(name):
    d = np.genfromtxt(os.path.join(ROOT, "samples", name), delimiter=",")[:1000]
    return d[:, 0], d[:, 1]


@pytest.fixture
def sine():
    x = np.linspace(0, 10, 1000)
    rng = np.random.default_rng(0)
    return x, np.sin(x) + 0.01 * rng.normal(size=x.size)


# --------------------------------------------------------------- import hygiene

def test_import_has_no_gui_deps():
    code = ("import sys, fitting; bad = [m for m in ('PyQt5', 'pyqtgraph', 'matplotlib') "
            "if m in sys.modules]; assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


# ---------------------------------------------------------------- chunk_bounds

@pytest.mark.parametrize("n,k", [(10, 3), (1000, 7), (5, 5), (11, 1), (13, 4)])
def test_chunk_bounds_matches_array_split(n, k):
    expected = [(int(a[0]), int(a[-1]) + 1) for a in np.array_split(np.arange(n), k)]
    assert F.chunk_bounds(n, k) == expected


def test_chunk_bounds_overlap_rules():
    # base: (0,4) (4,7) (7,10); ov = int(len * 50/100) -> 2, 1, 1
    assert F.chunk_bounds(10, 3, 50) == [(0, 6), (4, 8), (6, 10)]
    # 4 chunks of 25 with 20% -> ov 5
    assert F.chunk_bounds(100, 4, 20) == [(0, 30), (23, 53), (48, 78), (70, 100)]


def test_chunk_bounds_clipped():
    b = F.chunk_bounds(10, 2, 300)
    assert all(0 <= s < e <= 10 for s, e in b)
    assert b == [(0, 10), (0, 10)]


@pytest.mark.parametrize("k", [0, -1, 11])
def test_chunk_bounds_invalid(k):
    with pytest.raises(ValueError):
        F.chunk_bounds(10, k)


# ------------------------------------------------------------------ fit_chunks

def test_fit_chunks_unequal_chunks_use_correct_slices():
    # 10 points in 3 chunks -> lengths 4,3,3 (the old code sliced at i*len(chunk)).
    x = np.arange(10.0)
    y = np.where(x < 4, 1.0, np.where(x < 7, 2.0, 3.0))
    fits = F.fit_chunks(x, y, 3, 0)
    assert [(f.start, f.stop) for f in fits] == [(0, 4), (4, 7), (7, 10)]
    assert [f.coeffs[-1] for f in fits] == pytest.approx([1, 2, 3])
    assert all(f.rmse == pytest.approx(0, abs=1e-12) for f in fits)


def test_fit_chunks_exact_polynomial_and_coeffs():
    x = np.linspace(-3, 7, 200)
    true = [0.5, -2.0, 1.0, 3.0]
    y = np.polyval(true, x)
    (fit,) = F.fit_chunks(x, y, 1, 3)
    assert fit.coeffs == pytest.approx(true, rel=1e-8, abs=1e-8)
    assert fit.x_min == x[0] and fit.x_max == x[-1]
    assert fit.predict(x) == pytest.approx(y, abs=1e-9)


def test_fit_chunks_high_order_is_stable_on_large_x():
    # Unscaled polyfit is badly conditioned here; the scaled fit is not.
    x = np.linspace(1e4, 1e4 + 1, 1000)
    y = np.sin(2 * np.pi * (x - 1e4))
    (fit,) = F.fit_chunks(x, y, 1, 10)
    assert F.rmse(y, fit.predict(x)) < 1e-4


def test_fit_chunks_reduces_order_on_short_chunks():
    x = np.arange(6.0)
    y = x ** 2
    fits = F.fit_chunks(x, y, 3, 5)  # 2 points per chunk
    assert all(len(f.coeffs) == 2 for f in fits)
    assert all(f.rmse == pytest.approx(0, abs=1e-12) for f in fits)


def test_fit_chunks_overlap_slices():
    x, y = sample("sine.csv")
    fits = F.fit_chunks(x, y, 4, 3, overlap_pct=20)
    assert [(f.start, f.stop) for f in fits] == F.chunk_bounds(1000, 4, 20)


# ---------------------------------------------------------- evaluate_piecewise

def test_evaluate_piecewise_no_blend_uses_base_chunk():
    x = np.arange(10.0)
    y = np.where(x < 5, 0.0, 10.0)
    fits = F.fit_chunks(x, y, 2, 0, overlap_pct=40)
    out = F.evaluate_piecewise(fits, x, blend=False)
    # knot at x=5 belongs to the right piece
    assert out[4] == pytest.approx(fits[0].predict(4.0))
    assert out[5] == pytest.approx(fits[1].predict(5.0))


def test_evaluate_piecewise_blend_is_continuous(sine):
    x, y = sine
    fits = F.fit_chunks(x, y, 5, 2, overlap_pct=30)
    xq = np.linspace(x[0], x[-1], 200001)
    hard = F.evaluate_piecewise(fits, xq, blend=False)
    soft = F.evaluate_piecewise(fits, xq, blend=True)
    step = np.diff(xq)[0]
    # blended curve: no jumps beyond what the slope allows
    assert np.max(np.abs(np.diff(soft))) < 50 * step
    assert np.max(np.abs(np.diff(hard))) > np.max(np.abs(np.diff(soft)))
    # far from seams both agree
    assert soft[0] == hard[0] and soft[-1] == hard[-1]


def test_evaluate_piecewise_preserves_shape(sine):
    x, y = sine
    fits = F.fit_chunks(x, y, 3, 3)
    assert F.evaluate_piecewise(fits, np.ones((2, 3))).shape == (2, 3)


# --------------------------------------------------------------------- metrics

def test_rmse_and_r2():
    assert F.rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))
    assert F.r2([1, 2, 3], [1, 2, 3]) == 1.0
    assert F.r2([1, 2, 3], [2, 2, 2]) == pytest.approx(0.0)
    assert F.r2([2, 2], [2, 2]) == 1.0


# ------------------------------------------------------------------- error_map

def brute_rmse(x, y, order, n_chunks, ov):
    fits = F.fit_chunks(x, y, n_chunks, order, ov)
    return F.rmse(y, F.evaluate_piecewise(fits, x, blend=False))


def test_error_map_shapes_values_and_axes():
    x, y = sample("ECG2.csv")
    m = F.error_map(x, y, [1, 2, 3], [2, 4], [0])
    assert m.shape == (3, 2)
    assert m[2, 1] == pytest.approx(brute_rmse(x, y, 3, 4, 0))
    m2 = F.error_map(x, y, [4], [1, 3, 5], [0, 10])
    assert m2.shape == (3, 2)
    assert m2[1, 1] == pytest.approx(brute_rmse(x, y, 4, 3, 10))
    m3 = F.error_map(x, y, [1, 2, 3, 4], [3], [0, 25])
    assert m3.shape == (4, 2)
    assert m3[3, 1] == pytest.approx(brute_rmse(x, y, 4, 3, 25))


def test_error_map_is_unnormalized_rmse_and_workers_agree():
    x, y = sample("sine.csv")
    a = F.error_map(x, y, range(1, 6), range(1, 6), [0], workers=1)
    b = F.error_map(x, y, range(1, 6), range(1, 6), [0], workers=4)
    assert np.allclose(a, b)
    assert a.max() < 100 and not np.isclose(a.max(), 100)
    # more chunks or higher order never hurts much on a smooth signal
    assert a[-1, -1] < a[0, 0]


@pytest.mark.parametrize("args", [([1], [1], [0]), ([1, 2], [1, 2], [0, 5]), ([1, 2], [], [0])])
def test_error_map_rejects_bad_axes(args):
    x, y = sample("sine.csv")
    with pytest.raises(ValueError):
        F.error_map(x, y, *args)


def test_error_map_speed():
    x, y = sample("ECG2.csv")
    t = time.perf_counter()
    F.error_map(x, y, range(1, 11), range(1, 11), [0])
    assert time.perf_counter() - t < 2.0


# ----------------------------------------------------------------- extrapolate

def test_extrapolate_uses_prefix_only():
    x = np.linspace(0, 1, 100)
    y = np.where(x < 0.5, 2 * x, 100.0)  # junk after the fitted half
    coeffs, pred = F.extrapolate(x, y, 0.5, 1)
    assert coeffs == pytest.approx([2, 0], abs=1e-9)
    assert pred == pytest.approx(2 * x, abs=1e-9)
    assert pred.shape == x.shape


def test_extrapolate_min_points_and_validation():
    x = np.arange(20.0)
    y = x ** 3
    coeffs, _ = F.extrapolate(x, y, 0.01, 3)  # 0.2 points -> uses order+1 = 4
    assert coeffs == pytest.approx([1, 0, 0, 0], abs=1e-6)
    with pytest.raises(ValueError):
        F.extrapolate(x, y, 0, 3)
    with pytest.raises(ValueError):
        F.extrapolate(x, y, 1.5, 3)


# ------------------------------------------------------------------ fit_spline

@pytest.mark.parametrize("cont", [-1, 0, 1, 2])
def test_spline_continuity(cont, sine):
    x, y = sine
    s = F.fit_spline(x, y, 8, 4, continuity=cont)
    assert len(s.knots) == 7
    assert s.knots[0] == x[F.chunk_bounds(len(x), 8)[1][0]]
    eps = 1e-7
    for k in s.knots:
        for m in range(cont + 1):
            left = s.derivative(k - eps, m)
            right = s.derivative(k, m)
            scale = 1 + abs(right)
            assert abs(left - right) < 1e-4 * scale * 10 ** m


def test_spline_no_constraint_matches_independent_fits(sine):
    x, y = sine
    s = F.fit_spline(x, y, 5, 3, continuity=-1)
    fits = F.fit_chunks(x, y, 5, 3)
    assert s.predict(x) == pytest.approx(F.evaluate_piecewise(fits, x, blend=False), abs=1e-9)


def test_spline_is_joint_least_squares():
    # A global cubic is a C2 cubic spline, so the joint fit must recover it exactly.
    x = np.linspace(-1, 2, 300)
    y = 2 * x ** 3 - x + 1
    s = F.fit_spline(x, y, 6, 3, continuity=2)
    assert s.predict(x) == pytest.approx(y, abs=1e-9)
    assert s.derivative(x, 1) == pytest.approx(6 * x ** 2 - 1, abs=1e-7)
    assert s.derivative(x, 3) == pytest.approx(np.full_like(x, 12), abs=1e-5)
    assert s.derivative(x, 4) == pytest.approx(np.zeros_like(x), abs=1e-6)


def test_spline_beats_patched_fit_on_constrained_objective(sine):
    # The joint C1 solution must have SSE <= any other C1 spline, e.g. one made by
    # a C1 fit of fewer degrees of freedom (C2 is a subset of C1).
    x, y = sine
    c1 = F.fit_spline(x, y, 6, 3, continuity=1)
    c2 = F.fit_spline(x, y, 6, 3, continuity=2)
    assert np.sum((y - c1.predict(x)) ** 2) <= np.sum((y - c2.predict(x)) ** 2) + 1e-12


def test_spline_stable_high_order_many_chunks():
    x = np.linspace(0, 1000, 3000)
    y = np.sin(x / 30) + 0.001 * np.cos(x)
    s = F.fit_spline(x, y, 50, 10, continuity=2)
    pred = s.predict(x)
    assert np.all(np.isfinite(pred))
    assert F.rmse(y, pred) < 1e-2
    for k in s.knots:
        assert abs(s.derivative(k - 1e-6, 1) - s.derivative(k, 1)) < 1e-6


@pytest.mark.parametrize("order,cont", [(3, 3), (2, 5), (3, -2)])
def test_spline_invalid_continuity(order, cont, sine):
    x, y = sine
    with pytest.raises(ValueError):
        F.fit_spline(x, y, 3, order, continuity=cont)


# ------------------------------------------------------------- adaptive_chunks

def _check_partition(chunks, n):
    assert chunks[0][0] == 0 and chunks[-1][1] == n
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))
    assert all(s < e for s, e in chunks)


def test_adaptive_chunks_meets_tolerance(sine):
    x, y = sine
    order, tol = 3, 0.02
    chunks = F.adaptive_chunks(x, y, order, tol, 50)
    _check_partition(chunks, len(x))
    for s, e in chunks:
        fit = F.fit_chunks(x[s:e], y[s:e], 1, order)[0]
        assert fit.rmse <= tol or e - s < 2 * (order + 1)
    assert len(chunks) <= 6  # one cubic can't do sin over [0,10], a handful can


def test_adaptive_chunks_single_chunk_when_easy():
    x = np.linspace(0, 1, 500)
    assert F.adaptive_chunks(x, 3 * x ** 2, 2, 1e-6, 10) == [(0, 500)]


def test_adaptive_chunks_piecewise_signal_finds_breaks():
    x = np.arange(300.0)
    y = np.concatenate([np.zeros(100), np.full(100, 5.0), np.zeros(100)])
    chunks = F.adaptive_chunks(x, y, 0, 1e-9, 10)
    assert chunks == [(0, 100), (100, 200), (200, 300)]


def test_adaptive_chunks_respects_max_chunks(sine):
    x, y = sine
    chunks = F.adaptive_chunks(x, y, 1, 1e-6, 4)
    _check_partition(chunks, len(x))
    assert len(chunks) <= 4


# ----------------------------------------------------------- robust_fit_chunks

def test_robust_fit_tracks_clean_signal_with_outliers(sine):
    x, y = sine
    rng = np.random.default_rng(1)
    yo = y.copy()
    bad = rng.choice(len(x), size=len(x) // 10, replace=False)
    yo[bad] += rng.choice([-1, 1], size=bad.size) * rng.uniform(3, 10, size=bad.size)
    clean = np.sin(x)
    robust = F.robust_fit_chunks(x, yo, 4, 5)
    plain = F.fit_chunks(x, yo, 4, 5)
    err_r = F.rmse(clean, F.evaluate_piecewise(robust, x, blend=False))
    err_p = F.rmse(clean, F.evaluate_piecewise(plain, x, blend=False))
    assert err_r < 0.02
    assert err_r < err_p / 10


def test_robust_fit_same_contract(sine):
    x, y = sine
    fits = F.robust_fit_chunks(x, y, 3, 2, overlap_pct=20)
    assert [(f.start, f.stop) for f in fits] == F.chunk_bounds(len(x), 3, 20)
    assert all(isinstance(f, F.ChunkFit) for f in fits)
    short = F.robust_fit_chunks(np.arange(6.0), np.arange(6.0) ** 2, 3, 5)
    assert all(len(f.coeffs) == 2 for f in short)


# ---------------------------------------------------------------- select_order

@pytest.mark.parametrize("method", ["cv", "aic"])
def test_select_order_recovers_cubic(method):
    rng = np.random.default_rng(3)
    x = np.linspace(-2, 2, 400)
    y = x ** 3 - 2 * x + 0.05 * rng.normal(size=x.size)
    assert F.select_order(x, y, 1, 10, method=method) == 3


@pytest.mark.parametrize("method", ["cv", "aic"])
def test_select_order_constant(method):
    rng = np.random.default_rng(4)
    x = np.linspace(0, 1, 200)
    y = 1 + 0.1 * rng.normal(size=x.size)
    assert F.select_order(x, y, 2, 6, method=method) <= 1


def test_select_order_bad_method(sine):
    x, y = sine
    with pytest.raises(ValueError):
        F.select_order(x, y, 1, 3, method="bic")


# ------------------------------------------------------------------------- CLI

def test_cli(tmp_path):
    out = tmp_path / "out.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "fitting_cli.py"),
                        os.path.join(ROOT, "samples", "ECG2.csv"),
                        "--orders", "1-4", "--chunks", "1-3", "--json", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text())
    assert data["orders"] == [1, 2, 3, 4] and data["chunks"] == [1, 2, 3]
    assert data["overlap"] == 0
    em = np.array(data["error_map"])
    assert em.shape == (4, 3)
    x, y = sample("ECG2.csv")
    assert em[2, 1] == pytest.approx(brute_rmse(x, y, 3, 2, 0))
    i, j = np.unravel_index(np.argmin(em), em.shape)
    assert data["best"] == {"order": data["orders"][i], "chunks": data["chunks"][j],
                            "rmse": pytest.approx(em[i, j])}
    assert isinstance(data["seconds"], float)
