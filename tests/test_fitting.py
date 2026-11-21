import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import fitting
from fitting import (adaptive_chunks, chunk_bounds, error_map, evaluate_piecewise, extrapolate,
                     fit_chunks, fit_spline, r2, rmse, robust_fit_chunks, select_order)

ROOT = Path(__file__).resolve().parent.parent


def test_import_has_no_gui():
    code = ("import sys, fitting; "
            "bad=[m for m in ('PyQt5','pyqtgraph','matplotlib') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode == 0


@pytest.mark.parametrize("n,k", [(10, 3), (1000, 7), (7, 7), (100, 1)])
def test_chunk_bounds_partition(n, k):
    b = chunk_bounds(n, k)
    ref = np.array_split(np.arange(n), k)
    assert b == [(int(r[0]), int(r[-1]) + 1) for r in ref]


def test_chunk_bounds_overlap():
    b = chunk_bounds(100, 4, 20)  # base len 25 -> ov 5
    assert b == [(0, 30), (23, 53), (48, 78), (70, 100)]
    b = chunk_bounds(10, 3, 50)  # unequal base lengths 4,3,3
    assert b[0] == (0, 6) and b[1] == (4, 8) and b[2] == (6, 10)


def test_chunk_bounds_errors():
    for n, k in [(10, 0), (10, 11)]:
        with pytest.raises(ValueError):
            chunk_bounds(n, k)


def test_fit_chunks_exact_and_unscaled_high_order():
    x = np.linspace(1000, 1010, 200)
    y = 3 * (x - 1005) ** 3 - 2 * (x - 1005)
    f = fit_chunks(x, y, 1, 10)[0]
    assert np.max(np.abs(f.predict(x) - y)) < 1e-6
    assert f.rmse < 1e-6
    assert (f.start, f.stop) == (0, 200)


def test_coeffs_original_x():
    x = np.linspace(-1, 3, 50)
    y = 2 * x ** 2 - x + 4
    f = fit_chunks(x, y, 1, 2)[0]
    np.testing.assert_allclose(f.coeffs, [2, -1, 4], atol=1e-9)


def test_order_reduced_for_small_chunks():
    x = np.arange(6.0)
    y = x ** 2
    fits = fit_chunks(x, y, 3, 5)  # 2 points per chunk
    assert len(fits) == 3 and all(np.isfinite(f.coeffs).all() for f in fits)
    assert all(f.rmse < 1e-9 for f in fits)


def test_unequal_chunk_overlap_slices():
    x = np.arange(10.0)
    fits = fit_chunks(x, x, 3, 1, 50)
    assert [(f.start, f.stop) for f in fits] == chunk_bounds(10, 3, 50)
    assert fits[1].x_min == x[fits[1].start] and fits[1].x_max == x[fits[1].stop - 1]


def test_evaluate_noblend_uses_base_chunks():
    x = np.linspace(0, 1, 100)
    y = np.sin(6 * x)
    fits = fit_chunks(x, y, 4, 3, 20)
    got = evaluate_piecewise(fits, x, blend=False)
    exp = np.empty_like(y)
    for f, (a, b) in zip(fits, chunk_bounds(100, 4)):
        exp[a:b] = f.predict(x[a:b])
    np.testing.assert_allclose(got, exp)


def test_blend_is_continuous_across_overlap():
    x = np.linspace(0, 1, 400)
    rng = np.random.default_rng(0)
    y = np.sin(8 * x) + 0.05 * rng.standard_normal(400)
    fits = fit_chunks(x, y, 4, 3, 30)
    xq = np.linspace(0, 1, 20001)
    v = evaluate_piecewise(fits, xq, blend=True)
    assert np.max(np.abs(np.diff(v))) < 0.05
    # un-blended with the same data jumps at least as much
    nb = evaluate_piecewise(fits, xq, blend=False)
    assert np.max(np.abs(np.diff(nb))) > np.max(np.abs(np.diff(v)))


def test_blend_exact_when_both_exact():
    x = np.linspace(0, 1, 200)
    y = x ** 2
    fits = fit_chunks(x, y, 5, 2, 40)
    np.testing.assert_allclose(evaluate_piecewise(fits, x), y, atol=1e-9)


def test_metrics():
    a = np.array([1.0, 2, 3, 4])
    assert rmse(a, a + 2) == pytest.approx(2.0)
    assert r2(a, a) == 1.0
    assert r2(a, np.full(4, a.mean())) == pytest.approx(0.0)


def test_error_map_shape_values_and_axes():
    x = np.linspace(0, 1, 300)
    y = np.sin(5 * x)
    m = error_map(x, y, range(1, 4), range(1, 5), [0], workers=1)
    assert m.shape == (3, 4)
    f = fit_chunks(x, y, 2, 3)
    assert m[2, 1] == pytest.approx(rmse(y, evaluate_piecewise(f, x, blend=False)))
    m2 = error_map(x, y, [3], range(1, 5), [0, 10, 20])
    assert m2.shape == (4, 3)
    m3 = error_map(x, y, range(1, 4), [4], [0, 10])
    assert m3.shape == (3, 2)
    f = fit_chunks(x, y, 4, 2, 10)
    assert m3[1, 1] == pytest.approx(rmse(y, evaluate_piecewise(f, x, blend=False)))
    np.testing.assert_allclose(error_map(x, y, range(1, 4), range(1, 5), [0]), m)


def test_error_map_bad_args():
    x = np.arange(20.0)
    with pytest.raises(ValueError):
        error_map(x, x, [1], [1], [0, 5, 10][:1])
    with pytest.raises(ValueError):
        error_map(x, x, [1, 2], [1, 2], [0, 5])


def test_error_map_speed():
    x = np.linspace(0, 10, 1000)
    y = np.sin(x) + 0.1 * np.cos(7 * x)
    t = time.perf_counter()
    m = error_map(x, y, range(1, 11), range(1, 11), [0])
    assert time.perf_counter() - t < 2.0
    assert m.shape == (10, 10) and np.isfinite(m).all()
    assert m[9, 9] < m[0, 0]


def test_extrapolate():
    x = np.linspace(0, 10, 1000)
    y = 0.5 * x ** 2 - x
    c, p = extrapolate(x, y, 0.3, 2)
    np.testing.assert_allclose(c, [0.5, -1, 0], atol=1e-7)
    assert p.shape == x.shape
    np.testing.assert_allclose(p, y, atol=1e-7)
    c, p = extrapolate(x[:50], y[:50], 0.01, 4)  # at least order+1 points
    assert p.shape == (50,)
    c, p = extrapolate(x, y, 1.0, 2)
    assert p.shape == x.shape
    with pytest.raises(ValueError):
        extrapolate(x, y, 0.0, 2)


# ------------------------------------------------------------------ phase 2

@pytest.mark.parametrize("cont", [-1, 0, 1, 2])
def test_spline_continuity(cont):
    x = np.linspace(0, 6, 600)
    y = np.sin(x) + 0.1 * np.cos(9 * x)
    s = fit_spline(x, y, 6, 4, cont)
    assert len(s.knots) == 5
    eps = 1e-9
    for m in range(cont + 1):
        left = s.derivative(s.knots - eps, m)
        right = s.derivative(s.knots + eps, m)
        scale = 1 + np.abs(left).max()
        np.testing.assert_allclose(left, right, atol=1e-4 * scale * max(1, 10 ** m))
    if cont == -1:
        assert np.max(np.abs(s.predict(s.knots - 1e-9) - s.predict(s.knots + 1e-9))) > 1e-4


def test_spline_exact_cubic_and_derivative():
    x = np.linspace(-2, 2, 400)
    y = x ** 3 - 2 * x
    s = fit_spline(x, y, 8, 3, 2)
    np.testing.assert_allclose(s.predict(x), y, atol=1e-8)
    np.testing.assert_allclose(s.derivative(x, 1), 3 * x ** 2 - 2, atol=1e-6)
    np.testing.assert_allclose(s.derivative(x, 2), 6 * x, atol=1e-5)
    np.testing.assert_allclose(s.derivative(x, 0), s.predict(x))
    np.testing.assert_allclose(s.derivative(x, 5), 0, atol=1e-12)


def test_spline_knot_belongs_to_right_piece_and_stability():
    x = np.linspace(0, 100, 1000)
    y = np.sin(x / 5)
    s = fit_spline(x, y, 50, 10, 2)
    assert np.isfinite(s.predict(x)).all()
    assert rmse(y, s.predict(x)) < 1e-3
    assert len(s.knots) == 49
    assert s.knots[0] == x[20]


def test_spline_constrained_beats_nothing_but_not_better_than_free():
    x = np.linspace(0, 1, 300)
    y = np.sin(10 * x)
    free = rmse(y, fit_spline(x, y, 5, 3, -1).predict(x))
    c2 = rmse(y, fit_spline(x, y, 5, 3, 2).predict(x))
    assert free <= c2 + 1e-12
    pieces = rmse(y, evaluate_piecewise(fit_chunks(x, y, 5, 3), x, blend=False))
    assert free == pytest.approx(pieces, rel=1e-6)


def test_spline_bad_continuity():
    x = np.linspace(0, 1, 50)
    for c in (-2, 3):
        with pytest.raises(ValueError):
            fit_spline(x, x, 3, 3, c)


def test_adaptive_chunks():
    x = np.linspace(0, 1, 1000)
    y = np.where(x < 0.5, x, 0.5 + 5 * (x - 0.5) ** 2) + 0.3 * np.sin(30 * x) ** 3
    order, tol = 3, 0.02
    parts = adaptive_chunks(x, y, order, tol, 60)
    assert parts[0][0] == 0 and parts[-1][1] == 1000
    assert all(a[1] == b[0] for a, b in zip(parts, parts[1:]))
    for a, b in parts:
        if b - a >= 2 * (order + 1):
            p = np.polynomial.Polynomial.fit(x[a:b], y[a:b], order)
            assert rmse(y[a:b], p(x[a:b])) <= tol + 1e-12
    fewer = adaptive_chunks(x, y, order, 0.2, 60)
    assert len(fewer) <= len(parts)
    assert len(adaptive_chunks(x, y, order, 1e-9, 7)) <= 7


def test_adaptive_single_chunk_when_easy():
    x = np.linspace(0, 1, 200)
    assert adaptive_chunks(x, x ** 2, 2, 1e-6, 10) == [(0, 200)]


def test_robust_fit_tracks_signal():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 6, 600)
    clean = np.sin(x)
    y = clean + 0.02 * rng.standard_normal(600)
    out = rng.choice(600, 60, replace=False)
    y[out] += rng.choice([-1, 1], 60) * rng.uniform(5, 15, 60)
    plain = rmse(clean, evaluate_piecewise(fit_chunks(x, y, 3, 5), x, blend=False))
    rob = rmse(clean, evaluate_piecewise(robust_fit_chunks(x, y, 3, 5), x, blend=False))
    assert rob < 0.05 and rob < plain / 5


def test_robust_equals_plain_on_exact_data_and_small_chunks():
    x = np.linspace(0, 1, 60)
    y = x ** 3
    f = robust_fit_chunks(x, y, 3, 3, 10)
    assert [(a.start, a.stop) for a in f] == chunk_bounds(60, 3, 10)
    np.testing.assert_allclose(evaluate_piecewise(f, x), y, atol=1e-9)
    assert len(robust_fit_chunks(np.arange(4.0), np.arange(4.0), 2, 6)) == 2


def test_select_order():
    rng = np.random.default_rng(2)
    x = np.linspace(-1, 1, 400)
    exact = 1 + x - 2 * x ** 3
    assert select_order(x, exact, 1, 8, "cv") == 3
    assert select_order(x, exact, 1, 8, "aic") == 3
    noisy = exact + 0.05 * rng.standard_normal(400)
    assert 3 <= select_order(x, noisy, 1, 8, "cv") <= 5
    assert 3 <= select_order(x, noisy, 1, 8, "aic") <= 5
    with pytest.raises(ValueError):
        select_order(x, exact, 1, 5, "bogus")


def test_select_order_piecewise():
    x = np.linspace(0, 1, 500)
    y = np.sin(12 * x)
    assert select_order(x, y, 5, 8) <= 8


# ---------------------------------------------------------------------- CLI

def test_cli(tmp_path):
    out = tmp_path / "o.json"
    r = subprocess.run([sys.executable, str(ROOT / "fitting_cli.py"), str(ROOT / "samples" / "sine.csv"),
                        "--orders", "1-4", "--chunks", "1-3", "--json", str(out)], cwd=tmp_path)
    assert r.returncode == 0
    d = json.loads(out.read_text())
    assert d["orders"] == [1, 2, 3, 4] and d["chunks"] == [1, 2, 3] and d["overlap"] == 0
    em = np.array(d["error_map"])
    assert em.shape == (4, 3) and d["seconds"] >= 0
    i, j = np.unravel_index(em.argmin(), em.shape)
    assert d["best"] == {"order": i + 1, "chunks": j + 1, "rmse": em[i, j]}
    data = np.genfromtxt(ROOT / "samples" / "sine.csv", delimiter=",")[:1000]
    expect = error_map(data[:, 0], data[:, 1], [1, 2, 3, 4], [1, 2, 3], [0])
    np.testing.assert_allclose(em, expect)


def test_cli_missing_file(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "fitting_cli.py"), "nope.csv", "--json", str(tmp_path / "o")],
                       cwd=tmp_path, capture_output=True)
    assert r.returncode != 0
