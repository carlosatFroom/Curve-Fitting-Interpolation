import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from fitting import error_map

REPO = Path(__file__).resolve().parent.parent
PYTHON = "/Users/carlos/code/teamclaude_experiment/.venv/bin/python"


@pytest.fixture(scope="module")
def data():
    x = np.linspace(0.0, 10.0, 1000)
    y = np.sin(x)
    return x, y


def test_shape_orders_by_chunks(data):
    x, y = data
    m = error_map(x, y, [1, 2, 3], [1, 2, 3, 4], [0.0])
    assert m.shape == (3, 4)


def test_shape_orders_by_overlap(data):
    x, y = data
    m = error_map(x, y, [1, 2], [3], [0.0, 5.0, 10.0])
    assert m.shape == (2, 3)


def test_shape_chunks_by_overlap(data):
    x, y = data
    m = error_map(x, y, [2], [1, 2, 3, 4, 5], [0.0, 10.0])
    assert m.shape == (5, 2)


def test_orientation_matches_direct_computation(data):
    from fitting import evaluate_piecewise, fit_chunks, rmse

    x, y = data
    m = error_map(x, y, [1, 3], [2, 5, 7], [0.0], workers=1)
    for i, order in enumerate([1, 3]):
        for j, n_chunks in enumerate([2, 5, 7]):
            expected = rmse(y, evaluate_piecewise(fit_chunks(x, y, n_chunks, order, 0.0), x, blend=False))
            assert m[i, j] == pytest.approx(expected)


def test_chunks_by_overlap_orientation(data):
    from fitting import evaluate_piecewise, fit_chunks, rmse

    x, y = data
    m = error_map(x, y, [4], [2, 6], [0.0, 20.0], workers=1)
    for i, n_chunks in enumerate([2, 6]):
        for j, ov in enumerate([0.0, 20.0]):
            expected = rmse(y, evaluate_piecewise(fit_chunks(x, y, n_chunks, 4, ov), x, blend=False))
            assert m[i, j] == pytest.approx(expected)


def test_values_finite_and_non_negative(data):
    x, y = data
    m = error_map(x, y, list(range(1, 11)), list(range(1, 11)), [0.0])
    assert np.all(np.isfinite(m))
    assert np.all(m >= 0.0)


def test_failing_cell_does_not_crash(data):
    x, y = data
    # 2000 chunks on 1000 points is invalid; that cell must come back NaN, not raise.
    m = error_map(x, y, [1, 2], [2, 2000], [0.0], workers=1)
    assert m.shape == (2, 2)
    assert np.isnan(m[:, 1]).all()
    assert np.isfinite(m[:, 0]).all()


def test_timing_orders_by_chunks_under_2s(data):
    x, y = data
    start = time.perf_counter()
    m = error_map(x, y, list(range(1, 11)), list(range(1, 11)), [0.0])
    elapsed = time.perf_counter() - start
    assert m.shape == (10, 10)
    assert elapsed < 2.0, f"took {elapsed:.2f}s"


def test_workers_1_and_2_agree(data):
    x, y = data
    serial = error_map(x, y, [1, 2, 3, 4], [1, 2, 3, 4, 5], [0.0], workers=1)
    threaded = error_map(x, y, [1, 2, 3, 4], [1, 2, 3, 4, 5], [0.0], workers=2)
    np.testing.assert_allclose(serial, threaded, rtol=1e-9, atol=1e-12)


def test_cli_end_to_end(tmp_path):
    x = np.linspace(0.0, 4.0, 200)
    y = np.cos(x) + 0.05 * x
    csv = tmp_path / "data.csv"
    lines = ["x,y"] + [f"{a},{b}" for a, b in zip(x, y)]
    csv.write_text("\n".join(lines) + "\n")
    out = tmp_path / "out.json"

    proc = subprocess.run(
        [
            PYTHON,
            str(REPO / "fitting_cli.py"),
            str(csv),
            "--orders", "1-3",
            "--chunks", "1-4",
            "--overlap", "0",
            "--json", str(out),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    result = json.loads(out.read_text())
    assert set(result) >= {"orders", "chunks", "overlap", "error_map", "seconds", "best"}
    assert result["orders"] == [1, 2, 3]
    assert result["chunks"] == [1, 2, 3, 4]
    assert result["overlap"] == 0
    assert len(result["error_map"]) == 3
    assert all(len(row) == 4 for row in result["error_map"])
    assert result["seconds"] >= 0

    best = result["best"]
    assert set(best) == {"order", "chunks", "rmse"}
    i = result["orders"].index(best["order"])
    j = result["chunks"].index(best["chunks"])
    assert result["error_map"][i][j] == pytest.approx(best["rmse"])
    flat = [v for row in result["error_map"] for v in row if v is not None]
    assert best["rmse"] == pytest.approx(min(flat))
