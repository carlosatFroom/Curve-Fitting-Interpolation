import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import fitting  # noqa: E402

CSV = os.path.join(ROOT, "samples", "sine.csv")
CLI = os.path.join(ROOT, "fitting_cli.py")


def _run(tmp_path, *args):
    out = tmp_path / "out.json"
    result = subprocess.run(
        [sys.executable, CLI, CSV, *args, "--json", str(out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    return result, out


def test_cli_full_grid_schema_and_best(tmp_path):
    result, out = _run(tmp_path, "--orders", "1-10", "--chunks", "1-10")
    assert result.returncode == 0, result.stderr
    d = json.loads(out.read_text())
    assert set(d) == {"orders", "chunks", "overlap", "error_map", "seconds", "best"}
    assert d["orders"] == list(range(1, 11))
    assert d["chunks"] == list(range(1, 11))
    assert d["overlap"] == 0
    grid = np.array(d["error_map"])
    assert grid.shape == (10, 10)
    assert isinstance(d["seconds"], float) and d["seconds"] >= 0
    i, j = np.unravel_index(np.argmin(grid), grid.shape)
    assert d["best"]["order"] == d["orders"][i]
    assert d["best"]["chunks"] == d["chunks"][j]
    assert d["best"]["rmse"] == grid[i, j]


def test_cli_matches_library_error_map(tmp_path):
    result, out = _run(tmp_path, "--orders", "2-4", "--chunks", "2,3,6")
    assert result.returncode == 0, result.stderr
    d = json.loads(out.read_text())
    data = np.genfromtxt(CSV, delimiter=",")[:1000]
    expected = fitting.error_map(data[:, 0], data[:, 1], [2, 3, 4], [2, 3, 6], [0])
    np.testing.assert_allclose(np.array(d["error_map"]), expected)


def test_cli_single_order_gives_one_row(tmp_path):
    result, out = _run(tmp_path, "--orders", "3", "--chunks", "1-4")
    assert result.returncode == 0, result.stderr
    grid = np.array(json.loads(out.read_text())["error_map"])
    assert grid.shape == (1, 4)


def test_cli_rejects_chunks_above_point_count(tmp_path):
    result, _ = _run(tmp_path, "--orders", "1", "--chunks", "5000")
    assert result.returncode == 2
