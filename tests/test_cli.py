import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "fitting_cli.py"
SINE = ROOT / "samples" / "sine.csv"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(CLI), *args], capture_output=True, text=True, cwd=ROOT
    )


def test_cli_writes_json_with_expected_shape_and_best(tmp_path):
    out = tmp_path / "out.json"
    proc = run_cli(str(SINE), "--orders", "1-3", "--chunks", "1-4", "--json", str(out))
    assert proc.returncode == 0, proc.stderr

    data = json.loads(out.read_text())
    assert set(data) >= {"orders", "chunks", "overlap", "error_map", "seconds", "best"}
    assert data["orders"] == [1, 2, 3]
    assert data["chunks"] == [1, 2, 3, 4]
    assert data["overlap"] == 0

    em = np.array(data["error_map"], dtype=float)
    assert em.shape == (3, 4)
    assert data["seconds"] >= 0

    i, j = np.unravel_index(np.argmin(em), em.shape)
    assert data["best"]["order"] == data["orders"][i]
    assert data["best"]["chunks"] == data["chunks"][j]
    assert data["best"]["rmse"] == pytest.approx(em[i, j])


def test_cli_error_map_matches_library(tmp_path):
    import fitting

    raw = np.genfromtxt(SINE, delimiter=",", usecols=(0, 1), max_rows=1000)
    expected = fitting.error_map(raw[:, 0], raw[:, 1], [1, 2], [2, 3], [0.0])

    out = tmp_path / "out.json"
    proc = run_cli(str(SINE), "--orders", "1,2", "--chunks", "2,3", "--json", str(out))
    assert proc.returncode == 0, proc.stderr
    got = np.array(json.loads(out.read_text())["error_map"])
    np.testing.assert_allclose(got, expected, rtol=1e-9)


def test_cli_accepts_comma_lists_and_single_ints_and_overlap(tmp_path):
    out = tmp_path / "out.json"
    proc = run_cli(str(SINE), "--orders", "1,3,5", "--chunks", "2", "--overlap", "10",
                   "--json", str(out))
    # single chunk count gives one varying argument -> must be rejected cleanly
    assert proc.returncode != 0
    proc = run_cli(str(SINE), "--orders", "1,3,5", "--chunks", "2,4", "--overlap", "10",
                   "--json", str(out))
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert data["orders"] == [1, 3, 5]
    assert data["chunks"] == [2, 4]
    assert data["overlap"] == 10


def test_cli_rejects_bad_range(tmp_path):
    proc = run_cli(str(SINE), "--orders", "x-3", "--chunks", "1-2")
    assert proc.returncode != 0
