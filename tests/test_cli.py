import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fitting_cli  # noqa: E402

SINE = ROOT / "samples" / "sine.csv"


def test_cli_writes_schema_and_best_cell(tmp_path):
    out = tmp_path / "out.json"
    code = fitting_cli.main([str(SINE), "--orders", "1-3", "--chunks", "1-4", "--json", str(out)])
    assert code == 0

    result = json.loads(out.read_text())
    assert set(result) == {"orders", "chunks", "overlap", "error_map", "seconds", "best"}
    assert result["orders"] == [1, 2, 3]
    assert result["chunks"] == [1, 2, 3, 4]
    assert result["overlap"] == 0
    grid = np.array(result["error_map"])
    assert grid.shape == (3, 4)
    assert isinstance(result["seconds"], float)

    best = result["best"]
    assert set(best) == {"order", "chunks", "rmse"}
    assert best["rmse"] == pytest.approx(grid.min())
    i = result["orders"].index(best["order"])
    j = result["chunks"].index(best["chunks"])
    assert grid[i, j] == pytest.approx(best["rmse"])


def test_cli_single_overlap_value(tmp_path):
    out = tmp_path / "out.json"
    code = fitting_cli.main([str(SINE), "--orders", "1-2", "--chunks", "2-3", "--overlap", "10", "--json", str(out)])
    assert code == 0
    result = json.loads(out.read_text())
    assert result["overlap"] == 10
    assert np.array(result["error_map"]).shape == (2, 2)


def test_cli_subprocess_exit_code_zero(tmp_path):
    out = tmp_path / "sub.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "fitting_cli.py"), str(SINE),
         "--orders", "1-2", "--chunks", "1-2", "--json", str(out)],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()


def test_load_xy_skips_header_and_limits_rows(tmp_path):
    csv = tmp_path / "with_header.csv"
    rows = [f"{i},{i * 2}" for i in range(1500)]
    csv.write_text("time,value\n" + "\n".join(rows) + "\n")
    x, y = fitting_cli.load_xy(csv)
    assert len(x) == fitting_cli.MAX_ROWS == len(y)
    assert x[0] == 0 and y[0] == 0
    assert y[-1] == 2 * (fitting_cli.MAX_ROWS - 1)


def test_parse_range_inclusive_and_single():
    assert fitting_cli.parse_range("1-10", 1) == list(range(1, 11))
    assert fitting_cli.parse_range("4", 1) == [4]


@pytest.mark.parametrize("bad", ["5-2", "0-3", "a-b", ""])
def test_parse_range_rejects_bad_input(bad):
    with pytest.raises(Exception):
        fitting_cli.parse_range(bad, 1)


def test_cli_requires_two_varying_axes(tmp_path):
    with pytest.raises(SystemExit) as exc:
        fitting_cli.main([str(SINE), "--orders", "3", "--chunks", "1-4", "--json", str(tmp_path / "x.json")])
    assert exc.value.code == 2
