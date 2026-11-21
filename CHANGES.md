# CHANGES

## What was broken

From TASK.md:
- `from msilib import datasizemask` (Windows-only): the app crashed on import elsewhere.
- Overlap slicing used `chunkIndex * len(chunk)`, wrong when `array_split` gives unequal chunks.
- Third error map used `range(1, maxOrder)` (off by one, max order missing).
- Transposed error maps were built from un-normalized data.
- `np.polyfit` on raw x: ill-conditioned at high order.
- "error %" was not an RMSE (sum divided by chunk count, then `sqrt(/(n-2))`).
- Extrapolation hard-coded 1000 points (`slider*100` and `/1000`).
- Matplotlib drawing ran on the error-map worker thread.

Found while reading:
- `from pyqtgraph import *` star import; `os`/`sys` never imported (NameError in `read_file` and `__main__`); bare `numpy.` used in `chunks()` without importing it.
- `normalizeData` normalized the caller's array in place.
- `chunks()` overwrote `xAxisValuesofsplit[i]` with a sympy symbol, destroying the chunk data.
- `chunks()` fitted each chunk twice and used the loop variable `order` after the loop.
- Cancel set a flag that `get2DErrorArray` returned `None` on, so the next step crashed.
- Cancelling the file dialog (empty path) or loading a non-CSV crashed on `min()` of empty data; a header row became NaN.
- Slider states of 0 (chunks, order, error-map sliders) were not guarded.
- Closing the window while the error map was computing could destroy a running thread.

## What changed

- `fitting.py` (new, Qt-free, numpy only): chunk bounds with exact overlap rules, `ChunkFit` with scaled-domain fits (`Polynomial.fit`), `fit_chunks` (order drops for short chunks), `evaluate_piecewise` (optional cross-fade blend), `rmse`/`r2`, `error_map` (un-normalized, `concurrent.futures`), `extrapolate`.
- Phase 2 in `fitting.py`: `fit_spline` (joint constrained least squares, continuity 0-2), `adaptive_chunks`, `robust_fit_chunks` (IRLS with robust scale), `select_order` (CV or AIC).
- `app.py` rewritten onto `fitting.py`:
  - No `msilib`, no star import; explicit imports.
  - Error maps: a `QThread` worker computes the three maps and emits plain arrays via a signal. The GUI thread alone draws. Cancel requests interruption; the worker checks it between maps. Closing the window waits for the worker.
  - Each map is normalized to percent of its max (copy). Transposed copies are made from the normalized data.
  - Third map is orders `1..maxOrder` inclusive (was off by one).
  - Maps that do not vary order or chunks keep the old fixed values (order 5, chunks 5).
  - Table `error%` = `ChunkFit.rmse / (max(y) - min(y)) * 100`, i.e. the chunk's real RMSE as a percentage of the signal's peak-to-peak range. Shows `nan` for a constant signal.
  - Extrapolation: `fraction = chunks_slider / 10`, passed to `fitting.extrapolate`. Slider 0 draws nothing. The red line marks the last fitted x.
  - No data, non-CSV, cancelled dialog, bad file, and slider-0 states show a status-bar message instead of crashing.
  - Widgets and layout are unchanged (`mainapp.py` untouched).
- `fitting_cli.py`: `python fitting_cli.py <csv> --orders 1-10 --chunks 1-10 [--overlap 0] --json out.json`. Reads the first two columns, first 1000 numeric rows (header skipped). Writes the JSON schema from TASK.md. `--overlap` takes a single number. Exit 0 on success, 1 on data or fit errors, 2 on bad arguments.
- `requirements.txt`: every package from the project venv (`pip freeze`), pinned.
- `tests/test_cli.py`: CLI schema, best cell, overlap, subprocess exit code, header handling, argument validation.

## Why

- Crashes (`msilib`, missing imports, empty-data `min()`) block the app from running at all.
- Off-thread drawing and in-place mutation make the GUI race-prone.
- The numeric fixes (scaled fits, real RMSE, overlap slicing, off-by-one) change the numbers the app shows, so the old outputs were wrong, not just slow.

## Not done

- The GUI does not yet expose spline, adaptive chunks, robust fit or order selection. They are in `fitting.py` and the CLI does not use them.
