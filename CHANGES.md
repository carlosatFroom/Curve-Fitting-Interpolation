# CHANGES

## Broken in the original `app.py`

- `from msilib import datasizemask`: Windows-only, crashes import on macOS/Linux. Removed.
- Missing imports: `sys`, `os`, bare `numpy` (used as `numpy.polyfit` / `numpy.sum`).
- Overlap slicing used `chunkIndex * len(chunk)` as the start index, so overlapping chunks were offset wrongly.
- Error map used `range(1, maxOrder)` (off by one, drops the top order) and the transposed maps were not normalised the same way.
- Unscaled-x `np.polyfit` is ill-conditioned for large x.
- "error %" was a fake RMSE: `sqrt(SSE/n_chunks/(N-2)) * 100`.
- Extrapolation hard-coded 1000 points (`*100/1000`).
- Matplotlib drawing (`axes.clear()`, `colorbar`, `draw()`) ran on the worker thread.

## Changed

- All fitting math now goes through `fitting.py`:
  - `chunks()` uses `fit_chunks`, `ChunkFit.predict` for the curve and `ChunkFit.rmse` for the table. Guards `n_chunks == 0` and empty data.
  - `updateExtraPolation()` uses `extrapolate`. The point count comes from the data length, not 1000. Fraction is `slider / slider.maximum()`.
  - The error-map worker uses `error_map` for the three grids. Each grid is normalised to percent of max, and the transposed view is derived from the same array.
- Error map worker computes only. It emits `errorMapReady`, connected to `showErrorMap` on the GUI thread (queued), so canvas drawing never runs on the worker.
- Off-by-one fixed: order range is `1..maxOrder` inclusive.
- Error map skips (no crash) when either error-map slider is below 2.
- Table column `error%` renamed `RMSE` (it is now a real RMSE, not a percent).
- Imports grouped stdlib / third-party / local.

## Why

Single source of fitting math (tested in `fitting.py`), correct numerics on normalised x, real RMSE, and no Qt/matplotlib drawing off the GUI thread.

## Not changed

- `read_file` still truncates to the first 1000 CSV rows.
- `read_file` with no file selected still fails on `min()` of empty data.
