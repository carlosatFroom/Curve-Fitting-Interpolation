# CHANGES

## Broken before

- `from msilib import datasizemask` (Windows-only): app crashed on import on macOS/Linux. Removed.
- `os`, `sys`, `numpy` (as `numpy.` in `chunks`) used but not imported; `from pyqtgraph import *` shadowed names. Removed star import; explicit imports only.
- `read_file`: cancelled dialog gave `''` path and crashed on `min([])`. Now returns early.
- Overlap slicing used `chunkIndex * len(chunk)`, wrong for unequal `np.array_split` chunks. Replaced by `fitting.chunk_bounds`.
- Third error map used `range(1, maxOrder)` (off by one). Now `fitting.error_map` over `1..maxOrder`.
- Transposed error maps were built from un-normalized data. Now each map is normalized (`raw / max * 100`) before transposing.
- Error-map worker read Qt slider values and wrote attributes from a background thread. Slider values and array copies are now taken on the GUI thread; the worker computes numpy only and emits `errorMapComputed`; the GUI slot stores results and calls `showErrorMap()`.
- Cancel flag was only set in `errorMapStart`, so `cancel` before start was unsafe. Now a per-run `threading.Event`, checked between maps and after each computation.
- `np.polyfit` on raw x: ill-conditioned for high orders. `fitting.fit_chunks` / `ChunkFit.predict` fit in a scaled domain (`numpy.polynomial.Polynomial.fit`).
- "error %" was `sqrt(sum(res^2)/chunks/(n-2))` times 100, not an RMSE. Now `error%` = chunk RMSE / whole-signal y-range * 100 (0 if range is 0); a separate `rmse` column shows the raw RMSE.
- Extrapolation hard-coded 1000 points (`chunks*100` / `/1000`). Now `fitting.extrapolate` with `fit_fraction = slider/10`; slider 0 skips the fit. Vertical marker placed at the last fitted x.
- Chunks view with 0 chunks or no data: `numpy.polyfit` / `array_split` raised. Now guarded (raw signal only, empty table).
- Equation code overwrote `self.xAxisValuesofsplit[chunkidx]` with a sympy symbol. Symbol is local now.
- `showErrorMap` with the overlap/order maps used a hand-written if-chain; kept same axis/extent/label semantics via a lookup.

## Other

- `requirements.txt`: pinned to the versions in the experiment venv for the modules actually imported (numpy, PyQt5, pyqtgraph, matplotlib, sympy, pandas) plus pytest. scipy not used by app or fitting.
- Error-map computation time for orders 1..4 × chunks 1..4 on 1000 points: ~0.2 s (offscreen smoke test).

## Known limits (not changed)

- Only `.csv` is loaded; reading is still capped at the first 1000 rows.
- `errorMapStart` does not block a second start while a run is active; the older worker's results are still stored.
- Error maps skip a map when its two varying axes don't both have length > 1 (e.g. slider max order = 1).
