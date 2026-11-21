# Changes

## Broken
- `from msilib import datasizemask`: Windows-only import crashed the app elsewhere. Missing `os`/`sys` imports and a stray `numpy.` reference also broke `read_file`/startup.
- Overlap slicing used `chunkIndex * len(chunk)`, wrong for unequal `np.array_split` chunks.
- Third error map used `range(1, maxOrder)` (off by one); transposed maps were built from un-normalised data; `error%` was not an RMSE (divided per-chunk SSE by chunk count and `n-2`).
- `np.polyfit` on raw x is ill-conditioned for high orders.
- Extrapolation hard-coded 1000 points; chunks slider at 0 crashed.
- Matplotlib drawing ran on a worker thread; cancel did nothing inside the computation.

## Changed
- New pure-numpy `fitting.py` (no GUI imports): `chunk_bounds`, `fit_chunks`/`ChunkFit` (fit in scaled domain via `Polynomial.fit`), `evaluate_piecewise` (smoothstep cross-fade in overlaps), `rmse`, `r2`, `error_map` (thread pool, real RMSE against all points), `extrapolate`.
- Phase 2: `fit_spline` (joint constrained least squares via null-space of the knot constraints, Chebyshev local basis), `adaptive_chunks` (greedy grow+bisect, raising tolerance only if `max_chunks` would be exceeded), `robust_fit_chunks` (Huber then Tukey IRLS, MAD scale), `select_order` (interleaved 5-fold CV or AICc, ties favour lower order).
- `fitting_cli.py` benchmark CLI writing the JSON error map.
- `app.py` uses `fitting` for all math; error map is computed on a worker thread and handed to the GUI thread by a Qt signal (all drawing on the GUI thread); cancel works inside `error_map`. Error-map orientations come from one un-normalised RMSE array, normalised once. Map extents now centred on cell values. Table column `error%` is now `RMSE`. Dropped sympy/pandas (own LaTeX formatter).
- Added pinned `requirements.txt` and `tests/`.

## Notes
- No scipy was available, so everything is numpy-only.
- `evaluate_piecewise` assumes ascending x.
