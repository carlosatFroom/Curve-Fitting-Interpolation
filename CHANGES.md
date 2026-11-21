# Changes

## Broken
- `app.py` imported Windows-only `msilib` and used `os`/`sys`/`numpy` names that were never imported properly (crash on macOS/Linux).
- Overlap slicing used `chunkIndex * len(chunk)`, wrong for unequal `np.array_split` chunks.
- Third error map used `range(1, maxOrder)` (off by one); transposed maps were built from un-normalized data.
- `np.polyfit` on raw x is ill-conditioned at high order.
- "error %" was not an RMSE (divided by chunk count and N-2).
- Extrapolation hard-coded 1000 points.
- Matplotlib drawing ran on a worker thread; error map ran in-GUI loops with no reuse.

## Changed
- New pure-numpy `fitting.py` (no Qt/matplotlib): `chunk_bounds`, `ChunkFit` (fits in a scaled domain via `Polynomial.fit`), `fit_chunks`, `evaluate_piecewise` (linear cross-fade in overlaps), `rmse`, `r2`, parallel `error_map`, `extrapolate`.
- Phase 2: joint constrained `fit_spline` (C0..Cn), `adaptive_chunks`, `robust_fit_chunks` (IRLS), `select_order` (CV/AIC).
- `app.py` delegates all math to `fitting.py`; the error map computes on a worker thread and hands results to the GUI thread by signal, which does all drawing. Table column is now a true RMSE.
- `fitting_cli.py` benchmark CLI, pinned `requirements.txt`, tests in `tests/`.
