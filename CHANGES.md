# Changes

## What was broken

- `from msilib import datasizemask` (Windows-only, unused) crashed the app on macOS/Linux.
  `os`, `sys` and `numpy` (used as `numpy.polyfit`) were never imported.
- Overlap slicing used `chunkIndex * len(chunk)` as chunk start, which is wrong whenever
  `np.array_split` yields chunks of unequal length (chunks shifted, points skipped/duplicated).
- The orders × overlap error map used `range(1, maxOrder)` (one order missing); the transposed
  maps were built from un-normalized data while the direct maps were normalized (and
  `normalizeData` mutated its input in place).
- `np.polyfit` on raw x is ill-conditioned for high orders (RankWarning, garbage fits).
- "error %" was `sqrt(SSE / n_chunks / (N - 2)) * 100` per chunk: not an RMSE of anything.
- Extrapolation hard-coded 1000 points for slice index and the vertical line position.
- The error map was drawn with matplotlib from a `threading.Thread` (Qt widgets touched off
  the GUI thread → crashes / undefined behaviour).
- A chunk with `<= order` points made `polyfit` fail/warn; `n_chunks > N` was not handled.

## What changed

- New `fitting.py`: pure numpy, no GUI imports. Chunk boundaries computed explicitly
  (`chunk_bounds`) from the `np.array_split` partition, so unequal chunks and overlaps are exact.
  Fits use `numpy.polynomial.Polynomial.fit` (scaled domain) for stability; `coeffs` in original
  x are only for display. Order is reduced per chunk when it has too few points. Real `rmse`/`r2`.
  `evaluate_piecewise` cross-fades (smoothstep) adjacent chunk predictions inside overlaps so the
  curve has no jumps. `error_map` returns un-normalized RMSE over all points, parallel via
  `concurrent.futures`. `extrapolate` uses a fraction of the data instead of fixed 1000 points.
- Phase 2: `fit_spline` (joint equality-constrained least squares with C^k continuity at knots,
  solved in per-piece scaled Legendre bases via the null-space method), `adaptive_chunks`
  (automatic boundaries to meet an RMSE tolerance), `robust_fit_chunks` (IRLS with a robust
  MAD scale), `select_order` (k-fold CV or AIC).
- `fitting_cli.py`: benchmark CLI writing the error map + best cell as JSON.
- `app.py`: uses `fitting.py` for all math; the error-map worker only computes numbers and hands
  them to the GUI thread via a Qt signal, where they are normalized (on a copy) and drawn.
  Table shows the real per-chunk RMSE. The third map now covers orders 1..maxOrder; transposed
  views come from the normalized maps; `imshow` uses `origin='lower'` with extents matching the
  real axis values. Extrapolation fits a fraction of the loaded data (slider / 10) and places the
  marker at the last fitted point. Cancelled file dialogs and `n_chunks > N` are handled; closing
  the window waits for a running worker.
- `requirements.txt` with pinned versions; `tests/` with pytest tests (headless).
