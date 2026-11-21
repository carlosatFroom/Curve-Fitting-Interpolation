# Changes

## What was broken

- `from msilib import datasizemask` (Windows-only, unused) crashed the app on macOS/Linux;
  `os`, `sys`, `numpy` were used but never imported.
- Overlap slicing used `chunkIndex * len(chunk)` as the chunk start, which is wrong when
  `np.array_split` yields chunks of unequal length (chunks shifted, points skipped/duplicated).
- The orders × overlap error map used `range(1, maxOrder)` (last order missing), and the
  transposed maps were built from un-normalized data while the others were normalized.
- `np.polyfit` on raw x is ill-conditioned for high orders (e.g. x ≈ 1000, order 10).
- The "error %" was `sqrt(sum_sq / n_chunks / (N - 2)) * 100`, not an RMSE of anything.
- Extrapolation hard-coded 1000 points (`slider * 100`, `max(x) * idx / 1000`).
- Matplotlib drawing (`showErrorMap`) ran on a `threading.Thread`, which Qt forbids.

## What changed

- New `fitting.py`: pure numpy, no GUI imports. `chunk_bounds` computes exact
  half-open ranges from the `array_split` partition; `fit_chunks` fits each chunk with
  `Polynomial.fit` (scaled domain, stable at high order; `coeffs` converted to original x
  for display) and lowers the order for tiny chunks; `evaluate_piecewise` evaluates per
  base chunk or smoothstep-blends inside overlaps so seams have no jumps; `rmse`/`r2` are
  the standard definitions; `error_map` returns un-normalized RMSE maps, computed in a
  thread pool; `extrapolate` works for any number of points.
- Phase 2 in `fitting.py`: `fit_spline` (one joint equality-constrained least-squares
  solve via the constraint null space, Legendre basis per piece in a scaled variable, so
  order 10 × 50 chunks is stable; `C^continuity` at knots), `adaptive_chunks` (greedy
  longest-feasible chunk via binary search, capped at `max_chunks`), `robust_fit_chunks`
  (Tukey bisquare IRLS, MAD scale), `select_order` (5-fold CV or AIC).
- `fitting_cli.py`: benchmark CLI writing the error map, timing and best cell as JSON.
- `app.py`: all math delegated to `fitting.py`; error maps computed in a worker and
  delivered to the GUI thread by a Qt signal, where all drawing happens; maps normalized
  once and transposed from the same normalized data; per-chunk RMSE shown in the table;
  extrapolation uses the actual fitted x range.
- `requirements.txt` with pinned versions; `tests/` with pytest coverage of `fitting.py`
  and the CLI.
