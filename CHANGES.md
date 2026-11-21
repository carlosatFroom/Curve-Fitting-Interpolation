# Changes

## What was broken

- `from msilib import datasizemask` (Windows-only, unused) crashed the app on macOS/Linux.
  `os`/`sys`/`numpy` were only available through `from pyqtgraph import *`.
- Overlap chunk slicing used `chunkIndex * len(chunk)` as the start index, which is wrong
  whenever `np.array_split` yields unequal chunk lengths (chunks were shifted/overlapping).
- The order-vs-overlap error map used `range(1, maxOrder)` (one order short), and every
  "transposed" map was built from the un-normalized data while its partner was normalized
  (`normalizeData` also mutated its input in place).
- `np.polyfit` on raw x is ill-conditioned for high orders (and large x offsets).
- The "error %" was `sqrt(SSE / n_chunks / (N - 2)) * 100` — not an RMSE of anything.
- Extrapolation assumed exactly 1000 points (`slider * 100`, `max(x) * idx / 1000`).
- The error map was computed *and drawn with matplotlib* on a `threading.Thread`.
- Chunk equations went through sympy/pandas just to format a table.

## What changed

- **`fitting.py`** (new, pure numpy, no GUI imports) holds all math:
  - `chunk_bounds` — exact `array_split` partition plus the specified overlap rule.
  - `fit_chunks` / `ChunkFit` — fits via `numpy.polynomial.Polynomial.fit` (scaled domain);
    `coeffs` converted to original-x power basis for display only; order is reduced on short
    chunks; per-chunk RMSE.
  - `evaluate_piecewise` — base-chunk evaluation, or smoothstep cross-fade over overlaps
    (no jumps at seams).
  - `rmse`, `r2`, `error_map` (un-normalized RMSE, thread pool; 10×10 on 1000 pts ≈ 0.2 s),
    `extrapolate` (fraction of the data, any length).
  - Phase 2: `fit_spline` (one joint equality-constrained least-squares problem solved by
    null-space projection, Legendre basis per piece in a local [-1,1] variable, C⁻¹…C²),
    `adaptive_chunks` (greedy longest-feasible segment via exponential+binary search, then
    least-damage merging down to `max_chunks`), `robust_fit_chunks` (IRLS: Huber then Tukey
    bisquare with MAD scale), `select_order` (interleaved k-fold CV or AIC).
- **`fitting_cli.py`** — benchmark CLI writing the error map + best cell as JSON.
- **`app.py`** — uses `fitting` for everything; error maps are computed in a `QThread`
  worker that only does numpy and emits the result back to the GUI thread, where all
  drawing happens. Maps are normalized for display only (both orientations), axes come
  from the actual parameter values, table shows real per-chunk RMSE, equation LaTeX is
  formatted locally (sympy and pandas dropped). Extrapolation uses a fraction of the data.
- **`requirements.txt`** pinned; **`tests/`** covers every public function plus a headless
  (`QT_QPA_PLATFORM=offscreen`) app smoke test.
