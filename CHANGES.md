# Changes

## What was broken (in the original `app.py`)

- `from msilib import datasizemask` — a Windows-only module; the app crashed on
  import on macOS/Linux.
- Overlap chunk slicing used `chunkIndex * len(chunk)` as the start index, which
  is wrong whenever `np.array_split` produces chunks of unequal length (the
  common case): chunk starts no longer line up with the data.
- The order×overlap error map used `range(1, maxOrder)` (off by one, dropping the
  top order), and the transposed maps were built from **un-normalized** data
  while the originals were normalized — so paired views disagreed.
- `np.polyfit` was run on the raw, unscaled x. For high orders and large x this
  is badly ill-conditioned and the fit/prediction blow up.
- The "error %" metric was `sqrt(sum_sq / numberOfChunks / (N-2)) * 100` — not a
  real RMSE (nonsensical normalization by chunk count and `N-2`).
- Extrapolation hard-coded `1000` points and indexed with magic numbers.
- Matplotlib drawing (`draw()`, table/colorbar building) happened on a worker
  thread, which is not allowed — Qt/matplotlib drawing must be on the GUI thread.

## What was changed and why

- **New pure module `fitting.py`** holds all the math. Importing it does not
  import PyQt5, pyqtgraph or matplotlib (verified by a test), so the algorithm is
  testable headless and reusable.
  - `chunk_bounds` returns correct half-open `(start, stop)` ranges based on
    `np.array_split`, with the specified overlap extension rules and clipping,
    and validates `n_chunks`.
  - `ChunkFit.predict` fits in a **scaled domain** (`numpy.polynomial.Polynomial`)
    so it is numerically stable for high orders; `coeffs` (power basis, highest
    degree first) is kept for display only.
  - `fit_chunks` reduces a chunk's order to `len-1` when a chunk has too few
    points instead of failing.
  - `evaluate_piecewise` assigns each point to the chunk whose **base** range
    contains it (point at a knot → right piece); with `blend=True` it does a
    linear cross-fade across overlap regions so there are no seam jumps.
  - `rmse`/`r2` are real metrics.
  - `error_map` returns the un-normalized RMSE over all points, uses
    `concurrent.futures` (threads) and runs 10×10 on 1000 points in well under 2s.
  - `extrapolate` fits the first `fit_fraction` of the data (no hard-coded 1000).
- **Phase 2** (also in `fitting.py`): `fit_spline` (jointly-solved constrained
  least squares with C0..C2 continuity via a KKT system, per-piece scaling for
  conditioning up to order 10 / 50 chunks), `adaptive_chunks` (greedy
  tolerance-driven partition, capped at `max_chunks`), `robust_fit_chunks`
  (Huber IRLS with a MAD scale estimate), and `select_order` (k-fold CV / AIC).
- **`fitting_cli.py`** benchmark CLI writing the required JSON.
- **`app.py`** now uses `fitting.py` for every computation, drops the `msilib`
  import, adds the missing explicit imports (`os`, `sys`, `numpy`, …), and moves
  all error-map drawing onto the GUI thread via a `pyqtSignal` emitted from the
  worker thread. GUI behaviour is otherwise unchanged.
- **`requirements.txt`** pins every dependency.
- **`tests/`** has pytest coverage for `fitting.py`; `pytest -q` passes with no
  display.
