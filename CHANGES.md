# CHANGES

## What was broken (in `app.py`)

- **`from msilib import datasizemask`** — Windows-only stdlib module; the app
  crashed on import on macOS/Linux. The import was also unused.
- **Wrong overlap slicing** — `chunkStartIndex = chunkIndex * len(chunk)` assumes
  every chunk has the same length, but `np.array_split` produces unequal chunks,
  so the computed index was wrong for all but the trivial case.
- **Off-by-one error map** — the order×overlap map used `range(1, maxOrder)`,
  dropping the top order; the transposed maps were also built from data that had
  already been normalized in place, corrupting them.
- **Ill-conditioned fits** — `np.polyfit` on raw (unscaled) x is unstable for high
  orders (Vandermonde blow-up).
- **Fake "error %"** — the reported metric was `sqrt(SSE / numChunks / (N-2)) * 100`,
  which is neither a per-chunk RMSE nor a percentage of anything meaningful.
- **Hard-coded extrapolation** — the fit window was `sliderValue * 100` points with a
  magic `1000` denominator instead of a fraction of the data.
- **Drawing off the GUI thread** — the error map ran in a `threading.Thread` that
  then called matplotlib draw code directly, i.e. Qt/matplotlib drawing from a
  worker thread (undefined behaviour / crashes).

## What changed

- **New pure module `fitting.py`** (numpy only; importing it does **not** import
  PyQt5, pyqtgraph or matplotlib). It holds all the math:
  - `chunk_bounds` — half-open index ranges from `np.array_split`, with the exact
    overlap rule from the spec (`ov = int(base_len * pct/100)`; first extends stop,
    last extends start, middle splits `ov//2` / `ov-ov//2`), clipped to `[0, n]`,
    `ValueError` outside `1 <= n_chunks <= n_points`.
  - `ChunkFit` + `fit_chunks` — one polynomial per (possibly overlapping) chunk.
    Fitting and `predict` go through `numpy.polynomial.Polynomial.fit`, which works
    in a scaled domain, so high orders are stable; `coeffs` (power basis in original
    x, highest degree first) is carried for display only. A chunk with `<= order`
    points drops to `len-1` instead of failing.
  - `evaluate_piecewise` — base-range ownership for `blend=False`; a linear
    cross-fade across the x-overlap of adjacent chunks for `blend=True`, so seams
    have no jumps.
  - `rmse`, `r2` — the real metrics.
  - `error_map` — un-normalized RMSE grid over the two varying parameter sequences,
    `concurrent.futures` thread pool unless `workers == 1`. Orders 1..10 × chunks
    1..10 on 1000 points runs in ~0.2 s.
  - `extrapolate` — fit the first `fit_fraction` of the points (>= order+1), predict
    over all x.
  - **Phase 2:** `fit_spline`/`SplineFit` (one degree-`order` polynomial per base
    chunk solved *jointly* as one constrained least-squares problem with C0..C`k`
    continuity at the knots, via a null-space solve with per-piece local scaling —
    stable to order 10 × 50 chunks; exposes `knots`, `predict`, `derivative`),
    `adaptive_chunks` (greedy gapless partition within a tolerance),
    `robust_fit_chunks` (Huber IRLS, tracks the signal through 10% gross outliers),
    `select_order` (k-fold CV or AIC).
- **New `fitting_cli.py`** — benchmark CLI that reads the first two CSV columns
  (first 1000 rows), computes `error_map` and writes the JSON contract from the task
  (`orders`, `chunks`, `overlap`, `error_map`, `seconds`, `best`). Exit code 0.
- **`app.py` rewired** — uses `fitting.py` for every calculation; the `msilib` import
  is gone; the error map is computed in a `QThread` worker that **emits a signal**,
  and all matplotlib drawing happens in the GUI-thread slot; cancel sets a flag the
  worker checks between maps; the per-chunk table now shows a real **RMSE** column;
  extrapolation maps the slider to a fraction. GUI behaviour is otherwise unchanged.
- **`requirements.txt`** — all runtime deps pinned.
- **`tests/`** — pytest for `fitting.py` (`test_fitting.py`, `test_spline.py`,
  `test_phase2.py`); `pytest -q` passes headless.

## Why

The algorithm was welded to the GUI, untestable, numerically fragile, and reported a
meaningless error metric. Splitting the math into a pure, scaled-domain numpy module
makes it correct, importable without a display, fast, and unit-tested, while the GUI
keeps its behaviour and no longer draws from a worker thread.
