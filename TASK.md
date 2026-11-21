# Task: make the curve-fitting algorithm correct, testable, fast and better

`app.py` welds the fitting algorithm to the PyQt GUI and has several bugs.
Extract the algorithm into a pure module, fix it, improve it, and test it.
Work only inside this repository.

## Known problems (not exhaustive, find others)

- `from msilib import datasizemask` is Windows-only; the app crashes on import elsewhere.
- Overlap chunk slicing uses `chunkIndex * len(chunk)`, which is wrong when
  `np.array_split` produces chunks of unequal length.
- The third error map uses `range(1, maxOrder)` (off by one); the transposed maps
  are built from un-normalized data.
- `np.polyfit` on unscaled x is ill-conditioned for high orders.
- The "error %" metric is not a real RMSE.
- Extrapolation hard-codes 1000 points.
- Matplotlib drawing happens on a worker thread.

## Required deliverable: `fitting.py` (repo root)

Pure numpy (scipy allowed). Importing it must NOT import PyQt5, pyqtgraph or matplotlib.
Must expose exactly these names with these semantics:

```python
chunk_bounds(n_points: int, n_chunks: int, overlap_pct: float = 0.0) -> list[tuple[int, int]]
```
Half-open `(start, stop)` index ranges, one per chunk. Base chunks are the
partition produced by `np.array_split(np.arange(n_points), n_chunks)`. With
`overlap_pct > 0`, let `ov = int(base_len_of_that_chunk * overlap_pct / 100)`;
the first chunk extends `stop` by `ov`, the last extends `start` back by `ov`,
middle chunks extend `start` back by `ov // 2` and `stop` forward by `ov - ov // 2`.
Always clipped to `[0, n_points]`. Raise `ValueError` if `n_chunks < 1` or
`n_chunks > n_points`.

```python
class ChunkFit:  # dataclass or similar
    start: int; stop: int          # the slice that was fitted
    x_min: float; x_max: float     # x range of the fitted slice
    coeffs: np.ndarray             # power-basis coefficients in ORIGINAL x, highest degree first (np.polyfit convention)
    rmse: float                    # RMSE of this chunk's fit on its own slice
    def predict(self, xq) -> np.ndarray
```
`predict` must be numerically stable for high orders (fit in a scaled domain,
e.g. `numpy.polynomial.Polynomial.fit`); `coeffs` is provided for display only.

```python
fit_chunks(x, y, n_chunks: int, order: int, overlap_pct: float = 0.0) -> list[ChunkFit]
```
If a chunk has `<= order` points, reduce that chunk's order to `len - 1` instead of failing.

```python
evaluate_piecewise(fits: list[ChunkFit], xq, blend: bool = True) -> np.ndarray
```
Evaluate the piecewise model at `xq`. With `blend=False` each point uses the
chunk whose base range contains it. With `blend=True`, inside the overlap of two
adjacent chunks the result is a smooth (linear or better) cross-fade of the two
predictions, so the curve has no jumps at seams.

```python
rmse(y_true, y_pred) -> float          # sqrt(mean((y_true - y_pred)**2))
r2(y_true, y_pred) -> float
```

```python
error_map(x, y, orders, chunk_counts, overlap_pcts, workers: int | None = None) -> np.ndarray
```
Exactly two of the three sequences have length > 1 (the third has length 1).
Return a 2-D array of the **un-normalized** RMSE of `evaluate_piecewise(fit_chunks(...), x, blend=False)`
against `y` over all points. Axis 0 = first varying argument in the order
(orders, chunk_counts, overlap_pcts), axis 1 = second varying argument.
Use `concurrent.futures` when `workers` is not 1. Must be fast: orders 1..10 ×
chunks 1..10 on 1000 points should take well under 2 s.

```python
extrapolate(x, y, fit_fraction: float, order: int) -> tuple[np.ndarray, np.ndarray]
```
Fit one polynomial to the first `fit_fraction` (0 < f <= 1) of the points
(at least `order + 1` points), return `(coeffs, y_pred_over_all_x)`.


## Phase 2: algorithm improvements (also required, also in `fitting.py`)

```python
fit_spline(x, y, n_chunks: int, order: int, continuity: int = 2) -> SplineFit
```
One polynomial of degree `order` per base chunk (`chunk_bounds(n, n_chunks, 0)`), solved
**jointly** as a single constrained least-squares problem (minimise total squared error over
all points subject to the constraints, not fit-then-patch). Knot `i` is `x[start of base chunk i+1]`;
a point at a knot belongs to the right-hand piece. At every knot the value and the first
`continuity` derivatives of adjacent pieces must be equal (`continuity` = -1 means no
constraint, 0 = C0, 1 = C1, 2 = C2). Raise `ValueError` unless `-1 <= continuity < order`.
Must be numerically stable for order up to 10 and 50 chunks.
`SplineFit` exposes `knots` (np.ndarray), `predict(xq)` and `derivative(xq, m=1)`
(m-th derivative w.r.t. original x, m >= 0).

```python
adaptive_chunks(x, y, order: int, tol: float, max_chunks: int) -> list[tuple[int, int]]
```
Choose chunk boundaries automatically: return a contiguous partition of `[0, len(x))`
(half-open, sorted) such that a degree-`order` fit on every chunk has RMSE <= `tol`,
using as few chunks as reasonably possible and never more than `max_chunks`. A chunk with
fewer than `2 * (order + 1)` points may exceed `tol`.

```python
robust_fit_chunks(x, y, n_chunks: int, order: int, overlap_pct: float = 0.0) -> list[ChunkFit]
```
Same contract as `fit_chunks`, but resistant to outliers (e.g. Huber/Tukey IRLS with a
robust scale estimate). With 10% gross outliers the fit should track the clean signal.

```python
select_order(x, y, n_chunks: int, max_order: int, method: str = "cv") -> int
```
Pick the polynomial order in `[0, max_order]` for `fit_chunks` with `n_chunks` chunks.
`method="cv"`: k-fold cross-validation; `method="aic"`: Akaike information criterion.

### Benchmark CLI: `fitting_cli.py` (repo root)

`python fitting_cli.py <csv> --orders 1-10 --chunks 1-10 [--overlap 0] --json out.json`
reads the first two CSV columns as x, y (first 1000 rows), computes `error_map` and writes JSON:
`{"orders": [...], "chunks": [...], "overlap": 0, "error_map": [[...]], "seconds": float,
"best": {"order": int, "chunks": int, "rmse": float}}` where `error_map[i][j]` is for
`orders[i]`, `chunks[j]` and `best` is the minimum-RMSE cell. Exit code 0 on success.

## Also required

- `app.py` uses `fitting.py` for all math, runs on macOS/Linux, and does no Qt/matplotlib
  drawing off the GUI thread. Keep the GUI behaviour otherwise the same.
- `requirements.txt` with every dependency pinned (`pkg==x.y.z`).
- `tests/` with pytest tests for `fitting.py`. `pytest -q` must pass with no display.
- Short `CHANGES.md`: what was broken, what you changed, and why.

## Done means

`pytest -q` passes, `python -c "import fitting"` works without PyQt installed,
everything committed on the current branch.
