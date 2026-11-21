"""Benchmark CLI: RMSE over polynomial order x chunk count on a CSV (no GUI imports)."""
import argparse
import json
import sys
import time

import numpy as np

import fitting


def parse_values(text):
    """'a-b' (inclusive), 'a,b,c' or a single int -> list of ints."""
    values = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    if not values or min(values) < 0:
        raise argparse.ArgumentTypeError(f"invalid range: {text!r}")
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark polynomial chunk fitting error map.")
    parser.add_argument("csv")
    parser.add_argument("--orders", type=parse_values, required=True)
    parser.add_argument("--chunks", type=parse_values, required=True)
    parser.add_argument("--overlap", type=float, default=0.0, help="overlap percent (single value)")
    parser.add_argument("--json", required=True, help="output file")
    args = parser.parse_args(argv)

    data = np.genfromtxt(args.csv, delimiter=",")
    if data.ndim != 2 or data.shape[1] < 2:
        parser.error("CSV needs at least two columns")
    x, y = data[:1000, 0], data[:1000, 1]
    if max(args.chunks) > len(x) or min(args.chunks) < 1:
        parser.error(f"--chunks must be in [1, {len(x)}]")

    start = time.perf_counter()
    if len(args.orders) > 1 and len(args.chunks) > 1:
        grid = fitting.error_map(x, y, args.orders, args.chunks, [args.overlap])
    else:
        # error_map needs two varying sequences; compute the (possibly 1xN / Nx1 / 1x1) cells directly.
        grid = np.empty((len(args.orders), len(args.chunks)))
        for i, order in enumerate(args.orders):
            for j, n_chunks in enumerate(args.chunks):
                fits = fitting.fit_chunks(x, y, n_chunks, order, args.overlap)
                grid[i, j] = fitting.rmse(y, fitting.evaluate_piecewise(fits, x, blend=False))
    seconds = time.perf_counter() - start

    i, j = np.unravel_index(np.argmin(grid), grid.shape)
    result = {
        "orders": args.orders,
        "chunks": args.chunks,
        "overlap": args.overlap,
        "error_map": grid.tolist(),
        "seconds": seconds,
        "best": {"order": args.orders[i], "chunks": args.chunks[j], "rmse": float(grid[i, j])},
    }
    with open(args.json, "w") as fh:
        json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
