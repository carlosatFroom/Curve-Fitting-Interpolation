"""Benchmark CLI: error map of piecewise polynomial fits over a CSV signal.

Usage:
    python fitting_cli.py <csv> --orders 1-10 --chunks 1-10 [--overlap 0] --json out.json
"""

import argparse
import csv
import json
import time

import numpy as np

import fitting


def parse_range(text):
    """Parse '1-10' (inclusive) or a single integer '5' into a list of ints."""
    text = text.strip()
    if "-" in text:
        lo, hi = text.split("-", 1)
        lo, hi = int(lo), int(hi)
        if hi < lo:
            raise argparse.ArgumentTypeError(f"empty range: {text}")
        return list(range(lo, hi + 1))
    return [int(text)]


def load_xy(path, max_rows=1000):
    xs, ys = [], []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(xs) >= max_rows:
                break
            if len(row) < 2:
                continue
            try:
                x, y = float(row[0]), float(row[1])
            except ValueError:
                continue  # header or non-numeric row
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark fitting.error_map")
    parser.add_argument("csv")
    parser.add_argument("--orders", type=parse_range, required=True)
    parser.add_argument("--chunks", type=parse_range, required=True)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--json", required=True, dest="json_path")
    args = parser.parse_args(argv)

    x, y = load_xy(args.csv)

    start = time.perf_counter()
    grid = fitting.error_map(x, y, args.orders, args.chunks, [args.overlap])
    seconds = time.perf_counter() - start

    # error_map(orders, chunks, [overlap]) has axis 0 = orders, axis 1 = chunks
    grid = np.asarray(grid, dtype=float)
    i, j = np.unravel_index(int(np.argmin(grid)), grid.shape)
    result = {
        "orders": args.orders,
        "chunks": args.chunks,
        "overlap": args.overlap,
        "error_map": grid.tolist(),
        "seconds": seconds,
        "best": {
            "order": int(args.orders[i]),
            "chunks": int(args.chunks[j]),
            "rmse": float(grid[i, j]),
        },
    }
    with open(args.json_path, "w") as fh:
        json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
