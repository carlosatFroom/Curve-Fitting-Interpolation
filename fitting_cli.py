"""Benchmark CLI: python fitting_cli.py data.csv --orders 1-10 --chunks 1-10 --json out.json"""
import argparse
import json
import sys
import time

import numpy as np

from fitting import _error_cells

MAX_ROWS = 1000


def parse_range(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part[1:]:
            lo, hi = part.split("-", 1)
            values.extend(range(int(lo), int(hi) + 1))
        else:
            values.append(int(part))
    if not values:
        raise ValueError(f"empty range: {text!r}")
    return values


def load_xy(path):
    data = np.genfromtxt(path, delimiter=",", usecols=(0, 1), ndmin=2)
    data = data[np.isfinite(data).all(axis=1)][:MAX_ROWS]
    return data[:, 0], data[:, 1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--orders", default="1-10")
    ap.add_argument("--chunks", default="1-10")
    ap.add_argument("--overlap", type=float, default=0.0)
    ap.add_argument("--json", required=True, help="output file")
    args = ap.parse_args(argv)

    try:
        orders, chunks = parse_range(args.orders), parse_range(args.chunks)
        x, y = load_xy(args.csv)
        if len(x) == 0:
            raise ValueError("no numeric rows found")
        cells = [(o, c, args.overlap) for o in orders for c in chunks]
        start = time.perf_counter()
        vals = _error_cells(x, y, cells)
        seconds = time.perf_counter() - start
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    grid = np.array(vals).reshape(len(orders), len(chunks))
    i, j = np.unravel_index(int(np.argmin(grid)), grid.shape)
    overlap = int(args.overlap) if args.overlap == int(args.overlap) else args.overlap
    result = {
        "orders": orders, "chunks": chunks, "overlap": overlap,
        "error_map": grid.tolist(), "seconds": seconds,
        "best": {"order": orders[i], "chunks": chunks[j], "rmse": float(grid[i, j])},
    }
    with open(args.json, "w") as fh:
        json.dump(result, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
