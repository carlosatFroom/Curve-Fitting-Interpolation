"""Benchmark CLI for the piecewise fitting error map.

Usage:
    python fitting_cli.py <csv> --orders 1-10 --chunks 1-10 [--overlap 0] --json out.json
"""

import argparse
import json
import sys
import time

import numpy as np

import fitting


def parse_range(spec: str):
    spec = spec.strip()
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Piecewise polynomial error map benchmark")
    parser.add_argument("csv")
    parser.add_argument("--orders", default="1-10")
    parser.add_argument("--chunks", default="1-10")
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--json", dest="json_out", required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)

    orders = parse_range(args.orders)
    chunks = parse_range(args.chunks)

    data = np.genfromtxt(args.csv, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    x = data[:, 0][:1000].astype(float)
    y = data[:, 1][:1000].astype(float)

    t0 = time.perf_counter()
    emap = fitting.error_map(x, y, orders, chunks, [args.overlap], workers=args.workers)
    seconds = time.perf_counter() - t0

    i, j = np.unravel_index(int(np.argmin(emap)), emap.shape)
    best = {
        "order": int(orders[i]),
        "chunks": int(chunks[j]),
        "rmse": float(emap[i, j]),
    }

    out = {
        "orders": [int(o) for o in orders],
        "chunks": [int(c) for c in chunks],
        "overlap": args.overlap,
        "error_map": emap.tolist(),
        "seconds": float(seconds),
        "best": best,
    }
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
