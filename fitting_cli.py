"""Benchmark CLI: RMSE error map over polynomial order, chunk count and overlap.

    python fitting_cli.py <csv> --orders 1-10 --chunks 1-10 [--overlap 0] --json out.json
"""

import argparse
import functools
import json
import sys
import time

import numpy as np

import fitting

MAX_ROWS = 1000


def load_xy(path, max_rows=MAX_ROWS):
    """First two CSV columns as x, y, first `max_rows` numeric rows. A non-numeric header row is skipped."""
    data = np.genfromtxt(path, delimiter=',', usecols=(0, 1), ndmin=2)
    data = data[~np.isnan(data).any(axis=1)][:max_rows]
    if len(data) < 2:
        raise ValueError(f"{path}: need at least 2 numeric rows, found {len(data)}")
    return data[:, 0], data[:, 1]


def parse_range(text, minimum):
    """'N' or 'A-B' (inclusive) -> list of ints."""
    try:
        if "-" in text:
            low_text, high_text = text.split("-", 1)
            low, high = int(low_text), int(high_text)
        else:
            low = high = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected N or A-B, got {text!r}")
    if low < minimum or high < low:
        raise argparse.ArgumentTypeError(f"range {text!r} must start at >= {minimum} and not be descending")
    return list(range(low, high + 1))


def parse_overlap(text):
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}")
    if not 0 <= value < 100:
        raise argparse.ArgumentTypeError("overlap must be in [0, 100)")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description="RMSE error map benchmark for piecewise polynomial fits.")
    parser.add_argument("csv", help="CSV file; first two columns are x and y")
    parser.add_argument("--orders", type=functools.partial(parse_range, minimum=0), required=True,
                        help="polynomial orders, e.g. 1-10")
    parser.add_argument("--chunks", type=functools.partial(parse_range, minimum=1), required=True,
                        help="chunk counts, e.g. 1-10")
    parser.add_argument("--overlap", type=parse_overlap, default=0.0,
                        help="overlap percentage (single number, default 0)")
    parser.add_argument("--json", required=True, help="output JSON file")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.orders) < 2 or len(args.chunks) < 2:
        parser.error("--orders and --chunks must each cover at least 2 values")

    try:
        x, y = load_xy(args.csv)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    start = time.perf_counter()
    try:
        error_map = fitting.error_map(x, y, args.orders, args.chunks, [args.overlap])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    seconds = time.perf_counter() - start

    # error_map is shaped (orders, chunks); pick the minimum-RMSE cell.
    i, j = np.unravel_index(np.nanargmin(error_map), error_map.shape)
    overlap = int(args.overlap) if float(args.overlap).is_integer() else args.overlap
    result = {
        "orders": args.orders,
        "chunks": args.chunks,
        "overlap": overlap,
        "error_map": error_map.tolist(),
        "seconds": seconds,
        "best": {"order": args.orders[i], "chunks": args.chunks[j], "rmse": float(error_map[i, j])},
    }
    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"best: order={result['best']['order']} chunks={result['best']['chunks']} "
          f"rmse={result['best']['rmse']:.6g} ({seconds:.3f} s) -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
