"""Benchmark error_map over polynomial orders and chunk counts for a CSV file."""
import argparse
import json
import sys
import time

import numpy as np

import fitting

MAX_ROWS = 1000


def parse_ints(text: str) -> list[int]:
    """Parse '1-10', '1,3,5', '4' or any mix ('1-3,7') into a list of ints."""
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        try:
            if "-" in part:
                lo, hi = (int(v) for v in part.split("-", 1))
                values.extend(range(lo, hi + 1))
            else:
                values.append(int(part))
        except ValueError:
            raise argparse.ArgumentTypeError(f"bad integer spec: {part!r}")
    if not values or min(values) < 0:
        raise argparse.ArgumentTypeError(f"values must be >= 0: {text!r}")
    return values


def load_xy(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", usecols=(0, 1), max_rows=MAX_ROWS)
    data = data[~np.isnan(data).any(axis=1)]  # drops header/blank lines
    if len(data) < 2:
        raise SystemExit(f"{path}: need at least 2 numeric rows in the first two columns")
    return data[:, 0], data[:, 1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV file; first two columns are x and y")
    parser.add_argument("--orders", type=parse_ints, required=True, help="e.g. 1-10 or 1,3,5")
    parser.add_argument("--chunks", type=parse_ints, required=True, help="e.g. 1-10 or 1,3,5")
    parser.add_argument("--overlap", type=float, default=0.0, help="overlap percent (default 0)")
    parser.add_argument("--json", dest="json_path", help="write results to this JSON file")
    args = parser.parse_args(argv)

    if len(args.orders) < 2 or len(args.chunks) < 2:
        parser.error("--orders and --chunks must each cover more than one value")

    x, y = load_xy(args.csv)

    start = time.perf_counter()
    try:
        em = fitting.error_map(x, y, args.orders, args.chunks, [args.overlap])
    except ValueError as exc:
        parser.error(str(exc))
    seconds = time.perf_counter() - start

    i, j = np.unravel_index(np.nanargmin(em), em.shape)
    best = {"order": int(args.orders[i]), "chunks": int(args.chunks[j]), "rmse": float(em[i, j])}
    result = {
        "orders": args.orders,
        "chunks": args.chunks,
        "overlap": int(args.overlap) if float(args.overlap).is_integer() else args.overlap,
        "error_map": [[None if np.isnan(v) else float(v) for v in row] for row in em],
        "seconds": seconds,
        "best": best,
    }

    if args.json_path:
        with open(args.json_path, "w") as fh:
            json.dump(result, fh, indent=2)

    print(
        f"{len(x)} points, {em.size} cells in {seconds:.3f} s; "
        f"best order={best['order']} chunks={best['chunks']} rmse={best['rmse']:.6g}"
    )
    if args.json_path:
        print(f"wrote {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
