import argparse
import json
import time

import numpy as np

from fitting import error_map


def parse_range(text: str) -> list[int]:
    """Parse 'A-B' (inclusive) or a single integer into a list of ints."""
    if "-" in text:
        start, stop = (int(part) for part in text.split("-", 1))
        if stop < start:
            raise argparse.ArgumentTypeError(f"empty range: {text}")
        return list(range(start, stop + 1))
    return [int(text)]


def load_xy(path: str, max_rows: int = 1000):
    data = np.genfromtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if np.isnan(data[0, :2]).any():  # non-numeric first row: header
        data = data[1:]
    data = data[:max_rows, :2]
    return data[:, 0], data[:, 1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RMSE map over polynomial order x chunk count.")
    parser.add_argument("csv", help="CSV file; first two columns are x and y")
    parser.add_argument("--orders", type=parse_range, required=True, help="order range A-B or single int")
    parser.add_argument("--chunks", type=parse_range, required=True, help="chunk count range A-B or single int")
    parser.add_argument("--overlap", type=float, default=0.0, help="overlap percent (default 0)")
    parser.add_argument("--json", required=True, help="output JSON path")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    x, y = load_xy(args.csv)

    start = time.perf_counter()
    emap = error_map(x, y, args.orders, args.chunks, [args.overlap])
    seconds = time.perf_counter() - start

    i, j = np.unravel_index(np.nanargmin(emap), emap.shape)
    result = {
        "orders": args.orders,
        "chunks": args.chunks,
        "overlap": args.overlap,
        # NaN (failed cell) -> null so the JSON stays strict.
        "error_map": [[float(v) if np.isfinite(v) else None for v in row] for row in emap],
        "seconds": seconds,
        "best": {
            "order": int(args.orders[i]),
            "chunks": int(args.chunks[j]),
            "rmse": float(emap[i, j]),
        },
    }
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
