"""Benchmark CLI: compute an order x chunks error map for a CSV and write JSON."""
import argparse
import json
import sys
import time

import numpy as np

import fitting


def parse_range(text: str) -> list[int]:
    """'1-10' -> [1..10], '3' -> [3], '1,4,7' -> [1, 4, 7]."""
    values = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            values.extend(range(lo, hi + 1))
        elif part:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError(f"empty range: {text!r}")
    return values


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--orders", type=parse_range, default=parse_range("1-10"))
    ap.add_argument("--chunks", type=parse_range, default=parse_range("1-10"))
    ap.add_argument("--overlap", type=float, default=0.0)
    ap.add_argument("--json", dest="json_out", required=True)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)

    data = np.genfromtxt(args.csv, delimiter=",", usecols=(0, 1))
    data = data[~np.isnan(data).any(axis=1)][:1000]
    x, y = data[:, 0], data[:, 1]

    t0 = time.perf_counter()
    if len(args.orders) > 1 and len(args.chunks) > 1:
        emap = fitting.error_map(x, y, args.orders, args.chunks, [args.overlap], workers=args.workers)
    else:
        # error_map needs two varying axes; a degenerate grid is computed directly.
        emap = np.array([[fitting.error_map(x, y, [o, o], [c, c], [args.overlap])[0, 0]
                          for c in args.chunks] for o in args.orders])
    seconds = time.perf_counter() - t0

    i, j = np.unravel_index(np.nanargmin(emap), emap.shape)
    overlap = int(args.overlap) if float(args.overlap).is_integer() else args.overlap
    result = {
        "orders": args.orders,
        "chunks": args.chunks,
        "overlap": overlap,
        "error_map": [[None if np.isnan(v) else float(v) for v in row] for row in emap],
        "seconds": seconds,
        "best": {"order": int(args.orders[i]), "chunks": int(args.chunks[j]),
                 "rmse": float(emap[i, j])},
    }
    with open(args.json_out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"best: order={result['best']['order']} chunks={result['best']['chunks']} "
          f"rmse={result['best']['rmse']:.6g} ({seconds:.3f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
