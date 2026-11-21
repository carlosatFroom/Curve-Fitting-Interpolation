import argparse
import json
import sys
import time

import numpy as np

import fitting


def parse_range(text):
    if "-" in text:
        lo, hi = text.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(v) for v in text.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark piecewise polynomial error map")
    ap.add_argument("csv")
    ap.add_argument("--orders", default="1-10")
    ap.add_argument("--chunks", default="1-10")
    ap.add_argument("--overlap", type=float, default=0)
    ap.add_argument("--json", required=True, dest="json_path")
    args = ap.parse_args(argv)

    data = np.genfromtxt(args.csv, delimiter=",")[:1000]
    x, y = data[:, 0], data[:, 1]
    orders, chunks = parse_range(args.orders), parse_range(args.chunks)

    t0 = time.perf_counter()
    emap = fitting.error_map(x, y, orders, chunks, [args.overlap])
    seconds = time.perf_counter() - t0

    i, j = np.unravel_index(int(np.argmin(emap)), emap.shape)
    overlap = int(args.overlap) if float(args.overlap).is_integer() else args.overlap
    out = {
        "orders": orders,
        "chunks": chunks,
        "overlap": overlap,
        "error_map": emap.tolist(),
        "seconds": seconds,
        "best": {"order": orders[i], "chunks": chunks[j], "rmse": float(emap[i, j])},
    }
    with open(args.json_path, "w") as f:
        json.dump(out, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
