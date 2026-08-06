#!/usr/bin/env python3
"""Estimate recall of a fixed-row kNN graph against exact Faiss L2 search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss  # type: ignore
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--knng", required=True, type=Path)
    parser.add_argument("--dim", required=True, type=int)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path,
                        help="optional JSON diagnostic report")
    args = parser.parse_args()

    row_bytes = args.dim * np.dtype("<f4").itemsize
    if args.dim <= 0 or args.k <= 0 or args.samples <= 0 or args.threads <= 0:
        raise ValueError("dim, k, samples, and threads must be positive")
    if args.base.stat().st_size % row_bytes:
        raise ValueError("base is not a headerless float32 matrix of the requested dim")
    rows = args.base.stat().st_size // row_bytes
    expected_knng_bytes = rows * (args.k + 1) * np.dtype("<u4").itemsize
    if args.knng.stat().st_size != expected_knng_bytes:
        raise ValueError("kNN graph shape does not match base rows and k")
    if args.k >= rows or args.samples > rows:
        raise ValueError("require k < rows and samples <= rows")

    base = np.memmap(args.base, dtype="<f4", mode="r", shape=(rows, args.dim))
    graph = np.memmap(args.knng, dtype="<u4", mode="r", shape=(rows, args.k + 1))
    if not np.all(graph[:, 0] == args.k):
        raise ValueError("kNN graph contains a row with the wrong degree header")

    rng = np.random.default_rng(args.seed)
    sample_ids = np.sort(rng.choice(rows, size=args.samples, replace=False))
    queries = np.ascontiguousarray(base[sample_ids])
    faiss.omp_set_num_threads(args.threads)
    exact = faiss.IndexFlatL2(args.dim)
    exact.add(base)
    _, exact_ids = exact.search(queries, args.k + 1)

    recalls: list[float] = []
    for offset, row_id in enumerate(sample_ids):
        truth = [int(value) for value in exact_ids[offset] if int(value) != int(row_id)]
        truth = truth[: args.k]
        observed = [int(value) for value in graph[row_id, 1:]]
        recalls.append(len(set(truth).intersection(observed)) / args.k)
    values = np.asarray(recalls, dtype=np.float64)
    mean = float(values.mean())
    minimum = float(values.min())
    p05 = float(np.quantile(values, 0.05))
    p50 = float(np.quantile(values, 0.50))
    print(f"rows={rows}")
    print(f"dim={args.dim}")
    print(f"k={args.k}")
    print(f"samples={args.samples}")
    print(f"threads={args.threads}")
    print(f"recall_mean={mean:.10f}")
    print(f"recall_min={minimum:.10f}")
    print(f"recall_p05={p05:.10f}")
    print(f"recall_p50={p50:.10f}")
    print("acceptance_policy=diagnostic_only_no_threshold")
    report = {
        "schema": "chronos_knng_validation_v1",
        "base": str(args.base.resolve()),
        "knng": str(args.knng.resolve()),
        "rows": rows,
        "dim": args.dim,
        "k": args.k,
        "samples": args.samples,
        "threads": args.threads,
        "seed": args.seed,
        "sample_ids": [int(value) for value in sample_ids],
        "sample_recalls": recalls,
        "recall_mean": mean,
        "recall_min": minimum,
        "recall_p05": p05,
        "recall_p50": p50,
        "acceptance_policy": "diagnostic_only_no_threshold",
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_name(args.report.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.report)
        print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
