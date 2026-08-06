# ip-NSW baseline

This directory contains the original core implementation released for
*Non-metric Similarity Graphs for Maximum Inner Product Search* (NeurIPS 2018).

- Upstream: <https://github.com/stanis-morozov/ip-nsw>
- Paper implementation files: `L2space.h`, `hnswalg.h`, `hnswlib.h`,
  `visited_list_pool.h`, and the original command-line program `main.cpp`

These files retain the code released by the paper authors, with nonfunctional
formatting, unused-macro, and dead-debug-code cleanup plus validation that
rejects out-of-range candidate IDs before array access. Chronos does not add
SIMD kernels, graph-selection changes, counters, thread-safety patches, or
persistence changes inside those sources. Chronos integration lives in
`../runners/ip_nsw.cpp`. The original `main.cpp` is retained for provenance and
reference, but Chronos workloads use `tdvs_ip_nsw` rather than the upstream CLI.

## Experiment contract

The runner indexes the exact transformed MIPS vectors produced by
`tdvs_workload_builder/`. For multiplicative TDVS, row `i` is

```text
y_i = exp(-lambda * (T - t_i)) * x_i.
```

For additive TDVS, it indexes the exported `dim+1` augmented base and receives
the matching `dim+1` augmented query. The transformed base must not be
L2-normalized in either mode because its norm or final coordinate carries the
temporal signal. Query normalization occurs before the timed search interval
and does not change the inner-product ranking.

The defaults match the upstream recommended experiment:

```text
M = 32
efConstruction = 1024
construction seed = 100
```

The upstream core fixes the RNG seed at 100. The Chronos runner therefore
rejects any other `--random-seed` value instead of modifying the core.

## Build and run

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 64 --target tdvs_ip_nsw
./build/baselines/tdvs_ip_nsw --help
```

Invoke the runner directly with `--mode build`, `--mode search`, and
`--mode footprint`; [`../README.md`](../README.md) gives a complete command
sequence. Search is serial, has no warmup, and emits one CSV row per requested
`efSearch` value, aggregated across the requested repeats, without altering the
upstream graph implementation.

## Reproducibility notes

- The index embeds all vectors and is not compatible with indexes produced by
  the superseded local implementation. A small `INDEX.ip_nsw.meta` sidecar
  records source provenance and build shape; it is produced and validated by
  the experiment runner, not the paper implementation.
- The upstream build path uses OpenMP insertion. Its source does not guarantee
  byte-for-byte deterministic parallel construction.
