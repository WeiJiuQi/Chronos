# MAG baseline

This directory vendors the original implementation of **Metric-Amphibious
Graph (MAG)** and **Adaptive Navigation with Metric Switch (ANMS)** released by
the authors of the SIGIR 2025 paper *Stitching Inner Product and Euclidean
Metrics for Topology-aware Maximum Inner Product Search*.

- Upstream: <https://github.com/ZJU-DAILY/MAG>
- Paper: <https://doi.org/10.1145/3726302.3730091>

Chronos uses MAG as a MIPS comparison method. The index consumes the
mode-matched transformed TDVS base and a fixed-row kNN graph. Multiplicative
search uses the canonical semantic query; additive search uses the paired
`dim+1` transformed query. Both use the corresponding TDVS ground-truth IDs.

## Chronos integration

Build the unified runner from the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 64 --target tdvs_mag
```

The runner supports:

```text
tdvs_mag --mode build ...
tdvs_mag --mode footprint ...
tdvs_mag --mode search ...
```

Run `./build/baselines/tdvs_mag --help` for the complete interface. Invoke
`--mode build`, `--mode search`, and `--mode footprint` directly;
[`../README.md`](../README.md) gives a complete command sequence. Shared runner
code provides consistent binary I/O, construction thread configuration,
single-threaded query timing, repeated query passes, Recall, footprint
measurement, and CSV output.

## Required kNN graph

MAG and PSP share the same headerless fixed-row graph:

```text
[uint32 degree][degree x uint32 neighbor IDs]
```

The released MAG code does not generate this graph: `Build` loads the supplied
path, and the upstream example script consumes an existing `*.knng` file. The
paper describes IVF-PQ for this preprocessing step, but the repository does not
publish an IVF-PQ generator or its parameters. Chronos can prepare an external
graph with `../knng/`; that method and its parameters are local protocol choices.
The generated manifest lets the runner report KNNG time separately and include
it in end-to-end index time.

## Default parameters

The unified runner uses the upstream script defaults:

| Parameter | Default |
| --- | ---: |
| construction `L` | 60 |
| Euclidean degree `R` | 48 |
| candidate budget `C` | 300 |
| inner-product degree `R_IP` | 20 |
| auxiliary parameter `M` | 64 |
| threshold | 8 |

The authors' ANMS query uses a fixed five-element Euclidean candidate pool
(`L_NN=5`) and runs that pool to convergence before switching to inner-product
search. `L_search` is the only query-time sweep parameter; its numeric value is
not equivalent to HNSW `ef`.
