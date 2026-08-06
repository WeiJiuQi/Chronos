# Chronos

**Chronos** is a research framework for **time-decayed vector search (TDVS)**,
which retrieves timestamped vectors by jointly considering semantic similarity
and temporal freshness at query time. Unlike hard time filters or
post-retrieval reranking, TDVS incorporates continuous temporal decay directly
into the search objective.

**TANGO** (**Time-Aware Navigable Graph with Query-Orthogonal TimeLift**) is
the TDVS-native hierarchical graph index built on Chronos. It uses
layer-specific TimeLift geometries to preserve temporal locality at the base
layer and strengthen long-range semantic connectivity in upper layers, while
traversing the graph with the original TDVS score. TANGO also caches temporal
factors and supports efficient online insertion.

## Repository layout

```text
tango/                   TANGO headers and command-line tools
tdvs_workload_builder/   canonical workload and exact ground-truth tools
baselines/               comparison methods and common C++ runners
  runners/               build, footprint, and search entry points
  knng/                  shared MAG/PSP NN-Descent graph preparation
  ip-nsw/                stanis-morozov/ip-nsw source snapshot
  ip-nsw-plus/           GraphMIPS ip-NSW+ source snapshot
  napg/                   Chronos NAPG implementation
  mag/                    MAG/ANMS source snapshot
  psp/                    PSP source snapshot
```

## Requirements

- Linux or macOS for TANGO and the workload builder;
- CMake 3.21 or newer;
- a C++17 compiler;
- POSIX threads; and
- OpenMP for parallel workload generation, dynamic-checkpoint ground truth,
  MAG, and PSP (optional but recommended).

Python 3.10 or newer, NumPy, and the documented Faiss build are needed only for
the shared MAG/PSP kNN-graph generator. See
[baselines/knng/README.md](baselines/knng/README.md).

The complete baseline suite targets x86-64 Linux. On Apple Silicon or another
non-x86 host, configure with `-DCHRONOS_BUILD_BASELINES=OFF`.

On Ubuntu, the ordinary C++ build dependencies are:

```bash
sudo apt-get update
sudo apt-get install -y build-essential g++ cmake python3 libgomp1
```

## Build

Configure a portable Release build:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 7
```

`-DCHRONOS_NATIVE_OPTIMIZATION=ON` adds `-march=native` on supported compilers
for machine-specific benchmark builds. It is disabled by default so published
binaries are not tied to the build host's CPU.

Run public CLI smoke checks with:

```bash
make check JOBS=7
```

Build only one component by disabling the others:

```bash
cmake -S . -B build-tango \
  -DCMAKE_BUILD_TYPE=Release \
  -DCHRONOS_BUILD_BASELINES=OFF \
  -DCHRONOS_BUILD_TDVS_WORKLOAD_BUILDER=OFF
cmake --build build-tango --parallel 7
```

## Binary data contract

Canonical Chronos arrays are **headerless**, row-major binary files:

| Artifact | Type and shape |
| --- | --- |
| semantic base | `float32[N, dim]` |
| semantic query | `float32[Q, dim]` |
| timestamps | `float32[N]` |
| TDVS ground truth | `int64[Q, gt_width]` |
| source row IDs | `uint64[N]` or `uint64[Q]` |

Some historical output names use the `.fbin` suffix even though these canonical
files have no global header. Do not treat them as standard headered fbin files.
The workload builder accepts raw float32, fvecs, and standard headered fbin
sources and writes the canonical format above.

Row identity is invariant across every representation: base row `i`, timestamp
`i`, transformed-base row `i`, and every ground-truth ID refer to the same
logical object. Timestamps are creation times; larger values mean newer
objects, not larger ages. Timestamps, query times, and half-life values use
days throughout the current persisted-data contract.

## Build a workload

The workload builder implements four explicit stages:

```text
source base/query
    -> canonical unit-vector split
    -> topic-independent or topic-correlated timestamps
    -> exact TDVS ground truth
    -> optional transformed MIPS inputs
```

The five tools are:

```text
prepare_vector_subset
generate_time_metadata
generate_tdvs_groundtruth
export_mips_transforms
verify_dataset_alignment
```

Run any tool with `--help`, and see
[tdvs_workload_builder/README.md](tdvs_workload_builder/README.md) for complete
contracts and examples.

## Build and query TANGO

TANGO indexes the canonical semantic base and timestamps directly. The static
`tango_index` construction CLI requires an explicit kappa schedule. The
following values are the paper configuration, not hidden `tango_index`
defaults:

```bash
./build/tango/tango_index \
  --index-mode create --build-only \
  --index-prefix /data/indexes/tango_h90 \
  --mode multiplicative \
  --base /data/canonical/base.fbin \
  --timestamps /data/meta/topic-correlated/timestamp_days.f32bin \
  --n 1000000 --dim 1536 --half-life 90 \
  --M 25 --ef-construction 200 --threads 64 \
  --random-seed 100 --normalize 1 \
  --kappa-base 0.015873015873 \
  --kappa-nav 0.066666666667 \
  --tango-decay-fastpath all
```

The saved index consists of two files that must be moved together:

```text
PREFIX.hnsw
PREFIX.tango.meta
```

Query it in a fresh process:

```bash
./build/tango/tango_query \
  --index /data/indexes/tango_h90 \
  --query /data/canonical/query.fbin \
  --gt /data/gt/gt_chronos_multiplicative_cosine.hl90.k100.bin \
  --queries 1000 --dim 1536 --gt-width 100 --k 50 \
  --ef-list 50,75,100,150,200,300,500,700,1000,1500 \
  --query-time 365 --repeats 3 \
  --max-search-expansions 0 \
  --tango-decay-fastpath qd \
  --output-csv /data/results/tango_k50.csv
```

`tango_query` is serial by design. Input loading, normalization, query packing,
and Recall calculation are outside the measured search interval. Additional
commands are available through `tango_dynamic` and `tango_decay_profile`.

## Run comparison methods

The comparison methods consume mode-matched MIPS inputs produced by the
workload builder; they must not index TANGO's raw base directly.

| Method | Executable | Additional construction input |
| --- | --- | --- |
| ip-NSW | `tdvs_ip_nsw` | none |
| ip-NSW+ | `tdvs_ip_nsw_plus` | none |
| NAPG | `tdvs_napg` | none |
| MAG | `tdvs_mag` | transformed-base kNN graph |
| PSP | `tdvs_psp` | transformed-base kNN graph |

Each runner supports separate `build`, `search`, and `footprint` modes and
prints its complete current interface with `--help`:

```bash
./build/baselines/tdvs_ip_nsw --help
./build/baselines/tdvs_ip_nsw_plus --help
./build/baselines/tdvs_napg --help
./build/baselines/tdvs_mag --help
./build/baselines/tdvs_psp --help
```

## Reproducibility rules

- Create one immutable canonical base/query split per dataset and preserve its
  source row IDs. Every timestamp, transformed row, and ground-truth ID must
  refer to that same row order.
- Treat timestamps as creation times and use one time unit consistently for
  timestamps, query time, time range, and half-life. Record the timestamp
  distribution, seed, TDVS mode, half-life, query time, and additive alpha for
  every workload.
- Derive every method-specific artifact from the same canonical split and
  timestamp assignment. TANGO consumes the canonical base plus timestamps;
  comparison methods consume the matching MIPS transform. Do not L2-normalize
  a transformed base.
- Generate exact ground truth from the original TDVS score for every workload
  configuration. Regenerate it whenever the subset, timestamps, mode,
  half-life, query time, or alpha changes, and use a width at least as large as
  the maximum reported `k`.
- Record construction parameters, thread counts, seeds, compiler options,
  software revisions, and hardware. Report the shared kNN-graph construction
  cost separately when MAG or PSP uses one.
- Run query evaluation with one thread and zero warmup. Exclude file loading,
  normalization, query preparation, and Recall computation from the measured
  search interval, and use the same query set and repeat policy for all methods.
- Keep commands, manifests, and results together, and verify artifact shapes,
  row IDs, and score parameters before reusing an artifact.

## License

Chronos-owned source code is licensed under the [Apache License 2.0](LICENSE).
Bundled third-party components remain subject to their upstream licenses.
