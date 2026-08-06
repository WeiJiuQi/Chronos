# Baseline methods

This directory contains the five MIPS comparison methods used with TANGO. Each
method has a command-line runner in `runners/` with the same three-stage
lifecycle: build an index, measure its load-time footprint, and sweep a
single-threaded query budget. The runners share binary I/O, validation, timing,
Recall, footprint, provenance, and CSV code from `include/evaluation.h`.

## Implementations

- [ip-NSW](https://github.com/stanis-morozov/ip-nsw)
- [ip-NSW+ (GraphMIPS)](https://github.com/Jerry-liujie/ip-nsw/tree/GraphMIPS)
- [PSP](https://github.com/ZJU-DAILY/PSP)
- [MAG](https://github.com/ZJU-DAILY/MAG)
- NAPG has no publicly released author implementation. The implementation in
  this repository was developed on top of
  [hnswlib](https://github.com/nmslib/hnswlib).

## Directory layout

| Path | Role |
| --- | --- |
| `runners/` | build, footprint, and search CLI for each method |
| `include/evaluation.h` | shared CLI, binary I/O, validation, timing, Recall, CSV, and footprint support |
| `include/boost/dynamic_bitset.hpp` | minimal Boost-compatible container used by MAG and PSP |
| `knng/` | shared MAG/PSP kNN-graph construction and validation |
| `ip-nsw/`, `ip-nsw-plus/`, `napg/`, `mag/`, `psp/` | method implementations and retained upstream material |

| Method | Implementation | Executable | Query budget |
| --- | --- | --- | --- |
| ip-NSW | `ip-nsw/` | `tdvs_ip_nsw` | `efSearch` |
| ip-NSW+ / GraphMIPS | `ip-nsw-plus/` | `tdvs_ip_nsw_plus` | `efSearch` |
| NAPG | `napg/` | `tdvs_napg` | `efSearch` |
| MAG / ANMS | `mag/` | `tdvs_mag` | `L_search` |
| PSP | `psp/` | `tdvs_psp` | `L_search` |

## Input data contract

All five methods index the transformed MIPS representation produced by
`tdvs_workload_builder/`, not the raw base and timestamps consumed by TANGO.
Files named `*.fbin` in the examples are headerless, row-major `float32`
matrices; the dimension is supplied explicitly. Ground truth is headerless
`int64[query_count, gt_width]` by default. Base row IDs must be preserved
through transformation and ground-truth generation.

For multiplicative TDVS, transformed row `i` is

```text
y_i = exp(-lambda * (T - t_i)) * x_i.
```

The runner receives this `dim`-dimensional base and the canonical semantic
query. For additive TDVS, the paired transform is

```text
x'_i = [sqrt(alpha) * x_i, sqrt(1-alpha) * decay_i]
q'   = [sqrt(alpha) * q,   sqrt(1-alpha)].
```

The runner receives both exported `dim+1` matrices. In either mode, do not
L2-normalize the transformed base: its norm or final coordinate carries the
temporal signal. The default `--normalize-query true` only scales each query by
a positive constant and therefore preserves its MIPS ranking. See
[`tdvs_workload_builder/README.md`](../tdvs_workload_builder/README.md) for the
producer commands and alignment verifier.

MAG and PSP additionally consume a fixed-row L2 kNN graph built from the exact
transformed base used by their index. Multiplicative and additive transforms
therefore require different graph files.

## Build

A C++17 compiler, CMake 3.21 or newer, and pthreads are required. OpenMP is
required for multithreaded construction; without it, pass `--build-threads 1`
for an explicitly serial build. From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 64 --target \
  tdvs_ip_nsw tdvs_ip_nsw_plus tdvs_napg tdvs_mag tdvs_psp
```

The directory can also be built independently:

```bash
cmake -S baselines -B build-baselines \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-baselines --parallel 64
```

With a root build the executables are in `build/baselines/`; with a standalone
build they are in `build-baselines/`. Inspect the authoritative interface of
every runner with:

```bash
for runner in tdvs_ip_nsw tdvs_ip_nsw_plus tdvs_napg tdvs_mag tdvs_psp; do
  ./build/baselines/$runner --help
done
```

## Direct runner workflow

All runners implement these modes:

```text
--mode build       construct and serialize an index
--mode footprint   load serving artifacts in a fresh process
--mode search      run a single-threaded search-budget sweep
```

For ip-NSW, ip-NSW+, and NAPG, a complete direct run has this shape (replace
`tdvs_ip_nsw` and the index suffix for the other methods):

```bash
./build/baselines/tdvs_ip_nsw \
  --mode build --base "$BASE" --index "$INDEX" \
  --base-count "$N" --dim "$DIM" --build-threads 64

./build/baselines/tdvs_ip_nsw \
  --mode search --query "$QUERY" --gt "$GT" --index "$INDEX" \
  --base-count "$N" --dim "$DIM" --query-count "$Q" \
  --gt-width 100 --gt-type int64 --k 50 \
  --ef-list 50,75,100,150,200,300,500,700,1000,1500 \
  --normalize-query true --repeats 3 --output-csv results.csv

./build/baselines/tdvs_ip_nsw \
  --mode footprint --index "$INDEX" --base-count "$N" --dim "$DIM" \
  --output-csv footprint.csv
```

MAG and PSP first need the shared graph. The Python tools require NumPy and a
Faiss build that provides `IndexNNDescentFlat`:

```bash
python3 baselines/knng/build_knng.py \
  --base "$BASE" --output "$KNNG" --dim "$DIM" --threads 64

python3 baselines/knng/validate_knng.py \
  --base "$BASE" --knng "$KNNG" --dim "$DIM" --k 400 \
  --threads 64 --report knng_validation.json

./build/baselines/tdvs_mag \
  --mode build --base "$BASE" --knng "$KNNG" --index "$MAG_INDEX" \
  --dim "$DIM" --build-threads 64

./build/baselines/tdvs_mag \
  --mode search --base "$BASE" --query "$QUERY" --gt "$GT" \
  --index "$MAG_INDEX" --dim "$DIM" --query-count "$Q" \
  --gt-width 100 --gt-type int64 --k 50 \
  --search-l 50,75,100,150,200,300,500,700,1000,1500 \
  --normalize-query true --repeats 3 --output-csv mag.csv

./build/baselines/tdvs_mag \
  --mode footprint --base "$BASE" --index "$MAG_INDEX" \
  --base-count "$N" --dim "$DIM" --output-csv mag_footprint.csv
```

Use `tdvs_psp` and a PSP index path for the analogous PSP commands. Its build
also checks that the supplied graph has degree 400 unless `--knng-k` explicitly
selects another degree.

Construction defaults to 64 threads. Search forcibly configures OpenMP for one
thread, has no warmup, and measures every query in every repeat. Dataset
loading, query normalization, index loading, and Recall computation are outside
the timed search interval. The search CSV records every requested budget;
build metrics are written to standard output, and footprint mode writes its
own CSV when `--output-csv` is supplied.

## Method-specific notes

- MAG and PSP use the same KNNG only when their transformed base is identical.
  The KNNG is a construction input, not a serving artifact; its manifest lets
  the runner report preprocessing and method-specific construction separately.
- MAG and PSP keep the transformed base resident during search, so their
  serving footprint includes the external base and serialized method graph.
  They also require the `INDEX.mag.meta` or `INDEX.psp.meta` sidecar produced
  at build time; the input KNNG is not a serving artifact.
- ip-NSW, ip-NSW+, and NAPG embed transformed vectors in their serialized
  indexes. ip-NSW also requires `INDEX.ip_nsw.meta`; NAPG requires
  `INDEX.napg.meta`.
- ip-NSW defaults to `M=32` and `efConstruction=1024`; its retained core fixes
  the construction seed at 100.
- ip-NSW+ defaults to `M=32`, `efConstruction=1024`, angular `M=10`, and
  angular `efConstruction=100`. Its parallel-construction backport prevents
  data races, but thread scheduling can still change graph topology.
- NAPG defaults to `M=16`, `efConstruction=100`, five equal-count norm ranges,
  100 sampled sources per range, and 100 exact MIPS neighbors per source. The
  range/sample defaults are explicit local reproducibility choices where the
  paper does not prescribe a universal value.
- MAG and PSP retain their upstream OpenMP construction paths; parallel builds
  are not claimed to be byte-for-byte deterministic. Their `L_search` values
  are method-specific budgets and are not numerically equivalent to HNSW
  `efSearch`.
- PSP always uses the No-SN query function enabled by the authors' public test;
  the released code does not provide a reproducible SN artifact pipeline.
- When built with CMake, each of the five `tdvs_*` runner targets embeds a
  SHA-256 fingerprint of its registered provenance source set. CMake writes the
  corresponding per-file manifest to the target build directory's
  `source_provenance/`.
