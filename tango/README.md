# TANGO

TANGO is the native Chronos index for time-decayed vector search. The name
stands for **Time-Aware Navigable Graph with Query-Orthogonal TimeLift**. The
implementation extends the HNSW graph construction interface
with separate data-to-data and query-to-data distances while retaining a
single hierarchical graph and a single query traversal.

## Scoring model

Each object contains an L2-normalized semantic vector `x_i` and timestamp
`t_i`. A query contains normalized vector `q` and time `tau`. With
`lambda = ln(2) / half_life`, TANGO supports:

```text
F_mul(q, i, tau) = dot(q, x_i) * exp(-lambda * (tau - t_i))

F_add(q, i, tau) = alpha * dot(q, x_i)
                   + (1 - alpha) * exp(-lambda * (tau - t_i))
```

The current interface requires `tau >= max_i(t_i)`. It does not implement
historical visibility filtering for objects created after a query snapshot.

## Anchor-free temporal geometry

TANGO uses the positive-definite temporal kernel

```text
k_ij = exp(-lambda * abs(t_i - t_j))
```

and semantic similarity `c_ij = dot(x_i, x_j)`. The multiplicative
data-to-data squared distance at HNSW level `l` is

```text
D_mul_DD(i, j; kappa_l)
    = 2 * (1 - k_ij) + 2 * (k_ij + kappa_l) * (1 - c_ij).
```

For fixed-alpha additive TDVS:

```text
D_add_DD(i, j; kappa_l)
    = 2 * (alpha + kappa_l) * (1 - c_ij)
      + 2 * (1 - alpha) * (1 - k_ij).
```

These are squared Hilbert-space distances. The query-to-data distance used by
the implementation is

```text
D_QD(q, i) = 1 - F(q, i, tau).
```

It differs from the corresponding lifted squared distance only by a positive
scale and an object-independent constant. Consequently, every `kappa_l`
produces exactly the same query ranking; `kappa_l` only shapes graph edges.

The temporal feature map is never materialized. TANGO computes the kernel
value directly, so the representation introduces no random-feature or
truncation error.

## Kappa schedules

TANGO supports three schedule forms:

- **fixed:** `kappa_base == kappa_nav`;
- **two-tier:** level 0 uses `kappa_base`, upper levels use `kappa_nav`; and
- **per-level:** `--kappa-levels k0,k1,...` supplies a non-decreasing list.

Every value must be finite and non-negative. For a two-tier schedule,
`kappa_base <= kappa_nav`. Levels above the end of an explicit list use its
last value.

The paper configuration uses the two-tier schedule:

```text
kappa_base = 1 / 63
kappa_nav  = 1 / 15
```

They can be expressed by the dimensionless horizon parameter `C`:

```text
kappa(C) = 1 / (2^C - 1).
```

This parameterization is independent of the timestamp unit and current data
distribution. The static `tango_index` construction CLI deliberately requires
an explicit schedule; it has no hidden kappa default. The dynamic-workload CLI
uses `1/63` and `1/15` by default and accepts explicit overrides. A graph still
fixes one half-life because its data-to-data kernel depends on `lambda`.

## Decay factorization fast path

The direct implementation evaluates `exp(-lambda * delta_t)` in each temporal
distance callback. An optional algebraically equivalent factorization caches

```text
z_i = exp(lambda * (t_i - T0))
```

per internal ID. Query-to-data decay becomes `g_tau * z_i`; data-to-data decay
becomes a ratio of the smaller and larger basis values. Cache entries use
float64. A callback falls back to direct evaluation when its cached factors are
unavailable, non-finite, or outside the valid decay range. The algebraically
equivalent paths can differ in their final float rounding; optional sampled
verification checks that difference against the implementation tolerance.

Available modes are:

| Mode | Query distance | Construction distance |
| --- | --- | --- |
| `off` | direct | direct |
| `qd` | cached when safe | direct |
| `dd` | direct | cached when safe |
| `all` | cached when safe | cached when safe |

The cache is derived runtime state and is not serialized. When enabled, loading
an index rebuilds it outside the timed query loop. Verification and
instrumentation are disabled by default because they add callback overhead.

## Build

From the repository root:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 7 --target \
  tango_index tango_query tango_dynamic tango_decay_profile
```

Standalone build:

```bash
cmake -S tango -B build-tango \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-tango --parallel 7
```

The reusable index wrapper is header-only under `include/`. The public commands
are:

| Executable | Purpose |
| --- | --- |
| `tango_index` | construct, save, load, diagnose, and optionally query an index |
| `tango_query` | load an index and measure serial query performance |
| `tango_dynamic` | run timestamp-ordered online insertion checkpoints |
| `tango_decay_profile` | profile temporal-distance factorization fast paths |

Set `-DCHRONOS_NATIVE_OPTIMIZATION=ON` for a machine-specific benchmark build.
Portable builds leave it disabled.

## Input files

All inputs are headerless and row-major:

| Option | Type and shape |
| --- | --- |
| `--base` | `float32[N, dim]` semantic vectors |
| `--timestamps` | `float32[N]` timestamps aligned with base rows |
| `--query` | `float32[Q, dim]` semantic queries |
| `--gt` | `int64[Q, gt_width]` ground-truth IDs |
| `--query-times` | optional `float32[Q]` per-query timestamps for the combined `tango_index` path |

Timestamps are creation times measured in days; query times and half-life values
use the same unit. The persisted sidecar records and validates this day-based
time contract.

TANGO must receive the raw semantic base, not a decay-scaled MIPS base. With
`--normalize 1`, base and query semantic coordinates are normalized; the
timestamp is never included in normalization or semantic dot products.

The TDVS workload builder may also export a decay-weighted base for comparison methods.
That file is aligned row-for-row with the raw base but must never be passed to
`tango_index`: doing so would apply temporal decay twice. TANGO and the MIPS
baselines share the same logical objects, semantic query workload, and TDVS
ground-truth IDs; only their physical index inputs differ. See the data-flow
contract in [../tdvs_workload_builder/README.md](../tdvs_workload_builder/README.md).

## Construct an index

Use `--build-only` to keep file loading and construction separate from query
measurement:

```bash
./build/tango/tango_index \
  --index-mode create --build-only \
  --index-prefix /data/indexes/tango_hl30 \
  --mode multiplicative \
  --base /data/base.fbin \
  --timestamps /data/timestamp_days.f32bin \
  --n 999000 --dim 1536 \
  --half-life 30 \
  --M 25 --ef-construction 200 \
  --threads 64 --random-seed 100 \
  --normalize 1 \
  --kappa-base 0.015873015873 \
  --kappa-nav 0.066666666667 \
  --tango-decay-fastpath all \
  --build-report-csv /data/results/tango_build.csv
```

Index construction defaults to 64 threads; override `--threads` explicitly when
reproducing an older build configuration. The serving index consists of two files:

```text
PREFIX.hnsw
PREFIX.tango.meta
```

The sidecar records the scoring mode, data-to-data construction mode, dimension,
half-life, additive alpha, kappa schedule, normalization setting, maximum
timestamp, HNSW parameters, and format version. The two files must be copied
together. Versions 1 and 2 load as the default `timelift` construction mode.

For additive TDVS, add:

```text
--mode additive --alpha 0.8
```

An additive index fixes its alpha value at construction time.

For the semantic-graph ablation, add `--dd-mode semantic`. This uses ordinary
semantic inner-product distance during every construction operation while
retaining the original TDVS QD distance at query time. The default
`--dd-mode timelift` is full TANGO. Build and query CSVs record this setting.

An explicit level schedule replaces the two-tier options:

```text
--kappa-levels 0.015873015873,0.025433984143,0.040986265908,0.066666666667
```

## Query an index

Use the dedicated query executable for reported latency:

```bash
./build/tango/tango_query \
  --index /data/indexes/tango_hl30 \
  --query /data/query.fbin \
  --gt /data/gt_tdvs_hl30.k100.bin \
  --queries 1000 --dim 1536 --gt-width 100 \
  --k 10 \
  --ef-list 10,20,30,50,100,200,400,700,1000,1500,2000 \
  --query-time 365 \
  --repeats 3 \
  --max-search-expansions 0 \
  --tango-decay-fastpath qd \
  --output-csv /data/results/tango_hl30_k10.csv
```

The query loop is serial. Every query in every repeat is measured; non-zero
warm-up is rejected. The timed interval covers graph search and top-k ID
materialization. Input loading, query normalization, query-record preparation,
and Recall computation are outside the interval.

`--max-search-expansions 0` disables the optional base-layer expansion cap.
Apply the same policy to every compared HNSW-derived method.

## Dynamic insertion workload

`tango_dynamic` implements a fixed arrival protocol without changing the index
algorithm. It stable-sorts rows by `(timestamp, row_id)`, builds the oldest 50%,
and, by default, inserts ten newer 5% batches. `--initial-percent` and
`--batch-percent` select another exact schedule that ends at 100%. The initial build uses
`--build-threads` (default 64); each online batch is deliberately inserted with
one thread. Query execution is always one thread with zero warmup. Checkpoint
ground-truth generation uses `--gt-threads` when OpenMP is available and one
thread otherwise.

```bash
./build/tango/tango_dynamic \
  --base /data/base.fbin --timestamps /data/timestamp_days.f32bin \
  --query /data/query.fbin --output-dir /data/results/dynamic \
  --n 1000000 --queries 1000 --dim 3072 \
  --mode additive --alpha 0.7 --half-life 90 --query-time 365 \
  --gt-width 100 --k-values 10,50,100 \
  --ef-list 10,20,30,40,50,75,100,125,150,200,300,500,700,1000,1500,2000 \
  --M 25 --ef-construction 200 --build-threads 64 --gt-threads 64 \
  --kappa-base 0.015873015873 --kappa-nav 0.066666666667 \
  --dd-mode timelift --tango-decay-fastpath all \
  --max-search-expansions 0 --initial-percent 50 --batch-percent 5
```

The output directory contains `arrival_order.u64bin`, one exact int64 GT file
per checkpoint, `dynamic_results.csv`, and `run_manifest.json`. Exact GT is
maintained incrementally: each newly arrived row is scored once per query and
merged into that query's retained top-`gt-width`. `--k-values` evaluates multiple
Recall@k curves on the same insertion trajectory; each curve uses requested
`ef` values that are at least `k`. The legacy `--k` option remains available for
a single curve. GT computation is recorded separately
and excluded from insertion and ANN query timers. Result rows include insertion
throughput, Recall/latency, average layer-0 degree, indegree p95/p99/max, and
indegree Gini. `--final-index-prefix` optionally saves only the final graph; the
runner does not retain a separate index copy at every checkpoint.

As in `tango_query`, each latency sample covers graph search and heap-to-ID
materialization. Query normalization and record packing are completed before
the timed scans, and Recall computation remains outside the timer.

`--max-search-expansions 0` disables the optional base-layer query expansion
cap, matching the static query protocol. A nonzero value applies an explicit cap.
For targeted operating-point experiments, `--stop-recall R` stops each
increasing-`ef` scan once Recall@K reaches `R`; `--min-ef-points N` can require
at least `N` measured settings before this early stop is applied.

## Footprint measurement

Run footprint measurement in a fresh process:

```bash
./build/tango/tango_query \
  --index /data/indexes/tango_hl30 \
  --dim 1536 --footprint-only \
  --tango-decay-fastpath qd \
  --output-csv /data/results/tango_footprint.csv
```

The output distinguishes serialized index bytes, serving artifact bytes,
fresh-process RSS delta, peak RSS, and the known decay-cache payload. RSS is
OS- and allocator-dependent and should be compared only across fresh processes
on the same machine.

## Dynamic insertion

The header API supports online insertion of objects with later timestamps.
Existing vector and timestamp records need no temporal re-encoding because the
data-to-data kernel depends only on timestamp differences. HNSW neighbor lists
evolve normally as new nodes are inserted, using the schedule stored by the
index. Query contexts are stack-local, and concurrent queries do not share a
mutable query-time scale.

Online insertion preserves representation stability; as with standard HNSW,
the graph produced by concurrent insertion may depend on insertion
interleaving.
