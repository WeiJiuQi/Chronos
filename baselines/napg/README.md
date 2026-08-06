# NAPG baseline

This directory implements **Norm Adjusted Proximity Graph for Fast Inner
Product Retrieval** according to the original KDD 2021 paper
([DOI 10.1145/3447548.3467412](https://doi.org/10.1145/3447548.3467412)). The
authors have not published implementation code, so the NAPG estimator and
neighbor-selection rule follow the paper rather than claiming unavailable
source code. The graph substrate under `include/` is derived from Apache-2.0
hnswlib commit `3e006ea4b0ffdcd3fb1319370859a24da897ddd0`; see the root
`LICENSE`.

NAPG is implemented as the paper describes: an HNSW/ip-NSW graph whose query
search is unchanged, with a norm-range-specific factor applied only to the
neighbor-diversification test during construction. It consumes the same
mode-matched, **unnormalized** MIPS base as the other baselines: weighted
`dim`-dimensional rows for multiplicative TDVS or augmented `dim+1` rows for
additive TDVS. Additive search must use the matching augmented query file.

## Paper algorithm

Given `N` norm ranges, the estimator:

1. orders base vectors by L2 norm and divides that order into `N` equal-count
   ranges;
2. samples `Z` source vectors from each range;
3. obtains each source's exact top-`n` MIPS neighbors from the entire base;
4. computes the paper's range factor

   ```text
   alpha_r = Avg(dot(p_i, p_j)) / Avg(dot(x, p_i)).
   ```

The implementation excludes the source itself from its top-`n` neighbor set
and averages distinct unordered candidate pairs in the numerator. These are
explicit interpretations of the graph-neighbor comparisons in the paper.
Factors are not clamped to one.

During edge selection, for a source `x`, candidate `p_i`, and already selected
neighbor `p`, NAPG rejects the candidate when

```text
alpha_range(x) * dot(x, p_i) < dot(p_i, p).
```

All candidate generation, hierarchy construction, serialization, and query
search otherwise follow the standard HNSW/ip-NSW procedure used by the paper.
Query priority is ordinary inner product and the query budget is the usual
HNSW `ef`.

## Artifact defaults

The runner defaults to:

| Parameter | CLI | Default | Provenance |
| --- | --- | ---: | --- |
| graph degree | `--m` | 16 | value fixed in the paper's parameter study |
| construction budget | `--ef-construction` | 100 | paper study's `kConstruction` value |
| norm ranges | `--norm-ranges` | 5 | representative favorable value from the paper's range-count sensitivity study |
| samples per range | `--samples-per-range` | 100 | local deterministic artifact choice; the paper gives no universal default |
| exact neighbors per sample | `--sample-top-n` | 100 | value used to illustrate the paper estimator |
| estimator seed | `--sampling-seed` | 42 | local reproducibility choice |

The paper tunes its parameters by dataset and does not define a single global
`N` or `Z`. Consequently, these defaults are starting points for reproducible
comparison, not values attributed to the authors. Every build writes the
learned norm boundaries, adjustment factors, estimator inputs, and seeds to
`INDEX.napg.meta`.

## Build and run

Build the target from the repository root and inspect the direct interface:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 64 --target tdvs_napg
./build/baselines/tdvs_napg --help
```

Invoke `--mode build`, `--mode search`, and `--mode footprint` directly;
[`../README.md`](../README.md) gives a complete command sequence. The shared
runner keeps dataset loading, query normalization, index loading, and Recall
computation outside the serial timed-search interval and writes the requested
budget sweep to CSV.

The index plus its `.napg.meta` sidecar form the complete serialized NAPG
artifact. Search loads no external base because vectors are embedded in the
HNSW index.
