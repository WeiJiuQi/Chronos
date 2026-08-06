# PSP baseline

This directory vendors the original implementation of **Proximity Graph with
Spherical Pathway (PSP)** released by the authors of the PVLDB 2025 paper
*Maximum Inner Product is Query-Scaled Nearest Neighbor*.

- Upstream: <https://github.com/ZJU-DAILY/PSP>
- Paper: <https://www.vldb.org/pvldb/vol18/p1770-ke.pdf>

Chronos uses PSP as a MIPS comparison method. The index consumes the
mode-matched transformed TDVS base and the same fixed-row kNN graph used by
MAG. Multiplicative search uses the canonical semantic query; additive search
uses the paired `dim+1` transformed query.

## Chronos integration

Build the unified runner from the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 64 --target tdvs_psp
```

The runner supports:

```text
tdvs_psp --mode build ...
tdvs_psp --mode footprint ...
tdvs_psp --mode search ...
```

Run `./build/baselines/tdvs_psp --help` for the complete interface. Invoke
`--mode build`, `--mode search`, and `--mode footprint` directly;
[`../README.md`](../README.md) gives a complete command sequence. Shared runner
code standardizes input validation, construction threads, single-threaded
query timing, repeats, Recall, footprint measurement, and CSV output. The
runner accepts `--build-threads` and defaults to 64.

## Required kNN graph

PSP expects a fixed-row kNN graph with degree 400 by default. Generate and
inspect the shared MAG/PSP graph with `baselines/knng/`. The upstream README
only says that Faiss or another library may prepare the graph; neither the
repository nor its script contains a generator or a graph-recall acceptance
threshold. Chronos therefore records the selected external method and includes
its construction time in end-to-end index construction.

## Default parameters

| Parameter | Default |
| --- | ---: |
| construction `L` | 800 |
| degree `R` | 40 |
| angle | 60 |
| pathway parameter `M` | 5 |
| construction threads | 64 |

The authors' public test executable leaves the optional SN initialization
commented out and calls `Search_Mips_IP_Cal_with_No_SN`. It also does not
release a reproducible pipeline for generating the external SN entry-point
artifact. Chronos therefore uses that public No-SN path only; it does not build
or select between local SN variants.

`L_search` is PSP's query-time budget and must be compared through the
resulting recall-latency curve rather than numerically equated with HNSW `ef`.
