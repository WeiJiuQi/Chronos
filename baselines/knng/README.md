# External MAG/PSP kNN graph

MAG and PSP require a fixed-row L2 kNN graph before their method-specific graph
construction can begin. This directory owns that shared index-preparation
stage.

- `build_knng.py` constructs the graph directly with Faiss NN-Descent and
  records graph construction, extraction, and validated-write time in a
  manifest.
- `validate_knng.py` verifies graph shape and row degree and reports sampled
  recall against exact Faiss L2 neighbors.

The generated kNN graph is a construction input, not a serving artifact. The
generator writes `OUTPUT.json`; passing the graph to a MAG or PSP runner makes
that runner discover the manifest automatically, report KNNG time separately,
and include it in end-to-end index time. TDVS metadata, ground truth, and
transformed vectors remain the responsibility of `tdvs_workload_builder/`.

The graph has no global header. Each base row is one little-endian record:

```text
[uint32 K][K x uint32 neighbor IDs]
```

## Upstream boundary

Neither author repository contains a kNN-graph generator. Their build methods
load a graph supplied by path, and their example scripts also consume an
already-built graph.

- The PSP README says to prepare a graph with Faiss or another library, but it
  does not prescribe an index family, construction parameters, or a recall
  threshold.
- The MAG paper describes IVF-PQ for this preprocessing step. The MAG repository
  does not include that IVF-PQ pipeline or publish its IVF/PQ parameters; its
  README likewise accepts an externally prepared graph.

Consequently, `build_knng.py` is Chronos experiment infrastructure, not vendored
MAG or PSP code. Its selected Faiss method and every parameter must be reported
as a Chronos protocol choice. The generator uses `IndexNNDescentFlat` because
NN-Descent directly produces a fixed-width approximate KNNG. The fixed K400
configuration follows the public SSG parameter family used for NSSG's base
graph lineage:

```text
K=400, L=420, iterations=12, S=15, R=100, seed=42
```

Faiss 1.14.3 uses a 32-bit intermediate expression when flattening
`row * K + neighbor` even though `final_graph` itself is sized with 64-bit
arithmetic. This overflows for the formal 10M-by-K400 graphs. The experiment
environment therefore applies
`faiss_nndescent_large_graph_v1.14.3.patch`, which promotes only the flattened
container offsets to `size_t`; neighbor IDs, NN-Descent parameters, distances,
and graph contents retain the upstream representation and algorithm. The KNNG
manifest records the loaded Faiss native-library fingerprint and, when a
matching sidecar is present, its patch identifier. The patch remains under
Faiss's MIT license, reproduced in `FAISS_LICENSE`.

An unpatched Faiss build is accepted when `rows * K` fits in a signed 32-bit
offset. Larger graphs require both a patched Faiss 1.14.3 build and a matching
sidecar next to the loaded native library. Prepare that build in this order:

```bash
git -C /path/to/faiss-1.14.3 apply \
  /path/to/Chronos/baselines/knng/faiss_nndescent_large_graph_v1.14.3.patch

# Rebuild and install Faiss and its Python bindings before recording the binary.
python3 baselines/knng/write_faiss_patch_sidecar.py \
  --library /path/to/python/site-packages/faiss/libfaiss.so \
  --confirm-patch-applied
```

The helper hashes the supplied native library and atomically writes
`LIBRARY.chronos-patch.json`. It does not inspect the binary or prove that the
patch is present: `--confirm-patch-applied` is an explicit assertion by the
person who built Faiss. The graph builder prefers an adjacent `libfaiss.so` for
the custom shared-library layout; ordinary installations without that file are
fingerprinted through their loaded `faiss._swigfaiss*` native extension. Pass
that extension's path to the helper instead when it contains the rebuilt Faiss
implementation.

These are not MAG or PSP author defaults and must not be described as a
reproduction of the MAG paper's unspecified IVF-PQ setup. The previous local
HNSW/IVF plus all-points-query generator has been removed. Its manifests and
downstream MAG/PSP artifacts are incompatible with the current protocol.

Both scripts require NumPy and a Faiss Python package that provides
`IndexNNDescentFlat`. Build and validate a graph directly with:

```bash
.venv-faiss/bin/python baselines/knng/build_knng.py \
  --base transformed_base.fbin --output transformed_base.k400.knng \
  --dim 1536 --k 400 --threads 64 \
  --nndescent-l 420 --nndescent-iterations 12 \
  --nndescent-s 15 --nndescent-r 100 --seed 42

.venv-faiss/bin/python baselines/knng/validate_knng.py \
  --base transformed_base.fbin --knng transformed_base.k400.knng \
  --dim 1536 --k 400 --samples 256 --threads 64 --seed 42 \
  --report transformed_base.k400.validation.json
```

Faiss construction defaults to 64 OpenMP threads. `IndexNNDescentFlat` must
receive the complete aligned matrix in one `add()` call because additional
calls rebuild its graph. The writer validates ID bounds, self edges,
duplicates, row width, and final byte size before atomically publishing the
graph. Validation is an unmeasured, non-blocking diagnostic: malformed shapes
or row-degree headers fail, while sampled recall is recorded without an
acceptance threshold. The JSON report includes deterministic sample IDs and
per-row recalls. Additive and multiplicative workloads use different
transformed bases, so they also require different KNNG files.
