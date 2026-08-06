#!/usr/bin/env python3
"""Build the fixed-row L2 kNN graph consumed by MAG and PSP.

The input is a headerless row-major float32 matrix. The output repeats
``[uint32 k][k * uint32 neighbor_id]`` once per data vector; there is no global
header. The graph is built directly with Faiss NN-Descent. The input is never
normalized because transformed multiplicative-vector norms encode time decay.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import platform
import sys
import time
from pathlib import Path


DEFAULT_BUILD_THREADS = 64
DEFAULT_K = 400
DEFAULT_NNDESCENT_L = 420
DEFAULT_NNDESCENT_ITERATIONS = 12
DEFAULT_NNDESCENT_S = 15
DEFAULT_NNDESCENT_R = 100
DEFAULT_PARAMETER_BASIS = "SSG-published K400 NN-Descent settings"
SIGNED_INT32_MAX = (1 << 31) - 1
LARGE_GRAPH_PATCH_ID = "faiss_nndescent_size_t_offsets_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dim", required=True, type=int)
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="neighbors per row in the shared MAG/PSP graph")
    parser.add_argument("--threads", type=int, default=DEFAULT_BUILD_THREADS,
                        help="Faiss/OpenMP construction threads")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="rows validated and written per output batch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nndescent-l", type=int, default=DEFAULT_NNDESCENT_L,
                        help="NN-Descent construction candidate-pool size")
    parser.add_argument(
        "--nndescent-iterations", type=int,
        default=DEFAULT_NNDESCENT_ITERATIONS,
        help="NN-Descent join/update iterations",
    )
    parser.add_argument("--nndescent-s", type=int, default=DEFAULT_NNDESCENT_S,
                        help="new neighbors sampled per node and iteration")
    parser.add_argument("--nndescent-r", type=int, default=DEFAULT_NNDESCENT_R,
                        help="reverse-neighbor cap; zero disables reverse links")
    parser.add_argument(
        "--parameter-basis",
        default=DEFAULT_PARAMETER_BASIS,
        help="provenance note describing why this parameter set was selected",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_raw(path: Path, dim: int) -> np.memmap:
    if dim <= 0:
        raise ValueError("--dim must be positive")
    size = path.stat().st_size
    row_bytes = dim * np.dtype(np.float32).itemsize
    if size == 0 or size % row_bytes:
        raise ValueError(f"{path} is not a headerless float32 matrix with dim={dim}")
    rows = size // row_bytes
    if rows > np.iinfo(np.int32).max:
        raise ValueError("Faiss NN-Descent uses signed 32-bit graph IDs")
    return np.memmap(path, dtype="<f4", mode="r", shape=(rows, dim))


def validate_args(args: argparse.Namespace, rows: int) -> None:
    if args.k <= 0 or args.threads <= 0 or args.batch_size <= 0:
        raise ValueError("k, threads, and batch-size must be positive")
    if args.k >= rows:
        raise ValueError("k must be smaller than the number of vectors")
    if args.nndescent_l < args.k:
        raise ValueError("--nndescent-l must be at least --k")
    if args.nndescent_iterations <= 0 or args.nndescent_s <= 0:
        raise ValueError("nndescent-iterations and nndescent-s must be positive")
    if args.nndescent_r < 0:
        raise ValueError("--nndescent-r must be non-negative")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_faiss_native_library(faiss_module: object) -> Path:
    """Locate the native library whose bytes implement the loaded Faiss module."""
    package_file_text = str(getattr(faiss_module, "__file__", ""))
    if not package_file_text:
        raise RuntimeError("Cannot locate the Faiss package directory")
    package_file = Path(package_file_text).resolve()

    # Preserve the custom shared-library layout used for the large-graph build.
    adjacent_library = package_file.parent / "libfaiss.so"
    if adjacent_library.is_file():
        return adjacent_library

    package_name = str(getattr(faiss_module, "__name__", "faiss"))
    native_prefix = package_name + "._swigfaiss"
    candidates: list[tuple[str, Path]] = []
    for module_name, module in sys.modules.items():
        if not module_name.startswith(native_prefix):
            continue
        module_file_text = str(getattr(module, "__file__", ""))
        if not module_file_text:
            continue
        module_file = Path(module_file_text).resolve()
        if not module_file.is_file():
            continue
        if not any(
            str(module_file).endswith(suffix)
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
        ):
            continue
        candidates.append((module_name, module_file))

    candidate_paths = sorted({path for _, path in candidates})
    if len(candidate_paths) == 1:
        return candidate_paths[0]
    if not candidates:
        raise RuntimeError(
            "Cannot locate the loaded Faiss native library: expected an adjacent "
            "libfaiss.so or a loaded faiss._swigfaiss* extension module"
        )
    rendered = ", ".join(f"{name}={path}" for name, path in candidates)
    raise RuntimeError(
        "Multiple loaded Faiss native libraries are ambiguous: " + rendered
    )


def requires_large_graph_patch(rows: int, k: int) -> bool:
    """Return whether the largest flattened final_graph offset exceeds int32."""
    return rows * k - 1 > SIGNED_INT32_MAX


def inspect_faiss_runtime(faiss_module: object, rows: int, k: int) -> dict:
    library = locate_faiss_native_library(faiss_module)
    library_sha256 = sha256_file(library)
    sidecar_path = library.with_name(library.name + ".chronos-patch.json")
    sidecar = None
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("library_sha256") != library_sha256:
            raise RuntimeError(
                "Faiss patch sidecar does not match the loaded libfaiss.so SHA256"
            )

    large_graph = requires_large_graph_patch(rows, k)
    if large_graph and (
        sidecar is None or sidecar.get("patch_id") != LARGE_GRAPH_PATCH_ID
    ):
        max_safe_rows = (SIGNED_INT32_MAX + 1) // k
        raise RuntimeError(
            "Faiss IndexNNDescentFlat requires the Chronos size_t-offset patch "
            f"for rows={rows}, k={k}; upstream int32 flattening is safe for at "
            f"most {max_safe_rows} rows at this k. Loaded native library: "
            f"{library}; expected matching sidecar: {sidecar_path}"
        )

    compile_options = getattr(faiss_module, "get_compile_options", lambda: "unknown")()
    return {
        "library": str(library),
        "library_sha256": library_sha256,
        "compile_options": str(compile_options).strip(),
        "large_graph_patch_required": large_graph,
        "patch_sidecar": str(sidecar_path) if sidecar is not None else None,
        "patch": sidecar,
    }


def build_graph(
    data: np.memmap, args: argparse.Namespace
) -> tuple[np.ndarray, float, float]:
    index = faiss.IndexNNDescentFlat(args.dim, args.k, faiss.METRIC_L2)
    index.nndescent.L = args.nndescent_l
    index.nndescent.iter = args.nndescent_iterations
    index.nndescent.S = args.nndescent_s
    index.nndescent.R = args.nndescent_r
    index.nndescent.random_seed = args.seed
    index.verbose = True

    # NN-Descent is an offline graph builder. Multiple add() calls rebuild the
    # complete graph, so the aligned matrix must be added in one operation.
    build_start = time.perf_counter()
    index.add(data)
    graph_build_seconds = time.perf_counter() - build_start
    expected_ids = len(data) * args.k
    extract_start = time.perf_counter()
    graph = faiss.vector_to_array(index.nndescent.final_graph)
    graph_extract_seconds = time.perf_counter() - extract_start
    if graph.size != expected_ids:
        raise RuntimeError(
            f"Faiss NN-Descent returned {graph.size} IDs; expected {expected_ids}"
        )
    return (
        graph.reshape(len(data), args.k),
        graph_build_seconds,
        graph_extract_seconds,
    )


def validate_neighbor_batch(neighbors: np.ndarray, start: int, rows: int) -> None:
    if np.any(neighbors < 0) or np.any(neighbors >= rows):
        raise RuntimeError(f"NN-Descent returned an invalid ID near row {start}")
    row_ids = np.arange(start, start + len(neighbors), dtype=np.int32)[:, None]
    if np.any(neighbors == row_ids):
        raise RuntimeError(f"NN-Descent returned a self edge near row {start}")
    sorted_ids = np.sort(neighbors, axis=1)
    if np.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
        raise RuntimeError(f"NN-Descent returned a duplicate edge near row {start}")


def write_graph(graph: np.ndarray, output: Path, args: argparse.Namespace) -> None:
    rows = len(graph)
    with output.open("wb") as handle:
        for start in range(0, rows, args.batch_size):
            stop = min(start + args.batch_size, rows)
            neighbors = graph[start:stop]
            validate_neighbor_batch(neighbors, start, rows)
            records = np.empty((stop - start, args.k + 1), dtype="<u4")
            records[:, 0] = args.k
            records[:, 1:] = neighbors
            records.tofile(handle)
            if start == 0 or (start // args.batch_size) % 25 == 0:
                print(f"validate/write: {stop}/{rows}", flush=True)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    args = parse_args()
    global np, faiss
    try:
        import numpy as np  # type: ignore
        import faiss  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency message is the behavior
        raise SystemExit(
            "numpy and Faiss with IndexNNDescentFlat are required; use the "
            "project's pinned Faiss environment"
        ) from exc
    if not hasattr(faiss, "IndexNNDescentFlat"):
        raise RuntimeError("installed Faiss does not provide IndexNNDescentFlat")

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    data = load_raw(args.base, args.dim)
    validate_args(args, len(data))
    faiss_runtime = inspect_faiss_runtime(faiss, len(data), args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    faiss.omp_set_num_threads(args.threads)

    graph, graph_build_seconds, graph_extract_seconds = build_graph(data, args)

    write_start = time.perf_counter()
    write_graph(graph, temporary, args)
    graph_write_seconds = time.perf_counter() - write_start

    expected_bytes = len(data) * (args.k + 1) * np.dtype(np.uint32).itemsize
    if temporary.stat().st_size != expected_bytes:
        raise RuntimeError("Generated kNN graph has an unexpected size")
    temporary.replace(args.output)

    manifest = {
        "schema": "chronos_knng_manifest_v2",
        "format": "repeated [little-endian uint32 k][k little-endian uint32 ids]",
        "base": str(args.base.resolve()),
        "base_bytes": args.base.stat().st_size,
        "rows": len(data),
        "dim": args.dim,
        "k": args.k,
        "metric": "squared_l2",
        "input_normalized": False,
        "generator_scope": "chronos_external_preprocessing",
        "author_prescribed": False,
        "method": "faiss_nndescent",
        "faiss_index_class": "IndexNNDescentFlat",
        "direct_final_graph": True,
        "threads": args.threads,
        "seed": args.seed,
        "nndescent_l": args.nndescent_l,
        "nndescent_iterations": args.nndescent_iterations,
        "nndescent_s": args.nndescent_s,
        "nndescent_r": args.nndescent_r,
        "parameter_basis": args.parameter_basis,
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "faiss_runtime": faiss_runtime,
        "python": sys.version,
        "platform": platform.platform(),
        "graph_build_seconds": graph_build_seconds,
        "graph_extract_seconds": graph_extract_seconds,
        "graph_write_seconds": graph_write_seconds,
        "total_seconds": (
            graph_build_seconds + graph_extract_seconds + graph_write_seconds
        ),
        "output_bytes": args.output.stat().st_size,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    manifest_temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    manifest_temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
