#pragma once

/**
 * Chronos evaluation-compatible binary I/O.
 *
 * Vector files (load_data / load_query):
 *   - no file header;
 *   - row-major float32 values in native byte order; and
 *   - total size = num_vectors * dimension * sizeof(float).
 *
 * Ground truth:
 *   - no file header;
 *   - one fixed-width row of k int64 IDs per query; and
 *   - each ID is a row index in the current base file.
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace chronos {

/** Source revision captured when the workload-builder binaries were configured. */
const char* source_revision();
bool source_tree_was_dirty_at_configure();

enum class InputVectorFormat {
    RawFloat32,  // Headerless contiguous float32 rows.
    Fvecs,       // Standard .fvecs: int32 dimension followed by float32 values.
    Fbin         // Standard .fbin: uint32 row count and dimension, then float32 rows.
};

const char* input_vector_format_name(InputVectorFormat format);

/** Returns the file size in bytes, or zero if the file cannot be inspected. */
std::size_t file_size_bytes(const std::string& path);

/**
 * RawFloat32 uses file_bytes / (dim * sizeof(float)). Fvecs requires a full
 * record scan and is therefore more expensive for large files.
 */
std::size_t count_vectors_in_file(const std::string& path, int dim, InputVectorFormat fmt);

/**
 * Reads at most max_read vectors into contiguous storage and reports the
 * actual count. Returns false on format or I/O failure.
 */
bool read_vectors_prefix(const std::string& path, int dim, InputVectorFormat fmt,
                         std::size_t max_read, std::vector<float>& out, std::size_t& out_count);

/**
 * Reads the requested zero-based rows in the order supplied by indices.
 * RawFloat32 performs one seek per row. Fvecs rescans from the beginning for
 * each row; use read_vectors_sorted_indices for sorted, large index sets.
 */
bool read_vectors_at_indices(const std::string& path, int dim, InputVectorFormat fmt,
                             const std::vector<std::size_t>& indices, std::vector<float>& out);

/**
 * Reads rows identified by a strictly increasing index list in one forward
 * pass. RawFloat32 coalesces contiguous rows; Fvecs scans each record once.
 */
bool read_vectors_sorted_indices(const std::string& path, int dim, InputVectorFormat fmt,
                                 const std::vector<std::size_t>& indices_sorted, std::vector<float>& out);

/** Writes a headerless row-major float32 vector file. */
bool write_raw_vectors(const std::string& path, const float* data, std::size_t n, int dim);

/** Writes headerless uint64 source-row IDs. */
bool write_source_row_ids(const std::string& path, const std::vector<std::size_t>& indices);

/** Ground-truth identifier type used by every Chronos component. */
using GroundTruthId = std::int64_t;

bool write_groundtruth(const std::string& path, const GroundTruthId* flat,
                       std::size_t num_queries, int k);

/** Writes a text manifest to path. */
bool write_text_file(const std::string& path, const std::string& content);

}  // namespace chronos
