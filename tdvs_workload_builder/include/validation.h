#pragma once

#include <cstddef>
#include <string>

namespace chronos {

struct VectorRowsReport {
    std::size_t rows = 0;
    std::size_t coordinates = 0;
    std::size_t nonfinite_coordinates = 0;
    std::size_t zero_norm_rows = 0;
    std::size_t non_unit_rows = 0;
    std::size_t exact_duplicate_rows = 0;
    double min_norm = 0.0;
    double max_norm = 0.0;
    double max_unit_norm_error = 0.0;

    bool valid(bool require_unit_norm) const;
};

/** Audits finite values, row norms, and optional exact duplicate rows. */
VectorRowsReport audit_vector_rows(
    const float* data,
    std::size_t rows,
    int dim,
    bool require_unit_norm,
    double unit_norm_tolerance,
    bool count_exact_duplicates);

/** Streaming finite/norm audit for a headerless float32 row file. */
VectorRowsReport audit_vector_file(
    const std::string& path,
    int dim,
    bool require_unit_norm,
    double unit_norm_tolerance,
    std::size_t chunk_rows = 1024);

/** Counts query rows that are byte-identical to at least one base row. */
std::size_t count_exact_query_base_overlaps(
    const float* base,
    std::size_t base_rows,
    const float* query,
    std::size_t query_rows,
    int dim);

}  // namespace chronos
