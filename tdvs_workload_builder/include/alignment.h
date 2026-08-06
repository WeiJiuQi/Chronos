#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace chronos {

struct TransformAlignmentOptions {
    std::size_t dim = 0;
    double half_life_days = 0.0;
    double query_time = 0.0;
    bool cosine = true;
    double absolute_tolerance = 1e-7;
    double relative_tolerance = 1e-6;
    std::size_t chunk_rows = 1024;
    std::size_t progress_every_rows = 0;
    int threads = 1;
};

struct TransformAlignmentReport {
    std::size_t rows = 0;
    std::size_t coordinates = 0;
    std::size_t mismatched_rows = 0;
    std::size_t mismatched_coordinates = 0;
    std::size_t nonfinite_raw_coordinates = 0;
    std::size_t nonfinite_transformed_coordinates = 0;
    std::size_t nonfinite_timestamps = 0;
    std::size_t out_of_range_timestamps = 0;
    std::size_t zero_norm_rows = 0;
    std::size_t first_mismatch_row = static_cast<std::size_t>(-1);
    std::size_t first_mismatch_coordinate = static_cast<std::size_t>(-1);
    std::size_t max_absolute_error_row = 0;
    std::size_t max_absolute_error_coordinate = 0;
    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;
    double min_timestamp = 0.0;
    double max_timestamp = 0.0;
    double min_raw_norm = 0.0;
    double max_raw_norm = 0.0;

    bool passed() const;
};

TransformAlignmentReport verify_multiplicative_transform_files(
    const std::string& base_path,
    const std::string& timestamps_path,
    const std::string& transformed_base_path,
    const TransformAlignmentOptions& options);

struct AdditiveTransformAlignmentReport {
    std::size_t base_rows = 0;
    std::size_t query_rows = 0;
    std::size_t compared_coordinates = 0;
    std::size_t mismatched_coordinates = 0;
    std::size_t first_mismatch_row = static_cast<std::size_t>(-1);
    std::size_t first_mismatch_coordinate = static_cast<std::size_t>(-1);
    bool first_mismatch_is_query = false;
    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;

    bool passed() const;
};

AdditiveTransformAlignmentReport verify_additive_transform_files(
    const std::string& base_path,
    const std::string& query_path,
    const std::string& timestamps_path,
    const std::string& transformed_base_path,
    const std::string& transformed_query_path,
    const TransformAlignmentOptions& options,
    double alpha);

struct MappedRowsReport {
    std::size_t source_rows = 0;
    std::size_t subset_rows = 0;
    std::size_t mapped_rows = 0;
    std::size_t mismatched_rows = 0;
    std::size_t first_mismatch_subset_row = static_cast<std::size_t>(-1);
    std::size_t first_mismatch_source_row = static_cast<std::size_t>(-1);
    std::size_t contiguous_runs = 0;

    bool passed() const;
};

MappedRowsReport verify_mapped_raw_rows(
    const std::string& source_path,
    const std::string& subset_path,
    const std::string& original_indices_path,
    std::size_t dim,
    std::size_t chunk_rows = 1024);

struct GroundTruthStructureReport {
    std::size_t queries = 0;
    std::size_t width = 0;
    std::size_t ids = 0;
    std::size_t invalid_ids = 0;
    std::size_t rows_with_duplicate_ids = 0;
    std::size_t first_invalid_query = static_cast<std::size_t>(-1);
    std::size_t first_invalid_rank = static_cast<std::size_t>(-1);
    std::int64_t min_id = 0;
    std::int64_t max_id = 0;

    bool passed() const;
};

GroundTruthStructureReport verify_groundtruth_structure(
    const std::string& groundtruth_path,
    std::size_t queries,
    std::size_t width,
    std::size_t base_rows);

struct SourceRowSplitReport {
    std::size_t base_ids = 0;
    std::size_t query_ids = 0;
    std::size_t duplicate_base_ids = 0;
    std::size_t duplicate_query_ids = 0;
    std::size_t out_of_range_base_ids = 0;
    std::size_t out_of_range_query_ids = 0;
    std::size_t overlapping_ids = 0;
    bool same_source = false;

    bool passed() const;
};

SourceRowSplitReport verify_source_row_id_split(
    const std::string& base_ids_path,
    std::size_t base_rows,
    std::size_t base_source_rows,
    const std::string& query_ids_path,
    std::size_t query_rows,
    std::size_t query_source_rows,
    bool same_source);

}  // namespace chronos
