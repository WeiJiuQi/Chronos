#include "alignment.h"

#include "tdvs.h"
#include "io.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace chronos {
namespace {

constexpr double kNormEpsilon = 1e-12;
constexpr double kTimestampRangeEpsilon = 1e-9;
constexpr std::size_t kMissingIndex = static_cast<std::size_t>(-1);

void require_regular_payload(
        const std::string& path,
        std::size_t expected_bytes,
        const char* description) {
    const std::size_t actual = file_size_bytes(path);
    if (actual != expected_bytes) {
        std::ostringstream message;
        message << description << " size mismatch for " << path
                << ": expected " << expected_bytes << " bytes, got " << actual;
        throw std::runtime_error(message.str());
    }
}

void read_exact(
        std::ifstream& input,
        void* destination,
        std::size_t bytes,
        const char* description) {
    input.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
    if (!input) {
        throw std::runtime_error(std::string("failed reading ") + description);
    }
}

std::vector<std::size_t> read_indices(const std::string& path) {
    std::ifstream input(path.c_str());
    if (!input) throw std::runtime_error("cannot open original-index map: " + path);
    std::vector<std::size_t> indices;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        std::size_t consumed = 0;
        const unsigned long long value = std::stoull(line, &consumed);
        while (consumed < line.size() &&
               (line[consumed] == ' ' || line[consumed] == '\t' || line[consumed] == '\r')) {
            ++consumed;
        }
        if (consumed != line.size() || value > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("invalid original-index map line in " + path);
        }
        indices.push_back(static_cast<std::size_t>(value));
    }
    return indices;
}

struct RowTransformStats {
    std::size_t mismatch_coordinates = 0;
    std::size_t first_mismatch_coordinate = kMissingIndex;
    std::size_t nonfinite_raw = 0;
    std::size_t nonfinite_transformed = 0;
    bool nonfinite_timestamp = false;
    bool out_of_range_timestamp = false;
    bool zero_norm = false;
    double norm = 0.0;
    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;
    std::size_t max_absolute_error_coordinate = 0;
};

}  // namespace

bool TransformAlignmentReport::passed() const {
    return mismatched_rows == 0 && mismatched_coordinates == 0 &&
           nonfinite_raw_coordinates == 0 &&
           nonfinite_transformed_coordinates == 0 &&
           nonfinite_timestamps == 0 && out_of_range_timestamps == 0;
}

TransformAlignmentReport verify_multiplicative_transform_files(
        const std::string& base_path,
        const std::string& timestamps_path,
        const std::string& transformed_base_path,
        const TransformAlignmentOptions& options) {
    if (options.dim == 0 || options.half_life_days <= 0.0 || options.query_time <= 0.0 ||
        options.absolute_tolerance < 0.0 || options.relative_tolerance < 0.0 ||
        options.chunk_rows == 0 || options.threads <= 0) {
        throw std::invalid_argument("invalid transform-alignment options");
    }

    const std::size_t row_bytes = options.dim * sizeof(float);
    const std::size_t base_bytes = file_size_bytes(base_path);
    if (base_bytes == 0 || base_bytes % row_bytes != 0) {
        throw std::runtime_error("base is empty or not headerless float32 rows: " + base_path);
    }
    const std::size_t rows = base_bytes / row_bytes;
    require_regular_payload(
        transformed_base_path, base_bytes, "multiplicative transformed base");
    require_regular_payload(timestamps_path, rows * sizeof(float), "timestamp array");

    std::ifstream base(base_path.c_str(), std::ios::binary);
    std::ifstream timestamps(timestamps_path.c_str(), std::ios::binary);
    std::ifstream transformed(transformed_base_path.c_str(), std::ios::binary);
    if (!base || !timestamps || !transformed) {
        throw std::runtime_error("cannot open transform-alignment input files");
    }

    TransformAlignmentReport report;
    report.rows = rows;
    report.coordinates = rows * options.dim;
    report.min_timestamp = std::numeric_limits<double>::infinity();
    report.max_timestamp = -std::numeric_limits<double>::infinity();
    report.min_raw_norm = std::numeric_limits<double>::infinity();
    report.max_raw_norm = 0.0;

    const double lambda = half_life_to_lambda(options.half_life_days);
    std::vector<float> raw(options.chunk_rows * options.dim);
    std::vector<float> weighted(options.chunk_rows * options.dim);
    std::vector<float> times(options.chunk_rows);
    std::vector<RowTransformStats> row_stats(options.chunk_rows);

#ifdef _OPENMP
    omp_set_num_threads(options.threads);
#endif

    for (std::size_t begin = 0; begin < rows; begin += options.chunk_rows) {
        const std::size_t count = std::min(options.chunk_rows, rows - begin);
        read_exact(base, raw.data(), count * row_bytes, "raw base chunk");
        read_exact(transformed, weighted.data(), count * row_bytes, "transformed base chunk");
        read_exact(timestamps, times.data(), count * sizeof(float), "timestamp chunk");

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t local_signed = 0;
             local_signed < static_cast<std::ptrdiff_t>(count);
             ++local_signed) {
            const std::size_t local = static_cast<std::size_t>(local_signed);
            RowTransformStats stats;
            const float* raw_row = raw.data() + local * options.dim;
            const float* transformed_row = weighted.data() + local * options.dim;
            double squared_norm = 0.0;
            for (std::size_t d = 0; d < options.dim; ++d) {
                const double value = raw_row[d];
                if (!std::isfinite(value)) ++stats.nonfinite_raw;
                squared_norm += value * value;
                if (!std::isfinite(static_cast<double>(transformed_row[d]))) {
                    ++stats.nonfinite_transformed;
                }
            }
            stats.norm = std::sqrt(squared_norm);
            stats.zero_norm = !(stats.norm > kNormEpsilon);

            const double timestamp = times[local];
            stats.nonfinite_timestamp = !std::isfinite(timestamp);
            stats.out_of_range_timestamp = !stats.nonfinite_timestamp &&
                (timestamp < -kTimestampRangeEpsilon ||
                 timestamp > options.query_time + kTimestampRangeEpsilon);
            const float weight = stats.nonfinite_timestamp
                ? std::numeric_limits<float>::quiet_NaN()
                : static_cast<float>(chronos_decay_from_timestamp_days(
                      times[local], options.query_time, lambda));
            const float inverse_norm = stats.zero_norm
                ? 0.0f
                : static_cast<float>(1.0 / stats.norm);

            for (std::size_t d = 0; d < options.dim; ++d) {
                float expected = raw_row[d] * weight;
                if (options.cosine) {
                    expected = stats.zero_norm ? 0.0f : raw_row[d] * inverse_norm * weight;
                }
                const double actual = transformed_row[d];
                const double absolute_error = std::fabs(actual - static_cast<double>(expected));
                const double scale = std::max(
                    std::max(std::fabs(actual), std::fabs(static_cast<double>(expected))), 1e-30);
                const double relative_error = absolute_error / scale;
                if (absolute_error > stats.max_absolute_error) {
                    stats.max_absolute_error = absolute_error;
                    stats.max_absolute_error_coordinate = d;
                }
                stats.max_relative_error = std::max(stats.max_relative_error, relative_error);
                const double tolerance = options.absolute_tolerance +
                    options.relative_tolerance * scale;
                if (!std::isfinite(absolute_error) || absolute_error > tolerance) {
                    if (stats.first_mismatch_coordinate == kMissingIndex) {
                        stats.first_mismatch_coordinate = d;
                    }
                    ++stats.mismatch_coordinates;
                }
            }
            row_stats[local] = stats;
        }

        for (std::size_t local = 0; local < count; ++local) {
            const std::size_t row = begin + local;
            const RowTransformStats& stats = row_stats[local];
            report.nonfinite_raw_coordinates += stats.nonfinite_raw;
            report.nonfinite_transformed_coordinates += stats.nonfinite_transformed;
            report.nonfinite_timestamps += stats.nonfinite_timestamp ? 1 : 0;
            report.out_of_range_timestamps += stats.out_of_range_timestamp ? 1 : 0;
            report.zero_norm_rows += stats.zero_norm ? 1 : 0;
            if (std::isfinite(stats.norm)) {
                report.min_raw_norm = std::min(report.min_raw_norm, stats.norm);
                report.max_raw_norm = std::max(report.max_raw_norm, stats.norm);
            }
            if (std::isfinite(static_cast<double>(times[local]))) {
                report.min_timestamp = std::min(report.min_timestamp, static_cast<double>(times[local]));
                report.max_timestamp = std::max(report.max_timestamp, static_cast<double>(times[local]));
            }
            if (stats.mismatch_coordinates != 0) {
                if (report.first_mismatch_row == kMissingIndex) {
                    report.first_mismatch_row = row;
                    report.first_mismatch_coordinate = stats.first_mismatch_coordinate;
                }
                ++report.mismatched_rows;
                report.mismatched_coordinates += stats.mismatch_coordinates;
            }
            if (stats.max_absolute_error > report.max_absolute_error) {
                report.max_absolute_error = stats.max_absolute_error;
                report.max_absolute_error_row = row;
                report.max_absolute_error_coordinate = stats.max_absolute_error_coordinate;
            }
            report.max_relative_error = std::max(
                report.max_relative_error, stats.max_relative_error);
        }

        const std::size_t completed = begin + count;
        if (options.progress_every_rows != 0 &&
            (completed == rows || completed / options.progress_every_rows !=
                                  begin / options.progress_every_rows)) {
            std::fprintf(stderr, "[verify_dataset_alignment] transformed rows %zu / %zu\n",
                         completed, rows);
            std::fflush(stderr);
        }
    }

    if (!std::isfinite(report.min_timestamp)) report.min_timestamp = 0.0;
    if (!std::isfinite(report.max_timestamp)) report.max_timestamp = 0.0;
    if (!std::isfinite(report.min_raw_norm)) report.min_raw_norm = 0.0;
    return report;
}

bool AdditiveTransformAlignmentReport::passed() const {
    return mismatched_coordinates == 0;
}

AdditiveTransformAlignmentReport verify_additive_transform_files(
        const std::string& base_path,
        const std::string& query_path,
        const std::string& timestamps_path,
        const std::string& transformed_base_path,
        const std::string& transformed_query_path,
        const TransformAlignmentOptions& options,
        double alpha) {
    if (options.dim == 0 || options.half_life_days <= 0.0 || options.query_time <= 0.0 ||
        options.absolute_tolerance < 0.0 || options.relative_tolerance < 0.0 ||
        options.chunk_rows == 0 || options.threads <= 0 || alpha < 0.0 || alpha > 1.0) {
        throw std::invalid_argument("invalid additive transform-alignment options");
    }
    const std::size_t raw_row_bytes = options.dim * sizeof(float);
    const std::size_t transformed_row_bytes = (options.dim + 1) * sizeof(float);
    const std::size_t base_bytes = file_size_bytes(base_path);
    const std::size_t query_bytes = file_size_bytes(query_path);
    if (base_bytes == 0 || base_bytes % raw_row_bytes != 0 ||
        query_bytes == 0 || query_bytes % raw_row_bytes != 0) {
        throw std::runtime_error("base/query is empty or not headerless float32 rows");
    }
    AdditiveTransformAlignmentReport report;
    report.base_rows = base_bytes / raw_row_bytes;
    report.query_rows = query_bytes / raw_row_bytes;
    require_regular_payload(
        timestamps_path, report.base_rows * sizeof(float), "timestamp array");
    require_regular_payload(
        transformed_base_path, report.base_rows * transformed_row_bytes,
        "additive transformed base");
    require_regular_payload(
        transformed_query_path, report.query_rows * transformed_row_bytes,
        "additive transformed query");

    std::ifstream base(base_path.c_str(), std::ios::binary);
    std::ifstream query(query_path.c_str(), std::ios::binary);
    std::ifstream timestamps(timestamps_path.c_str(), std::ios::binary);
    std::ifstream transformed_base(transformed_base_path.c_str(), std::ios::binary);
    std::ifstream transformed_query(transformed_query_path.c_str(), std::ios::binary);
    if (!base || !query || !timestamps || !transformed_base || !transformed_query) {
        throw std::runtime_error("cannot open additive transform-alignment input files");
    }

    const double lambda = half_life_to_lambda(options.half_life_days);
    auto compare_values = [&](const std::vector<float>& expected,
                              const std::vector<float>& actual,
                              std::size_t rows,
                              std::size_t global_begin,
                              bool is_query) {
        const std::size_t dim_out = options.dim + 1;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t coordinate = 0; coordinate < dim_out; ++coordinate) {
                const std::size_t offset = row * dim_out + coordinate;
                const double expected_value = expected[offset];
                const double actual_value = actual[offset];
                const double absolute_error = std::fabs(expected_value - actual_value);
                const double scale = std::max(
                    std::max(std::fabs(expected_value), std::fabs(actual_value)), 1e-30);
                const double relative_error = absolute_error / scale;
                report.max_absolute_error = std::max(report.max_absolute_error, absolute_error);
                report.max_relative_error = std::max(report.max_relative_error, relative_error);
                ++report.compared_coordinates;
                if (!std::isfinite(absolute_error) ||
                    absolute_error > options.absolute_tolerance + options.relative_tolerance * scale) {
                    if (report.first_mismatch_row == kMissingIndex) {
                        report.first_mismatch_row = global_begin + row;
                        report.first_mismatch_coordinate = coordinate;
                        report.first_mismatch_is_query = is_query;
                    }
                    ++report.mismatched_coordinates;
                }
            }
        }
    };

    std::vector<float> raw(options.chunk_rows * options.dim);
    std::vector<float> times(options.chunk_rows);
    std::vector<float> actual(options.chunk_rows * (options.dim + 1));
    std::vector<float> expected;
    for (std::size_t begin = 0; begin < report.base_rows; begin += options.chunk_rows) {
        const std::size_t count = std::min(options.chunk_rows, report.base_rows - begin);
        read_exact(base, raw.data(), count * raw_row_bytes, "additive raw-base chunk");
        read_exact(timestamps, times.data(), count * sizeof(float), "additive timestamp chunk");
        read_exact(transformed_base, actual.data(), count * transformed_row_bytes,
                   "additive transformed-base chunk");
        build_additive_transformed_base(
            raw.data(), count, static_cast<int>(options.dim), times.data(), options.query_time,
            lambda, alpha, options.cosine, expected, options.threads);
        compare_values(expected, actual, count, begin, false);
    }
    for (std::size_t begin = 0; begin < report.query_rows; begin += options.chunk_rows) {
        const std::size_t count = std::min(options.chunk_rows, report.query_rows - begin);
        read_exact(query, raw.data(), count * raw_row_bytes, "additive raw-query chunk");
        read_exact(transformed_query, actual.data(), count * transformed_row_bytes,
                   "additive transformed-query chunk");
        build_additive_transformed_queries(
            raw.data(), count, static_cast<int>(options.dim), alpha, options.cosine,
            expected, options.threads);
        compare_values(expected, actual, count, begin, true);
    }
    return report;
}

bool MappedRowsReport::passed() const {
    return mapped_rows == subset_rows && mismatched_rows == 0;
}

MappedRowsReport verify_mapped_raw_rows(
        const std::string& source_path,
        const std::string& subset_path,
        const std::string& original_indices_path,
        std::size_t dim,
        std::size_t chunk_rows) {
    if (dim == 0 || chunk_rows == 0) {
        throw std::invalid_argument("dim and chunk_rows must be positive");
    }
    const std::size_t row_bytes = dim * sizeof(float);
    const std::size_t source_bytes = file_size_bytes(source_path);
    const std::size_t subset_bytes = file_size_bytes(subset_path);
    if (source_bytes == 0 || source_bytes % row_bytes != 0 ||
        subset_bytes == 0 || subset_bytes % row_bytes != 0) {
        throw std::runtime_error("source/subset is not headerless float32 rows");
    }

    MappedRowsReport report;
    report.source_rows = source_bytes / row_bytes;
    report.subset_rows = subset_bytes / row_bytes;
    const std::vector<std::size_t> indices = read_indices(original_indices_path);
    if (indices.size() != report.subset_rows) {
        throw std::runtime_error("original-index count does not match subset rows");
    }
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= report.source_rows || (i != 0 && indices[i] <= indices[i - 1])) {
            throw std::runtime_error("original-index map must be strictly increasing and in range");
        }
    }

    std::ifstream source(source_path.c_str(), std::ios::binary);
    std::ifstream subset(subset_path.c_str(), std::ios::binary);
    if (!source || !subset) throw std::runtime_error("cannot open mapped-row input files");
    std::vector<char> source_buffer(chunk_rows * row_bytes);
    std::vector<char> subset_buffer(chunk_rows * row_bytes);

    std::size_t subset_row = 0;
    while (subset_row < indices.size()) {
        std::size_t run_end = subset_row + 1;
        while (run_end < indices.size() && indices[run_end] == indices[run_end - 1] + 1) {
            ++run_end;
        }
        ++report.contiguous_runs;
        source.seekg(
            static_cast<std::streamoff>(indices[subset_row] * row_bytes), std::ios::beg);
        if (!source) throw std::runtime_error("failed seeking source row");

        for (std::size_t cursor = subset_row; cursor < run_end;) {
            const std::size_t count = std::min(chunk_rows, run_end - cursor);
            const std::size_t bytes = count * row_bytes;
            read_exact(source, source_buffer.data(), bytes, "source mapped rows");
            read_exact(subset, subset_buffer.data(), bytes, "subset mapped rows");
            for (std::size_t local = 0; local < count; ++local) {
                const char* expected = source_buffer.data() + local * row_bytes;
                const char* actual = subset_buffer.data() + local * row_bytes;
                if (std::memcmp(expected, actual, row_bytes) != 0) {
                    if (report.first_mismatch_subset_row == kMissingIndex) {
                        report.first_mismatch_subset_row = cursor + local;
                        report.first_mismatch_source_row = indices[cursor + local];
                    }
                    ++report.mismatched_rows;
                }
                ++report.mapped_rows;
            }
            cursor += count;
        }
        subset_row = run_end;
    }
    return report;
}

bool GroundTruthStructureReport::passed() const {
    return invalid_ids == 0 && rows_with_duplicate_ids == 0;
}

GroundTruthStructureReport verify_groundtruth_structure(
        const std::string& groundtruth_path,
        std::size_t queries,
        std::size_t width,
        std::size_t base_rows) {
    if (queries == 0 || width == 0 || base_rows == 0) {
        throw std::invalid_argument("GT queries, width, and base_rows must be positive");
    }
    const std::size_t ids = queries * width;
    require_regular_payload(
        groundtruth_path, ids * sizeof(GroundTruthId), "ground truth");
    std::ifstream input(groundtruth_path.c_str(), std::ios::binary);
    if (!input) throw std::runtime_error("cannot open ground truth: " + groundtruth_path);

    GroundTruthStructureReport report;
    report.queries = queries;
    report.width = width;
    report.ids = ids;
    report.min_id = std::numeric_limits<std::int64_t>::max();
    report.max_id = std::numeric_limits<std::int64_t>::min();
    std::vector<GroundTruthId> row(width);
    for (std::size_t query = 0; query < queries; ++query) {
        read_exact(input, row.data(), width * sizeof(GroundTruthId), "ground-truth row");
        std::unordered_set<GroundTruthId> unique;
        unique.reserve(width * 2);
        for (std::size_t rank = 0; rank < width; ++rank) {
            const GroundTruthId id = row[rank];
            report.min_id = std::min(report.min_id, id);
            report.max_id = std::max(report.max_id, id);
            if (id < 0 || static_cast<std::uint64_t>(id) >= base_rows) {
                if (report.first_invalid_query == kMissingIndex) {
                    report.first_invalid_query = query;
                    report.first_invalid_rank = rank;
                }
                ++report.invalid_ids;
            }
            unique.insert(id);
        }
        if (unique.size() != width) ++report.rows_with_duplicate_ids;
    }
    return report;
}

bool SourceRowSplitReport::passed() const {
    return duplicate_base_ids == 0 && duplicate_query_ids == 0 &&
           out_of_range_base_ids == 0 && out_of_range_query_ids == 0 &&
           (!same_source || overlapping_ids == 0);
}

SourceRowSplitReport verify_source_row_id_split(
        const std::string& base_ids_path,
        std::size_t base_rows,
        std::size_t base_source_rows,
        const std::string& query_ids_path,
        std::size_t query_rows,
        std::size_t query_source_rows,
        bool same_source) {
    if (base_rows == 0 || query_rows == 0 || base_source_rows == 0 || query_source_rows == 0) {
        throw std::invalid_argument("source row counts must be positive");
    }
    require_regular_payload(base_ids_path, base_rows * sizeof(std::uint64_t), "base source-row IDs");
    require_regular_payload(query_ids_path, query_rows * sizeof(std::uint64_t), "query source-row IDs");
    std::ifstream base_input(base_ids_path.c_str(), std::ios::binary);
    std::ifstream query_input(query_ids_path.c_str(), std::ios::binary);
    if (!base_input || !query_input) throw std::runtime_error("cannot open source-row ID maps");
    std::vector<std::uint64_t> base_ids(base_rows);
    std::vector<std::uint64_t> query_ids(query_rows);
    read_exact(base_input, base_ids.data(), base_ids.size() * sizeof(std::uint64_t), "base source-row IDs");
    read_exact(query_input, query_ids.data(), query_ids.size() * sizeof(std::uint64_t), "query source-row IDs");

    SourceRowSplitReport report;
    report.base_ids = base_rows;
    report.query_ids = query_rows;
    report.same_source = same_source;
    std::unordered_set<std::uint64_t> base_unique;
    std::unordered_set<std::uint64_t> query_unique;
    base_unique.reserve(base_rows * 2);
    query_unique.reserve(query_rows * 2);
    for (std::uint64_t id : base_ids) {
        if (id >= base_source_rows) ++report.out_of_range_base_ids;
        if (!base_unique.insert(id).second) ++report.duplicate_base_ids;
    }
    for (std::uint64_t id : query_ids) {
        if (id >= query_source_rows) ++report.out_of_range_query_ids;
        if (!query_unique.insert(id).second) ++report.duplicate_query_ids;
        if (same_source && base_unique.find(id) != base_unique.end()) ++report.overlapping_ids;
    }
    return report;
}

}  // namespace chronos
