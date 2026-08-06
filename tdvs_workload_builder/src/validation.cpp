#include "validation.h"

#include "io.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace chronos {
namespace {

constexpr double kZeroNormEpsilon = 1e-12;

std::uint64_t hash_row(const float* row, int dim) {
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(row);
    const std::size_t count = static_cast<std::size_t>(dim) * sizeof(float);
    std::uint64_t hash = UINT64_C(1469598103934665603);
    for (std::size_t i = 0; i < count; ++i) {
        hash ^= static_cast<std::uint64_t>(bytes[i]);
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

void merge_row_stats(
        VectorRowsReport& report,
        const float* data,
        std::size_t rows,
        int dim,
        bool require_unit_norm,
        double tolerance) {
    for (std::size_t row = 0; row < rows; ++row) {
        const float* values = data + row * static_cast<std::size_t>(dim);
        double squared_norm = 0.0;
        bool finite_row = true;
        for (int d = 0; d < dim; ++d) {
            const double value = values[d];
            if (!std::isfinite(value)) {
                ++report.nonfinite_coordinates;
                finite_row = false;
            } else {
                squared_norm += value * value;
            }
        }
        if (!finite_row) continue;
        const double norm = std::sqrt(squared_norm);
        report.min_norm = std::min(report.min_norm, norm);
        report.max_norm = std::max(report.max_norm, norm);
        if (!(norm > kZeroNormEpsilon)) ++report.zero_norm_rows;
        const double error = std::fabs(norm - 1.0);
        report.max_unit_norm_error = std::max(report.max_unit_norm_error, error);
        if (require_unit_norm && error > tolerance) ++report.non_unit_rows;
    }
    report.rows += rows;
    report.coordinates += rows * static_cast<std::size_t>(dim);
}

}  // namespace

bool VectorRowsReport::valid(bool require_unit_norm) const {
    return nonfinite_coordinates == 0 && zero_norm_rows == 0 &&
           (!require_unit_norm || non_unit_rows == 0);
}

VectorRowsReport audit_vector_rows(
        const float* data,
        std::size_t rows,
        int dim,
        bool require_unit_norm,
        double unit_norm_tolerance,
        bool count_exact_duplicates) {
    if (!data || rows == 0 || dim <= 0 || unit_norm_tolerance < 0.0) {
        throw std::invalid_argument("invalid vector-audit arguments");
    }
    VectorRowsReport report;
    report.min_norm = std::numeric_limits<double>::infinity();
    merge_row_stats(report, data, rows, dim, require_unit_norm, unit_norm_tolerance);

    if (count_exact_duplicates) {
        std::vector<std::pair<std::uint64_t, std::size_t>> hashes(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            hashes[row] = {hash_row(data + row * static_cast<std::size_t>(dim), dim), row};
        }
        std::sort(hashes.begin(), hashes.end());
        const std::size_t row_bytes = static_cast<std::size_t>(dim) * sizeof(float);
        for (std::size_t begin = 0; begin < hashes.size();) {
            std::size_t end = begin + 1;
            while (end < hashes.size() && hashes[end].first == hashes[begin].first) ++end;
            std::vector<std::size_t> representatives;
            for (std::size_t cursor = begin; cursor < end; ++cursor) {
                const std::size_t candidate = hashes[cursor].second;
                bool duplicate = false;
                for (std::size_t representative : representatives) {
                    if (std::memcmp(
                            data + candidate * static_cast<std::size_t>(dim),
                            data + representative * static_cast<std::size_t>(dim),
                            row_bytes) == 0) {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate) ++report.exact_duplicate_rows;
                else representatives.push_back(candidate);
            }
            begin = end;
        }
    }
    if (!std::isfinite(report.min_norm)) report.min_norm = 0.0;
    return report;
}

VectorRowsReport audit_vector_file(
        const std::string& path,
        int dim,
        bool require_unit_norm,
        double unit_norm_tolerance,
        std::size_t chunk_rows) {
    if (dim <= 0 || unit_norm_tolerance < 0.0 || chunk_rows == 0) {
        throw std::invalid_argument("invalid vector-file audit arguments");
    }
    const std::size_t row_bytes = static_cast<std::size_t>(dim) * sizeof(float);
    const std::size_t bytes = file_size_bytes(path);
    if (bytes == 0 || bytes % row_bytes != 0) {
        throw std::runtime_error("vector file is empty or not headerless float32 rows: " + path);
    }
    const std::size_t rows = bytes / row_bytes;
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input) throw std::runtime_error("cannot open vector file: " + path);

    VectorRowsReport report;
    report.min_norm = std::numeric_limits<double>::infinity();
    std::vector<float> buffer(chunk_rows * static_cast<std::size_t>(dim));
    for (std::size_t begin = 0; begin < rows; begin += chunk_rows) {
        const std::size_t count = std::min(chunk_rows, rows - begin);
        const std::size_t values = count * static_cast<std::size_t>(dim);
        input.read(
            reinterpret_cast<char*>(buffer.data()),
            static_cast<std::streamsize>(values * sizeof(float)));
        if (!input) throw std::runtime_error("short read while auditing vector file: " + path);
        merge_row_stats(report, buffer.data(), count, dim, require_unit_norm, unit_norm_tolerance);
    }
    if (!std::isfinite(report.min_norm)) report.min_norm = 0.0;
    return report;
}

std::size_t count_exact_query_base_overlaps(
        const float* base,
        std::size_t base_rows,
        const float* query,
        std::size_t query_rows,
        int dim) {
    if (!base || !query || dim <= 0) return 0;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> query_by_hash;
    query_by_hash.reserve(query_rows * 2 + 1);
    for (std::size_t row = 0; row < query_rows; ++row) {
        query_by_hash[hash_row(query + row * static_cast<std::size_t>(dim), dim)].push_back(row);
    }
    std::vector<unsigned char> matched(query_rows, 0);
    const std::size_t row_bytes = static_cast<std::size_t>(dim) * sizeof(float);
    for (std::size_t base_row = 0; base_row < base_rows; ++base_row) {
        const float* base_values = base + base_row * static_cast<std::size_t>(dim);
        const auto found = query_by_hash.find(hash_row(base_values, dim));
        if (found == query_by_hash.end()) continue;
        for (std::size_t query_row : found->second) {
            if (!matched[query_row] &&
                std::memcmp(
                    base_values,
                    query + query_row * static_cast<std::size_t>(dim),
                    row_bytes) == 0) {
                matched[query_row] = 1;
            }
        }
    }
    return static_cast<std::size_t>(std::count(matched.begin(), matched.end(), 1));
}

}  // namespace chronos
