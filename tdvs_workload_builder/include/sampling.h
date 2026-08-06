#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace chronos {

enum class SampleMode {
    Prefix,   // Select the first N rows in file order.
    Random    // Select N rows uniformly without replacement.
};

/**
 * Generates sample_count indices in [0, total). Prefix truncates at total;
 * Random uses mt19937_64(seed) for reproducible sampling without replacement.
 */
std::vector<std::size_t> build_sample_indices(std::size_t total, std::size_t sample_count,
                                              SampleMode mode, std::uint64_t seed);

/**
 * Selects query indices from the complement of idx_base_sorted. Prefix takes
 * the first available rows; Random performs reproducible reservoir sampling.
 * The input base indices and returned query indices are sorted. Returns an
 * empty vector when the complement is too small.
 */
std::vector<std::size_t> build_disjoint_query_indices(const std::vector<std::size_t>& idx_base_sorted,
                                                      std::size_t total, std::size_t sample_query,
                                                      SampleMode query_mode, std::uint64_t seed_query);

/** Formats sampling metadata for inclusion in a run manifest. */
std::string sampling_manifest_lines(const std::string& source_path, std::size_t total_in_file,
                                    std::size_t sample_count, SampleMode mode, std::uint64_t seed,
                                    const std::vector<std::size_t>& original_indices);

}  // namespace chronos
