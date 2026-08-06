#include "sampling.h"

#include <algorithm>
#include <random>
#include <sstream>

namespace chronos {

// Uniform reservoir sampling without replacement in O(total) time and
// O(sample_count) space.
static void reservoir_sample_indices(std::size_t total, std::size_t sample_count,
                                     std::mt19937_64& rng, std::vector<std::size_t>& out) {
    out.resize(sample_count);
    for (std::size_t i = 0; i < sample_count; ++i) out[i] = i;
    for (std::size_t i = sample_count; i < total; ++i) {
        std::uniform_int_distribution<std::size_t> uni(0, i);
        std::size_t j = uni(rng);
        if (j < sample_count) out[j] = i;
    }
}

std::vector<std::size_t> build_sample_indices(std::size_t total, std::size_t sample_count,
                                              SampleMode mode, std::uint64_t seed) {
    if (sample_count == 0 || total == 0) return {};
    std::size_t n = std::min(sample_count, total);
    std::vector<std::size_t> idx;
    if (mode == SampleMode::Prefix) {
        idx.resize(n);
        for (std::size_t i = 0; i < n; ++i) idx[i] = i;
        return idx;
    }
    std::mt19937_64 rng(seed);
    if (n == total) {
        idx.resize(n);
        for (std::size_t i = 0; i < n; ++i) idx[i] = i;
        return idx;
    }
    reservoir_sample_indices(total, n, rng, idx);
    std::sort(idx.begin(), idx.end());
    return idx;
}

static bool index_in_sorted_base(const std::vector<std::size_t>& sorted, std::size_t i) {
    return std::binary_search(sorted.begin(), sorted.end(), i);
}

std::vector<std::size_t> build_disjoint_query_indices(const std::vector<std::size_t>& idx_base_sorted,
                                                      std::size_t total, std::size_t sample_query,
                                                      SampleMode query_mode, std::uint64_t seed_query) {
    std::vector<std::size_t> out;
    if (sample_query == 0 || total == 0) return out;
    const std::size_t nb = idx_base_sorted.size();
    if (nb + sample_query > total) return out;

    if (query_mode == SampleMode::Prefix) {
        out.reserve(sample_query);
        for (std::size_t i = 0; i < total && out.size() < sample_query; ++i) {
            if (!index_in_sorted_base(idx_base_sorted, i)) out.push_back(i);
        }
        if (out.size() < sample_query) out.clear();
        return out;
    }

    // Random: reservoir on complement stream (uniform Q-subset of complement)
    std::mt19937_64 rng(seed_query);
    out.resize(sample_query);
    std::size_t seen = 0;
    for (std::size_t i = 0; i < total; ++i) {
        if (index_in_sorted_base(idx_base_sorted, i)) continue;
        if (seen < sample_query) {
            out[seen] = i;
        } else {
            std::uniform_int_distribution<std::size_t> uni(0, seen);
            std::size_t j = uni(rng);
            if (j < sample_query) out[j] = i;
        }
        ++seen;
    }
    if (seen < sample_query) {
        out.clear();
        return out;
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::string sampling_manifest_lines(const std::string& source_path, std::size_t total_in_file,
                                    std::size_t sample_count, SampleMode mode, std::uint64_t seed,
                                    const std::vector<std::size_t>& original_indices) {
    (void)original_indices;
    std::ostringstream os;
    os << "source_file: " << source_path << "\n";
    os << "vectors_in_source_file: " << total_in_file << "\n";
    os << "requested_sample_count: " << sample_count << "\n";
    os << "actual_sample_count: " << original_indices.size() << "\n";
    os << "sample_mode: " << (mode == SampleMode::Prefix ? "prefix" : "random") << "\n";
    os << "random_algorithm: reservoir_sampling_without_replacement\n";
    os << "rng_seed: " << seed << "\n";
    os << "id_scheme: output_rows_renumbered_0_to_N_minus_1\n";
    os << "groundtruth_ids_reference: same_renumbered_row_indices\n";
    os << "original_source_row_index_file: see companion *_original_indices.txt\n";
    return os.str();
}

}  // namespace chronos
