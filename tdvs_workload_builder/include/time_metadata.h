#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace chronos {

enum class TimeDistribution {
    TopicIndependent,
    TopicCorrelated
};

const char* time_distribution_name(TimeDistribution distribution);

struct TimeMetadataOptions {
    TimeDistribution distribution = TimeDistribution::TopicIndependent;
    std::uint64_t seed = 42;
    float max_timestamp_days = 365.0f;
    int num_clusters = 256;
    int kmeans_iters = 15;
    std::size_t training_sample_rows = 256000;
    float cluster_timestamp_std_days = 0.0f;  // 0 selects H / 12.
    int num_threads = 64;
};

struct TimeMetadataResult {
    std::vector<float> timestamps;
    std::vector<std::int32_t> cluster_ids;
    std::vector<float> cluster_timestamp_centers;
    std::size_t training_rows = 0;
};

/**
 * Generates timestamps in base-row order. Topic-independent mode samples Uniform(0,H).
 * Topic-correlated mode trains spherical k-means on a deterministic sample,
 * assigns every base row, samples a time center per cluster, and draws each
 * timestamp from a normal distribution truncated to [0,H].
 */
TimeMetadataResult generate_time_metadata(
    const float* base_data,
    std::size_t n_base,
    int dim,
    const TimeMetadataOptions& options);

/** Writes timestamp_days as headerless float32 values in base-row order. */
bool write_timestamps_binary(const std::string& path, const std::vector<float>& timestamps);

bool read_timestamps_binary(const std::string& path, std::vector<float>& timestamps);

/** Writes headerless int32 cluster IDs in base-row order. */
bool write_cluster_ids_binary(const std::string& path, const std::vector<std::int32_t>& cluster_ids);

}  // namespace chronos
