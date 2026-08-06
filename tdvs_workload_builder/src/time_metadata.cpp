#include "time_metadata.h"

#include "tdvs.h"
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace chronos {
namespace {

constexpr double kNormEpsilon = 1e-12;

double dot_double(const float* a, const float* b, int dim) {
    double sum = 0.0;
    for (int d = 0; d < dim; ++d) sum += static_cast<double>(a[d]) * b[d];
    return sum;
}

void assign_centers(
        const float* rows,
        std::size_t count,
        int dim,
        const float* centers,
        int clusters,
        std::vector<std::int32_t>& assignments,
        int threads) {
    assignments.resize(count);
    auto assign_one = [&](std::size_t row) {
        const float* values = rows + row * static_cast<std::size_t>(dim);
        int best = 0;
        double best_dot = dot_double(values, centers, dim);
        for (int cluster = 1; cluster < clusters; ++cluster) {
            const double similarity = dot_double(
                values, centers + static_cast<std::size_t>(cluster) * dim, dim);
            if (similarity > best_dot) {
                best_dot = similarity;
                best = cluster;
            }
        }
        assignments[row] = static_cast<std::int32_t>(best);
    };

#ifdef _OPENMP
    omp_set_num_threads(std::max(1, threads));
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(count); ++row) {
        assign_one(static_cast<std::size_t>(row));
    }
#else
    const unsigned worker_count = static_cast<unsigned>(std::max(1, threads));
    if (worker_count == 1 || count < 1024) {
        for (std::size_t row = 0; row < count; ++row) assign_one(row);
    } else {
        const std::size_t block = (count + worker_count - 1) / worker_count;
        std::vector<std::thread> workers;
        for (unsigned worker = 0; worker < worker_count; ++worker) {
            const std::size_t begin = static_cast<std::size_t>(worker) * block;
            if (begin >= count) break;
            const std::size_t end = std::min(count, begin + block);
            workers.emplace_back([&, begin, end]() {
                for (std::size_t row = begin; row < end; ++row) assign_one(row);
            });
        }
        for (std::thread& worker : workers) worker.join();
    }
#endif
}

std::vector<float> normalized_training_sample(
        const float* base,
        std::size_t n_base,
        int dim,
        std::size_t requested_rows,
        std::uint64_t seed,
        int threads,
        std::size_t& actual_rows) {
    const std::size_t count = std::min(n_base, std::max<std::size_t>(1, requested_rows));
    const std::vector<std::size_t> indices = build_sample_indices(
        n_base, count, SampleMode::Random, seed);
    std::vector<float> training(count * static_cast<std::size_t>(dim));
    for (std::size_t row = 0; row < count; ++row) {
        const float* source = base + indices[row] * static_cast<std::size_t>(dim);
        std::copy(source, source + dim, training.begin() + row * static_cast<std::size_t>(dim));
    }
    l2_normalize_rows_inplace(training.data(), count, dim, threads);
    actual_rows = count;
    return training;
}

std::vector<float> initialize_kmeans_plus_plus(
        const float* training,
        std::size_t rows,
        int dim,
        int clusters,
        std::mt19937_64& rng) {
    std::vector<float> centers(static_cast<std::size_t>(clusters) * dim);
    std::uniform_int_distribution<std::size_t> first_pick(0, rows - 1);
    const std::size_t first = first_pick(rng);
    std::copy(training + first * static_cast<std::size_t>(dim),
              training + (first + 1) * static_cast<std::size_t>(dim), centers.begin());

    std::vector<double> minimum_squared_distance(rows, std::numeric_limits<double>::infinity());
    for (int cluster = 1; cluster < clusters; ++cluster) {
        const float* newest_center = centers.data() + static_cast<std::size_t>(cluster - 1) * dim;
        double total = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            const double similarity = std::max(-1.0, std::min(1.0,
                dot_double(training + row * static_cast<std::size_t>(dim), newest_center, dim)));
            const double squared_distance = std::max(0.0, 2.0 * (1.0 - similarity));
            minimum_squared_distance[row] = std::min(minimum_squared_distance[row], squared_distance);
            total += minimum_squared_distance[row];
        }
        std::size_t selected = static_cast<std::size_t>(cluster) % rows;
        if (total > 0.0 && std::isfinite(total)) {
            std::uniform_real_distribution<double> weighted_pick(0.0, total);
            const double target = weighted_pick(rng);
            double cumulative = 0.0;
            for (std::size_t row = 0; row < rows; ++row) {
                cumulative += minimum_squared_distance[row];
                if (cumulative >= target) {
                    selected = row;
                    break;
                }
            }
        }
        std::copy(training + selected * static_cast<std::size_t>(dim),
                  training + (selected + 1) * static_cast<std::size_t>(dim),
                  centers.begin() + static_cast<std::size_t>(cluster) * dim);
    }
    return centers;
}

void update_centers(
        const float* training,
        std::size_t rows,
        int dim,
        const std::vector<std::int32_t>& assignments,
        int clusters,
        std::vector<float>& centers) {
    std::vector<double> sums(static_cast<std::size_t>(clusters) * dim, 0.0);
    std::vector<std::size_t> counts(static_cast<std::size_t>(clusters), 0);
    for (std::size_t row = 0; row < rows; ++row) {
        const int cluster = assignments[row];
        ++counts[static_cast<std::size_t>(cluster)];
        double* destination = sums.data() + static_cast<std::size_t>(cluster) * dim;
        const float* source = training + row * static_cast<std::size_t>(dim);
        for (int d = 0; d < dim; ++d) destination[d] += source[d];
    }
    for (int cluster = 0; cluster < clusters; ++cluster) {
        if (counts[static_cast<std::size_t>(cluster)] == 0) continue;
        double squared_norm = 0.0;
        const double* source = sums.data() + static_cast<std::size_t>(cluster) * dim;
        for (int d = 0; d < dim; ++d) squared_norm += source[d] * source[d];
        const double norm = std::sqrt(squared_norm);
        if (!(norm > kNormEpsilon)) continue;
        float* destination = centers.data() + static_cast<std::size_t>(cluster) * dim;
        for (int d = 0; d < dim; ++d) destination[d] = static_cast<float>(source[d] / norm);
    }
}

double sample_truncated_normal(
        std::mt19937_64& rng,
        double center,
        double standard_deviation,
        double lower,
        double upper) {
    std::normal_distribution<double> distribution(center, standard_deviation);
    for (;;) {
        const double value = distribution(rng);
        if (value >= lower && value <= upper) return value;
    }
}

}  // namespace

const char* time_distribution_name(TimeDistribution distribution) {
    return distribution == TimeDistribution::TopicIndependent
        ? "topic-independent" : "topic-correlated";
}

TimeMetadataResult generate_time_metadata(
        const float* base_data,
        std::size_t n_base,
        int dim,
        const TimeMetadataOptions& options) {
    if (!base_data || n_base == 0 || dim <= 0 || options.max_timestamp_days <= 0.0f ||
        options.num_threads <= 0) {
        throw std::invalid_argument("invalid time-metadata generation arguments");
    }
    TimeMetadataResult result;
    result.timestamps.resize(n_base);
    std::mt19937_64 rng(options.seed);
    const double horizon = options.max_timestamp_days;

    if (options.distribution == TimeDistribution::TopicIndependent) {
        std::uniform_real_distribution<double> timestamp(0.0, horizon);
        for (float& value : result.timestamps) value = static_cast<float>(timestamp(rng));
        return result;
    }

    if (options.num_clusters < 2 || options.kmeans_iters < 1 ||
        options.training_sample_rows == 0) {
        throw std::invalid_argument("topic-correlated generation requires K>=2, iterations>=1, and training rows");
    }
    const int clusters = static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(options.num_clusters), n_base));
    std::vector<float> training = normalized_training_sample(
        base_data, n_base, dim, options.training_sample_rows, options.seed,
        options.num_threads, result.training_rows);
    if (result.training_rows < static_cast<std::size_t>(clusters)) {
        throw std::invalid_argument("cluster training sample must contain at least K rows");
    }
    std::vector<float> centers = initialize_kmeans_plus_plus(
        training.data(), result.training_rows, dim, clusters, rng);
    std::vector<std::int32_t> training_assignments;
    std::fprintf(stderr,
        "[time_metadata] spherical k-means: training_rows=%zu k=%d dim=%d iters=%d\n",
        result.training_rows, clusters, dim, options.kmeans_iters);
    for (int iteration = 0; iteration < options.kmeans_iters; ++iteration) {
        std::fprintf(stderr, "[time_metadata] k-means iteration %d / %d\n",
                     iteration + 1, options.kmeans_iters);
        std::fflush(stderr);
        assign_centers(training.data(), result.training_rows, dim, centers.data(), clusters,
                       training_assignments, options.num_threads);
        update_centers(training.data(), result.training_rows, dim, training_assignments,
                       clusters, centers);
    }
    std::fprintf(stderr, "[time_metadata] assigning all %zu base rows\n", n_base);
    std::fflush(stderr);
    // Dividing every row's dot products by its norm would not change the
    // winning unit center, so final assignment can use canonical rows directly.
    assign_centers(base_data, n_base, dim, centers.data(), clusters,
                   result.cluster_ids, options.num_threads);

    std::uniform_real_distribution<double> center_distribution(0.0, horizon);
    result.cluster_timestamp_centers.resize(static_cast<std::size_t>(clusters));
    for (float& center : result.cluster_timestamp_centers) {
        center = static_cast<float>(center_distribution(rng));
    }
    const double sigma = options.cluster_timestamp_std_days > 0.0f
        ? options.cluster_timestamp_std_days : horizon / 12.0;
    if (!(sigma > 0.0) || !std::isfinite(sigma)) {
        throw std::invalid_argument("cluster timestamp standard deviation must be positive");
    }
    for (std::size_t row = 0; row < n_base; ++row) {
        const double center = result.cluster_timestamp_centers[
            static_cast<std::size_t>(result.cluster_ids[row])];
        result.timestamps[row] = static_cast<float>(
            sample_truncated_normal(rng, center, sigma, 0.0, horizon));
    }
    return result;
}

bool write_timestamps_binary(const std::string& path, const std::vector<float>& timestamps) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    const std::size_t written = std::fwrite(timestamps.data(), sizeof(float), timestamps.size(), fp);
    std::fclose(fp);
    return written == timestamps.size();
}

bool read_timestamps_binary(const std::string& path, std::vector<float>& timestamps) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    if (std::fseek(fp, 0, SEEK_END) != 0) {
        std::fclose(fp);
        return false;
    }
    const long size = std::ftell(fp);
    if (size < 0 || size % static_cast<long>(sizeof(float)) != 0) {
        std::fclose(fp);
        return false;
    }
    timestamps.resize(static_cast<std::size_t>(size) / sizeof(float));
    std::rewind(fp);
    const std::size_t read = std::fread(timestamps.data(), sizeof(float), timestamps.size(), fp);
    std::fclose(fp);
    return read == timestamps.size();
}

bool write_cluster_ids_binary(
        const std::string& path,
        const std::vector<std::int32_t>& cluster_ids) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    const std::size_t written = std::fwrite(
        cluster_ids.data(), sizeof(std::int32_t), cluster_ids.size(), fp);
    std::fclose(fp);
    return written == cluster_ids.size();
}

}  // namespace chronos
