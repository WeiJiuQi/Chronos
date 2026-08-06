#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hnswlib.h"

namespace chronos_napg {

// Clean-room implementation of Algorithms 3 and 4 in:
// S. Tan et al., "Norm Adjusted Proximity Graph for Fast Inner Product
// Retrieval", KDD 2021, DOI 10.1145/3447548.3467412.

struct EstimationConfig {
  size_t norm_ranges = 5;
  size_t samples_per_range = 100;
  size_t top_n = 100;
  uint64_t seed = 42;
  unsigned threads = 1;
};

struct RangeStatistics {
  size_t begin_rank = 0;
  size_t end_rank = 0;
  size_t samples_used = 0;
  double min_norm = 0.0;
  double max_norm = 0.0;
  double average_source_candidate_ip = 0.0;
  double average_candidate_pair_ip = 0.0;
  double alpha = 0.0;
};

struct EstimationResult {
  size_t effective_top_n = 0;
  std::vector<double> upper_norm_boundaries;
  std::vector<double> alphas;
  // Algorithm 3 explicitly records the norm-rank partition as a map R.  Keep
  // the exact per-row assignment so equal-norm vectors split by a rank
  // boundary still receive the factor estimated for their own stratum.
  std::vector<size_t> range_by_row;
  std::vector<RangeStatistics> ranges;
  uint64_t source_candidate_inner_products = 0;
  uint64_t candidate_pair_inner_products = 0;
};

inline double SquaredNorm(const float* vector, size_t dim) {
  double result = 0.0;
  for (size_t d = 0; d < dim; ++d) {
    const double value = vector[d];
    result += value * value;
  }
  return result;
}

inline float InnerProduct(const float* left, const float* right, size_t dim) {
  float result = 0.0f;
  for (size_t d = 0; d < dim; ++d) result += left[d] * right[d];
  return result;
}

inline std::vector<size_t> ReservoirSample(
    const std::vector<size_t>& norm_order, size_t begin, size_t end,
    size_t count, uint64_t seed) {
  const size_t population = end - begin;
  count = std::min(count, population);
  std::vector<size_t> sample(count);
  for (size_t i = 0; i < count; ++i) sample[i] = norm_order[begin + i];
  std::mt19937_64 generator(seed);
  for (size_t offset = count; offset < population; ++offset) {
    std::uniform_int_distribution<size_t> distribution(0, offset);
    const size_t position = distribution(generator);
    if (position < count) sample[position] = norm_order[begin + offset];
  }
  return sample;
}

struct SampleStatistics {
  double source_candidate_sum = 0.0;
  double candidate_pair_sum = 0.0;
  uint64_t source_candidate_count = 0;
  uint64_t candidate_pair_count = 0;
};

inline SampleStatistics EstimateOneSample(
    const float* base, size_t rows, size_t dim, size_t stride,
    size_t sample_id, size_t top_n) {
  using ScoredId = std::pair<float, size_t>;
  // The smallest retained score is at the top. A larger ID loses a score tie,
  // which makes preprocessing deterministic across thread counts.
  auto worse_first = [](const ScoredId& left, const ScoredId& right) {
    if (left.first != right.first) return left.first > right.first;
    return left.second < right.second;
  };
  std::priority_queue<ScoredId, std::vector<ScoredId>, decltype(worse_first)>
      top(worse_first);
  const float* source = base + sample_id * stride;
  for (size_t candidate = 0; candidate < rows; ++candidate) {
    if (candidate == sample_id) continue;
    const float score = InnerProduct(source, base + candidate * stride, dim);
    if (!std::isfinite(score)) {
      throw std::runtime_error("NAPG factor estimation encountered a non-finite inner product");
    }
    if (top.size() < top_n) {
      top.emplace(score, candidate);
    } else if (score > top.top().first ||
               (score == top.top().first && candidate < top.top().second)) {
      top.pop();
      top.emplace(score, candidate);
    }
  }

  std::vector<ScoredId> neighbors;
  neighbors.reserve(top.size());
  while (!top.empty()) {
    neighbors.push_back(top.top());
    top.pop();
  }
  if (neighbors.size() < 2) {
    throw std::runtime_error("NAPG factor estimation needs at least two MIPS neighbors");
  }

  SampleStatistics result;
  for (const ScoredId& neighbor : neighbors) {
    result.source_candidate_sum += neighbor.first;
    ++result.source_candidate_count;
  }
  // Eq. (6) asks for the average p_i^T p_j among the top-n results.
  // Edge selection compares distinct candidates, so use every unordered
  // distinct pair exactly once and exclude self-pairs.
  for (size_t i = 0; i < neighbors.size(); ++i) {
    const float* left = base + neighbors[i].second * stride;
    for (size_t j = i + 1; j < neighbors.size(); ++j) {
      result.candidate_pair_sum +=
          InnerProduct(left, base + neighbors[j].second * stride, dim);
      ++result.candidate_pair_count;
    }
  }
  return result;
}

inline EstimationResult EstimateAdjustingFactors(
    const float* base, size_t rows, size_t dim, size_t stride,
    const EstimationConfig& config) {
  if (base == nullptr || rows < 3 || dim == 0 || stride < dim) {
    throw std::invalid_argument("Invalid matrix for NAPG factor estimation");
  }
  if (config.norm_ranges == 0 || config.norm_ranges > rows) {
    throw std::invalid_argument("NAPG norm range count must be in [1, N]");
  }
  if (config.samples_per_range == 0) {
    throw std::invalid_argument("NAPG samples per range must be positive");
  }
  if (config.top_n < 2 || config.threads == 0) {
    throw std::invalid_argument("NAPG top-n must be >=2 and threads must be positive");
  }

  std::vector<double> norms(rows);
  std::vector<size_t> norm_order(rows);
  std::iota(norm_order.begin(), norm_order.end(), size_t{0});
  for (size_t row = 0; row < rows; ++row) {
    const double squared = SquaredNorm(base + row * stride, dim);
    if (!(squared > 0.0) || !std::isfinite(squared)) {
      throw std::runtime_error(
          "NAPG requires finite non-zero base vectors; invalid row " +
          std::to_string(row));
    }
    norms[row] = std::sqrt(squared);
  }
  std::stable_sort(norm_order.begin(), norm_order.end(),
                   [&](size_t left, size_t right) {
                     if (norms[left] != norms[right]) return norms[left] < norms[right];
                     return left < right;
                   });

  EstimationResult result;
  result.effective_top_n = std::min(config.top_n, rows - 1);
  result.ranges.resize(config.norm_ranges);
  result.alphas.resize(config.norm_ranges);
  result.range_by_row.resize(rows);
  if (config.norm_ranges > 1) {
    result.upper_norm_boundaries.reserve(config.norm_ranges - 1);
  }

  struct SampleTask {
    size_t range = 0;
    size_t id = 0;
  };
  std::vector<SampleTask> tasks;
  for (size_t range = 0; range < config.norm_ranges; ++range) {
    const size_t begin = range * rows / config.norm_ranges;
    const size_t end = (range + 1) * rows / config.norm_ranges;
    RangeStatistics& stats = result.ranges[range];
    stats.begin_rank = begin;
    stats.end_rank = end;
    stats.min_norm = norms[norm_order[begin]];
    stats.max_norm = norms[norm_order[end - 1]];
    for (size_t rank = begin; rank < end; ++rank) {
      result.range_by_row[norm_order[rank]] = range;
    }
    if (range + 1 < config.norm_ranges) {
      result.upper_norm_boundaries.push_back(stats.max_norm);
    }
    const auto samples = ReservoirSample(
        norm_order, begin, end, config.samples_per_range,
        config.seed + 0x9e3779b97f4a7c15ULL * (range + 1));
    stats.samples_used = samples.size();
    for (size_t id : samples) tasks.push_back(SampleTask{range, id});
  }

  std::vector<SampleStatistics> per_sample(tasks.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(config.threads) if(config.threads > 1)
#endif
  for (long long task = 0; task < static_cast<long long>(tasks.size()); ++task) {
    per_sample[static_cast<size_t>(task)] = EstimateOneSample(
        base, rows, dim, stride, tasks[static_cast<size_t>(task)].id,
        result.effective_top_n);
  }

  std::vector<double> source_sums(config.norm_ranges, 0.0);
  std::vector<double> pair_sums(config.norm_ranges, 0.0);
  std::vector<uint64_t> source_counts(config.norm_ranges, 0);
  std::vector<uint64_t> pair_counts(config.norm_ranges, 0);
  for (size_t task = 0; task < tasks.size(); ++task) {
    const size_t range = tasks[task].range;
    source_sums[range] += per_sample[task].source_candidate_sum;
    pair_sums[range] += per_sample[task].candidate_pair_sum;
    source_counts[range] += per_sample[task].source_candidate_count;
    pair_counts[range] += per_sample[task].candidate_pair_count;
  }

  for (size_t range = 0; range < config.norm_ranges; ++range) {
    if (source_counts[range] == 0 || pair_counts[range] == 0) {
      throw std::runtime_error("NAPG factor estimation produced an empty expectation");
    }
    RangeStatistics& stats = result.ranges[range];
    stats.average_source_candidate_ip =
        source_sums[range] / static_cast<double>(source_counts[range]);
    stats.average_candidate_pair_ip =
        pair_sums[range] / static_cast<double>(pair_counts[range]);
    if (std::abs(stats.average_source_candidate_ip) <=
        std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error("NAPG factor denominator is numerically zero");
    }
    stats.alpha = stats.average_candidate_pair_ip /
                  stats.average_source_candidate_ip;
    if (!(stats.alpha > 0.0) || !std::isfinite(stats.alpha)) {
      throw std::runtime_error(
          "NAPG estimated a non-positive or non-finite adjusting factor");
    }
    result.alphas[range] = stats.alpha;
    result.source_candidate_inner_products += source_counts[range];
    result.candidate_pair_inner_products += pair_counts[range];
  }
  return result;
}

class Runtime {
 public:
  Runtime(size_t max_elements, size_t dim,
          std::vector<double> upper_norm_boundaries,
          std::vector<double> alphas,
          std::vector<size_t> range_by_label = {})
      : dim_(dim),
        upper_norm_boundaries_(std::move(upper_norm_boundaries)),
        alphas_(std::move(alphas)),
        range_by_label_(std::move(range_by_label)),
        alpha_by_internal_id_(max_elements,
                              std::numeric_limits<double>::quiet_NaN()) {
    if (dim_ == 0 || alphas_.empty() ||
        upper_norm_boundaries_.size() + 1 != alphas_.size()) {
      throw std::invalid_argument("Invalid NAPG runtime model");
    }
    if (!std::is_sorted(upper_norm_boundaries_.begin(),
                        upper_norm_boundaries_.end())) {
      throw std::invalid_argument("NAPG norm boundaries must be sorted");
    }
    for (size_t range : range_by_label_) {
      if (range >= alphas_.size()) {
        throw std::invalid_argument("NAPG range map contains an invalid range");
      }
    }
  }

  void Prepare(hnswlib::tableint internal_id, hnswlib::labeltype label,
               const void* data) {
    if (internal_id >= alpha_by_internal_id_.size() || data == nullptr) {
      throw std::out_of_range("NAPG prepared element is out of range");
    }
    size_t range = 0;
    if (label < range_by_label_.size()) {
      range = range_by_label_[static_cast<size_t>(label)];
      if (range >= alphas_.size()) {
        throw std::logic_error("NAPG range map contains an invalid range");
      }
    } else {
      // This fallback follows the paper's norm-value map for a future element
      // not present during estimation. Existing build rows use the exact
      // rank-stratum assignment above.
      const double norm = std::sqrt(
          SquaredNorm(static_cast<const float*>(data), dim_));
      range = static_cast<size_t>(std::lower_bound(
          upper_norm_boundaries_.begin(), upper_norm_boundaries_.end(), norm) -
          upper_norm_boundaries_.begin());
    }
    alpha_by_internal_id_[internal_id] = alphas_[range];
  }

  void Prepare(hnswlib::tableint internal_id, const void* data) {
    Prepare(internal_id, static_cast<hnswlib::labeltype>(internal_id), data);
  }

  bool Reject(hnswlib::tableint source_id, float source_candidate_distance,
              float selected_candidate_distance) const {
    if (source_id >= alpha_by_internal_id_.size()) {
      throw std::out_of_range("NAPG edge-selection source is out of range");
    }
    const double alpha = alpha_by_internal_id_[source_id];
    if (!(alpha > 0.0) || !std::isfinite(alpha)) {
      throw std::logic_error("NAPG source factor was not prepared before edge selection");
    }
    // hnswlib IP distance is 1 - inner_product. This is exactly Eq. (4):
    // reject p_i iff alpha_r * x^T p_i < p^T p_i.
    const double source_candidate_ip = 1.0 - source_candidate_distance;
    const double selected_candidate_ip = 1.0 - selected_candidate_distance;
    edge_selection_comparisons_.fetch_add(1, std::memory_order_relaxed);
    return alpha * source_candidate_ip < selected_candidate_ip;
  }

  uint64_t edgeSelectionComparisons() const {
    return edge_selection_comparisons_.load(std::memory_order_relaxed);
  }

  static void PrepareCallback(hnswlib::tableint internal_id,
                              hnswlib::labeltype label, const void* data,
                              const void* parameter) {
    const_cast<Runtime*>(static_cast<const Runtime*>(parameter))
        ->Prepare(internal_id, label, data);
  }

  static bool RejectCallback(
      hnswlib::tableint source_id, float source_candidate_distance,
      hnswlib::tableint candidate_id, hnswlib::tableint selected_id,
      float selected_candidate_distance, int level, const void* parameter) {
    (void)candidate_id;
    (void)selected_id;
    (void)level;
    return static_cast<const Runtime*>(parameter)->Reject(
        source_id, source_candidate_distance, selected_candidate_distance);
  }

 private:
  size_t dim_;
  std::vector<double> upper_norm_boundaries_;
  std::vector<double> alphas_;
  std::vector<size_t> range_by_label_;
  std::vector<double> alpha_by_internal_id_;
  mutable std::atomic<uint64_t> edge_selection_comparisons_{0};
};

}  // namespace chronos_napg
