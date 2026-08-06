#pragma once

#include "io.h"

#include <cstddef>
#include <vector>

namespace chronos {

/** Supported time-decayed scoring modes. */
enum class ChronosScoreMode {
    Multiplicative,
    Additive
};

/** Converts a positive half-life in days to lambda = ln(2) / half_life. */
double half_life_to_lambda(double half_life_days);

/**
 * Converts a timestamp to its non-positive offset from max_timestamp_days.
 * The reference value must match metadata generation.
 */
double bounded_time_from_timestamp_days(float timestamp_days, double max_timestamp_days);

/**
 * Evaluates exp(lambda * bounded_time). A valid bounded_time is non-positive.
 */
double chronos_decay_from_bounded_time(double bounded_time, double lambda);

/**
 * Evaluates decay directly from timestamp_days and max_timestamp_days.
 */
double chronos_decay_from_timestamp_days(float timestamp_days, double max_timestamp_days, double lambda);

/** L2-normalizes rows; rows with norm below 1e-12 remain zero. */
void l2_normalize_rows_inplace(float* data, std::size_t n, int dim, int num_threads = 1);

/**
 * Computes exact inner-product top-k with double-precision accumulation.
 * Normalize base and query rows first for cosine semantics. A positive
 * progress_log_every_queries emits progress to stderr at that interval.
 */
void brute_force_ip_topk(const float* queries, std::size_t nq, const float* base, std::size_t nb,
                         int dim, int k, int num_threads, std::vector<GroundTruthId>& gt_ids,
                         std::vector<double>* out_scores = nullptr,
                         int progress_log_every_queries = 0);

/**
 * Computes exact TDVS top-k by exhaustive scoring.
 *
 * semantic = dot(q, base_i)
 * bounded_time_i = timestamp_days[i] - max_timestamp_days
 * decay_i = exp(ln(2) / half_life_days * bounded_time_i)
 *
 * Multiplicative: score = semantic * decay_i
 * Additive: score = alpha * semantic + (1 - alpha) * decay_i
 *
 * half_life_days and max_timestamp_days must be positive, timestamp_days must
 * contain nb values, and alpha must be in [0, 1] for additive scoring.
 */
void brute_force_chronos_topk(const float* queries, std::size_t nq, const float* base,
                              std::size_t nb, int dim, const float* timestamp_days,
                              double max_timestamp_days, double half_life_days, ChronosScoreMode mode,
                              double alpha, int k, int num_threads, std::vector<GroundTruthId>& gt_ids,
                              std::vector<double>* out_scores = nullptr,
                              int progress_log_every_queries = 0);

/** Scales unnormalized base rows: out[i] = weights[i] * base[i]. */
void scale_base_vectors_by_weight(const float* base, std::size_t nb, int dim, const float* weights,
                                  std::vector<float>& out, int num_threads = 1);

/**
 * Scales normalized base rows for inner-product indexing. Do not use a cosine
 * index, which would normalize away the decay-carrying vector norm.
 */
void scale_l2_normalized_base_by_weight(const float* base, std::size_t nb, int dim,
                                        const float* weights, std::vector<float>& out,
                                        int num_threads = 1);

/**
 * Builds the multiplicative MIPS transform. With cosine semantics, x_i is
 * normalized before multiplication by the same decay used for ground truth.
 */
void build_multiplicative_transformed_base(const float* base, std::size_t nb, int dim,
                                           const float* timestamp_days, double max_timestamp_days,
                                           double half_life_days, bool use_l2_normalized_base,
                                           std::vector<float>& out, int num_threads = 1);

/**
 * Builds the exact additive inner-product augmentation with dim + 1 columns.
 *
 * The first dim columns contain sqrt(alpha) times either normalized or raw
 * semantic coordinates. The final column contains sqrt(1-alpha) * decay_i.
 */
void build_additive_transformed_base(const float* base, std::size_t nb, int dim,
                                     const float* timestamp_days, double max_timestamp_days, double lambda,
                                     double alpha, bool normalize_for_semantic, std::vector<float>& out,
                                     int num_threads = 1);

/** Builds q' = [sqrt(alpha) * q, sqrt(1-alpha)] for each query row. */
void build_additive_transformed_queries(const float* queries, std::size_t nq, int dim, double alpha,
                                        bool normalize_for_semantic, std::vector<float>& out,
                                        int num_threads = 1);

/**
 * Samples query/base pairs and verifies that augmented inner products equal
 * the additive TDVS score.
 */
bool verify_additive_transformed_ip_equivalence(const float* base_raw, const float* query_raw,
                                                const float* aug_base, const float* aug_query, std::size_t nb,
                                                std::size_t nq, int dim, const float* timestamp_days,
                                                double max_timestamp_days, double lambda, double alpha,
                                                bool semantic_cosine, int max_samples_per_query);

}  // namespace chronos
