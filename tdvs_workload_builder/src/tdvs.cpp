#include "tdvs.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace chronos {

double half_life_to_lambda(double half_life_days) { return std::log(2.0) / half_life_days; }

double bounded_time_from_timestamp_days(float timestamp_days, double max_timestamp_days) {
    return static_cast<double>(timestamp_days) - max_timestamp_days;
}

double chronos_decay_from_bounded_time(double bounded_time, double lambda) {
    return std::exp(lambda * bounded_time);
}

double chronos_decay_from_timestamp_days(float timestamp_days, double max_timestamp_days, double lambda) {
    return chronos_decay_from_bounded_time(bounded_time_from_timestamp_days(timestamp_days, max_timestamp_days),
                                           lambda);
}

static constexpr double kL2NormEps = 1e-12;

void l2_normalize_rows_inplace(float* data, std::size_t n, int dim, int num_threads) {
    if (!data || n == 0 || dim <= 0) return;
#ifdef _OPENMP
    omp_set_num_threads(std::max(1, num_threads));
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i_signed = 0; i_signed < static_cast<std::ptrdiff_t>(n); ++i_signed) {
        const std::size_t i = static_cast<std::size_t>(i_signed);
        float* row = data + i * static_cast<std::size_t>(dim);
        double sumsq = 0.0;
        for (int d = 0; d < dim; ++d) sumsq += static_cast<double>(row[d]) * static_cast<double>(row[d]);
        double norm = std::sqrt(sumsq);
        if (norm > kL2NormEps) {
            float inv = static_cast<float>(1.0 / norm);
            for (int d = 0; d < dim; ++d) row[d] *= inv;
        }
    }
}

static double dot_double(const float* a, const float* b, int dim) {
    double s = 0.0;
    for (int d = 0; d < dim; ++d) s += static_cast<double>(a[d]) * static_cast<double>(b[d]);
    return s;
}

struct ScoredId {
    double score;
    GroundTruthId id;
};

// The top element is the worst retained item: lower score, then larger ID.
struct BetterScoredId {
    bool operator()(const ScoredId& left, const ScoredId& right) const {
        if (left.score != right.score) return left.score > right.score;
        return left.id < right.id;
    }
};

using TopKHeap = std::priority_queue<ScoredId, std::vector<ScoredId>, BetterScoredId>;

static bool better_than(const ScoredId& candidate, const ScoredId& incumbent) {
    return candidate.score > incumbent.score ||
           (candidate.score == incumbent.score && candidate.id < incumbent.id);
}

static void push_topk(TopKHeap& heap, int k, double score, GroundTruthId id) {
    const ScoredId candidate{score, id};
    if (static_cast<int>(heap.size()) < k) {
        heap.push(candidate);
        return;
    }
    if (better_than(candidate, heap.top())) {
        heap.pop();
        heap.push(candidate);
    }
}

void brute_force_ip_topk(const float* queries, std::size_t nq, const float* base, std::size_t nb,
                         int dim, int k, int num_threads, std::vector<GroundTruthId>& gt_ids,
                         std::vector<double>* out_scores, int progress_log_every_queries) {
    if (k <= 0 || nb == 0 || nq == 0) {
        gt_ids.clear();
        if (out_scores) out_scores->clear();
        return;
    }
    int kk = std::min(k, static_cast<int>(nb));
    gt_ids.resize(nq * static_cast<std::size_t>(kk));
    if (out_scores) out_scores->resize(nq * static_cast<std::size_t>(kk));

    std::atomic<std::size_t> queries_done{0};
    const int prog_step = progress_log_every_queries;

    auto run_query = [&](std::size_t qi) {
        const float* q = queries + qi * static_cast<std::size_t>(dim);
        TopKHeap heap;
        for (std::size_t bi = 0; bi < nb; ++bi) {
            const float* b = base + bi * static_cast<std::size_t>(dim);
            double s = dot_double(q, b, dim);
            push_topk(heap, kk, s, static_cast<GroundTruthId>(bi));
        }
        std::vector<ScoredId> tmp;
        tmp.reserve(heap.size());
        while (!heap.empty()) {
            tmp.push_back(heap.top());
            heap.pop();
        }
        std::sort(tmp.begin(), tmp.end(),
                  [](const ScoredId& a, const ScoredId& b) {
                      return a.score > b.score || (a.score == b.score && a.id < b.id);
                  });
        for (int t = 0; t < kk; ++t) {
            gt_ids[qi * static_cast<std::size_t>(kk) + static_cast<std::size_t>(t)] =
                tmp[static_cast<std::size_t>(t)].id;
            if (out_scores)
                (*out_scores)[qi * static_cast<std::size_t>(kk) + static_cast<std::size_t>(t)] =
                    tmp[static_cast<std::size_t>(t)].score;
        }
        if (prog_step > 0) {
            std::size_t c = queries_done.fetch_add(1, std::memory_order_relaxed) + 1;
            std::size_t step = static_cast<std::size_t>(prog_step);
            if (c % step == 0 || c == nq) {
                std::fprintf(stderr, "[brute_force_ip_topk] queries done %zu / %zu\n", c, nq);
                std::fflush(stderr);
            }
        }
    };

#ifdef _OPENMP
    omp_set_num_threads(std::max(1, num_threads));
#pragma omp parallel for schedule(dynamic, 1)
    for (std::ptrdiff_t qi = 0; qi < static_cast<std::ptrdiff_t>(nq); ++qi) {
        run_query(static_cast<std::size_t>(qi));
    }
#else
    {
        unsigned nt = static_cast<unsigned>(std::max(1, num_threads));
        if (nt == 1 || nq <= 1) {
            for (std::size_t qi = 0; qi < nq; ++qi) run_query(qi);
        } else {
            std::atomic<std::size_t> next_q{0};
            auto worker = [&]() {
                for (;;) {
                    std::size_t qi = next_q.fetch_add(1, std::memory_order_relaxed);
                    if (qi >= nq) break;
                    run_query(qi);
                }
            };
            std::vector<std::thread> workers;
            workers.reserve(nt);
            for (unsigned t = 0; t < nt; ++t) workers.emplace_back(worker);
            for (auto& w : workers) w.join();
        }
    }
#endif
}

void brute_force_chronos_topk(const float* queries, std::size_t nq, const float* base,
                              std::size_t nb, int dim, const float* timestamp_days,
                              double max_timestamp_days, double half_life_days, ChronosScoreMode mode,
                              double alpha, int k, int num_threads, std::vector<GroundTruthId>& gt_ids,
                              std::vector<double>* out_scores, int progress_log_every_queries) {
    if (half_life_days <= 0.0 || max_timestamp_days <= 0.0 || !timestamp_days) {
        gt_ids.clear();
        if (out_scores) out_scores->clear();
        return;
    }
    if (mode == ChronosScoreMode::Additive && (alpha < 0.0 || alpha > 1.0)) {
        gt_ids.clear();
        if (out_scores) out_scores->clear();
        return;
    }
    if (k <= 0 || nb == 0 || nq == 0) {
        gt_ids.clear();
        if (out_scores) out_scores->clear();
        return;
    }
    int kk = std::min(k, static_cast<int>(nb));
    gt_ids.resize(nq * static_cast<std::size_t>(kk));
    if (out_scores) out_scores->resize(nq * static_cast<std::size_t>(kk));

    const double lambda = half_life_to_lambda(half_life_days);

    std::atomic<std::size_t> queries_done{0};
    const int prog_step = progress_log_every_queries;

    auto run_query = [&](std::size_t qi) {
        const float* q = queries + qi * static_cast<std::size_t>(dim);
        TopKHeap heap;
        for (std::size_t bi = 0; bi < nb; ++bi) {
            const float* b = base + bi * static_cast<std::size_t>(dim);
            double semantic = dot_double(q, b, dim);
            double decay =
                chronos_decay_from_timestamp_days(timestamp_days[bi], max_timestamp_days, lambda);
            double s = 0.0;
            if (mode == ChronosScoreMode::Multiplicative) {
                s = semantic * decay;
            } else {
                s = alpha * semantic + (1.0 - alpha) * decay;
            }
            push_topk(heap, kk, s, static_cast<GroundTruthId>(bi));
        }
        std::vector<ScoredId> tmp;
        tmp.reserve(heap.size());
        while (!heap.empty()) {
            tmp.push_back(heap.top());
            heap.pop();
        }
        std::sort(tmp.begin(), tmp.end(),
                  [](const ScoredId& a, const ScoredId& b) {
                      return a.score > b.score || (a.score == b.score && a.id < b.id);
                  });
        for (int t = 0; t < kk; ++t) {
            gt_ids[qi * static_cast<std::size_t>(kk) + static_cast<std::size_t>(t)] =
                tmp[static_cast<std::size_t>(t)].id;
            if (out_scores)
                (*out_scores)[qi * static_cast<std::size_t>(kk) + static_cast<std::size_t>(t)] =
                    tmp[static_cast<std::size_t>(t)].score;
        }
        if (prog_step > 0) {
            std::size_t c = queries_done.fetch_add(1, std::memory_order_relaxed) + 1;
            std::size_t step = static_cast<std::size_t>(prog_step);
            if (c % step == 0 || c == nq) {
                std::fprintf(stderr, "[brute_force_chronos_topk] queries done %zu / %zu\n", c, nq);
                std::fflush(stderr);
            }
        }
    };

#ifdef _OPENMP
    omp_set_num_threads(std::max(1, num_threads));
#pragma omp parallel for schedule(dynamic, 1)
    for (std::ptrdiff_t qi = 0; qi < static_cast<std::ptrdiff_t>(nq); ++qi) {
        run_query(static_cast<std::size_t>(qi));
    }
#else
    {
        unsigned nt = static_cast<unsigned>(std::max(1, num_threads));
        if (nt == 1 || nq <= 1) {
            for (std::size_t qi = 0; qi < nq; ++qi) run_query(qi);
        } else {
            std::atomic<std::size_t> next_q{0};
            auto worker = [&]() {
                for (;;) {
                    std::size_t qi = next_q.fetch_add(1, std::memory_order_relaxed);
                    if (qi >= nq) break;
                    run_query(qi);
                }
            };
            std::vector<std::thread> workers;
            workers.reserve(nt);
            for (unsigned t = 0; t < nt; ++t) workers.emplace_back(worker);
            for (auto& w : workers) w.join();
        }
    }
#endif
}

void scale_base_vectors_by_weight(const float* base, std::size_t nb, int dim, const float* weights,
                                  std::vector<float>& out, int num_threads) {
    out.resize(nb * static_cast<std::size_t>(dim));
#ifdef _OPENMP
    omp_set_num_threads(std::max(1, num_threads));
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i_signed = 0; i_signed < static_cast<std::ptrdiff_t>(nb); ++i_signed) {
        const std::size_t i = static_cast<std::size_t>(i_signed);
        float w = weights[i];
        const float* src = base + i * static_cast<std::size_t>(dim);
        float* dst = out.data() + i * static_cast<std::size_t>(dim);
        for (int d = 0; d < dim; ++d) dst[d] = src[d] * w;
    }
}

static void l2_normalize_one_row_inplace(float* row, int dim) {
    double sumsq = 0.0;
    for (int d = 0; d < dim; ++d) sumsq += static_cast<double>(row[d]) * static_cast<double>(row[d]);
    double norm = std::sqrt(sumsq);
    if (norm > kL2NormEps) {
        float inv = static_cast<float>(1.0 / norm);
        for (int d = 0; d < dim; ++d) row[d] *= inv;
    }
}

static double semantic_dot_for_chronos(const float* q, const float* x, int dim, bool cosine_semantic) {
    if (!cosine_semantic) return dot_double(q, x, dim);
    std::vector<float> qc(static_cast<std::size_t>(dim)), xc(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        qc[static_cast<std::size_t>(d)] = q[d];
        xc[static_cast<std::size_t>(d)] = x[d];
    }
    l2_normalize_one_row_inplace(qc.data(), dim);
    l2_normalize_one_row_inplace(xc.data(), dim);
    return dot_double(qc.data(), xc.data(), dim);
}

void build_additive_transformed_base(const float* base, std::size_t nb, int dim,
                                     const float* timestamp_days, double max_timestamp_days, double lambda,
                                     double alpha, bool normalize_for_semantic, std::vector<float>& out,
                                     int num_threads) {
    if (!base || !timestamp_days || nb == 0 || dim <= 0 || alpha < 0.0 || alpha > 1.0) {
        out.clear();
        return;
    }
    const int dim_out = dim + 1;
    out.resize(nb * static_cast<std::size_t>(dim_out));
    const float sa = static_cast<float>(std::sqrt(alpha));
    const float sm = static_cast<float>(std::sqrt(1.0 - alpha));
#ifdef _OPENMP
    omp_set_num_threads(std::max(1, num_threads));
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i_signed = 0; i_signed < static_cast<std::ptrdiff_t>(nb); ++i_signed) {
        const std::size_t i = static_cast<std::size_t>(i_signed);
        const float* src = base + i * static_cast<std::size_t>(dim);
        float* dst = out.data() + i * static_cast<std::size_t>(dim_out);
        if (normalize_for_semantic) {
            double squared_norm = 0.0;
            for (int d = 0; d < dim; ++d) squared_norm += static_cast<double>(src[d]) * src[d];
            const double norm = std::sqrt(squared_norm);
            const float scale = norm > kL2NormEps ? static_cast<float>(sa / norm) : 0.0f;
            for (int d = 0; d < dim; ++d) dst[d] = scale * src[d];
        } else {
            for (int d = 0; d < dim; ++d) dst[d] = sa * src[d];
        }
        double decay =
            chronos_decay_from_timestamp_days(timestamp_days[i], max_timestamp_days, lambda);
        dst[dim] = sm * static_cast<float>(decay);
    }
}

void build_additive_transformed_queries(const float* queries, std::size_t nq, int dim, double alpha,
                                        bool normalize_for_semantic, std::vector<float>& out,
                                        int num_threads) {
    if (!queries || nq == 0 || dim <= 0 || alpha < 0.0 || alpha > 1.0) {
        out.clear();
        return;
    }
    const int dim_out = dim + 1;
    out.resize(nq * static_cast<std::size_t>(dim_out));
    const float sa = static_cast<float>(std::sqrt(alpha));
    const float sm = static_cast<float>(std::sqrt(1.0 - alpha));
#ifdef _OPENMP
    omp_set_num_threads(std::max(1, num_threads));
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t qi_signed = 0; qi_signed < static_cast<std::ptrdiff_t>(nq); ++qi_signed) {
        const std::size_t qi = static_cast<std::size_t>(qi_signed);
        const float* src = queries + qi * static_cast<std::size_t>(dim);
        float* dst = out.data() + qi * static_cast<std::size_t>(dim_out);
        if (normalize_for_semantic) {
            double squared_norm = 0.0;
            for (int d = 0; d < dim; ++d) squared_norm += static_cast<double>(src[d]) * src[d];
            const double norm = std::sqrt(squared_norm);
            const float scale = norm > kL2NormEps ? static_cast<float>(sa / norm) : 0.0f;
            for (int d = 0; d < dim; ++d) dst[d] = scale * src[d];
        } else {
            for (int d = 0; d < dim; ++d) dst[d] = sa * src[d];
        }
        dst[dim] = sm;
    }
}

bool verify_additive_transformed_ip_equivalence(const float* base_raw, const float* query_raw,
                                                const float* aug_base, const float* aug_query, std::size_t nb,
                                                std::size_t nq, int dim, const float* timestamp_days,
                                                double max_timestamp_days, double lambda, double alpha,
                                                bool semantic_cosine, int max_samples_per_query) {
    if (!base_raw || !query_raw || !aug_base || !aug_query || !timestamp_days || nb == 0 || nq == 0 ||
        dim <= 0)
        return false;
    const int dim_out = dim + 1;
    const int ncheck_q = static_cast<int>(std::min<std::size_t>(nq, 3));
    const int ncheck_b = std::max(1, max_samples_per_query);
    for (int qi = 0; qi < ncheck_q; ++qi) {
        const float* q = query_raw + static_cast<std::size_t>(qi) * static_cast<std::size_t>(dim);
        const float* qp = aug_query + static_cast<std::size_t>(qi) * static_cast<std::size_t>(dim_out);
        int nb_sample = static_cast<int>(std::min<std::size_t>(nb, static_cast<std::size_t>(ncheck_b)));
        for (int bi = 0; bi < nb_sample; ++bi) {
            const float* x = base_raw + static_cast<std::size_t>(bi) * static_cast<std::size_t>(dim);
            const float* xp = aug_base + static_cast<std::size_t>(bi) * static_cast<std::size_t>(dim_out);
            double semantic = semantic_dot_for_chronos(q, x, dim, semantic_cosine);
            double decay = chronos_decay_from_timestamp_days(timestamp_days[static_cast<std::size_t>(bi)],
                                                               max_timestamp_days, lambda);
            double expected = alpha * semantic + (1.0 - alpha) * decay;
            double got = dot_double(qp, xp, dim_out);
            const double tol = std::max(1e-6, 1e-5 * std::max(1.0, std::fabs(expected)));
            if (std::fabs(got - expected) > tol) {
                std::fprintf(stderr,
                             "[verify_additive_transformed_ip_equivalence] mismatch qi=%d bi=%d "
                             "expected=%.9g got=%.9g\n",
                             qi, bi, expected, got);
                std::fflush(stderr);
                return false;
            }
        }
    }
    return true;
}

void scale_l2_normalized_base_by_weight(const float* base, std::size_t nb, int dim,
                                         const float* weights, std::vector<float>& out,
                                         int num_threads) {
    out.resize(nb * static_cast<std::size_t>(dim));
#ifdef _OPENMP
    omp_set_num_threads(std::max(1, num_threads));
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i_signed = 0; i_signed < static_cast<std::ptrdiff_t>(nb); ++i_signed) {
        const std::size_t i = static_cast<std::size_t>(i_signed);
        const float* src = base + i * static_cast<std::size_t>(dim);
        float* dst = out.data() + i * static_cast<std::size_t>(dim);
        double sumsq = 0.0;
        for (int d = 0; d < dim; ++d) sumsq += static_cast<double>(src[d]) * static_cast<double>(src[d]);
        double norm = std::sqrt(sumsq);
        float w = weights[i];
        if (norm > kL2NormEps) {
            float invn = static_cast<float>(1.0 / norm);
            for (int d = 0; d < dim; ++d) dst[d] = src[d] * invn * w;
        } else {
            for (int d = 0; d < dim; ++d) dst[d] = 0.f;
        }
    }
}

void build_multiplicative_transformed_base(const float* base, std::size_t nb, int dim,
                                             const float* timestamp_days, double max_timestamp_days,
                                             double half_life_days, bool use_l2_normalized_base,
                                             std::vector<float>& out, int num_threads) {
    if (!base || !timestamp_days || nb == 0 || dim <= 0 || half_life_days <= 0.0 ||
        max_timestamp_days <= 0.0) {
        out.clear();
        return;
    }
    const double lambda = half_life_to_lambda(half_life_days);
    std::vector<float> w(nb);
#ifdef _OPENMP
    omp_set_num_threads(std::max(1, num_threads));
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i_signed = 0; i_signed < static_cast<std::ptrdiff_t>(nb); ++i_signed) {
        const std::size_t i = static_cast<std::size_t>(i_signed);
        w[i] = static_cast<float>(
            chronos_decay_from_timestamp_days(timestamp_days[i], max_timestamp_days, lambda));
    }
    if (use_l2_normalized_base)
        scale_l2_normalized_base_by_weight(base, nb, dim, w.data(), out, num_threads);
    else
        scale_base_vectors_by_weight(base, nb, dim, w.data(), out, num_threads);
}

}  // namespace chronos
