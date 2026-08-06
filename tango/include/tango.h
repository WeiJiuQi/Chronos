#pragma once

#include "hnswlib.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hnswlib {

enum class TangoMode {
    Multiplicative,
    Additive
};

enum class TangoDataDataMode {
    TimeLift,
    Semantic
};

enum class KappaSchedule {
    Fixed,
    TwoTier,
    PerLevel
};

enum class TangoDecayFastPath {
    Off,
    QueryData,
    DataData,
    All
};

inline const char *tangoDecayFastPathName(TangoDecayFastPath mode) {
    switch (mode) {
        case TangoDecayFastPath::Off: return "off";
        case TangoDecayFastPath::QueryData: return "qd";
        case TangoDecayFastPath::DataData: return "dd";
        case TangoDecayFastPath::All: return "all";
    }
    throw std::invalid_argument("TANGO decay fastpath mode is invalid");
}

inline TangoDecayFastPath parseTangoDecayFastPath(const std::string &value) {
    if (value == "off") return TangoDecayFastPath::Off;
    if (value == "qd") return TangoDecayFastPath::QueryData;
    if (value == "dd") return TangoDecayFastPath::DataData;
    if (value == "all") return TangoDecayFastPath::All;
    throw std::invalid_argument(
        "TANGO decay fastpath must be one of off, qd, dd, or all");
}

inline bool tangoFastPathUsesQD(TangoDecayFastPath mode) {
    return mode == TangoDecayFastPath::QueryData ||
           mode == TangoDecayFastPath::All;
}

inline bool tangoFastPathUsesDD(TangoDecayFastPath mode) {
    return mode == TangoDecayFastPath::DataData ||
           mode == TangoDecayFastPath::All;
}

struct TangoDecayFastPathOptions {
    TangoDecayFastPath mode = TangoDecayFastPath::Off;
    double reference_time = std::numeric_limits<double>::quiet_NaN();
    size_t verify_samples = 0;
    // Instrumentation adds relaxed atomic increments to distance callbacks.
    // It is off by default and can be enabled explicitly by command-line tools.
    bool instrumentation = false;
};

struct TangoDecayFastPathStats {
    uint64_t qd_callbacks = 0;
    uint64_t dd_callbacks = 0;
    uint64_t direct_qd_exp = 0;
    uint64_t direct_dd_exp = 0;
    uint64_t query_scale_exp = 0;
    uint64_t fast_qd = 0;
    uint64_t fast_dd = 0;
    uint64_t fallbacks = 0;
    uint64_t active_entries = 0;
    uint64_t cache_bytes = 0;
    double cache_init_ms = 0.0;
    double verify_qd_max_abs = 0.0;
    double verify_qd_max_rel = 0.0;
    double verify_dd_max_abs = 0.0;
    double verify_dd_max_rel = 0.0;
};

namespace tango_detail {

inline void atomicMax(std::atomic<double> &target, double value) {
    double observed = target.load(std::memory_order_relaxed);
    while (value > observed && !target.compare_exchange_weak(
            observed, value, std::memory_order_relaxed,
            std::memory_order_relaxed)) {
    }
}

class TangoDecayRuntime {
 public:
    TangoDecayRuntime()
        : mode_(static_cast<int>(TangoDecayFastPath::Off)),
          instrumentation_(false), reference_time_(0.0), lambda_(0.0),
          capacity_(0), cache_inverse_(false), verify_samples_(0), active_entries_(0),
          cache_init_ms_(0.0), verify_qd_remaining_(0),
          verify_dd_remaining_(0), qd_callbacks_(0), dd_callbacks_(0),
          direct_qd_exp_(0), direct_dd_exp_(0), query_scale_exp_(0),
          fast_qd_(0), fast_dd_(0), fallbacks_(0),
          verify_qd_max_abs_(0.0), verify_qd_max_rel_(0.0),
          verify_dd_max_abs_(0.0), verify_dd_max_rel_(0.0) {}

    TangoDecayRuntime(const TangoDecayRuntime &) = delete;
    TangoDecayRuntime &operator=(const TangoDecayRuntime &) = delete;

    void configure(
            const TangoDecayFastPathOptions &options,
            double reference_time,
            double lambda,
            size_t capacity) {
        if (!std::isfinite(reference_time) || !std::isfinite(lambda) ||
                lambda <= 0.0) {
            throw std::invalid_argument(
                "TANGO fastpath reference time/lambda is invalid");
        }
        mode_.store(
            static_cast<int>(TangoDecayFastPath::Off),
            std::memory_order_release);
        instrumentation_ = options.instrumentation;
        reference_time_ = reference_time;
        lambda_ = lambda;
        capacity_ = capacity;
        cache_inverse_ = tangoFastPathUsesDD(options.mode);
        verify_samples_ = options.verify_samples;
        active_entries_.store(0, std::memory_order_relaxed);
        verify_qd_remaining_.store(
            tangoFastPathUsesQD(options.mode) ? options.verify_samples : 0,
            std::memory_order_relaxed);
        verify_dd_remaining_.store(
            tangoFastPathUsesDD(options.mode) ? options.verify_samples : 0,
            std::memory_order_relaxed);

        basis_.reset();
        inverse_.reset();
        state_.reset();
        if (options.mode != TangoDecayFastPath::Off) {
            basis_.reset(new double[capacity_]);
            if (cache_inverse_) inverse_.reset(new double[capacity_]);
            state_.reset(new std::atomic<uint8_t>[capacity_]);
            for (size_t i = 0; i < capacity_; ++i) {
                state_[i].store(0, std::memory_order_relaxed);
            }
        }
        mode_.store(static_cast<int>(options.mode), std::memory_order_release);
    }

    TangoDecayFastPath mode() const {
        return static_cast<TangoDecayFastPath>(
            mode_.load(std::memory_order_acquire));
    }

    bool instrumentation() const { return instrumentation_; }
    bool verificationEnabled() const { return verify_samples_ != 0; }
    double referenceTime() const { return reference_time_; }
    bool cachesInverse() const { return cache_inverse_; }
    size_t capacity() const { return capacity_; }

    void initializeEntry(size_t internal_id, double timestamp) noexcept {
        if (!state_ || internal_id >= capacity_) return;
        const double exponent = lambda_ * (timestamp - reference_time_);
        const double basis = std::exp(exponent);
        uint8_t flags = 0;
        if (std::isfinite(basis) && basis > 0.0) {
            basis_[internal_id] = basis;
            flags |= kBasisValid;
            if (cache_inverse_) {
                const double inverse = 1.0 / basis;
                if (std::isfinite(inverse) && inverse > 0.0) {
                    inverse_[internal_id] = inverse;
                    flags |= kInverseValid;
                }
            }
        }
        const uint8_t prior = state_[internal_id].load(std::memory_order_relaxed);
        state_[internal_id].store(
            static_cast<uint8_t>(kInitialized | flags),
            std::memory_order_release);
        if ((prior & kInitialized) == 0) {
            active_entries_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    bool loadBasis(size_t internal_id, double &basis) const {
        if (!state_ || internal_id >= capacity_) return false;
        const uint8_t flags = state_[internal_id].load(std::memory_order_acquire);
        if ((flags & (kInitialized | kBasisValid)) !=
                (kInitialized | kBasisValid)) return false;
        basis = basis_[internal_id];
        return true;
    }

    bool loadBasisAndInverse(
            size_t internal_id,
            double &basis,
            double &inverse) const {
        if (!state_ || !inverse_ || internal_id >= capacity_) return false;
        const uint8_t flags = state_[internal_id].load(std::memory_order_acquire);
        if ((flags & (kInitialized | kBasisValid | kInverseValid)) !=
                (kInitialized | kBasisValid | kInverseValid)) return false;
        basis = basis_[internal_id];
        inverse = inverse_[internal_id];
        return true;
    }

    void setCacheInitMs(double value) { cache_init_ms_ = value; }

    void resetCounters() {
        verify_qd_remaining_.store(
            tangoFastPathUsesQD(mode()) ? verify_samples_ : 0,
            std::memory_order_relaxed);
        verify_dd_remaining_.store(
            tangoFastPathUsesDD(mode()) ? verify_samples_ : 0,
            std::memory_order_relaxed);
        qd_callbacks_.store(0, std::memory_order_relaxed);
        dd_callbacks_.store(0, std::memory_order_relaxed);
        direct_qd_exp_.store(0, std::memory_order_relaxed);
        direct_dd_exp_.store(0, std::memory_order_relaxed);
        query_scale_exp_.store(0, std::memory_order_relaxed);
        fast_qd_.store(0, std::memory_order_relaxed);
        fast_dd_.store(0, std::memory_order_relaxed);
        fallbacks_.store(0, std::memory_order_relaxed);
        verify_qd_max_abs_.store(0.0, std::memory_order_relaxed);
        verify_qd_max_rel_.store(0.0, std::memory_order_relaxed);
        verify_dd_max_abs_.store(0.0, std::memory_order_relaxed);
        verify_dd_max_rel_.store(0.0, std::memory_order_relaxed);
    }

    TangoDecayFastPathStats stats() const {
        TangoDecayFastPathStats result;
        result.qd_callbacks = qd_callbacks_.load(std::memory_order_relaxed);
        result.dd_callbacks = dd_callbacks_.load(std::memory_order_relaxed);
        result.direct_qd_exp = direct_qd_exp_.load(std::memory_order_relaxed);
        result.direct_dd_exp = direct_dd_exp_.load(std::memory_order_relaxed);
        result.query_scale_exp = query_scale_exp_.load(std::memory_order_relaxed);
        result.fast_qd = fast_qd_.load(std::memory_order_relaxed);
        result.fast_dd = fast_dd_.load(std::memory_order_relaxed);
        result.fallbacks = fallbacks_.load(std::memory_order_relaxed);
        result.active_entries = active_entries_.load(std::memory_order_relaxed);
        result.cache_bytes = state_
            ? static_cast<uint64_t>(capacity_) *
                (sizeof(double) + sizeof(std::atomic<uint8_t>) +
                 (cache_inverse_ ? sizeof(double) : 0))
            : 0;
        result.cache_init_ms = cache_init_ms_;
        result.verify_qd_max_abs = verify_qd_max_abs_.load(std::memory_order_relaxed);
        result.verify_qd_max_rel = verify_qd_max_rel_.load(std::memory_order_relaxed);
        result.verify_dd_max_abs = verify_dd_max_abs_.load(std::memory_order_relaxed);
        result.verify_dd_max_rel = verify_dd_max_rel_.load(std::memory_order_relaxed);
        return result;
    }

    void noteQDCallback() const { note(qd_callbacks_); }
    void noteDDCallback() const { note(dd_callbacks_); }
    void noteDirectQD(bool force = false) const {
        note(direct_qd_exp_, force);
    }
    void noteDirectDD(bool force = false) const {
        note(direct_dd_exp_, force);
    }
    void noteQueryScale() const { note(query_scale_exp_); }
    void noteFastQD() const { note(fast_qd_); }
    void noteFastDD() const { note(fast_dd_); }
    void noteFallback() const {
        // Fallbacks are exceptional and must remain observable even when the
        // regular hot-path instrumentation is disabled.
        fallbacks_.fetch_add(1, std::memory_order_relaxed);
    }

    bool takeQDVerificationSample() const {
        return takeSample(verify_qd_remaining_);
    }
    bool takeDDVerificationSample() const {
        return takeSample(verify_dd_remaining_);
    }

    void recordQDVerification(double fast, double direct) const {
        recordVerification(
            fast, direct, verify_qd_max_abs_, verify_qd_max_rel_, "QD");
    }
    void recordDDVerification(double fast, double direct) const {
        recordVerification(
            fast, direct, verify_dd_max_abs_, verify_dd_max_rel_, "DD");
    }

 private:
    static const uint8_t kInitialized = 1;
    static const uint8_t kBasisValid = 2;
    static const uint8_t kInverseValid = 4;

    void note(std::atomic<uint64_t> &counter, bool force = false) const {
        if (instrumentation_ || force) {
            counter.fetch_add(1, std::memory_order_relaxed);
        }
    }

    static bool takeSample(std::atomic<uint64_t> &remaining) {
        uint64_t observed = remaining.load(std::memory_order_relaxed);
        while (observed != 0) {
            if (remaining.compare_exchange_weak(
                    observed, observed - 1, std::memory_order_relaxed,
                    std::memory_order_relaxed)) return true;
        }
        return false;
    }

    static void recordVerification(
            double fast,
            double direct,
            std::atomic<double> &max_abs,
            std::atomic<double> &max_rel,
            const char *kind) {
        const double abs_error = std::fabs(fast - direct);
        const double rel_error = abs_error / std::max(1e-300, std::fabs(direct));
        atomicMax(max_abs, abs_error);
        atomicMax(max_rel, rel_error);
        // Distance callbacks return float; different but algebraically
        // equivalent double evaluation orders may round one float ULP apart.
        if (abs_error > 2e-6 + 2e-6 * std::fabs(direct)) {
            throw std::runtime_error(
                std::string("TANGO fastpath ") + kind +
                " verification exceeded tolerance");
        }
    }

    std::atomic<int> mode_;
    bool instrumentation_;
    double reference_time_;
    double lambda_;
    size_t capacity_;
    bool cache_inverse_;
    size_t verify_samples_;
    std::unique_ptr<double[]> basis_;
    std::unique_ptr<double[]> inverse_;
    std::unique_ptr<std::atomic<uint8_t>[]> state_;
    std::atomic<uint64_t> active_entries_;
    double cache_init_ms_;
    mutable std::atomic<uint64_t> verify_qd_remaining_;
    mutable std::atomic<uint64_t> verify_dd_remaining_;
    mutable std::atomic<uint64_t> qd_callbacks_;
    mutable std::atomic<uint64_t> dd_callbacks_;
    mutable std::atomic<uint64_t> direct_qd_exp_;
    mutable std::atomic<uint64_t> direct_dd_exp_;
    mutable std::atomic<uint64_t> query_scale_exp_;
    mutable std::atomic<uint64_t> fast_qd_;
    mutable std::atomic<uint64_t> fast_dd_;
    mutable std::atomic<uint64_t> fallbacks_;
    mutable std::atomic<double> verify_qd_max_abs_;
    mutable std::atomic<double> verify_qd_max_rel_;
    mutable std::atomic<double> verify_dd_max_abs_;
    mutable std::atomic<double> verify_dd_max_rel_;
};

}  // namespace tango_detail

struct TangoConfig {
    size_t dim = 0;
    TangoMode mode = TangoMode::Multiplicative;
    TangoDataDataMode data_data_mode = TangoDataDataMode::TimeLift;
    double half_life = 0.0;
    double lambda = 0.0;
    double alpha = 1.0;
    double kappa_base = 0.0;
    double kappa_nav = 0.0;
    KappaSchedule schedule = KappaSchedule::Fixed;
    std::vector<double> kappa_levels;
    bool normalize_input = true;
    double future_time_epsilon = 1e-9;

    inline double kappaForLevel(int level) const {
        if (level < 0) {
            throw std::invalid_argument("TANGO level must be non-negative");
        }
        if (schedule == KappaSchedule::PerLevel) {
            if (kappa_levels.empty()) {
                throw std::logic_error(
                    "TANGO per-level schedule has no kappa values");
            }
            const size_t index = std::min(
                static_cast<size_t>(level), kappa_levels.size() - 1);
            return kappa_levels[index];
        }
        return (schedule == KappaSchedule::TwoTier && level > 0)
            ? kappa_nav : kappa_base;
    }

    void validate() const {
        if (dim == 0) throw std::invalid_argument("TANGO dim must be positive");
        if (dim > (std::numeric_limits<size_t>::max() - sizeof(double)) /
                sizeof(float)) {
            throw std::invalid_argument("TANGO dim overflows the packed record size");
        }
        if (mode != TangoMode::Multiplicative && mode != TangoMode::Additive) {
            throw std::invalid_argument("TANGO mode is invalid");
        }
        if (data_data_mode != TangoDataDataMode::TimeLift &&
                data_data_mode != TangoDataDataMode::Semantic) {
            throw std::invalid_argument("TANGO data-data mode is invalid");
        }
        if (schedule != KappaSchedule::Fixed &&
                schedule != KappaSchedule::TwoTier &&
                schedule != KappaSchedule::PerLevel) {
            throw std::invalid_argument("TANGO kappa schedule is invalid");
        }
        if (!std::isfinite(half_life) || half_life <= 0.0) {
            throw std::invalid_argument("TANGO half_life must be finite and positive");
        }
        if (!std::isfinite(lambda) || lambda <= 0.0) {
            throw std::invalid_argument("TANGO lambda must be finite and positive");
        }
        const double expected_lambda = std::log(2.0) / half_life;
        const double lambda_tolerance = 1e-10 * std::max(1.0, std::fabs(expected_lambda));
        if (std::fabs(lambda - expected_lambda) > lambda_tolerance) {
            throw std::invalid_argument("TANGO lambda must equal ln(2)/half_life");
        }
        if (!std::isfinite(alpha) || alpha < 0.0 || alpha > 1.0) {
            throw std::invalid_argument("TANGO alpha must lie in [0, 1]");
        }
        if (mode == TangoMode::Multiplicative && std::fabs(alpha - 1.0) > 1e-12) {
            throw std::invalid_argument("TANGO multiplicative mode requires alpha=1");
        }
        if (!std::isfinite(kappa_base) || !std::isfinite(kappa_nav) ||
                kappa_base < 0.0 || kappa_nav < kappa_base) {
            throw std::invalid_argument(
                "TANGO requires finite 0 <= kappa_base <= kappa_nav");
        }
        if (schedule == KappaSchedule::Fixed &&
                std::fabs(kappa_base - kappa_nav) > 1e-12) {
            throw std::invalid_argument(
                "TANGO fixed schedule requires kappa_base == kappa_nav");
        }
        if (schedule == KappaSchedule::PerLevel) {
            if (kappa_levels.empty()) {
                throw std::invalid_argument(
                    "TANGO per-level schedule requires at least one kappa");
            }
            for (size_t i = 0; i < kappa_levels.size(); ++i) {
                if (!std::isfinite(kappa_levels[i]) || kappa_levels[i] < 0.0) {
                    throw std::invalid_argument(
                        "TANGO per-level kappas must be finite and non-negative");
                }
                if (i != 0 && kappa_levels[i] < kappa_levels[i - 1]) {
                    throw std::invalid_argument(
                        "TANGO per-level kappas must be nondecreasing");
                }
            }
            const double endpoint_tolerance = 1e-12 * std::max(
                1.0, std::max(std::fabs(kappa_base), std::fabs(kappa_nav)));
            if (std::fabs(kappa_base - kappa_levels.front()) > endpoint_tolerance ||
                    std::fabs(kappa_nav - kappa_levels.back()) > endpoint_tolerance) {
                throw std::invalid_argument(
                    "TANGO per-level kappa endpoints must match kappa_base/kappa_nav");
            }
        } else if (!kappa_levels.empty()) {
            throw std::invalid_argument(
                "TANGO fixed/two-tier schedules must not contain per-level kappas");
        }
        if (!std::isfinite(future_time_epsilon) || future_time_epsilon < 0.0) {
            throw std::invalid_argument(
                "TANGO future_time_epsilon must be finite and non-negative");
        }
    }
};

inline const char *tangoModeName(TangoMode mode) {
    return mode == TangoMode::Multiplicative ? "multiplicative" : "additive";
}

inline const char *tangoDataDataModeName(TangoDataDataMode mode) {
    switch (mode) {
        case TangoDataDataMode::TimeLift: return "timelift";
        case TangoDataDataMode::Semantic: return "semantic";
    }
    throw std::invalid_argument("TANGO data-data mode is invalid");
}

inline const char *tangoScheduleName(KappaSchedule schedule) {
    switch (schedule) {
        case KappaSchedule::Fixed: return "fixed";
        case KappaSchedule::TwoTier: return "two_tier";
        case KappaSchedule::PerLevel: return "per_level";
    }
    throw std::invalid_argument("TANGO kappa schedule is invalid");
}

inline size_t tangoTimestampOffset(size_t dim) {
    return dim * sizeof(float);
}

inline size_t tangoRecordSize(size_t dim) {
    return tangoTimestampOffset(dim) + sizeof(double);
}

inline double readTangoTimestamp(const void *record, size_t timestamp_offset) {
    if (record == nullptr) throw std::invalid_argument("TANGO record is null");
    double timestamp = 0.0;
    std::memcpy(&timestamp,
                static_cast<const char *>(record) + timestamp_offset,
                sizeof(timestamp));
    return timestamp;
}

inline void writeTangoTimestamp(void *record, size_t timestamp_offset, double timestamp) {
    if (record == nullptr) throw std::invalid_argument("TANGO record is null");
    std::memcpy(static_cast<char *>(record) + timestamp_offset,
                &timestamp,
                sizeof(timestamp));
}

namespace tango_detail {

inline TangoConfig validatedConfig(const TangoConfig &config) {
    config.validate();
    return config;
}

inline void fillTangoRecordValidated(
        const float *semantic,
        double timestamp,
        const TangoConfig &config,
        uint8_t *record,
        size_t record_size) {
    if (semantic == nullptr) throw std::invalid_argument("TANGO vector is null");
    if (config.dim == 0 ||
            config.dim > (std::numeric_limits<size_t>::max() - sizeof(double)) /
                sizeof(float)) {
        throw std::invalid_argument("TANGO record dimension is invalid");
    }
    if (!std::isfinite(timestamp)) {
        throw std::invalid_argument("TANGO timestamp must be finite");
    }
    if (record == nullptr || record_size != tangoRecordSize(config.dim)) {
        throw std::invalid_argument("TANGO output record has the wrong size");
    }

    double norm2 = 0.0;
    for (size_t i = 0; i < config.dim; ++i) {
        const double value = semantic[i];
        if (!std::isfinite(value)) {
            throw std::invalid_argument("TANGO semantic vector contains non-finite values");
        }
        norm2 += value * value;
    }
    if (!std::isfinite(norm2) || norm2 <= 0.0) {
        throw std::invalid_argument("TANGO semantic vector must have non-zero finite norm");
    }
    const double norm = std::sqrt(norm2);
    if (!config.normalize_input && std::fabs(norm - 1.0) > 1e-5) {
        throw std::invalid_argument(
            "TANGO input is not unit normalized while normalize_input=false");
    }

    const double scale = config.normalize_input ? 1.0 / norm : 1.0;
    const size_t semantic_bytes = record_size - sizeof(double);
    if (!config.normalize_input) {
        std::memcpy(record, semantic, semantic_bytes);
    } else {
        for (size_t i = 0; i < config.dim; ++i) {
            const float normalized =
                static_cast<float>(static_cast<double>(semantic[i]) * scale);
            std::memcpy(record + i * sizeof(float), &normalized, sizeof(float));
        }
    }

    writeTangoTimestamp(record, semantic_bytes, timestamp);
}

inline std::vector<uint8_t> makeTangoRecordValidated(
        const float *semantic,
        double timestamp,
        const TangoConfig &config) {
    if (config.dim == 0 ||
            config.dim > (std::numeric_limits<size_t>::max() - sizeof(double)) /
                sizeof(float)) {
        throw std::invalid_argument("TANGO record dimension is invalid");
    }
    std::vector<uint8_t> record(tangoRecordSize(config.dim));
    fillTangoRecordValidated(
        semantic, timestamp, config, record.data(), record.size());
    return record;
}

}  // namespace tango_detail

inline std::vector<uint8_t> makeTangoRecord(
        const float *semantic,
        double timestamp,
        const TangoConfig &config) {
    config.validate();
    return tango_detail::makeTangoRecordValidated(semantic, timestamp, config);
}

inline std::vector<uint8_t> makeDataRecord(
        const float *semantic,
        double timestamp,
        const TangoConfig &config) {
    return makeTangoRecord(semantic, timestamp, config);
}

inline std::vector<uint8_t> makeQueryRecord(
        const float *semantic,
        double query_time,
        const TangoConfig &config) {
    return makeTangoRecord(semantic, query_time, config);
}

class TangoDistanceParams {
 public:
    explicit TangoDistanceParams(const TangoConfig &input_config)
        : config(tango_detail::validatedConfig(input_config)),
          semantic_space(config.dim),
          semantic_distance(semantic_space.get_dist_func()),
          semantic_distance_param(semantic_space.get_dist_func_param()),
          timestamp_offset(tangoTimestampOffset(config.dim)),
          record_size(tangoRecordSize(config.dim)) {}

    TangoDistanceParams(const TangoDistanceParams &) = delete;
    TangoDistanceParams &operator=(const TangoDistanceParams &) = delete;

    TangoConfig config;
    InnerProductSpace semantic_space;
    DISTFUNC<float> semantic_distance;
    void *semantic_distance_param;
    size_t timestamp_offset;
    size_t record_size;
    tango_detail::TangoDecayRuntime decay_runtime;
};

struct TangoQueryContext {
    const TangoDistanceParams *owner = nullptr;
    double query_time = std::numeric_limits<double>::quiet_NaN();
    double query_scale = std::numeric_limits<double>::quiet_NaN();
    bool valid = false;
};

inline TangoQueryContext makeTangoQueryContext(
        double query_time,
        TangoDistanceParams &params) {
    TangoQueryContext context;
    context.owner = &params;
    context.query_time = query_time;
    if (!tangoFastPathUsesQD(params.decay_runtime.mode())) return context;
    params.decay_runtime.noteQueryScale();
    context.query_scale = std::exp(
        -params.config.lambda *
        (query_time - params.decay_runtime.referenceTime()));
    context.valid = std::isfinite(context.query_scale) &&
                    context.query_scale > 0.0;
    return context;
}

inline double tangoSemanticDot(
        const void *left,
        const void *right,
        const TangoDistanceParams &params) {
    const double dot = 1.0 - static_cast<double>(params.semantic_distance(
        left, right, params.semantic_distance_param));
    return std::max(-1.0, std::min(1.0, dot));
}

inline float tangoDataDataDistance(
        const void *left,
        const void *right,
        int level,
        const void *opaque_params) {
    if (opaque_params == nullptr) {
        throw std::invalid_argument("TANGO DD parameters are null");
    }
    const TangoDistanceParams &params =
        *static_cast<const TangoDistanceParams *>(opaque_params);
    if (params.config.data_data_mode == TangoDataDataMode::Semantic) {
        return static_cast<float>(1.0 - tangoSemanticDot(left, right, params));
    }
    const double ti = readTangoTimestamp(left, params.timestamp_offset);
    const double tj = readTangoTimestamp(right, params.timestamp_offset);
    if (!std::isfinite(ti) || !std::isfinite(tj)) {
        throw std::invalid_argument("TANGO DD timestamp is non-finite");
    }

    const double semantic = tangoSemanticDot(left, right, params);
    const double exponent = params.config.lambda * std::fabs(ti - tj);
    const double temporal = std::exp(-exponent);
    const double one_minus_temporal = -std::expm1(-exponent);
    const double kappa = params.config.kappaForLevel(level);
    const double one_minus_semantic = 1.0 - semantic;

    double distance = 0.0;
    if (params.config.mode == TangoMode::Multiplicative) {
        distance = 2.0 * one_minus_temporal +
                   2.0 * (temporal + kappa) * one_minus_semantic;
    } else {
        distance = 2.0 * (params.config.alpha + kappa) * one_minus_semantic +
                   2.0 * (1.0 - params.config.alpha) * one_minus_temporal;
    }
    if (distance < 0.0 && distance > -1e-8) distance = 0.0;
    if (!std::isfinite(distance) || distance < 0.0) {
        throw std::runtime_error("TANGO DD distance is negative or non-finite");
    }
    return static_cast<float>(distance);
}

inline float tangoQueryDataDistance(
        const void *query,
        const void *data,
        const void *opaque_params) {
    if (opaque_params == nullptr) {
        throw std::invalid_argument("TANGO QD parameters are null");
    }
    const TangoDistanceParams &params =
        *static_cast<const TangoDistanceParams *>(opaque_params);
    const double query_time = readTangoTimestamp(query, params.timestamp_offset);
    const double data_time = readTangoTimestamp(data, params.timestamp_offset);
    if (!std::isfinite(query_time) || !std::isfinite(data_time)) {
        throw std::invalid_argument("TANGO QD timestamp is non-finite");
    }

    double age = query_time - data_time;
    if (age < -params.config.future_time_epsilon) {
        throw std::invalid_argument(
            "TANGO query sees an object whose timestamp is in the future");
    }
    if (age < 0.0) age = 0.0;
    const double freshness = std::exp(-params.config.lambda * age);
    const double semantic = tangoSemanticDot(query, data, params);
    const double score = params.config.mode == TangoMode::Multiplicative
        ? semantic * freshness
        : params.config.alpha * semantic +
          (1.0 - params.config.alpha) * freshness;

    double distance = 1.0 - score;
    if (distance < 0.0 && distance > -1e-7) distance = 0.0;
    if (!std::isfinite(distance) || distance < 0.0) {
        throw std::runtime_error("TANGO QD distance is negative or non-finite");
    }
    return static_cast<float>(distance);
}

inline float tangoQueryDataDistanceById(
        const void *query,
        const void *data,
        tableint data_id,
        const void *opaque_query_context,
        const void *opaque_params) {
    if (opaque_params == nullptr) {
        throw std::invalid_argument("TANGO QD parameters are null");
    }
    TangoDistanceParams &params = *const_cast<TangoDistanceParams *>(
        static_cast<const TangoDistanceParams *>(opaque_params));
    tango_detail::TangoDecayRuntime &runtime = params.decay_runtime;
    runtime.noteQDCallback();
    if (!tangoFastPathUsesQD(runtime.mode())) {
        runtime.noteDirectQD();
        return tangoQueryDataDistance(query, data, opaque_params);
    }

    const double query_time = readTangoTimestamp(query, params.timestamp_offset);
    const double data_time = readTangoTimestamp(data, params.timestamp_offset);
    if (!std::isfinite(query_time) || !std::isfinite(data_time)) {
        throw std::invalid_argument("TANGO QD timestamp is non-finite");
    }
    double age = query_time - data_time;
    if (age < -params.config.future_time_epsilon) {
        throw std::invalid_argument(
            "TANGO query sees an object whose timestamp is in the future");
    }

    bool fast_valid = false;
    double freshness = 1.0;
    if (age < 0.0) {
        // Preserve the direct path's epsilon-window semantics exactly.
        age = 0.0;
        fast_valid = true;
    } else {
        const TangoQueryContext *context =
            static_cast<const TangoQueryContext *>(opaque_query_context);
        double basis = 0.0;
        if (context != nullptr && context->owner == &params && context->valid &&
                context->query_time == query_time &&
                runtime.loadBasis(data_id, basis)) {
            freshness = context->query_scale * basis;
            const double tolerance = 32.0 * std::numeric_limits<double>::epsilon();
            if (std::isfinite(freshness) && freshness >= 0.0 &&
                    freshness <= 1.0 + tolerance) {
                if (freshness > 1.0) freshness = 1.0;
                fast_valid = true;
            }
        }
    }

    if (!fast_valid) {
        runtime.noteFallback();
        runtime.noteDirectQD(true);
        return tangoQueryDataDistance(query, data, opaque_params);
    }
    const double semantic = tangoSemanticDot(query, data, params);
    const double score = params.config.mode == TangoMode::Multiplicative
        ? semantic * freshness
        : params.config.alpha * semantic +
          (1.0 - params.config.alpha) * freshness;
    double distance = 1.0 - score;
    if (distance < 0.0 && distance > -1e-7) distance = 0.0;
    if (!std::isfinite(distance) || distance < 0.0) {
        throw std::runtime_error(
            "TANGO fast QD distance is negative or non-finite");
    }
    const float result = static_cast<float>(distance);
    if (runtime.verificationEnabled() && runtime.takeQDVerificationSample()) {
        // Verification performs a direct evaluation and records its exponential
        // in the direct-path counter.
        runtime.noteDirectQD(true);
        runtime.recordQDVerification(
            result, tangoQueryDataDistance(query, data, opaque_params));
    }
    runtime.noteFastQD();
    return result;
}

inline float tangoDataDataDistanceById(
        const void *left,
        tableint left_id,
        const void *right,
        tableint right_id,
        int level,
        const void *opaque_params) {
    if (opaque_params == nullptr) {
        throw std::invalid_argument("TANGO DD parameters are null");
    }
    TangoDistanceParams &params = *const_cast<TangoDistanceParams *>(
        static_cast<const TangoDistanceParams *>(opaque_params));
    tango_detail::TangoDecayRuntime &runtime = params.decay_runtime;
    runtime.noteDDCallback();
    if (params.config.data_data_mode == TangoDataDataMode::Semantic) {
        return static_cast<float>(1.0 - tangoSemanticDot(left, right, params));
    }
    if (!tangoFastPathUsesDD(runtime.mode())) {
        runtime.noteDirectDD();
        return tangoDataDataDistance(left, right, level, opaque_params);
    }

    const double ti = readTangoTimestamp(left, params.timestamp_offset);
    const double tj = readTangoTimestamp(right, params.timestamp_offset);
    if (!std::isfinite(ti) || !std::isfinite(tj)) {
        throw std::invalid_argument("TANGO DD timestamp is non-finite");
    }

    double zi = 0.0;
    double zj = 0.0;
    double inverse_i = 0.0;
    double inverse_j = 0.0;
    bool fast_valid = runtime.loadBasisAndInverse(left_id, zi, inverse_i) &&
                      runtime.loadBasisAndInverse(right_id, zj, inverse_j);
    double temporal = 1.0;
    double one_minus_temporal = 0.0;
    if (fast_valid && ti != tj) {
        // Distinct timestamps can round to an identical cached basis at very
        // small deltas.  Returning k=1 would then differ from direct exp, so
        // use the preserved direct path instead of silently approximating.
        if (zi == zj) {
            fast_valid = false;
        } else {
            if (zi < zj) {
                temporal = zi * inverse_j;
                one_minus_temporal = std::fma(-zi, inverse_j, 1.0);
            } else {
                temporal = zj * inverse_i;
                one_minus_temporal = std::fma(-zj, inverse_i, 1.0);
            }
            const double tolerance = 64.0 * std::numeric_limits<double>::epsilon();
            fast_valid = std::isfinite(temporal) &&
                         std::isfinite(one_minus_temporal) &&
                         temporal >= 0.0 && temporal <= 1.0 + tolerance &&
                         one_minus_temporal >= -tolerance &&
                         one_minus_temporal <= 1.0 + tolerance;
            if (fast_valid) {
                if (temporal > 1.0) temporal = 1.0;
                if (one_minus_temporal < 0.0) one_minus_temporal = 0.0;
                if (one_minus_temporal > 1.0) one_minus_temporal = 1.0;
            }
        }
    }
    if (!fast_valid) {
        runtime.noteFallback();
        runtime.noteDirectDD(true);
        return tangoDataDataDistance(left, right, level, opaque_params);
    }
    const double semantic = tangoSemanticDot(left, right, params);
    const double kappa = params.config.kappaForLevel(level);
    const double one_minus_semantic = 1.0 - semantic;
    double distance = 0.0;
    if (params.config.mode == TangoMode::Multiplicative) {
        distance = 2.0 * one_minus_temporal +
                   2.0 * (temporal + kappa) * one_minus_semantic;
    } else {
        distance = 2.0 * (params.config.alpha + kappa) * one_minus_semantic +
                   2.0 * (1.0 - params.config.alpha) * one_minus_temporal;
    }
    if (distance < 0.0 && distance > -1e-8) distance = 0.0;
    if (!std::isfinite(distance) || distance < 0.0) {
        throw std::runtime_error(
            "TANGO fast DD distance is negative or non-finite");
    }
    const float result = static_cast<float>(distance);
    if (runtime.verificationEnabled() && runtime.takeDDVerificationSample()) {
        runtime.noteDirectDD(true);
        runtime.recordDDVerification(
            result,
            tangoDataDataDistance(left, right, level, opaque_params));
    }
    runtime.noteFastDD();
    return result;
}

inline void tangoElementPreparedById(
        tableint internal_id,
        const void *stored_data,
        const void *opaque_params) {
    if (opaque_params == nullptr || stored_data == nullptr) return;
    TangoDistanceParams &params = *const_cast<TangoDistanceParams *>(
        static_cast<const TangoDistanceParams *>(opaque_params));
    double timestamp = 0.0;
    std::memcpy(
        &timestamp,
        static_cast<const char *>(stored_data) + params.timestamp_offset,
        sizeof(timestamp));
    params.decay_runtime.initializeEntry(internal_id, timestamp);
}

class TangoSpace : public SpaceInterface<float> {
 public:
    explicit TangoSpace(const TangoConfig &config) : params_(config) {}

    size_t get_data_size() { return params_.record_size; }

    DISTFUNC<float> get_dist_func() { return forbiddenFallbackDistance; }

    void *get_dist_func_param() { return &params_; }

    TangoDistanceParams &params() { return params_; }
    const TangoDistanceParams &params() const { return params_; }

 private:
    static float forbiddenFallbackDistance(const void *, const void *, const void *) {
        throw std::logic_error(
            "TANGO reached the single-distance fallback; a QD/DD call site was not adapted");
    }

    TangoDistanceParams params_;
};

namespace tango_detail {

inline std::string requiredMeta(
        const std::map<std::string, std::string> &values,
        const std::string &key) {
    std::map<std::string, std::string>::const_iterator it = values.find(key);
    if (it == values.end() || it->second.empty()) {
        throw std::runtime_error("TANGO sidecar is missing key: " + key);
    }
    return it->second;
}

inline size_t parseSize(const std::string &text, const std::string &key) {
    if (text.empty() || text[0] == '-') {
        throw std::runtime_error("Invalid TANGO sidecar integer: " + key);
    }
    std::istringstream input(text);
    unsigned long long value = 0;
    char trailing = 0;
    if (!(input >> value) || (input >> trailing) ||
            value > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Invalid TANGO sidecar integer: " + key);
    }
    return static_cast<size_t>(value);
}

inline double parseDouble(const std::string &text, const std::string &key) {
    std::istringstream input(text);
    double value = 0.0;
    char trailing = 0;
    if (!(input >> value) || (input >> trailing) || !std::isfinite(value)) {
        throw std::runtime_error("Invalid TANGO sidecar number: " + key);
    }
    return value;
}

inline std::vector<double> parseDoubleList(
        const std::string &text,
        const std::string &key) {
    if (text.empty() || text[text.size() - 1] == ',') {
        throw std::runtime_error("Invalid TANGO sidecar list: " + key);
    }
    std::vector<double> values;
    std::istringstream input(text);
    std::string token;
    while (std::getline(input, token, ',')) {
        if (token.empty()) {
            throw std::runtime_error("Invalid TANGO sidecar list: " + key);
        }
        values.push_back(parseDouble(token, key));
    }
    if (values.empty()) {
        throw std::runtime_error("Invalid TANGO sidecar list: " + key);
    }
    return values;
}

inline std::string serializeDoubleList(const std::vector<double> &values) {
    std::ostringstream output;
    output << std::setprecision(17);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) output << ',';
        output << values[i];
    }
    return output.str();
}

inline bool parseBool(const std::string &text, const std::string &key) {
    if (text == "1" || text == "true") return true;
    if (text == "0" || text == "false") return false;
    throw std::runtime_error("Invalid TANGO sidecar boolean: " + key);
}

inline std::map<std::string, std::string> readMeta(const std::string &path) {
    std::ifstream input(path.c_str());
    if (!input) throw std::runtime_error("Cannot open TANGO sidecar: " + path);
    std::map<std::string, std::string> values;
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') continue;
        const size_t equals = line.find('=');
        if (equals == std::string::npos || equals == 0) {
            std::ostringstream error;
            error << "Malformed TANGO sidecar line " << line_number;
            throw std::runtime_error(error.str());
        }
        const std::string key = line.substr(0, equals);
        const std::string value = line.substr(equals + 1);
        if (!values.insert(std::make_pair(key, value)).second) {
            throw std::runtime_error("Duplicate TANGO sidecar key: " + key);
        }
    }
    if (!input.eof()) throw std::runtime_error("Failed reading TANGO sidecar");
    return values;
}

}  // namespace tango_detail

class TANGOIndex {
 public:
    TANGOIndex(
            const TangoConfig &config,
            size_t max_elements,
            size_t M,
            size_t ef_construction,
            size_t random_seed = 100,
            const TangoDecayFastPathOptions &fastpath =
                TangoDecayFastPathOptions())
        : config_(config),
          M_(M),
          ef_construction_(std::max(ef_construction, M)),
          random_seed_(random_seed),
          space_(nullptr),
          index_(nullptr),
          max_timestamp_(-std::numeric_limits<double>::infinity()),
          committed_count_(0) {
        config_.validate();
        if (max_elements == 0) {
            throw std::invalid_argument("TANGO max_elements must be positive");
        }
        if (M < 2) throw std::invalid_argument("TANGO M must be at least 2");
        space_.reset(new TangoSpace(config_));
        index_.reset(new HierarchicalNSW<float>(
            space_.get(), max_elements, M_, ef_construction_, random_seed_, false));
        installDualDistances();
        configureFastPath(fastpath);
    }

    TANGOIndex(const TANGOIndex &) = delete;
    TANGOIndex &operator=(const TANGOIndex &) = delete;

    void addPoint(const float *semantic, double timestamp, labeltype label) {
        std::vector<uint8_t> record =
            tango_detail::makeTangoRecordValidated(semantic, timestamp, config_);
        reserveNewLabel(label);
        try {
            index_->addPoint(record.data(), label, false);
        } catch (...) {
            finishPendingLabel(label, false);
            throw;
        }
        updateMaximumTimestamp(timestamp);
        finishPendingLabel(label, true);
    }

    std::priority_queue<std::pair<float, labeltype> > searchKnn(
            const float *query,
            double query_time,
            size_t k) const {
        if (k == 0) throw std::invalid_argument("TANGO k must be positive");
        if (!std::isfinite(query_time)) {
            throw std::invalid_argument("TANGO query time must be finite");
        }
        const double maximum = max_timestamp_.load();
        if (size() != 0 && query_time < maximum - config_.future_time_epsilon) {
            throw std::invalid_argument(
                "TANGO query time is earlier than the maximum indexed timestamp");
        }
        static thread_local std::vector<uint8_t> query_record;
        if (query_record.size() != space_->params().record_size) {
            query_record.resize(space_->params().record_size);
        }
        tango_detail::fillTangoRecordValidated(
            query, query_time, config_, query_record.data(), query_record.size());
        TangoQueryContext query_context = makeQueryContext(query_time);
        return index_->searchKnnWithContext(
            query_record.data(), k, &query_context);
    }

    void setEf(size_t ef) {
        if (ef == 0) throw std::invalid_argument("TANGO ef must be positive");
        index_->setEf(ef);
    }

    void setMaxSearchExpansions(size_t max_expansions) {
        index_->setMaxSearchExpansions(max_expansions);
    }

    size_t size() const { return index_->getCurrentElementCount(); }
    size_t capacity() const { return index_->getMaxElements(); }
    double maxTimestamp() const { return max_timestamp_.load(); }
    size_t randomSeed() const { return random_seed_; }
    const TangoConfig &config() const { return config_; }
    const TangoDistanceParams &distanceParams() const { return space_->params(); }
    TangoDecayFastPath decayFastPathMode() const {
        return space_->params().decay_runtime.mode();
    }
    double decayReferenceTime() const {
        return space_->params().decay_runtime.referenceTime();
    }
    bool decayCacheStoresInverse() const {
        return space_->params().decay_runtime.cachesInverse();
    }
    TangoDecayFastPathStats decayFastPathStats() const {
        return space_->params().decay_runtime.stats();
    }
    void resetDecayFastPathCounters() {
        space_->params().decay_runtime.resetCounters();
    }
    TangoQueryContext makeQueryContext(double query_time) const {
        if (!std::isfinite(query_time)) {
            throw std::invalid_argument("TANGO query time must be finite");
        }
        return makeTangoQueryContext(
            query_time,
            const_cast<TangoDistanceParams &>(space_->params()));
    }

    HierarchicalNSW<float> &rawIndex() { return *index_; }
    const HierarchicalNSW<float> &rawIndex() const { return *index_; }

    std::vector<uint8_t> copyRecord(labeltype label) const {
        std::unique_lock<std::mutex> lock(index_->label_lookup_lock);
        std::unordered_map<labeltype, tableint>::const_iterator found =
            index_->label_lookup_.find(label);
        if (found == index_->label_lookup_.end()) {
            throw std::invalid_argument("TANGO label not found");
        }
        std::vector<uint8_t> record(space_->params().record_size);
        std::memcpy(record.data(), index_->getDataByInternalId(found->second), record.size());
        return record;
    }

    // Alias using the public wrapper's record terminology.
    std::vector<uint8_t> getDataRecord(labeltype label) const {
        return copyRecord(label);
    }

    void save(const std::string &prefix) const {
        if (prefix.empty()) throw std::invalid_argument("TANGO save prefix is empty");
        if (size() == 0) throw std::runtime_error("TANGO cannot save an empty index");
        index_->saveIndex(prefix + ".hnsw");

        std::ofstream output((prefix + ".tango.meta").c_str(), std::ios::trunc);
        if (!output) throw std::runtime_error("Cannot create TANGO sidecar");
        const size_t sidecar_version = 3;
        output << std::setprecision(17)
               << "magic=TANGO\n"
               << "version=" << sidecar_version << "\n"
               << "mode=" << tangoModeName(config_.mode) << "\n"
               << "data_data_mode=" << tangoDataDataModeName(config_.data_data_mode) << "\n"
               << "dim=" << config_.dim << "\n"
               << "half_life=" << config_.half_life << "\n"
               << "lambda=" << config_.lambda << "\n"
               << "alpha=" << config_.alpha << "\n"
               << "kappa_base=" << config_.kappa_base << "\n"
               << "kappa_nav=" << config_.kappa_nav << "\n"
               << "schedule=" << tangoScheduleName(config_.schedule) << "\n";
        if (config_.schedule == KappaSchedule::PerLevel) {
            output << "kappa_levels="
                   << tango_detail::serializeDoubleList(config_.kappa_levels)
                   << "\n";
        }
        output << "normalize_input=" << (config_.normalize_input ? 1 : 0) << "\n"
               << "future_time_epsilon=" << config_.future_time_epsilon << "\n"
               << "record_timestamp_type=float64\n"
               << "timestamp_unit=day\n"
               << "record_size=" << space_->params().record_size << "\n"
               << "max_timestamp=" << max_timestamp_.load() << "\n"
               << "max_elements=" << index_->getMaxElements() << "\n"
               << "element_count=" << size() << "\n"
               << "M=" << M_ << "\n"
               << "ef_construction=" << ef_construction_ << "\n"
               << "random_seed=" << random_seed_ << "\n";
        output.close();
        if (!output) throw std::runtime_error("Failed writing TANGO sidecar");
    }

    static std::unique_ptr<TANGOIndex> load(
            const std::string &prefix,
            size_t max_elements = 0,
            const TangoDecayFastPathOptions &fastpath =
                TangoDecayFastPathOptions()) {
        if (prefix.empty()) throw std::invalid_argument("TANGO load prefix is empty");
        const std::map<std::string, std::string> meta =
            tango_detail::readMeta(prefix + ".tango.meta");
        if (tango_detail::requiredMeta(meta, "magic") != "TANGO") {
            throw std::runtime_error("Invalid TANGO sidecar magic");
        }
        const size_t sidecar_version = tango_detail::parseSize(
            tango_detail::requiredMeta(meta, "version"), "version");
        if (sidecar_version != 1 && sidecar_version != 2 && sidecar_version != 3) {
            throw std::runtime_error("Unsupported TANGO sidecar version");
        }

        TangoConfig config;
        config.dim = tango_detail::parseSize(
            tango_detail::requiredMeta(meta, "dim"), "dim");
        const std::string mode = tango_detail::requiredMeta(meta, "mode");
        if (mode == "multiplicative") config.mode = TangoMode::Multiplicative;
        else if (mode == "additive") config.mode = TangoMode::Additive;
        else throw std::runtime_error("Invalid TANGO sidecar mode");
        if (sidecar_version >= 3) {
            const std::string data_data_mode =
                tango_detail::requiredMeta(meta, "data_data_mode");
            if (data_data_mode == "timelift") {
                config.data_data_mode = TangoDataDataMode::TimeLift;
            } else if (data_data_mode == "semantic") {
                config.data_data_mode = TangoDataDataMode::Semantic;
            } else {
                throw std::runtime_error("Invalid TANGO sidecar data-data mode");
            }
        }
        config.half_life = tango_detail::parseDouble(
            tango_detail::requiredMeta(meta, "half_life"), "half_life");
        config.lambda = tango_detail::parseDouble(
            tango_detail::requiredMeta(meta, "lambda"), "lambda");
        config.alpha = tango_detail::parseDouble(
            tango_detail::requiredMeta(meta, "alpha"), "alpha");
        config.kappa_base = tango_detail::parseDouble(
            tango_detail::requiredMeta(meta, "kappa_base"), "kappa_base");
        config.kappa_nav = tango_detail::parseDouble(
            tango_detail::requiredMeta(meta, "kappa_nav"), "kappa_nav");
        const std::string schedule = tango_detail::requiredMeta(meta, "schedule");
        if (schedule == "fixed") config.schedule = KappaSchedule::Fixed;
        else if (schedule == "two_tier") config.schedule = KappaSchedule::TwoTier;
        else if (schedule == "per_level") {
            if (sidecar_version < 2) {
                throw std::runtime_error(
                    "TANGO per-level schedule requires sidecar version 2");
            }
            config.schedule = KappaSchedule::PerLevel;
            config.kappa_levels = tango_detail::parseDoubleList(
                tango_detail::requiredMeta(meta, "kappa_levels"),
                "kappa_levels");
        }
        else throw std::runtime_error("Invalid TANGO sidecar schedule");
        config.normalize_input = tango_detail::parseBool(
            tango_detail::requiredMeta(meta, "normalize_input"), "normalize_input");
        config.future_time_epsilon = tango_detail::parseDouble(
            tango_detail::requiredMeta(meta, "future_time_epsilon"),
            "future_time_epsilon");
        config.validate();

        if (tango_detail::requiredMeta(meta, "record_timestamp_type") != "float64") {
            throw std::runtime_error("Unsupported TANGO timestamp record type");
        }
        if (tango_detail::requiredMeta(meta, "timestamp_unit") != "day") {
            throw std::runtime_error("Unsupported TANGO timestamp unit");
        }
        const size_t stored_record_size = tango_detail::parseSize(
            tango_detail::requiredMeta(meta, "record_size"), "record_size");
        if (stored_record_size != tangoRecordSize(config.dim)) {
            throw std::runtime_error("TANGO sidecar record size does not match dim");
        }
        const size_t M = tango_detail::parseSize(
            tango_detail::requiredMeta(meta, "M"), "M");
        const size_t ef_construction = tango_detail::parseSize(
            tango_detail::requiredMeta(meta, "ef_construction"), "ef_construction");
        const size_t random_seed = tango_detail::parseSize(
            tango_detail::requiredMeta(meta, "random_seed"), "random_seed");
        const double stored_max_timestamp = tango_detail::parseDouble(
            tango_detail::requiredMeta(meta, "max_timestamp"), "max_timestamp");
        const size_t stored_max_elements = tango_detail::parseSize(
            tango_detail::requiredMeta(meta, "max_elements"), "max_elements");
        const size_t stored_count = tango_detail::parseSize(
            tango_detail::requiredMeta(meta, "element_count"), "element_count");
        if (M < 2 || ef_construction == 0 || stored_count == 0 ||
                stored_max_elements < stored_count) {
            throw std::runtime_error("Invalid TANGO sidecar index parameters");
        }
        if (max_elements != 0 && max_elements < stored_count) {
            throw std::invalid_argument(
                "TANGO load capacity override is smaller than the stored element count");
        }
        const size_t effective_max_elements =
            max_elements == 0 ? stored_max_elements : max_elements;

        std::unique_ptr<TANGOIndex> result(new TANGOIndex(
            config, prefix + ".hnsw", effective_max_elements, M, ef_construction,
            random_seed, stored_max_timestamp, fastpath));
        if (stored_count != result->size()) {
            throw std::runtime_error("TANGO sidecar element count mismatch");
        }
        if (result->capacity() != effective_max_elements) {
            throw std::runtime_error("TANGO loaded capacity mismatch");
        }
        return result;
    }

 private:
    TANGOIndex(
            const TangoConfig &config,
            const std::string &index_path,
            size_t max_elements,
            size_t M,
            size_t ef_construction,
            size_t random_seed,
            double stored_max_timestamp,
            const TangoDecayFastPathOptions &fastpath)
        : config_(config),
          M_(M),
          ef_construction_(ef_construction),
          random_seed_(random_seed),
          space_(nullptr),
          index_(nullptr),
          max_timestamp_(stored_max_timestamp),
          committed_count_(0) {
        config_.validate();
        space_.reset(new TangoSpace(config_));
        index_.reset(new HierarchicalNSW<float>(
            space_.get(), index_path, false, max_elements, false));
        index_->restoreRandomGeneratorState(random_seed_, index_->getCurrentElementCount());
        installDualDistances();
        validateLoadedIndex(stored_max_timestamp);
        configureFastPath(fastpath);
        committed_count_ = index_->getCurrentElementCount();
    }

    void installDualDistances() {
        index_->setDualDistanceFunctions(
            tangoQueryDataDistance,
            tangoDataDataDistance,
            &space_->params());
    }

    void configureFastPath(const TangoDecayFastPathOptions &options) {
        // Validate enum values even when no cache/callback is requested.
        (void)tangoDecayFastPathName(options.mode);
        if (options.verify_samples != 0 && options.mode == TangoDecayFastPath::Off) {
            throw std::invalid_argument(
                "TANGO fastpath verification requires qd, dd, or all mode");
        }
        if (options.mode == TangoDecayFastPath::Off &&
                !options.instrumentation && options.verify_samples == 0) {
            return;
        }
        double reference_time = options.reference_time;
        if (!std::isfinite(reference_time)) {
            reference_time = size() == 0 ? 0.0 : max_timestamp_.load();
        }
        const std::chrono::steady_clock::time_point start =
            std::chrono::steady_clock::now();
        space_->params().decay_runtime.configure(
            options, reference_time, config_.lambda, capacity());
        if (options.mode != TangoDecayFastPath::Off) {
            for (size_t i = 0; i < size(); ++i) {
                tangoElementPreparedById(
                    static_cast<tableint>(i),
                    index_->getDataByInternalId(static_cast<tableint>(i)),
                    &space_->params());
            }
        }
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        space_->params().decay_runtime.setCacheInitMs(elapsed_ms);
        const bool instrument_all = options.instrumentation;
        index_->setIdAwareDistanceFunctions(
            (tangoFastPathUsesQD(options.mode) || instrument_all)
                ? tangoQueryDataDistanceById : nullptr,
            (tangoFastPathUsesDD(options.mode) || instrument_all)
                ? tangoDataDataDistanceById : nullptr,
            options.mode != TangoDecayFastPath::Off
                ? tangoElementPreparedById : nullptr,
            &space_->params());
    }

    void reserveNewLabel(labeltype label) {
        std::lock_guard<std::mutex> state_lock(add_state_mutex_);
        if (pending_labels_.count(label) != 0) {
            throw std::invalid_argument("TANGO duplicate label insertion");
        }
        {
            std::unique_lock<std::mutex> lookup_lock(index_->label_lookup_lock);
            if (index_->label_lookup_.count(label) != 0) {
                throw std::invalid_argument(
                    "TANGO only supports online insertion of new labels");
            }
        }
        if (committed_count_ + pending_labels_.size() >= index_->getMaxElements()) {
            throw std::runtime_error("TANGO index capacity exceeded");
        }
        pending_labels_.insert(label);
    }

    void finishPendingLabel(labeltype label, bool committed) {
        std::lock_guard<std::mutex> lock(add_state_mutex_);
        const size_t erased = pending_labels_.erase(label);
        if (erased != 1) {
            throw std::logic_error("TANGO insertion reservation was lost");
        }
        if (committed) ++committed_count_;
    }

    void updateMaximumTimestamp(double timestamp) {
        double observed = max_timestamp_.load();
        while (timestamp > observed &&
               !max_timestamp_.compare_exchange_weak(observed, timestamp)) {
        }
    }

    void validateLoadedIndex(double stored_max_timestamp) {
        if (index_->data_size_ != space_->params().record_size ||
                index_->label_offset_ < index_->offsetData_ ||
                index_->label_offset_ - index_->offsetData_ != space_->params().record_size) {
            throw std::runtime_error(
                "TANGO HNSW record layout does not match sidecar configuration");
        }
        if (index_->M_ != M_ || index_->ef_construction_ != ef_construction_) {
            throw std::runtime_error("TANGO HNSW parameters do not match sidecar");
        }
        if (index_->getCurrentElementCount() == 0) {
            throw std::runtime_error("TANGO loaded index is empty");
        }

        double observed_max = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < index_->getCurrentElementCount(); ++i) {
            const double timestamp = readTangoTimestamp(
                index_->getDataByInternalId(static_cast<tableint>(i)),
                space_->params().timestamp_offset);
            if (!std::isfinite(timestamp)) {
                throw std::runtime_error("TANGO loaded record has non-finite timestamp");
            }
            observed_max = std::max(observed_max, timestamp);
        }
        const double tolerance = config_.future_time_epsilon +
            1e-12 * std::max(1.0, std::fabs(observed_max));
        if (!std::isfinite(stored_max_timestamp) ||
                std::fabs(observed_max - stored_max_timestamp) > tolerance) {
            throw std::runtime_error("TANGO sidecar max timestamp mismatch");
        }
        max_timestamp_.store(observed_max);
    }

    TangoConfig config_;
    size_t M_;
    size_t ef_construction_;
    size_t random_seed_;
    std::unique_ptr<TangoSpace> space_;
    std::unique_ptr<HierarchicalNSW<float> > index_;
    std::atomic<double> max_timestamp_;
    mutable std::mutex add_state_mutex_;
    std::unordered_set<labeltype> pending_labels_;
    size_t committed_count_;
};

}  // namespace hnswlib
