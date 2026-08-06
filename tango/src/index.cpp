// TANGO index construction, persistence, diagnostics, and validation CLI.
//
// Ground-truth validation uses an independent scorer instead of reusing the
// index distance callback.
#include "tango.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr size_t kDefaultBuildThreads = 64;

using Clock = std::chrono::steady_clock;
using hnswlib::TangoConfig;
using hnswlib::TANGOIndex;
using hnswlib::TangoMode;
using hnswlib::TangoDataDataMode;
using hnswlib::TangoDecayFastPathOptions;
using hnswlib::TangoDecayFastPathStats;
using hnswlib::KappaSchedule;
using hnswlib::labeltype;

struct Options {
    std::string mode = "multiplicative";
    std::string data_data_mode = "timelift";
    std::string base_path;
    std::string timestamps_path;
    std::string query_path;
    std::string gt_path;
    std::string query_times_path;
    std::string index_mode;
    std::string index_prefix;
    std::string output_csv;
    std::string build_report_csv;
    std::string graph_stats_csv;
    std::string dataset = "unknown";
    std::string time_distribution = "unknown";
    std::string tango_decay_fastpath = "off";

    size_t n = 0;
    size_t queries = 0;
    size_t dim = 0;
    size_t gt_width = 100;
    size_t k = 10;
    size_t M = 16;
    size_t ef_construction = 200;
    size_t threads = kDefaultBuildThreads;
    size_t random_seed = 100;
    size_t validate_bruteforce = 0;
    size_t max_search_expansions = 0;
    size_t decay_verify_samples = 0;
    std::vector<size_t> ef_list;
    std::vector<double> kappa_levels;

    double query_time = std::numeric_limits<double>::quiet_NaN();
    double half_life = std::numeric_limits<double>::quiet_NaN();
    double alpha = 1.0;
    double kappa_base = 0.0;
    double kappa_nav = 0.0;
    bool normalize = true;
    bool decay_instrumentation = false;
    bool build_only = false;

    bool dim_supplied = false;
    bool mode_supplied = false;
    bool data_data_mode_supplied = false;
    bool half_life_supplied = false;
    bool alpha_supplied = false;
    bool kappa_base_supplied = false;
    bool kappa_nav_supplied = false;
    bool kappa_levels_supplied = false;
    bool M_supplied = false;
    bool ef_construction_supplied = false;
    bool random_seed_supplied = false;
    bool normalize_supplied = false;
};

struct BuildMetrics {
    double input_seconds = 0.0;
    double preprocess_seconds = 0.0;
    double core_build_seconds = 0.0;
    double serialize_seconds = 0.0;
    double serving_index_pipeline_seconds = 0.0;
    uint64_t index_hnsw_bytes = 0;
    uint64_t index_meta_bytes = 0;
    uint64_t serialized_index_bytes = 0;
    std::string completed_at_utc;
};

// A low-contention 0--100% progress display. Only threads that cross one of
// the 50 display thresholds contend on the output lock.
class ProgressDisplay {
public:
    explicit ProgressDisplay(size_t expected_count, std::ostream &out = std::cerr)
        : out_(out),
          expected_count_(std::max<size_t>(1, expected_count)),
          count_(0),
          displayed_tics_(0),
          next_tic_count_(nextTicCount(1)),
          finished_(false) {
        out_ << "0%   10   20   30   40   50   60   70   80   90   100%\n"
             << "|----|----|----|----|----|----|----|----|----|----|\n";
    }

    void advance() {
        const size_t count = count_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (count < next_tic_count_.load(std::memory_order_relaxed) &&
                count < expected_count_) {
            return;
        }

        std::lock_guard<std::mutex> lock(output_mutex_);
        const size_t bounded_count = std::min(count, expected_count_);
        unsigned target = static_cast<unsigned>(
            static_cast<long double>(bounded_count) * kTotalTics / expected_count_);
        if (count >= expected_count_) target = kTotalTics;

        unsigned displayed = displayed_tics_.load(std::memory_order_relaxed);
        while (displayed < target) {
            out_ << '*';
            ++displayed;
        }
        displayed_tics_.store(displayed, std::memory_order_relaxed);

        if (displayed == kTotalTics) {
            if (!finished_.exchange(true, std::memory_order_relaxed)) out_ << std::endl;
            next_tic_count_.store(
                std::numeric_limits<size_t>::max(), std::memory_order_relaxed);
        } else {
            next_tic_count_.store(nextTicCount(displayed + 1), std::memory_order_relaxed);
        }
    }

private:
    enum { kTotalTics = 50 };

    size_t nextTicCount(unsigned tic) const {
        const long double position =
            static_cast<long double>(tic) * expected_count_ / kTotalTics;
        return std::max<size_t>(1, static_cast<size_t>(std::ceil(position)));
    }

    std::ostream &out_;
    const size_t expected_count_;
    std::atomic<size_t> count_;
    std::atomic<unsigned> displayed_tics_;
    std::atomic<size_t> next_tic_count_;
    std::atomic<bool> finished_;
    std::mutex output_mutex_;
};

void printUsage(const char *argv0) {
    std::cerr
        << "Usage (build + query): " << argv0
        << " --index-mode {create|load}\n"
        << "  --index-prefix PREFIX --query FILE --gt FILE --queries N --gt-width W --k K\n"
        << "  (--query-time TAU | --query-times FILE) --ef-list E1,E2,... [options]\n\n"
        << "Usage (create + save only): " << argv0
        << " --index-mode create --build-only\n"
        << "  --index-prefix PREFIX [create options]\n\n"
        << "Create additionally requires:\n"
        << "  --mode {multiplicative|additive} --base FILE --timestamps FILE\n"
        << "  --n N --dim D --half-life H and either:\n"
        << "    --kappa-base X --kappa-nav Y\n"
        << "    --kappa-levels K0,K1,...\n\n"
        << "Options:\n"
        << "  --alpha A                    additive weight (default 1)\n"
        << "  --dd-mode {timelift|semantic} construction distance (default timelift)\n"
        << "  --kappa-levels K0,K1,...     nondecreasing per-level kappas; high levels clamp to last\n"
        << "  --M N                        HNSW degree (default 16)\n"
        << "  --ef-construction N          build beam (default 200)\n"
        << "  --threads N                  parallel insertion threads (default 64)\n"
        << "  --random-seed N              HNSW seed (default 100)\n"
        << "  --normalize {0|1}            normalize semantic vectors (default 1)\n"
        << "  --max-search-expansions N    0 means unlimited (default 0)\n"
        << "  --tango-decay-fastpath M   factorized decay path: off|qd|dd|all (default off)\n"
        << "  --tango-decay-fastpath-verify-samples N\n"
        << "                              compare first N fast QD/DD callback distances\n"
        << "  --tango-decay-fastpath-instrumentation {0|1}\n"
        << "                              enable callback counters (default 0)\n"
        << "  --validate-bruteforce N      independently validate first N queries\n"
        << "  --output-csv FILE            append recall/latency rows\n"
        << "  --build-only                 create and save without loading query/GT files\n"
        << "  --build-report-csv FILE      append one phase-timing/footprint row per build\n"
        << "  --graph-stats-csv FILE       write per-layer topology statistics\n"
        << "  --dataset NAME --time-distribution NAME\n";
}

std::string requireValue(int &i, int argc, char **argv) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value after ") + argv[i]);
    }
    return argv[++i];
}

size_t parseSize(const std::string &text, const std::string &name, bool allow_zero = false) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (text.empty() || text[0] == '-' || errno != 0 || end == text.c_str() || *end != '\0' ||
        (!allow_zero && value == 0) || value > std::numeric_limits<size_t>::max()) {
        throw std::invalid_argument("invalid " + name + ": " + text);
    }
    return static_cast<size_t>(value);
}

double parseDouble(const std::string &text, const std::string &name) {
    char *end = NULL;
    errno = 0;
    double value = std::strtod(text.c_str(), &end);
    if (errno != 0 || end == text.c_str() || *end != '\0' || !std::isfinite(value)) {
        throw std::invalid_argument("invalid " + name + ": " + text);
    }
    return value;
}

bool parseBool(const std::string &text, const std::string &name) {
    if (text == "1" || text == "true") return true;
    if (text == "0" || text == "false") return false;
    throw std::invalid_argument("invalid " + name + " (expected 0 or 1): " + text);
}

std::vector<size_t> parseEfList(const std::string &text) {
    std::vector<size_t> result;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) throw std::invalid_argument("empty entry in --ef-list");
        result.push_back(parseSize(token, "ef"));
    }
    if (result.empty()) throw std::invalid_argument("--ef-list must not be empty");
    return result;
}

std::vector<double> parseKappaLevels(const std::string &text) {
    if (text.empty() || text[text.size() - 1] == ',') {
        throw std::invalid_argument("invalid --kappa-levels list");
    }
    std::vector<double> result;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) throw std::invalid_argument("empty entry in --kappa-levels");
        result.push_back(parseDouble(token, "kappa-levels"));
    }
    if (result.empty()) throw std::invalid_argument("--kappa-levels must not be empty");
    return result;
}

Options parseOptions(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string flag(argv[i]);
        if (flag == "--help" || flag == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (flag == "--mode") {
            o.mode = requireValue(i, argc, argv);
            o.mode_supplied = true;
        } else if (flag == "--dd-mode") {
            o.data_data_mode = requireValue(i, argc, argv);
            o.data_data_mode_supplied = true;
        } else if (flag == "--base") {
            o.base_path = requireValue(i, argc, argv);
        } else if (flag == "--timestamps") {
            o.timestamps_path = requireValue(i, argc, argv);
        } else if (flag == "--query") {
            o.query_path = requireValue(i, argc, argv);
        } else if (flag == "--gt") {
            o.gt_path = requireValue(i, argc, argv);
        } else if (flag == "--query-times") {
            o.query_times_path = requireValue(i, argc, argv);
        } else if (flag == "--index-mode") {
            o.index_mode = requireValue(i, argc, argv);
        } else if (flag == "--index-prefix") {
            o.index_prefix = requireValue(i, argc, argv);
        } else if (flag == "--output-csv") {
            o.output_csv = requireValue(i, argc, argv);
        } else if (flag == "--build-report-csv") {
            o.build_report_csv = requireValue(i, argc, argv);
        } else if (flag == "--graph-stats-csv") {
            o.graph_stats_csv = requireValue(i, argc, argv);
        } else if (flag == "--build-only") {
            o.build_only = true;
        } else if (flag == "--dataset") {
            o.dataset = requireValue(i, argc, argv);
        } else if (flag == "--time-distribution") {
            o.time_distribution = requireValue(i, argc, argv);
        } else if (flag == "--n") {
            o.n = parseSize(requireValue(i, argc, argv), "n");
        } else if (flag == "--queries") {
            o.queries = parseSize(requireValue(i, argc, argv), "queries");
        } else if (flag == "--dim") {
            o.dim = parseSize(requireValue(i, argc, argv), "dim");
            o.dim_supplied = true;
        } else if (flag == "--gt-width") {
            o.gt_width = parseSize(requireValue(i, argc, argv), "gt-width");
        } else if (flag == "--k") {
            o.k = parseSize(requireValue(i, argc, argv), "k");
        } else if (flag == "--M") {
            o.M = parseSize(requireValue(i, argc, argv), "M");
            o.M_supplied = true;
        } else if (flag == "--ef-construction") {
            o.ef_construction = parseSize(requireValue(i, argc, argv), "ef-construction");
            o.ef_construction_supplied = true;
        } else if (flag == "--threads") {
            o.threads = parseSize(requireValue(i, argc, argv), "threads");
        } else if (flag == "--random-seed") {
            o.random_seed = parseSize(requireValue(i, argc, argv), "random-seed", true);
            o.random_seed_supplied = true;
        } else if (flag == "--validate-bruteforce") {
            o.validate_bruteforce = parseSize(requireValue(i, argc, argv), "validate-bruteforce", true);
        } else if (flag == "--max-search-expansions") {
            o.max_search_expansions = parseSize(requireValue(i, argc, argv), "max-search-expansions", true);
        } else if (flag == "--tango-decay-fastpath") {
            o.tango_decay_fastpath = requireValue(i, argc, argv);
            (void)hnswlib::parseTangoDecayFastPath(o.tango_decay_fastpath);
        } else if (flag == "--tango-decay-fastpath-verify-samples") {
            o.decay_verify_samples = parseSize(
                requireValue(i, argc, argv),
                "tango-decay-fastpath-verify-samples", true);
        } else if (flag == "--tango-decay-fastpath-instrumentation") {
            o.decay_instrumentation = parseBool(
                requireValue(i, argc, argv),
                "tango-decay-fastpath-instrumentation");
        } else if (flag == "--ef-list") {
            o.ef_list = parseEfList(requireValue(i, argc, argv));
        } else if (flag == "--query-time") {
            o.query_time = parseDouble(requireValue(i, argc, argv), "query-time");
        } else if (flag == "--half-life") {
            o.half_life = parseDouble(requireValue(i, argc, argv), "half-life");
            o.half_life_supplied = true;
        } else if (flag == "--alpha") {
            o.alpha = parseDouble(requireValue(i, argc, argv), "alpha");
            o.alpha_supplied = true;
        } else if (flag == "--kappa-base") {
            o.kappa_base = parseDouble(requireValue(i, argc, argv), "kappa-base");
            o.kappa_base_supplied = true;
        } else if (flag == "--kappa-nav") {
            o.kappa_nav = parseDouble(requireValue(i, argc, argv), "kappa-nav");
            o.kappa_nav_supplied = true;
        } else if (flag == "--kappa-levels") {
            o.kappa_levels = parseKappaLevels(requireValue(i, argc, argv));
            o.kappa_levels_supplied = true;
        } else if (flag == "--normalize") {
            o.normalize = parseBool(requireValue(i, argc, argv), "normalize");
            o.normalize_supplied = true;
        } else {
            throw std::invalid_argument("unknown argument: " + flag);
        }
    }

    if (o.index_mode != "create" && o.index_mode != "load") {
        throw std::invalid_argument("--index-mode must be create or load");
    }
    if (o.index_prefix.empty()) throw std::invalid_argument("--index-prefix is required");
    if (o.build_only) {
        if (o.index_mode != "create") {
            throw std::invalid_argument("--build-only requires --index-mode create");
        }
        if (o.validate_bruteforce != 0) {
            throw std::invalid_argument(
                "--validate-bruteforce requires query execution and cannot be used with --build-only");
        }
        if (!o.output_csv.empty()) {
            throw std::invalid_argument(
                "--output-csv contains query rows; use --build-report-csv with --build-only");
        }
    } else {
        if (o.query_path.empty() || o.gt_path.empty() || o.queries == 0 || o.ef_list.empty()) {
            throw std::invalid_argument(
                "--query, --gt, --queries, and --ef-list are required unless --build-only is set");
        }
        if (o.k > o.gt_width) throw std::invalid_argument("--k must be <= --gt-width");
        if ((!o.query_times_path.empty()) == (!std::isnan(o.query_time))) {
            throw std::invalid_argument("specify exactly one of --query-time and --query-times");
        }
    }
    if (!o.build_report_csv.empty() && o.index_mode != "create") {
        throw std::invalid_argument("--build-report-csv requires --index-mode create");
    }
    if (o.index_mode == "create") {
        if (o.base_path.empty() || o.timestamps_path.empty() || o.n == 0 || !o.dim_supplied ||
            !o.mode_supplied || !o.half_life_supplied) {
            throw std::invalid_argument(
                "create requires --base, --timestamps, --n, --dim, --mode, and --half-life");
        }
        if (o.kappa_levels_supplied) {
            if (o.kappa_base_supplied || o.kappa_nav_supplied) {
                throw std::invalid_argument(
                    "create accepts either --kappa-levels or --kappa-base/--kappa-nav, not both");
            }
        } else if (!o.kappa_base_supplied || !o.kappa_nav_supplied) {
            throw std::invalid_argument(
                "create requires either --kappa-levels or both --kappa-base and --kappa-nav");
        }
    }
    if (o.mode != "multiplicative" && o.mode != "additive") {
        throw std::invalid_argument("--mode must be multiplicative or additive");
    }
    if (o.data_data_mode != "timelift" && o.data_data_mode != "semantic") {
        throw std::invalid_argument("--dd-mode must be timelift or semantic");
    }
    return o;
}

uint64_t fileSize(const std::string &path) {
    std::ifstream in(path.c_str(), std::ios::binary | std::ios::ate);
    if (!in) return 0;
    const std::streamoff n = in.tellg();
    return n < 0 ? 0 : static_cast<uint64_t>(n);
}

std::string iso8601NowUtc() {
    const std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
    std::tm utc;
#if defined(_WIN32)
    gmtime_s(&utc, &seconds);
#else
    gmtime_r(&seconds, &utc);
#endif
    const long long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count() % 1000;
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.'
        << std::setfill('0') << std::setw(3) << milliseconds << 'Z';
    return out.str();
}

template <typename T>
std::vector<T> loadBinary(const std::string &path, size_t count, const std::string &what) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::overflow_error(what + " byte-size overflow");
    }
    const size_t required = count * sizeof(T);
    if (required > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::overflow_error(what + " exceeds the stream-size limit");
    }
    std::ifstream in(path.c_str(), std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open " + what + ": " + path);
    const std::streamoff bytes = in.tellg();
    if (bytes < 0 || static_cast<uint64_t>(bytes) != static_cast<uint64_t>(required)) {
        std::ostringstream msg;
        msg << what << " size mismatch: expected " << required
            << " bytes, found " << bytes;
        throw std::runtime_error(msg.str());
    }
    in.seekg(0);
    std::vector<T> result(count);
    if (required != 0) {
        in.read(reinterpret_cast<char *>(result.data()),
                static_cast<std::streamsize>(required));
    }
    if (!in) throw std::runtime_error("failed reading " + what + ": " + path);
    return result;
}

size_t checkedElementCount(size_t rows, size_t width, const std::string &what) {
    if (width != 0 && rows > std::numeric_limits<size_t>::max() / width) {
        throw std::overflow_error(what + " element-count overflow");
    }
    return rows * width;
}

void validateGroundTruthIds(
        const std::vector<int64_t> &ground_truth,
        size_t queries,
        size_t width,
        size_t index_elements) {
    for (size_t query = 0; query < queries; ++query) {
        std::unordered_set<int64_t> seen;
        seen.reserve(width);
        for (size_t rank = 0; rank < width; ++rank) {
            const int64_t id = ground_truth[query * width + rank];
            if (id < 0 || static_cast<uint64_t>(id) >= index_elements) {
                std::ostringstream message;
                message << "ground truth ID out of range at query " << query
                        << ", rank " << rank << ": " << id
                        << " (index elements " << index_elements << ')';
                throw std::invalid_argument(message.str());
            }
            if (!seen.insert(id).second) {
                std::ostringstream message;
                message << "duplicate ground truth ID at query " << query
                        << ", rank " << rank << ": " << id;
                throw std::invalid_argument(message.str());
            }
        }
    }
}

void normalizeVector(float *v, size_t dim) {
    double norm2 = 0.0;
    for (size_t j = 0; j < dim; ++j) norm2 += static_cast<double>(v[j]) * v[j];
    if (!(norm2 > 0.0) || !std::isfinite(norm2)) {
        throw std::invalid_argument("encountered zero or non-finite semantic vector");
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(norm2));
    for (size_t j = 0; j < dim; ++j) v[j] *= inv;
}

void normalizeBatch(std::vector<float> &values, size_t count, size_t dim) {
    for (size_t i = 0; i < count; ++i) normalizeVector(values.data() + i * dim, dim);
}

template <class Function>
void parallelFor(size_t begin, size_t end, size_t thread_count, Function fn) {
    if (begin >= end) return;
    thread_count = std::max<size_t>(1, std::min(thread_count, end - begin));
    if (thread_count == 1) {
        for (size_t i = begin; i < end; ++i) fn(i);
        return;
    }
    std::atomic<size_t> next(begin);
    std::atomic<bool> stopped(false);
    std::mutex exception_mutex;
    std::exception_ptr exception;
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (size_t t = 0; t < thread_count; ++t) {
        workers.emplace_back([&]() {
            while (!stopped.load()) {
                const size_t i = next.fetch_add(1);
                if (i >= end) break;
                try {
                    fn(i);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    if (!exception) exception = std::current_exception();
                    stopped.store(true);
                    break;
                }
            }
        });
    }
    for (size_t t = 0; t < workers.size(); ++t) workers[t].join();
    if (exception) std::rethrow_exception(exception);
}

double dot(const float *a, const float *b, size_t dim) {
    double value = 0.0;
    for (size_t j = 0; j < dim; ++j) value += static_cast<double>(a[j]) * b[j];
    return std::max(-1.0, std::min(1.0, value));
}

double independentScore(
        const float *query,
        const float *base,
        double query_time,
        double timestamp,
        const TangoConfig &config) {
    double age = query_time - timestamp;
    if (age < -config.future_time_epsilon) {
        throw std::invalid_argument("brute-force query sees an object from the future");
    }
    if (age < 0.0) age = 0.0;
    const double freshness = std::exp(-config.lambda * age);
    const double semantic = dot(query, base, config.dim);
    return config.mode == TangoMode::Multiplicative
        ? semantic * freshness
        : config.alpha * semantic + (1.0 - config.alpha) * freshness;
}

std::vector<labeltype> independentTopK(
        const float *query,
        double query_time,
        const std::vector<float> &base,
        const std::vector<float> &timestamps,
        size_t n,
        size_t k,
        const TangoConfig &config) {
    std::vector<std::pair<double, labeltype> > scores;
    scores.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        scores.push_back(std::make_pair(
            independentScore(query, base.data() + i * config.dim, query_time, timestamps[i], config),
            static_cast<labeltype>(i)));
    }
    const size_t keep = std::min(k, scores.size());
    std::partial_sort(scores.begin(), scores.begin() + keep, scores.end(),
        [](const std::pair<double, labeltype> &a, const std::pair<double, labeltype> &b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });
    std::vector<labeltype> result;
    result.reserve(keep);
    for (size_t i = 0; i < keep; ++i) result.push_back(scores[i].second);
    return result;
}

void validateGroundTruth(
        const Options &o,
        const TangoConfig &config,
        const std::vector<float> &base,
        const std::vector<float> &timestamps,
        const std::vector<float> &queries,
        const std::vector<double> &query_times,
        const std::vector<int64_t> &gt) {
    const size_t count = std::min(o.validate_bruteforce, o.queries);
    for (size_t qi = 0; qi < count; ++qi) {
        const std::vector<labeltype> exact = independentTopK(
            queries.data() + qi * config.dim, query_times[qi], base, timestamps,
            o.n, o.k, config);
        std::unordered_set<labeltype> expected(exact.begin(), exact.end());
        size_t overlap = 0;
        for (size_t j = 0; j < o.k; ++j) {
            const int64_t id = gt[qi * o.gt_width + j];
            if (id >= 0 && expected.count(static_cast<labeltype>(id))) ++overlap;
        }
        if (overlap != o.k) {
            std::ostringstream msg;
            msg << "ground truth mismatch for query " << qi << ": independent top-" << o.k
                << " overlaps GT by " << overlap;
            throw std::runtime_error(msg.str());
        }
    }
    std::cout << "Independent brute-force validation passed for " << count << " queries\n";
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = q * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(position));
    const size_t hi = static_cast<size_t>(std::ceil(position));
    const double fraction = position - lo;
    return values[lo] * (1.0 - fraction) + values[hi] * fraction;
}

double mean(const std::vector<double> &values) {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < values.size(); ++i) sum += values[i];
    return sum / values.size();
}

std::string csvQuote(const std::string &value) {
    std::string out = "\"";
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '"') out += '"';
        out += value[i];
    }
    out += '"';
    return out;
}

std::string modeName(TangoMode mode) {
    return mode == TangoMode::Multiplicative ? "multiplicative" : "additive";
}

void emitBuildReportHeader(std::ostream &out) {
    out << "run_completed_at_utc,graph_mode,mode,data_data_mode,dataset,time_distribution,n,dim,"
           "half_life,alpha,kappa_base,kappa_nav,kappa_schedule,kappa_levels,M,"
           "ef_construction,build_threads,random_seed,normalize,tango_decay_fastpath,"
           "input_seconds,preprocess_seconds,core_build_seconds,serialize_seconds,"
           "serving_index_pipeline_seconds,index_hnsw_bytes,index_meta_bytes,"
           "serialized_index_bytes,"
           "index_prefix\n";
}

void emitBuildReportRow(
        std::ostream &out,
        const Options &o,
        const TangoConfig &config,
        const BuildMetrics &metrics) {
    out << std::setprecision(12)
        << csvQuote(metrics.completed_at_utc) << ",tango,"
        << modeName(config.mode) << ',' << hnswlib::tangoDataDataModeName(config.data_data_mode)
        << ',' << csvQuote(o.dataset) << ','
        << csvQuote(o.time_distribution) << ',' << o.n << ',' << config.dim << ','
        << config.half_life << ',' << config.alpha << ',' << config.kappa_base << ','
        << config.kappa_nav << ',' << hnswlib::tangoScheduleName(config.schedule) << ','
        << csvQuote(hnswlib::tango_detail::serializeDoubleList(
               config.schedule == KappaSchedule::PerLevel
                   ? config.kappa_levels
                   : std::vector<double>{config.kappa_base, config.kappa_nav}))
        << ',' << o.M << ',' << o.ef_construction << ',' << o.threads << ','
        << o.random_seed << ',' << (o.normalize ? 1 : 0) << ','
        << hnswlib::tangoDecayFastPathName(
               hnswlib::parseTangoDecayFastPath(o.tango_decay_fastpath))
        << ',' << metrics.input_seconds << ',' << metrics.preprocess_seconds << ','
        << metrics.core_build_seconds << ',' << metrics.serialize_seconds << ','
        << metrics.serving_index_pipeline_seconds << ',' << metrics.index_hnsw_bytes << ','
        << metrics.index_meta_bytes << ',' << metrics.serialized_index_bytes << ','
        << csvQuote(o.index_prefix) << '\n';
}

void emitBuildReportKeyValues(
        std::ostream &out,
        const Options &o,
        const TangoConfig &config,
        const BuildMetrics &metrics) {
    out << std::setprecision(12)
        << "build_report_schema=tango_build_v1\n"
        << "run_completed_at_utc=" << metrics.completed_at_utc << '\n'
        << "build_method=tango\n"
        << "build_mode=" << modeName(config.mode) << '\n'
        << "build_data_data_mode=" << hnswlib::tangoDataDataModeName(config.data_data_mode) << '\n'
        << "build_elements=" << o.n << '\n'
        << "build_dim=" << config.dim << '\n'
        << "build_M=" << o.M << '\n'
        << "build_ef_construction=" << o.ef_construction << '\n'
        << "build_threads=" << o.threads << '\n'
        << "input_seconds=" << metrics.input_seconds << '\n'
        << "preprocess_seconds=" << metrics.preprocess_seconds << '\n'
        << "core_build_seconds=" << metrics.core_build_seconds << '\n'
        << "serialize_seconds=" << metrics.serialize_seconds << '\n'
        << "serving_index_pipeline_seconds="
        << metrics.serving_index_pipeline_seconds << '\n'
        << "index_hnsw_bytes=" << metrics.index_hnsw_bytes << '\n'
        << "index_meta_bytes=" << metrics.index_meta_bytes << '\n'
        << "serialized_index_bytes=" << metrics.serialized_index_bytes << '\n';
}

void writeBuildReportCsv(
        const std::string &path,
        const Options &o,
        const TangoConfig &config,
        const BuildMetrics &metrics) {
    if (path.empty()) return;
    const bool needs_header = fileSize(path) == 0;
    std::ofstream out(path.c_str(), std::ios::app);
    if (!out) throw std::runtime_error("cannot open build report CSV: " + path);
    if (needs_header) emitBuildReportHeader(out);
    emitBuildReportRow(out, o, config, metrics);
    out.close();
    if (!out) throw std::runtime_error("failed writing build report CSV: " + path);
    std::cerr << "Wrote build report to " << path << '\n';
}

void emitResultHeader(std::ostream &out) {
    out << "graph_mode,mode,data_data_mode,dataset,time_distribution,half_life,alpha,"
           "kappa_base,kappa_nav,M,ef_construction,ef,k,recall,qps,mean_ms,"
           "p50_ms,p95_ms,p99_ms,avg_distance_computations,avg_hops,index_bytes,"
           "build_seconds,kappa_schedule,kappa_levels,tango_decay_fastpath,"
           "decay_instrumentation,"
           "decay_reference_time,decay_cache_precision,decay_cache_inverse,"
           "decay_cache_active,decay_cache_init_ms,decay_cache_bytes,qd_callbacks,"
           "dd_callbacks,direct_qd_exp,direct_dd_exp,query_scale_exp,fast_qd,fast_dd,"
           "decay_fallbacks,verify_qd_max_abs,verify_qd_max_rel,verify_dd_max_abs,"
           "verify_dd_max_rel\n";
}

void emitResultRow(
        std::ostream &out,
        const Options &o,
        const TangoConfig &config,
        size_t ef,
        double recall,
        const std::vector<double> &latency_ms,
        double distance_computations,
        double hops,
        uint64_t index_bytes,
        double build_seconds,
        const TANGOIndex &index) {
    const TangoDecayFastPathStats decay = index.decayFastPathStats();
    const double mean_ms = mean(latency_ms);
    const double qps = mean_ms > 0.0 ? 1000.0 / mean_ms : 0.0;
    out << std::setprecision(12)
        << "tango," << modeName(config.mode) << ','
        << hnswlib::tangoDataDataModeName(config.data_data_mode) << ','
        << csvQuote(o.dataset) << ',' << csvQuote(o.time_distribution) << ','
        << config.half_life << ',' << config.alpha << ','
        << config.kappa_base << ',' << config.kappa_nav << ','
        << o.M << ',' << o.ef_construction << ',' << ef << ',' << o.k << ','
        << recall << ',' << qps << ',' << mean_ms << ','
        << percentile(latency_ms, 0.50) << ',' << percentile(latency_ms, 0.95) << ','
        << percentile(latency_ms, 0.99) << ',' << distance_computations << ','
        << hops << ',' << index_bytes << ',' << build_seconds << ','
        << hnswlib::tangoScheduleName(config.schedule) << ','
        << csvQuote(hnswlib::tango_detail::serializeDoubleList(
               config.schedule == KappaSchedule::PerLevel
                   ? config.kappa_levels
                   : std::vector<double>{config.kappa_base, config.kappa_nav}))
        << ',' << hnswlib::tangoDecayFastPathName(index.decayFastPathMode())
        << ',' << (o.decay_instrumentation ? 1 : 0)
        << ',' << index.decayReferenceTime() << ",float64,"
        << (index.decayCacheStoresInverse() ? 1 : 0) << ','
        << decay.active_entries << ',' << decay.cache_init_ms << ','
        << decay.cache_bytes << ',' << decay.qd_callbacks << ','
        << decay.dd_callbacks << ',' << decay.direct_qd_exp << ','
        << decay.direct_dd_exp << ',' << decay.query_scale_exp << ','
        << decay.fast_qd << ',' << decay.fast_dd << ',' << decay.fallbacks << ','
        << decay.verify_qd_max_abs << ',' << decay.verify_qd_max_rel << ','
        << decay.verify_dd_max_abs << ',' << decay.verify_dd_max_rel << '\n';
}

double readRecordTimestamp(const void *record, size_t dim) {
    double timestamp = 0.0;
    std::memcpy(&timestamp, static_cast<const char *>(record) + dim * sizeof(float), sizeof(double));
    return timestamp;
}

double gini(std::vector<size_t> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    long double weighted = 0.0;
    long double sum = 0.0;
    const long double n = static_cast<long double>(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        sum += values[i];
        weighted += (2.0L * (i + 1) - n - 1.0L) * values[i];
    }
    return sum == 0.0 ? 0.0 : static_cast<double>(weighted / (n * sum));
}

double sizePercentile(std::vector<size_t> values, double q) {
    std::vector<double> converted(values.begin(), values.end());
    return percentile(converted, q);
}

void dumpGraphStats(const TANGOIndex &index, const std::string &path) {
    const hnswlib::HierarchicalNSW<float> &raw = index.rawIndex();
    const size_t n = raw.getCurrentElementCount();
    const size_t dim = index.config().dim;
    std::vector<double> timestamps(n);
    for (size_t i = 0; i < n; ++i) {
        timestamps[i] = readRecordTimestamp(raw.getDataByInternalId(static_cast<hnswlib::tableint>(i)), dim);
    }
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (timestamps[a] != timestamps[b]) return timestamps[a] < timestamps[b];
        return a < b;
    });
    std::vector<int> decile(n, 0);
    for (size_t rank = 0; rank < n; ++rank) {
        decile[order[rank]] = n == 0 ? 0 : std::min<int>(9, static_cast<int>(10 * rank / n));
    }

    std::ofstream out(path.c_str());
    if (!out) throw std::runtime_error("cannot open graph stats CSV: " + path);
    out << "row_type,level,timestamp_decile,nodes,edges,avg_outdegree,indegree_mean,"
           "indegree_p50,indegree_p95,indegree_p99,indegree_max,indegree_gini,"
           "time_delta_mean,time_delta_p50,time_delta_p95,time_delta_p99,"
           "semantic_cosine_mean,semantic_cosine_p50,semantic_cosine_p95,"
           "cross_timestamp_decile_ratio\n";

    for (int level = 0; level <= raw.maxlevel_; ++level) {
        std::vector<size_t> indegree(n, 0);
        std::vector<double> time_deltas;
        std::vector<double> cosines;
        size_t layer_nodes = 0;
        size_t edges = 0;
        size_t cross_decile = 0;
        for (size_t i = 0; i < n; ++i) {
            if (raw.element_levels_[i] < level) continue;
            ++layer_nodes;
            hnswlib::linklistsizeint *links = raw.get_linklist_at_level(
                static_cast<hnswlib::tableint>(i), level);
            const size_t degree = raw.getListCount(links);
            hnswlib::tableint *neighbors = reinterpret_cast<hnswlib::tableint *>(links + 1);
            const float *xi = reinterpret_cast<const float *>(raw.getDataByInternalId(
                static_cast<hnswlib::tableint>(i)));
            for (size_t j = 0; j < degree; ++j) {
                const size_t neighbor = neighbors[j];
                if (neighbor >= n) throw std::runtime_error("invalid graph neighbor while dumping stats");
                ++edges;
                ++indegree[neighbor];
                time_deltas.push_back(std::fabs(timestamps[i] - timestamps[neighbor]));
                const float *xj = reinterpret_cast<const float *>(raw.getDataByInternalId(
                    static_cast<hnswlib::tableint>(neighbor)));
                cosines.push_back(dot(xi, xj, dim));
                if (decile[i] != decile[neighbor]) ++cross_decile;
            }
        }
        std::vector<size_t> layer_indegree;
        for (size_t i = 0; i < n; ++i) {
            if (raw.element_levels_[i] >= level) layer_indegree.push_back(indegree[i]);
        }
        const size_t max_in = layer_indegree.empty()
            ? 0 : *std::max_element(layer_indegree.begin(), layer_indegree.end());
        out << std::setprecision(12) << "layer_summary," << level << ",-1,"
            << layer_nodes << ',' << edges << ','
            << (layer_nodes ? static_cast<double>(edges) / layer_nodes : 0.0) << ','
            << (layer_nodes ? static_cast<double>(edges) / layer_nodes : 0.0) << ','
            << sizePercentile(layer_indegree, 0.50) << ','
            << sizePercentile(layer_indegree, 0.95) << ','
            << sizePercentile(layer_indegree, 0.99) << ',' << max_in << ','
            << gini(layer_indegree) << ',' << mean(time_deltas) << ','
            << percentile(time_deltas, 0.50) << ',' << percentile(time_deltas, 0.95) << ','
            << percentile(time_deltas, 0.99) << ',' << mean(cosines) << ','
            << percentile(cosines, 0.50) << ',' << percentile(cosines, 0.95) << ','
            << (edges ? static_cast<double>(cross_decile) / edges : 0.0) << '\n';

        for (int d = 0; d < 10; ++d) {
            size_t nodes = 0;
            size_t in_sum = 0;
            for (size_t i = 0; i < n; ++i) {
                if (raw.element_levels_[i] >= level && decile[i] == d) {
                    ++nodes;
                    in_sum += indegree[i];
                }
            }
            out << "timestamp_decile," << level << ',' << d << ',' << nodes
                << ",0,0," << (nodes ? static_cast<double>(in_sum) / nodes : 0.0)
                << ",0,0,0,0,0,0,0,0,0,0,0,0,0\n";
        }
    }
    std::cout << "Wrote graph statistics to " << path << '\n';
}

TangoConfig makeCreateConfig(const Options &o) {
    TangoConfig config;
    config.dim = o.dim;
    config.mode = o.mode == "multiplicative" ? TangoMode::Multiplicative : TangoMode::Additive;
    config.data_data_mode = o.data_data_mode == "semantic"
        ? TangoDataDataMode::Semantic : TangoDataDataMode::TimeLift;
    config.half_life = o.half_life;
    config.lambda = std::log(2.0) / o.half_life;
    config.alpha = config.mode == TangoMode::Multiplicative ? 1.0 : o.alpha;
    if (o.kappa_levels_supplied) {
        config.kappa_levels = o.kappa_levels;
        config.kappa_base = config.kappa_levels.front();
        config.kappa_nav = config.kappa_levels.back();
        config.schedule = KappaSchedule::PerLevel;
    } else {
        config.kappa_base = o.kappa_base;
        config.kappa_nav = o.kappa_nav;
        config.schedule = std::fabs(o.kappa_base - o.kappa_nav) <= 1e-15
            ? KappaSchedule::Fixed : KappaSchedule::TwoTier;
    }
    config.normalize_input = o.normalize;
    config.future_time_epsilon = 1e-9;
    config.validate();
    return config;
}

TangoDecayFastPathOptions makeDecayFastPathOptions(
        const Options &o,
        double reference_time = std::numeric_limits<double>::quiet_NaN()) {
    TangoDecayFastPathOptions options;
    options.mode = hnswlib::parseTangoDecayFastPath(o.tango_decay_fastpath);
    options.reference_time = reference_time;
    options.verify_samples = o.decay_verify_samples;
    options.instrumentation = o.decay_instrumentation;
    return options;
}

void logDecayFastPath(const TANGOIndex &index, const char *phase) {
    const TangoDecayFastPathStats stats = index.decayFastPathStats();
    std::cerr << "TANGO decay fastpath [" << phase << "]: mode="
              << hnswlib::tangoDecayFastPathName(index.decayFastPathMode())
              << " T0=" << std::setprecision(17) << index.decayReferenceTime()
              << " precision=float64 inverse="
              << (index.decayCacheStoresInverse() ? 1 : 0)
              << " active=" << stats.active_entries
              << " cache_init_ms=" << stats.cache_init_ms
              << " cache_bytes=" << stats.cache_bytes
              << " qd=" << stats.qd_callbacks << " dd=" << stats.dd_callbacks
              << " direct_qd_exp=" << stats.direct_qd_exp
              << " direct_dd_exp=" << stats.direct_dd_exp
              << " query_scale_exp=" << stats.query_scale_exp
              << " fast_qd=" << stats.fast_qd << " fast_dd=" << stats.fast_dd
              << " fallback=" << stats.fallbacks
              << " verify_qd_max_abs=" << stats.verify_qd_max_abs
              << " verify_qd_max_rel=" << stats.verify_qd_max_rel
              << " verify_dd_max_abs=" << stats.verify_dd_max_abs
              << " verify_dd_max_rel=" << stats.verify_dd_max_rel << '\n';
    if (std::string(phase) == "load" &&
            index.decayFastPathMode() == hnswlib::TangoDecayFastPath::DataData) {
        std::cerr << "TANGO decay fastpath: dd affects later insert/update/repair; "
                     "a static load+query run still uses direct QD decay.\n";
    }
}

void validateLoadOverrides(const Options &o, const TANGOIndex &index) {
    const TangoConfig &c = index.config();
    const double eps = 1e-10;
    if (o.dim_supplied && o.dim != c.dim) throw std::invalid_argument("--dim conflicts with sidecar");
    if (o.mode_supplied && o.mode != modeName(c.mode)) throw std::invalid_argument("--mode conflicts with sidecar");
    if (o.data_data_mode_supplied &&
            o.data_data_mode != hnswlib::tangoDataDataModeName(c.data_data_mode)) {
        throw std::invalid_argument("--dd-mode conflicts with sidecar");
    }
    if (o.half_life_supplied && std::fabs(o.half_life - c.half_life) > eps * std::max(1.0, c.half_life))
        throw std::invalid_argument("--half-life conflicts with sidecar");
    if (o.alpha_supplied && std::fabs(o.alpha - c.alpha) > eps)
        throw std::invalid_argument("--alpha conflicts with sidecar");
    if (o.kappa_base_supplied && std::fabs(o.kappa_base - c.kappa_base) > eps)
        throw std::invalid_argument("--kappa-base conflicts with sidecar");
    if (o.kappa_nav_supplied && std::fabs(o.kappa_nav - c.kappa_nav) > eps)
        throw std::invalid_argument("--kappa-nav conflicts with sidecar");
    if (o.kappa_levels_supplied) {
        if (c.schedule != KappaSchedule::PerLevel ||
                o.kappa_levels.size() != c.kappa_levels.size()) {
            throw std::invalid_argument("--kappa-levels conflicts with sidecar");
        }
        for (size_t i = 0; i < o.kappa_levels.size(); ++i) {
            if (std::fabs(o.kappa_levels[i] - c.kappa_levels[i]) >
                    eps * std::max(1.0, std::fabs(c.kappa_levels[i]))) {
                throw std::invalid_argument("--kappa-levels conflicts with sidecar");
            }
        }
    }
    if (o.normalize_supplied && o.normalize != c.normalize_input)
        throw std::invalid_argument("--normalize conflicts with sidecar");
    if (o.M_supplied && o.M != index.rawIndex().M_)
        throw std::invalid_argument("--M conflicts with sidecar");
    if (o.ef_construction_supplied &&
            o.ef_construction != index.rawIndex().ef_construction_)
        throw std::invalid_argument("--ef-construction conflicts with sidecar");
    if (o.random_seed_supplied && o.random_seed != index.randomSeed())
        throw std::invalid_argument("--random-seed conflicts with sidecar");
}

int run(int argc, char **argv) {
    Options o = parseOptions(argc, argv);

    std::unique_ptr<TANGOIndex> index;
    std::vector<float> base;
    std::vector<float> timestamps;
    double build_seconds = 0.0;
    BuildMetrics build_metrics;

    if (o.index_mode == "create") {
        TangoConfig config = makeCreateConfig(o);

        const Clock::time_point input_start = Clock::now();
        base = loadBinary<float>(
            o.base_path, checkedElementCount(o.n, config.dim, "base vectors"),
            "base vectors");
        timestamps = loadBinary<float>(o.timestamps_path, o.n, "timestamps");
        build_metrics.input_seconds =
            std::chrono::duration<double>(Clock::now() - input_start).count();

        // Keep the construction-time boundary aligned with the baseline
        // adapters: file I/O is reported separately, while the serving-index
        // pipeline starts at method-specific preprocessing.
        const Clock::time_point pipeline_start = Clock::now();
        const Clock::time_point preprocess_start = pipeline_start;
        if (o.normalize) normalizeBatch(base, o.n, config.dim);

        const double reference_time = *std::max_element(
            timestamps.begin(), timestamps.end());

        const size_t headroom = std::max<size_t>(16, o.n / 10);
        index.reset(new TANGOIndex(
            config, o.n + headroom, o.M, o.ef_construction, o.random_seed,
            makeDecayFastPathOptions(o, reference_time)));
        index->setMaxSearchExpansions(o.max_search_expansions);
        build_metrics.preprocess_seconds =
            std::chrono::duration<double>(Clock::now() - preprocess_start).count();

        std::cerr << "Building index with " << o.threads << " thread(s):\n";
        ProgressDisplay progress(o.n);
        const Clock::time_point start = Clock::now();
        index->addPoint(base.data(), timestamps[0], 0);
        progress.advance();
        parallelFor(1, o.n, o.threads, [&](size_t i) {
            index->addPoint(base.data() + i * config.dim, timestamps[i], static_cast<labeltype>(i));
            progress.advance();
        });
        build_seconds = std::chrono::duration<double>(Clock::now() - start).count();
        build_metrics.core_build_seconds = build_seconds;

        const Clock::time_point serialize_start = Clock::now();
        index->save(o.index_prefix);
        build_metrics.serialize_seconds =
            std::chrono::duration<double>(Clock::now() - serialize_start).count();
        build_metrics.serving_index_pipeline_seconds =
            std::chrono::duration<double>(Clock::now() - pipeline_start).count();
        build_metrics.index_hnsw_bytes = fileSize(o.index_prefix + ".hnsw");
        build_metrics.index_meta_bytes = fileSize(o.index_prefix + ".tango.meta");
        build_metrics.serialized_index_bytes =
            build_metrics.index_hnsw_bytes + build_metrics.index_meta_bytes;
        build_metrics.completed_at_utc = iso8601NowUtc();
        std::cerr << "Built " << o.n << " points in " << build_seconds << " s and saved "
                  << o.index_prefix << "\n";
        logDecayFastPath(*index, "build");
        emitBuildReportKeyValues(std::cerr, o, config, build_metrics);
        writeBuildReportCsv(o.build_report_csv, o, config, build_metrics);
    } else {
        index = TANGOIndex::load(
            o.index_prefix, 0, makeDecayFastPathOptions(o));
        validateLoadOverrides(o, *index);
        index->setMaxSearchExpansions(o.max_search_expansions);
        o.n = index->size();
        o.dim = index->config().dim;
        o.M = index->rawIndex().M_;
        o.ef_construction = index->rawIndex().ef_construction_;
        std::cerr << "Loaded " << index->size() << " points from " << o.index_prefix << "\n";
        logDecayFastPath(*index, "load");
    }

    if (o.build_only) {
        if (!o.graph_stats_csv.empty()) dumpGraphStats(*index, o.graph_stats_csv);
        return 0;
    }

    const TangoConfig &config = index->config();
    const std::vector<float> query_data = loadBinary<float>(
        o.query_path, checkedElementCount(o.queries, config.dim, "query vectors"),
        "query vectors");
    std::vector<float> queries = query_data;
    if (config.normalize_input) normalizeBatch(queries, o.queries, config.dim);
    const std::vector<int64_t> gt = loadBinary<int64_t>(
        o.gt_path, checkedElementCount(o.queries, o.gt_width, "ground truth"),
        "ground truth");
    if (o.k > index->size()) {
        throw std::invalid_argument("k exceeds the number of indexed elements");
    }
    validateGroundTruthIds(gt, o.queries, o.gt_width, index->size());

    std::vector<double> query_times(o.queries, o.query_time);
    if (!o.query_times_path.empty()) {
        const std::vector<float> times = loadBinary<float>(
            o.query_times_path, o.queries, "query times");
        for (size_t i = 0; i < o.queries; ++i) query_times[i] = times[i];
    }
    for (size_t i = 0; i < query_times.size(); ++i) {
        if (query_times[i] < index->maxTimestamp() - config.future_time_epsilon) {
            throw std::invalid_argument("query time is earlier than the index maximum timestamp");
        }
    }

    if (o.validate_bruteforce != 0) {
        if (o.index_mode != "create") {
            throw std::invalid_argument("--validate-bruteforce currently requires --index-mode create");
        }
        validateGroundTruth(o, config, base, timestamps, queries, query_times, gt);
    }

    if (!o.graph_stats_csv.empty()) dumpGraphStats(*index, o.graph_stats_csv);

    base.clear();
    base.shrink_to_fit();
    timestamps.clear();
    timestamps.shrink_to_fit();

    const uint64_t index_bytes = fileSize(o.index_prefix + ".hnsw") +
                                 fileSize(o.index_prefix + ".tango.meta");
    std::unique_ptr<std::ofstream> csv_file;
    std::ostream *csv = &std::cout;
    bool needs_header = true;
    if (!o.output_csv.empty()) {
        needs_header = fileSize(o.output_csv) == 0;
        csv_file.reset(new std::ofstream(o.output_csv.c_str(), std::ios::app));
        if (!*csv_file) throw std::runtime_error("cannot open output CSV: " + o.output_csv);
        csv = csv_file.get();
    }
    if (needs_header) emitResultHeader(*csv);

    for (size_t ei = 0; ei < o.ef_list.size(); ++ei) {
        const size_t ef = std::max(o.ef_list[ei], o.k);
        index->setEf(ef);
        index->rawIndex().metric_distance_computations = 0;
        index->rawIndex().metric_hops = 0;
        index->resetDecayFastPathCounters();
        std::vector<double> latency_ms;
        latency_ms.reserve(o.queries);
        size_t correct = 0;
        for (size_t qi = 0; qi < o.queries; ++qi) {
            const Clock::time_point start = Clock::now();
            std::priority_queue<std::pair<float, labeltype> > result = index->searchKnn(
                queries.data() + qi * config.dim, query_times[qi], o.k);
            latency_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
            std::unordered_set<labeltype> expected;
            for (size_t j = 0; j < o.k; ++j) {
                const int64_t id = gt[qi * o.gt_width + j];
                if (id >= 0) expected.insert(static_cast<labeltype>(id));
            }
            while (!result.empty()) {
                if (expected.count(result.top().second)) ++correct;
                result.pop();
            }
        }
        const double recall = static_cast<double>(correct) / (o.queries * o.k);
        const double avg_dist = static_cast<double>(index->rawIndex().metric_distance_computations.load()) /
                                o.queries;
        const double avg_hops = static_cast<double>(index->rawIndex().metric_hops.load()) / o.queries;
        emitResultRow(*csv, o, config, ef, recall, latency_ms, avg_dist, avg_hops,
                         index_bytes, build_seconds, *index);
        if (csv != &std::cout) {
            std::cout << "ef=" << ef << " recall=" << recall
                      << " mean_ms=" << mean(latency_ms) << '\n';
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "tango_index: " << e.what() << '\n';
        return 1;
    }
}
