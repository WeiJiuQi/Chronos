// TANGO query, evaluation, and serving-footprint command-line interface.
//
// Queries run serially over a saved TANGO index. Every executed query is timed;
// file loading, normalization, prepared-record construction, and Recall@k are
// outside the graph-search timing interval.
#include "tango.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#elif defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using hnswlib::TANGOIndex;
using hnswlib::TangoDecayFastPathOptions;
using hnswlib::TangoDecayFastPathStats;
using hnswlib::labeltype;

typedef std::priority_queue<std::pair<float, labeltype> > SearchResult;

struct Options {
    std::string index_path;
    std::string query_path;
    std::string gt_path;
    std::string output_csv;
    std::string result_ids_output;
    std::string run_label = "default";
    std::string tango_decay_fastpath = "off";
    bool overwrite_output = false;
    bool decay_instrumentation = false;
    bool footprint_only = false;
    size_t queries = 0;
    size_t dim = 0;
    size_t gt_width = 100;
    size_t k = 0;
    // Kept in the CSV schema as an explicit audit field.  Non-zero warm-up is
    // rejected at the CLI so every executed query contributes to performance.
    size_t warmup_queries = 0;
    size_t repeats = 3;
    size_t max_search_expansions = 0;
    size_t decay_verify_samples = 0;
    std::vector<size_t> ef_list;
    double query_time = std::numeric_limits<double>::quiet_NaN();
};

void usage(const char *program) {
    std::cerr
        << "Query mode:\n  " << program
        << " --index PREFIX --query FILE --gt FILE --queries N --dim D\n"
        << "    --gt-width W --k K --ef-list E1,E2,... --query-time TAU [options]\n\n"
        << "Footprint mode:\n  " << program
        << " --index PREFIX --dim D --footprint-only [options]\n\n"
        << "Options:\n"
        << "  --query-time TAU             required for query mode\n"
        << "  --warmup-queries 0           compatibility-only audit option; non-zero is\n"
        << "                              rejected (all queries are timed)\n"
        << "  --repeats N                  timed full-query passes per ef (default 3)\n"
        << "  --max-search-expansions N    base-layer cap; 0 is unlimited (default 0)\n"
        << "  --tango-decay-fastpath M     factorized path: off|qd|dd|all (default off)\n"
        << "  --tango-decay-fastpath-verify-samples N\n"
        << "                              verify first N fast QD/DD callback distances\n"
        << "  --tango-decay-fastpath-instrumentation {0|1}\n"
        << "                              enable callback counters (default 0)\n"
        << "  --footprint-only             load only the serving index artifacts\n"
        << "  --output-csv FILE            append rows; stdout if omitted\n"
        << "  --result-ids-output FILE     write first-repeat uint64 IDs by (ef,query,rank)\n"
        << "  --overwrite-output           truncate output CSV before writing\n"
        << "  --run-label TEXT             label for repeated process runs\n\n"
        << "PREFIX omits the .hnsw/.tango.meta suffixes. Queries are L2 normalized.\n"
        << "The query loop is serial and every ef must be >= k. Footprint mode\n"
        << "does not load query/GT data and should run in a fresh process.\n";
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
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (text.empty() || text[0] == '-' || errno != 0 || end == text.c_str() ||
            *end != '\0' || (!allow_zero && value == 0) ||
            value > std::numeric_limits<size_t>::max()) {
        throw std::invalid_argument("invalid " + name + ": " + text);
    }
    return static_cast<size_t>(value);
}

double parseDouble(const std::string &text, const std::string &name) {
    char *end = NULL;
    errno = 0;
    const double value = std::strtod(text.c_str(), &end);
    if (text.empty() || errno != 0 || end == text.c_str() || *end != '\0' ||
            !std::isfinite(value)) {
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
    std::vector<size_t> values;
    std::stringstream input(text);
    std::string token;
    while (std::getline(input, token, ',')) {
        if (token.empty()) throw std::invalid_argument("empty value in --ef-list");
        values.push_back(parseSize(token, "ef"));
    }
    if (values.empty()) throw std::invalid_argument("--ef-list must not be empty");
    return values;
}

Options parseOptions(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag(argv[i]);
        if (flag == "--help" || flag == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else if (flag == "--index") {
            options.index_path = requireValue(i, argc, argv);
        } else if (flag == "--query") {
            options.query_path = requireValue(i, argc, argv);
        } else if (flag == "--gt") {
            options.gt_path = requireValue(i, argc, argv);
        } else if (flag == "--queries") {
            options.queries = parseSize(requireValue(i, argc, argv), "queries");
        } else if (flag == "--dim") {
            options.dim = parseSize(requireValue(i, argc, argv), "dim");
        } else if (flag == "--gt-width") {
            options.gt_width = parseSize(requireValue(i, argc, argv), "gt-width");
        } else if (flag == "--k") {
            options.k = parseSize(requireValue(i, argc, argv), "k");
        } else if (flag == "--ef-list") {
            options.ef_list = parseEfList(requireValue(i, argc, argv));
        } else if (flag == "--query-time") {
            options.query_time = parseDouble(requireValue(i, argc, argv), "query-time");
        } else if (flag == "--warmup-queries") {
            const size_t requested_warmup = parseSize(
                requireValue(i, argc, argv), "warmup-queries", true);
            if (requested_warmup != 0) {
                throw std::invalid_argument(
                    "--warmup-queries must be 0; query warm-up is disabled and "
                    "every query is timed");
            }
        } else if (flag == "--repeats") {
            options.repeats = parseSize(requireValue(i, argc, argv), "repeats");
        } else if (flag == "--max-search-expansions") {
            options.max_search_expansions = parseSize(
                requireValue(i, argc, argv), "max-search-expansions", true);
        } else if (flag == "--tango-decay-fastpath") {
            options.tango_decay_fastpath = requireValue(i, argc, argv);
            (void)hnswlib::parseTangoDecayFastPath(
                options.tango_decay_fastpath);
        } else if (flag == "--tango-decay-fastpath-verify-samples") {
            options.decay_verify_samples = parseSize(
                requireValue(i, argc, argv),
                "tango-decay-fastpath-verify-samples", true);
        } else if (flag == "--tango-decay-fastpath-instrumentation") {
            options.decay_instrumentation = parseBool(
                requireValue(i, argc, argv),
                "tango-decay-fastpath-instrumentation");
        } else if (flag == "--footprint-only") {
            options.footprint_only = true;
        } else if (flag == "--output-csv") {
            options.output_csv = requireValue(i, argc, argv);
        } else if (flag == "--result-ids-output") {
            options.result_ids_output = requireValue(i, argc, argv);
        } else if (flag == "--overwrite-output") {
            options.overwrite_output = true;
        } else if (flag == "--run-label") {
            options.run_label = requireValue(i, argc, argv);
        } else {
            throw std::invalid_argument("unknown argument: " + flag);
        }
    }

    if (options.index_path.empty() || options.dim == 0) {
        throw std::invalid_argument("--index and --dim are required");
    }
    if (!options.footprint_only &&
            (options.query_path.empty() || options.gt_path.empty() ||
             options.queries == 0 || options.k == 0 || options.ef_list.empty())) {
        throw std::invalid_argument(
            "query mode requires --query, --gt, --queries, --k, and --ef-list");
    }
    if (!options.footprint_only && options.k > options.gt_width) {
        throw std::invalid_argument("--k must be <= --gt-width");
    }
    if (!options.footprint_only && !std::isfinite(options.query_time)) {
        throw std::invalid_argument("--query-time is required in query mode");
    }
    if (options.footprint_only && !options.result_ids_output.empty()) {
        throw std::invalid_argument("--result-ids-output is not valid with --footprint-only");
    }
    for (size_t i = 0; i < options.ef_list.size(); ++i) {
        if (options.ef_list[i] < options.k) {
            throw std::invalid_argument("every ef must be >= k");
        }
    }
    return options;
}

uint64_t fileSize(const std::string &path) {
    std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
    if (!input) return 0;
    const std::ifstream::pos_type position = input.tellg();
    return position < 0 ? 0 : static_cast<uint64_t>(position);
}

uint64_t currentRssBytes() {
#if defined(__APPLE__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    const kern_return_t status = task_info(
        mach_task_self(), MACH_TASK_BASIC_INFO,
        reinterpret_cast<task_info_t>(&info), &count);
    return status == KERN_SUCCESS ? static_cast<uint64_t>(info.resident_size) : 0;
#elif defined(__linux__)
    std::ifstream input("/proc/self/statm");
    uint64_t total_pages = 0;
    uint64_t resident_pages = 0;
    if (!(input >> total_pages >> resident_pages)) return 0;
    (void)total_pages;
    const long page_size = ::sysconf(_SC_PAGESIZE);
    return page_size > 0
        ? resident_pages * static_cast<uint64_t>(page_size)
        : 0;
#else
    return 0;
#endif
}

uint64_t peakRssBytes() {
#if defined(__APPLE__) || defined(__linux__)
    struct rusage usage;
    if (::getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
    // Darwin reports ru_maxrss in bytes.
    return static_cast<uint64_t>(usage.ru_maxrss);
#else
    // Linux reports ru_maxrss in KiB.
    return static_cast<uint64_t>(usage.ru_maxrss) * 1024ULL;
#endif
#else
    return 0;
#endif
}

int64_t signedByteDelta(uint64_t after, uint64_t before) {
    const uint64_t max_signed = static_cast<uint64_t>(
        std::numeric_limits<int64_t>::max());
    if (after >= before) {
        const uint64_t value = after - before;
        return value > max_signed
            ? std::numeric_limits<int64_t>::max()
            : static_cast<int64_t>(value);
    }
    const uint64_t value = before - after;
    return value > max_signed
        ? std::numeric_limits<int64_t>::min()
        : -static_cast<int64_t>(value);
}

template <typename T>
std::vector<T> loadExact(const std::string &path, size_t count, const char *description) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::overflow_error(std::string(description) + " size overflow");
    }
    const size_t required = count * sizeof(T);
    if (required > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::overflow_error(std::string(description) + " exceeds the stream-size limit");
    }
    std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open " + std::string(description) + ": " + path);
    const std::streamoff actual = input.tellg();
    if (actual < 0 || static_cast<uint64_t>(actual) != static_cast<uint64_t>(required)) {
        std::ostringstream error;
        error << description << " file size mismatch: " << path
              << " (expected " << required << " bytes, found " << actual << ')';
        throw std::runtime_error(error.str());
    }
    input.seekg(0);
    std::vector<T> values(count);
    input.read(reinterpret_cast<char *>(values.data()),
               static_cast<std::streamsize>(required));
    if (!input) throw std::runtime_error("failed reading " + std::string(description) + ": " + path);
    return values;
}

size_t checkedElementCount(size_t rows, size_t width, const char *description) {
    if (width != 0 && rows > std::numeric_limits<size_t>::max() / width) {
        throw std::overflow_error(std::string(description) + " element-count overflow");
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

void normalizeQueries(std::vector<float> &queries, size_t count, size_t dim) {
    for (size_t i = 0; i < count; ++i) {
        float *query = queries.data() + i * dim;
        double norm2 = 0.0;
        for (size_t j = 0; j < dim; ++j) {
            if (!std::isfinite(query[j])) {
                throw std::invalid_argument("query contains a non-finite value");
            }
            norm2 += static_cast<double>(query[j]) * query[j];
        }
        if (!(norm2 > 0.0) || !std::isfinite(norm2)) {
            throw std::invalid_argument("query has zero or non-finite norm");
        }
        const double scale = 1.0 / std::sqrt(norm2);
        for (size_t j = 0; j < dim; ++j) {
            query[j] = static_cast<float>(static_cast<double>(query[j]) * scale);
        }
    }
}

class LoadedTangoIndex {
 public:
    LoadedTangoIndex(
            const std::string &prefix,
            size_t dim,
            double query_time,
            const TangoDecayFastPathOptions &fastpath)
        : index_(TANGOIndex::load(prefix, 0, fastpath)),
          query_time_(query_time) {
        if (index_->config().dim != dim) {
            throw std::runtime_error("TANGO index dimension does not match --dim");
        }
        if (std::isfinite(query_time) &&
                query_time < index_->maxTimestamp() -
                    index_->config().future_time_epsilon) {
            throw std::invalid_argument("--query-time is earlier than the TANGO maximum timestamp");
        }
    }

    void setEf(size_t ef) { index_->setEf(ef); }
    void setMaxSearchExpansions(size_t value) { index_->setMaxSearchExpansions(value); }
    void prepareQueries(
            const std::vector<float> &queries, size_t count, size_t dim, double query_time) {
        if (dim != index_->config().dim || queries.size() < count * dim) {
            throw std::invalid_argument("invalid TANGO prepared-query dimensions");
        }
        const size_t record_size = index_->distanceParams().record_size;
        if (count > std::numeric_limits<size_t>::max() / record_size) {
            throw std::overflow_error("TANGO prepared-query storage overflow");
        }
        prepared_records_.resize(count * record_size);
        for (size_t i = 0; i < count; ++i) {
            uint8_t *record = prepared_records_.data() + i * record_size;
            // Normalization was completed before timing; copy the exact semantic
            // coordinates into a packed query record.
            std::memcpy(record, queries.data() + i * dim, dim * sizeof(float));
            hnswlib::writeTangoTimestamp(
                record, index_->distanceParams().timestamp_offset, query_time);
        }
        record_size_ = record_size;
        prepared_count_ = count;
    }
    SearchResult searchPrepared(size_t query_index, size_t k) const {
        if (record_size_ == 0 || query_index >= prepared_count_) {
            throw std::logic_error("TANGO query was not prepared");
        }
        // Deliberately bypass the wrapper here: validation, packing, timestamp
        // writing, and optional normalization were completed before any timed
        // ef/repeat. The timer covers the TANGO search entry. For the QD
        // fastpath this deliberately includes its one query-local scale exp.
        hnswlib::TangoQueryContext query_context =
            index_->makeQueryContext(query_time_);
        return index_->rawIndex().searchKnnWithContext(
            prepared_records_.data() + query_index * record_size_, k,
            &query_context);
    }
    void resetMetrics() {
        index_->rawIndex().metric_distance_computations = 0;
        index_->rawIndex().metric_hops = 0;
        index_->resetDecayFastPathCounters();
    }
    long distanceComputations() const {
        return index_->rawIndex().metric_distance_computations.load();
    }
    long hops() const { return index_->rawIndex().metric_hops.load(); }
    size_t elements() const { return index_->size(); }
    size_t M() const { return index_->rawIndex().M_; }
    size_t efConstruction() const { return index_->rawIndex().ef_construction_; }
    std::string halfLifeField() const { return number(index_->config().half_life); }
    std::string scoreModeField() const { return hnswlib::tangoModeName(index_->config().mode); }
    std::string alphaField() const { return number(index_->config().alpha); }
    std::string dataDataModeField() const {
        return hnswlib::tangoDataDataModeName(index_->config().data_data_mode);
    }
    std::string kappaBaseField() const { return number(index_->config().kappa_base); }
    std::string kappaNavField() const { return number(index_->config().kappa_nav); }
    std::string kappaScheduleField() const {
        return hnswlib::tangoScheduleName(index_->config().schedule);
    }
    std::string kappaLevelsField() const {
        const hnswlib::TangoConfig &config = index_->config();
        return hnswlib::tango_detail::serializeDoubleList(
            config.schedule == hnswlib::KappaSchedule::PerLevel
                ? config.kappa_levels
                : std::vector<double>{config.kappa_base, config.kappa_nav});
    }
    std::string decayFastPathField() const {
        return hnswlib::tangoDecayFastPathName(index_->decayFastPathMode());
    }
    std::string decayReferenceTimeField() const {
        return number(index_->decayReferenceTime());
    }
    std::string decayCachePrecisionField() const { return "float64"; }
    std::string decayCacheInverseField() const {
        return index_->decayCacheStoresInverse() ? "1" : "0";
    }
    TangoDecayFastPathStats decayStats() const {
        return index_->decayFastPathStats();
    }

 private:
    static std::string number(double value) {
        std::ostringstream output;
        output << std::setprecision(17) << value;
        return output.str();
    }
    std::unique_ptr<TANGOIndex> index_;
    std::vector<uint8_t> prepared_records_;
    size_t record_size_ = 0;
    size_t prepared_count_ = 0;
    double query_time_;
};

double mean(const std::vector<double> &values) {
    if (values.empty()) return 0.0;
    double total = 0.0;
    for (size_t i = 0; i < values.size(); ++i) total += values[i];
    return total / values.size();
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

std::string csvQuote(const std::string &value) {
    std::string result = "\"";
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '"') result += '"';
        result += value[i];
    }
    result += '"';
    return result;
}

struct RoundStats {
    std::vector<double> latencies;
    std::vector<uint64_t> result_ids;
    size_t correct = 0;
    long distance_computations = 0;
    long hops = 0;
    TangoDecayFastPathStats decay;
};

size_t countRecallHits(
        const labeltype *result_ids,
        const int64_t *ground_truth,
        size_t k) {
    size_t hits = 0;
    for (size_t i = 0; i < k; ++i) {
        const labeltype label = result_ids[i];
        for (size_t j = 0; j < k; ++j) {
            if (ground_truth[j] >= 0 && label == static_cast<labeltype>(ground_truth[j])) {
                ++hits;
                break;
            }
        }
    }
    return hits;
}

RoundStats runTimedRound(
        LoadedTangoIndex &index,
        const Options &options,
        const std::vector<int64_t> &ground_truth,
        bool capture_result_ids) {
    RoundStats stats;
    stats.latencies.reserve(options.queries);
    if (capture_result_ids) stats.result_ids.reserve(options.queries * options.k);
    // Reuse one output buffer so the timed interval includes exactly the
    // TANGO search and top-k materialization, but not allocator overhead.
    std::vector<labeltype> result_ids(options.k);
    index.resetMetrics();
    for (size_t qi = 0; qi < options.queries; ++qi) {
        const Clock::time_point start = Clock::now();
        SearchResult result = index.searchPrepared(qi, options.k);
        if (result.size() != options.k) {
            throw std::runtime_error("index returned fewer than k results");
        }
        // HNSW's heap exposes the worst retained item first.  Materialize the
        // output in best-first order inside the common timed scope.
        for (size_t position = options.k; position > 0; --position) {
            result_ids[position - 1] = result.top().second;
            result.pop();
        }
        const Clock::time_point finish = Clock::now();
        stats.latencies.push_back(
            std::chrono::duration<double, std::milli>(finish - start).count());

        // Recall remains outside the timed scope.
        stats.correct += countRecallHits(
            result_ids.data(), ground_truth.data() + qi * options.gt_width, options.k);
        if (capture_result_ids) {
            for (size_t position = 0; position < options.k; ++position) {
                stats.result_ids.push_back(static_cast<uint64_t>(result_ids[position]));
            }
        }
    }
    stats.distance_computations = index.distanceComputations();
    stats.hops = index.hops();
    stats.decay = index.decayStats();
    return stats;
}

std::string jsonEscape(const std::string &value) {
    std::string escaped;
    for (size_t i = 0; i < value.size(); ++i) {
        const char c = value[i];
        if (c == '\\' || c == '"') escaped += '\\';
        if (c == '\n') escaped += "\\n";
        else if (c == '\r') escaped += "\\r";
        else if (c == '\t') escaped += "\\t";
        else escaped += c;
    }
    return escaped;
}

void writeResultIds(
        const Options &options,
        const std::vector<uint64_t> &result_ids) {
    if (options.result_ids_output.empty()) return;
    const size_t expected = options.ef_list.size() * options.queries * options.k;
    if (result_ids.size() != expected) {
        throw std::logic_error("captured result ID count does not match ef/query/k shape");
    }
    std::ofstream output(
        options.result_ids_output.c_str(), std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "cannot create result ID output: " + options.result_ids_output);
    }
    output.write(
        reinterpret_cast<const char *>(result_ids.data()),
        static_cast<std::streamsize>(result_ids.size() * sizeof(uint64_t)));
    output.close();
    if (!output) {
        throw std::runtime_error(
            "failed writing result ID output: " + options.result_ids_output);
    }

    std::ofstream manifest((options.result_ids_output + ".json").c_str());
    if (!manifest) {
        throw std::runtime_error("cannot create result ID manifest");
    }
    manifest << "{\n"
             << "  \"schema\": \"chronos_tango_query_result_ids_v1\",\n"
             << "  \"index_prefix\": \"" << jsonEscape(options.index_path) << "\",\n"
             << "  \"query_file\": \"" << jsonEscape(options.query_path) << "\",\n"
             << "  \"dtype\": \"uint64\", \"byte_order\": \"native_little_endian\",\n"
             << "  \"layout\": \"ef_query_rank\", \"queries\": " << options.queries
             << ", \"k\": " << options.k << ",\n"
             << "  \"ef_values\": [";
    for (size_t i = 0; i < options.ef_list.size(); ++i) {
        if (i != 0) manifest << ',';
        manifest << options.ef_list[i];
    }
    manifest << "],\n"
             << "  \"ids_file\": \"" << jsonEscape(options.result_ids_output)
             << "\", \"elements\": " << result_ids.size() << "\n}\n";
    manifest.close();
    if (!manifest) throw std::runtime_error("failed writing result ID manifest");
}

void emitHeader(std::ostream &output) {
    output
        << "run_label,method,index_path,query_threads,query_normalized,clock,timed_scope,"
           "index_elements,index_M,index_ef_construction,score_mode,alpha,data_data_mode,"
           "half_life,kappa_base,kappa_nav,"
           "kappa_schedule,kappa_levels,tango_decay_fastpath,decay_instrumentation,"
           "decay_reference_time,"
           "decay_cache_precision,decay_cache_inverse,decay_cache_active,"
           "decay_cache_init_ms,decay_cache_bytes,"
           "query_time,dim,queries,gt_width,k,ef,max_search_expansions,warmup_queries,"
           "configured_repeats,row_type,repeat_index,samples,recall,qps,mean_ms,p50_ms,"
           "p95_ms,p99_ms,avg_distance_computations,avg_hops,qd_callbacks,dd_callbacks,"
           "direct_qd_exp,direct_dd_exp,query_scale_exp,fast_qd,fast_dd,decay_fallbacks,"
           "verify_qd_max_abs,verify_qd_max_rel,verify_dd_max_abs,verify_dd_max_rel\n";
}

void emitRow(
        std::ostream &output,
        const Options &options,
        const LoadedTangoIndex &index,
        size_t ef,
        const char *row_type,
        size_t repeat_index,
        const RoundStats &stats) {
    const double mean_ms = mean(stats.latencies);
    const double recall = static_cast<double>(stats.correct) /
                          (stats.latencies.size() * options.k);
    const double query_count = static_cast<double>(stats.latencies.size());
    output << std::setprecision(12)
           << csvQuote(options.run_label) << ",tango,"
           << csvQuote(options.index_path)
           << ",1,1,steady_clock,serial_search_and_topk_materialization,"
           << index.elements() << ',' << index.M() << ',' << index.efConstruction() << ','
           << index.scoreModeField() << ',' << index.alphaField() << ','
           << index.dataDataModeField() << ','
           << index.halfLifeField() << ',' << index.kappaBaseField() << ','
           << index.kappaNavField() << ',' << index.kappaScheduleField() << ','
           << csvQuote(index.kappaLevelsField()) << ','
           << index.decayFastPathField() << ','
           << (options.decay_instrumentation ? 1 : 0) << ','
           << index.decayReferenceTimeField() << ','
           << index.decayCachePrecisionField() << ','
           << index.decayCacheInverseField() << ','
           << stats.decay.active_entries << ',' << stats.decay.cache_init_ms << ','
           << stats.decay.cache_bytes << ',';
    if (std::isfinite(options.query_time)) output << options.query_time;
    output << ',' << options.dim << ',' << options.queries << ',' << options.gt_width << ','
           << options.k << ',' << ef << ',' << options.max_search_expansions << ','
           << options.warmup_queries << ',' << options.repeats << ',' << row_type << ','
           << repeat_index << ',' << stats.latencies.size() << ',' << recall << ','
           << (mean_ms > 0.0 ? 1000.0 / mean_ms : 0.0) << ',' << mean_ms << ','
           << percentile(stats.latencies, 0.50) << ','
           << percentile(stats.latencies, 0.95) << ','
           << percentile(stats.latencies, 0.99) << ','
           << (query_count > 0.0 ? stats.distance_computations / query_count : 0.0) << ','
           << (query_count > 0.0 ? stats.hops / query_count : 0.0) << ','
           << stats.decay.qd_callbacks << ',' << stats.decay.dd_callbacks << ','
           << stats.decay.direct_qd_exp << ',' << stats.decay.direct_dd_exp << ','
           << stats.decay.query_scale_exp << ',' << stats.decay.fast_qd << ','
           << stats.decay.fast_dd << ',' << stats.decay.fallbacks << ','
           << stats.decay.verify_qd_max_abs << ',' << stats.decay.verify_qd_max_rel << ','
           << stats.decay.verify_dd_max_abs << ',' << stats.decay.verify_dd_max_rel << '\n';
}

std::unique_ptr<LoadedTangoIndex> loadIndex(const Options &options) {
    TangoDecayFastPathOptions fastpath;
    fastpath.mode = hnswlib::parseTangoDecayFastPath(
        options.tango_decay_fastpath);
    fastpath.verify_samples = options.decay_verify_samples;
    fastpath.instrumentation = options.decay_instrumentation;
    return std::unique_ptr<LoadedTangoIndex>(new LoadedTangoIndex(
        options.index_path, options.dim, options.query_time, fastpath));
}

struct ServingArtifact {
    std::string files;
    uint64_t serialized_index_bytes = 0;
    uint64_t external_base_bytes = 0;
};

ServingArtifact servingArtifact(const Options &options) {
    ServingArtifact result;
    const std::string hnsw_path = options.index_path + ".hnsw";
    const std::string meta_path = options.index_path + ".tango.meta";
    const uint64_t hnsw_bytes = fileSize(hnsw_path);
    const uint64_t meta_bytes = fileSize(meta_path);
    if (hnsw_bytes > std::numeric_limits<uint64_t>::max() - meta_bytes) {
        throw std::overflow_error("TANGO serialized artifact size overflow");
    }
    result.files = hnsw_path + ";" + meta_path;
    result.serialized_index_bytes = hnsw_bytes + meta_bytes;
    result.external_base_bytes = 0;
    return result;
}

std::string footprintHeader() {
    return
        "run_label,method,index_path,footprint_scope,rss_measurement_supported,"
        "index_contains_base,serving_artifact_files,serialized_index_bytes,"
        "external_base_bytes,serving_artifact_bytes,rss_baseline_bytes,"
        "rss_after_load_bytes,rss_delta_bytes,peak_rss_bytes,index_elements,"
        "index_M,index_ef_construction,score_mode,alpha,data_data_mode,"
        "half_life,kappa_base,kappa_nav,"
        "kappa_schedule,kappa_levels,tango_decay_fastpath,"
        "decay_reference_time,decay_cache_precision,decay_cache_inverse,"
        "decay_cache_active,decay_cache_init_ms,decay_cache_known_bytes";
}

int runFootprint(const Options &options) {
    // This mode is intentionally entered before query/GT loading. Since each
    // invocation is a new process, the delta excludes all query and
    // recall buffers and captures the serving index load only.
    const uint64_t rss_baseline = currentRssBytes();
    std::unique_ptr<LoadedTangoIndex> index = loadIndex(options);
    const uint64_t rss_after_load = currentRssBytes();
    const uint64_t peak_rss = peakRssBytes();
    const ServingArtifact artifact = servingArtifact(options);
    const TangoDecayFastPathStats decay = index->decayStats();

    if (artifact.serialized_index_bytes >
            std::numeric_limits<uint64_t>::max() - artifact.external_base_bytes) {
        throw std::overflow_error("serving artifact size overflow");
    }
    const uint64_t serving_artifact_bytes =
        artifact.serialized_index_bytes + artifact.external_base_bytes;

    std::unique_ptr<std::ofstream> output_file;
    std::ostream *output = &std::cout;
    bool need_header = true;
    const std::string expected_header = footprintHeader();
    if (!options.output_csv.empty()) {
        const uint64_t existing_size = fileSize(options.output_csv);
        need_header = options.overwrite_output || existing_size == 0;
        if (!need_header) {
            std::ifstream existing(options.output_csv.c_str());
            std::string actual_header;
            std::getline(existing, actual_header);
            if (actual_header != expected_header) {
                throw std::runtime_error(
                    "footprint CSV schema mismatch; use a separate file or --overwrite-output");
            }
        }
        const std::ios_base::openmode mode = options.overwrite_output
            ? (std::ios::out | std::ios::trunc)
            : (std::ios::out | std::ios::app);
        output_file.reset(new std::ofstream(options.output_csv.c_str(), mode));
        if (!*output_file) {
            throw std::runtime_error("cannot open output CSV: " + options.output_csv);
        }
        output = output_file.get();
    }
    if (need_header) *output << expected_header << '\n';

    *output << std::setprecision(12)
            << csvQuote(options.run_label) << ",tango,"
            << csvQuote(options.index_path) << ','
            << "fresh_process_load_only_no_query_or_gt,"
            << ((rss_baseline != 0 && rss_after_load != 0) ? 1 : 0) << ','
            << "1," << csvQuote(artifact.files) << ','
            << artifact.serialized_index_bytes << ','
            << artifact.external_base_bytes << ','
            << serving_artifact_bytes << ','
            << rss_baseline << ',' << rss_after_load << ','
            << signedByteDelta(rss_after_load, rss_baseline) << ','
            << peak_rss << ',' << index->elements() << ',' << index->M() << ','
            << index->efConstruction() << ',' << index->scoreModeField() << ','
            << index->alphaField() << ',' << index->dataDataModeField() << ','
            << index->halfLifeField() << ','
            << index->kappaBaseField() << ',' << index->kappaNavField() << ','
            << index->kappaScheduleField() << ','
            << csvQuote(index->kappaLevelsField()) << ','
            << index->decayFastPathField() << ','
            << index->decayReferenceTimeField() << ','
            << index->decayCachePrecisionField() << ','
            << index->decayCacheInverseField() << ','
            << decay.active_entries << ',' << decay.cache_init_ms << ','
            << decay.cache_bytes << '\n';

    std::cerr << "Footprint loaded TANGO"
              << ": serialized_index_bytes=" << artifact.serialized_index_bytes
              << " external_base_bytes=0 serving_artifact_bytes="
              << serving_artifact_bytes << " rss_delta_bytes="
              << signedByteDelta(rss_after_load, rss_baseline)
              << " peak_rss_bytes=" << peak_rss
              << " decay_cache_known_bytes=" << decay.cache_bytes << '\n';
    return 0;
}

int run(int argc, char **argv) {
    const Options options = parseOptions(argc, argv);
    if (options.footprint_only) return runFootprint(options);

    std::vector<float> queries = loadExact<float>(
        options.query_path,
        checkedElementCount(options.queries, options.dim, "query"), "query");
    normalizeQueries(queries, options.queries, options.dim);
    const std::vector<int64_t> ground_truth = loadExact<int64_t>(
        options.gt_path,
        checkedElementCount(options.queries, options.gt_width, "ground truth"),
        "ground truth");

    std::unique_ptr<LoadedTangoIndex> index = loadIndex(options);
    if (options.k > index->elements()) {
        throw std::invalid_argument("k exceeds the number of indexed elements");
    }
    validateGroundTruthIds(
        ground_truth, options.queries, options.gt_width, index->elements());
    index->setMaxSearchExpansions(options.max_search_expansions);
    // Materialize packed query records before every ef/repeat, keeping wrapper
    // validation and normalization outside the graph-search timing interval.
    index->prepareQueries(queries, options.queries, options.dim, options.query_time);

    std::unique_ptr<std::ofstream> output_file;
    std::ostream *output = &std::cout;
    bool need_header = true;
    if (!options.output_csv.empty()) {
        need_header = options.overwrite_output || fileSize(options.output_csv) == 0;
        const std::ios_base::openmode mode = options.overwrite_output
            ? (std::ios::out | std::ios::trunc)
            : (std::ios::out | std::ios::app);
        output_file.reset(new std::ofstream(options.output_csv.c_str(), mode));
        if (!*output_file) {
            throw std::runtime_error("cannot open output CSV: " + options.output_csv);
        }
        output = output_file.get();
    }
    if (need_header) emitHeader(*output);

    std::cerr << "Loaded TANGO index with " << index->elements()
              << " elements; serial query_threads=1, normalized_queries=1, repeats="
              << options.repeats
              << ", warmup_queries=0, timed_all_queries=1; decay_fastpath="
              << index->decayFastPathField();
    {
        const TangoDecayFastPathStats cache = index->decayStats();
        std::cerr << " T0=" << index->decayReferenceTimeField()
                  << " precision=" << index->decayCachePrecisionField()
                  << " inverse=" << index->decayCacheInverseField()
                  << " active=" << cache.active_entries
                  << " cache_init_ms=" << cache.cache_init_ms
                  << " cache_bytes=" << cache.cache_bytes;
    }
    std::cerr << "\n";
    if (options.tango_decay_fastpath == "dd") {
        std::cerr << "TANGO decay fastpath: dd affects later insert/update/repair; "
                     "this static query run still uses direct QD decay.\n";
    }

    std::vector<uint64_t> captured_result_ids;
    if (!options.result_ids_output.empty()) {
        captured_result_ids.reserve(
            options.ef_list.size() * options.queries * options.k);
    }
    for (size_t ei = 0; ei < options.ef_list.size(); ++ei) {
        const size_t ef = options.ef_list[ei];
        index->setEf(ef);

        RoundStats aggregate;
        aggregate.latencies.reserve(options.queries * options.repeats);
        for (size_t repeat = 1; repeat <= options.repeats; ++repeat) {
            const RoundStats current = runTimedRound(
                *index, options, ground_truth,
                repeat == 1 && !options.result_ids_output.empty());
            if (repeat == 1 && !options.result_ids_output.empty()) {
                captured_result_ids.insert(
                    captured_result_ids.end(),
                    current.result_ids.begin(), current.result_ids.end());
            }
            emitRow(*output, options, *index, ef, "repeat", repeat, current);
            aggregate.latencies.insert(
                aggregate.latencies.end(), current.latencies.begin(), current.latencies.end());
            aggregate.correct += current.correct;
            aggregate.distance_computations += current.distance_computations;
            aggregate.hops += current.hops;
            aggregate.decay.qd_callbacks += current.decay.qd_callbacks;
            aggregate.decay.dd_callbacks += current.decay.dd_callbacks;
            aggregate.decay.direct_qd_exp += current.decay.direct_qd_exp;
            aggregate.decay.direct_dd_exp += current.decay.direct_dd_exp;
            aggregate.decay.query_scale_exp += current.decay.query_scale_exp;
            aggregate.decay.fast_qd += current.decay.fast_qd;
            aggregate.decay.fast_dd += current.decay.fast_dd;
            aggregate.decay.fallbacks += current.decay.fallbacks;
            aggregate.decay.active_entries = current.decay.active_entries;
            aggregate.decay.cache_bytes = current.decay.cache_bytes;
            aggregate.decay.cache_init_ms = current.decay.cache_init_ms;
            aggregate.decay.verify_qd_max_abs = std::max(
                aggregate.decay.verify_qd_max_abs,
                current.decay.verify_qd_max_abs);
            aggregate.decay.verify_qd_max_rel = std::max(
                aggregate.decay.verify_qd_max_rel,
                current.decay.verify_qd_max_rel);
            aggregate.decay.verify_dd_max_abs = std::max(
                aggregate.decay.verify_dd_max_abs,
                current.decay.verify_dd_max_abs);
            aggregate.decay.verify_dd_max_rel = std::max(
                aggregate.decay.verify_dd_max_rel,
                current.decay.verify_dd_max_rel);
        }
        emitRow(*output, options, *index, ef, "summary", 0, aggregate);
        if (output != &std::cout) {
            const double recall = static_cast<double>(aggregate.correct) /
                                  (aggregate.latencies.size() * options.k);
            std::cerr << "ef=" << ef << " recall=" << recall
                      << " mean_ms=" << mean(aggregate.latencies) << '\n';
        }
    }
    writeResultIds(options, captured_result_ids);
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
