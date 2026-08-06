// Dynamic-insertion evaluation for timestamp-ordered TANGO workloads.
#include "tango.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using hnswlib::KappaSchedule;
using hnswlib::TANGOIndex;
using hnswlib::TangoConfig;
using hnswlib::TangoDataDataMode;
using hnswlib::TangoDecayFastPathOptions;
using hnswlib::TangoMode;
using hnswlib::labeltype;

struct Options {
    std::string base_path;
    std::string timestamps_path;
    std::string query_path;
    std::string output_dir;
    std::string final_index_prefix;
    std::string mode = "multiplicative";
    std::string data_data_mode = "timelift";
    std::string fastpath = "all";
    size_t n = 0;
    size_t queries = 0;
    size_t dim = 0;
    size_t gt_width = 100;
    std::vector<size_t> k_values{10};
    size_t M = 16;
    size_t ef_construction = 100;
    size_t build_threads = 64;
    size_t gt_threads = 64;
    size_t repeats = 1;
    size_t random_seed = 100;
    size_t initial_percent = 50;
    size_t batch_percent = 5;
    size_t max_search_expansions = 0;
    double query_time = std::numeric_limits<double>::quiet_NaN();
    double half_life = std::numeric_limits<double>::quiet_NaN();
    double alpha = 1.0;
    double kappa_base = 1.0 / 63.0;
    double kappa_nav = 1.0 / 15.0;
    double stop_recall = std::numeric_limits<double>::quiet_NaN();
    size_t min_ef_points = 0;
    std::vector<size_t> ef_list{10, 50, 100};
};

void usage(const char *name) {
    std::cerr
        << "Usage: " << name << " --base FILE --timestamps FILE --query FILE\n"
        << "  --output-dir DIR --n N --queries Q --dim D --mode multiplicative|additive\n"
        << "  --half-life H --query-time TAU [options]\n\n"
        << "Options:\n"
        << "  --alpha A                    additive semantic weight (default 1)\n"
        << "  --gt-width W                 exact checkpoint GT width (default 100)\n"
        << "  --k K                        one reported Recall@K (default 10)\n"
        << "  --k-values CSV               multiple Recall@K values on one insertion trajectory\n"
        << "  --ef-list CSV                query ef values (default 10,50,100)\n"
        << "  --M N --ef-construction N    graph parameters (defaults 16,100)\n"
        << "  --build-threads N            initial active-set construction threads (default 64)\n"
        << "  --gt-threads N               exact-GT preprocessing threads (default 64;\n"
        << "                               effective value is 1 without OpenMP)\n"
        << "  --repeats N                  timed query passes; warmup is always 0 (default 1)\n"
        << "  --initial-percent P          oldest initial active-set percentage (default 50)\n"
        << "  --batch-percent P            single-threaded online batch percentage (default 5)\n"
        << "  --max-search-expansions N    base-layer query cap; 0 disables it (default 0)\n"
        << "  --stop-recall R              stop an ef scan after reaching Recall@K R\n"
        << "  --min-ef-points N            minimum ef points before --stop-recall applies\n"
        << "  --kappa-base X --kappa-nav Y two-tier TANGO schedule\n"
        << "                               (defaults 1/63 and 1/15)\n"
        << "  --dd-mode timelift|semantic  construction distance (default timelift)\n"
        << "  --tango-decay-fastpath M     off|qd|dd|all (default all)\n"
        << "  --random-seed N              HNSW seed (default 100)\n"
        << "  --final-index-prefix PREFIX  optionally persist the final checkpoint\n\n"
        << "The arrival order is stable (timestamp, row_id). The initial and batch\n"
        << "percentages must form exact row counts and finish at 100%. Queries are serial.\n";
}

std::string valueAfter(int &i, int argc, char **argv) {
    if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value after ") + argv[i]);
    return argv[++i];
}

size_t parseSize(const std::string &text, const char *name, bool zero_ok = false) {
    char *end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (text.empty() || text[0] == '-' || errno != 0 || end == text.c_str() || *end != '\0' ||
            (!zero_ok && value == 0) || value > std::numeric_limits<size_t>::max()) {
        throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
    }
    return static_cast<size_t>(value);
}

double parseDouble(const std::string &text, const char *name) {
    char *end = nullptr;
    errno = 0;
    const double value = std::strtod(text.c_str(), &end);
    if (errno != 0 || end == text.c_str() || *end != '\0' || !std::isfinite(value)) {
        throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
    }
    return value;
}

std::vector<size_t> parseSizes(const std::string &text, const char *name) {
    std::vector<size_t> result;
    std::stringstream input(text);
    std::string item;
    while (std::getline(input, item, ',')) result.push_back(parseSize(item, name));
    if (result.empty()) {
        throw std::invalid_argument(std::string("--") + name + " must not be empty");
    }
    std::sort(result.begin(), result.end());
    if (std::adjacent_find(result.begin(), result.end()) != result.end()) {
        throw std::invalid_argument(std::string("--") + name + " contains duplicates");
    }
    return result;
}

Options parseOptions(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string flag(argv[i]);
        if (flag == "--help" || flag == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else if (flag == "--base") o.base_path = valueAfter(i, argc, argv);
        else if (flag == "--timestamps") o.timestamps_path = valueAfter(i, argc, argv);
        else if (flag == "--query") o.query_path = valueAfter(i, argc, argv);
        else if (flag == "--output-dir") o.output_dir = valueAfter(i, argc, argv);
        else if (flag == "--final-index-prefix") o.final_index_prefix = valueAfter(i, argc, argv);
        else if (flag == "--mode") o.mode = valueAfter(i, argc, argv);
        else if (flag == "--dd-mode") o.data_data_mode = valueAfter(i, argc, argv);
        else if (flag == "--tango-decay-fastpath") o.fastpath = valueAfter(i, argc, argv);
        else if (flag == "--n") o.n = parseSize(valueAfter(i, argc, argv), "n");
        else if (flag == "--queries") o.queries = parseSize(valueAfter(i, argc, argv), "queries");
        else if (flag == "--dim") o.dim = parseSize(valueAfter(i, argc, argv), "dim");
        else if (flag == "--gt-width") o.gt_width = parseSize(valueAfter(i, argc, argv), "gt-width");
        else if (flag == "--k") {
            o.k_values.assign(1, parseSize(valueAfter(i, argc, argv), "k"));
        }
        else if (flag == "--k-values") {
            o.k_values = parseSizes(valueAfter(i, argc, argv), "k-values");
        }
        else if (flag == "--M") o.M = parseSize(valueAfter(i, argc, argv), "M");
        else if (flag == "--ef-construction") o.ef_construction = parseSize(valueAfter(i, argc, argv), "ef-construction");
        else if (flag == "--build-threads") o.build_threads = parseSize(valueAfter(i, argc, argv), "build-threads");
        else if (flag == "--gt-threads") o.gt_threads = parseSize(valueAfter(i, argc, argv), "gt-threads");
        else if (flag == "--repeats") o.repeats = parseSize(valueAfter(i, argc, argv), "repeats");
        else if (flag == "--random-seed") o.random_seed = parseSize(valueAfter(i, argc, argv), "random-seed", true);
        else if (flag == "--initial-percent") o.initial_percent = parseSize(valueAfter(i, argc, argv), "initial-percent");
        else if (flag == "--batch-percent") o.batch_percent = parseSize(valueAfter(i, argc, argv), "batch-percent");
        else if (flag == "--max-search-expansions") o.max_search_expansions = parseSize(valueAfter(i, argc, argv), "max-search-expansions", true);
        else if (flag == "--stop-recall") o.stop_recall = parseDouble(valueAfter(i, argc, argv), "stop-recall");
        else if (flag == "--min-ef-points") o.min_ef_points = parseSize(valueAfter(i, argc, argv), "min-ef-points", true);
        else if (flag == "--query-time") o.query_time = parseDouble(valueAfter(i, argc, argv), "query-time");
        else if (flag == "--half-life") o.half_life = parseDouble(valueAfter(i, argc, argv), "half-life");
        else if (flag == "--alpha") o.alpha = parseDouble(valueAfter(i, argc, argv), "alpha");
        else if (flag == "--kappa-base") o.kappa_base = parseDouble(valueAfter(i, argc, argv), "kappa-base");
        else if (flag == "--kappa-nav") o.kappa_nav = parseDouble(valueAfter(i, argc, argv), "kappa-nav");
        else if (flag == "--ef-list") o.ef_list = parseSizes(valueAfter(i, argc, argv), "ef-list");
        else throw std::invalid_argument("unknown argument: " + flag);
    }
    if (o.base_path.empty() || o.timestamps_path.empty() || o.query_path.empty() ||
            o.output_dir.empty() || o.n == 0 || o.queries == 0 || o.dim == 0 ||
            !std::isfinite(o.half_life) || !std::isfinite(o.query_time)) {
        throw std::invalid_argument("base, timestamps, query, output-dir, n, queries, dim, half-life, and query-time are required");
    }
    if (o.initial_percent >= 100 || o.batch_percent >= 100 ||
            (100 - o.initial_percent) % o.batch_percent != 0) {
        throw std::invalid_argument(
            "initial-percent and batch-percent must produce equal batches ending at 100%");
    }
    const size_t initial_divisor = 100 / std::gcd<size_t>(100, o.initial_percent);
    const size_t batch_divisor = 100 / std::gcd<size_t>(100, o.batch_percent);
    if (o.n % initial_divisor != 0 || o.n % batch_divisor != 0) {
        throw std::invalid_argument(
            "initial-percent and batch-percent must map to exact integer row counts");
    }
    const size_t initial_size = (o.n / initial_divisor) *
        (o.initial_percent / std::gcd<size_t>(100, o.initial_percent));
    if (initial_size < o.gt_width ||
            *std::max_element(o.k_values.begin(), o.k_values.end()) > o.gt_width) {
        throw std::invalid_argument(
            "initial active set must cover gt-width and every k must be <= gt-width");
    }
    if (o.build_threads > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            o.gt_threads > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("thread counts exceed the supported range");
    }
    if (o.mode != "multiplicative" && o.mode != "additive") {
        throw std::invalid_argument("--mode must be multiplicative or additive");
    }
    if (o.mode == "multiplicative" && std::fabs(o.alpha - 1.0) > 1e-12) {
        throw std::invalid_argument("multiplicative mode requires --alpha 1");
    }
    if (o.mode == "additive" && (o.alpha < 0.0 || o.alpha > 1.0)) {
        throw std::invalid_argument("additive alpha must lie in [0,1]");
    }
    if (o.data_data_mode != "timelift" && o.data_data_mode != "semantic") {
        throw std::invalid_argument("--dd-mode must be timelift or semantic");
    }
    if (std::isfinite(o.stop_recall) && (o.stop_recall <= 0.0 || o.stop_recall > 1.0)) {
        throw std::invalid_argument("--stop-recall must lie in (0,1]");
    }
    if (!std::isfinite(o.stop_recall) && o.min_ef_points != 0) {
        throw std::invalid_argument("--min-ef-points requires --stop-recall");
    }
    if (o.min_ef_points > o.ef_list.size()) {
        throw std::invalid_argument("--min-ef-points exceeds the ef-list length");
    }
    (void)hnswlib::parseTangoDecayFastPath(o.fastpath);
    for (size_t k : o.k_values) {
        if (std::none_of(o.ef_list.begin(), o.ef_list.end(),
                        [k](size_t ef) { return ef >= k; })) {
            throw std::invalid_argument("each k must have at least one ef >= k");
        }
    }
    return o;
}

template <typename T>
std::vector<T> loadBinary(const std::string &path, size_t count, const char *name) {
    std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error(std::string("cannot open ") + name + ": " + path);
    const std::streamoff bytes = input.tellg();
    const uint64_t expected = static_cast<uint64_t>(count) * sizeof(T);
    if (bytes < 0 || static_cast<uint64_t>(bytes) != expected) {
        throw std::runtime_error(std::string(name) + " byte size does not match the declared shape");
    }
    input.seekg(0);
    std::vector<T> values(count);
    input.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(expected));
    if (!input) throw std::runtime_error(std::string("failed reading ") + name);
    return values;
}

template <typename T>
void writeBinary(const std::filesystem::path &path, const std::vector<T> &values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create " + path.string());
    output.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!output) throw std::runtime_error("failed writing " + path.string());
}

void normalizeRows(std::vector<float> &values, size_t rows, size_t dim) {
    std::atomic<bool> invalid(false);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(rows); ++row) {
        float *x = values.data() + static_cast<size_t>(row) * dim;
        double norm2 = 0.0;
        for (size_t d = 0; d < dim; ++d) norm2 += static_cast<double>(x[d]) * x[d];
        if (!std::isfinite(norm2) || norm2 <= 0.0) {
            invalid.store(true, std::memory_order_relaxed);
            continue;
        }
        const double inverse = 1.0 / std::sqrt(norm2);
        for (size_t d = 0; d < dim; ++d) x[d] = static_cast<float>(x[d] * inverse);
    }
    if (invalid.load(std::memory_order_relaxed)) {
        throw std::runtime_error("zero or non-finite semantic vector");
    }
}

template <typename Function>
void parallelFor(size_t begin, size_t end, size_t threads, const Function &function) {
    if (begin >= end) return;
    const size_t count = end - begin;
    threads = std::min(threads, count);
    if (threads <= 1) {
        for (size_t i = begin; i < end; ++i) function(i);
        return;
    }
    std::atomic<size_t> next(begin);
    std::exception_ptr failure;
    std::mutex failure_mutex;
    std::vector<std::thread> workers;
    for (size_t t = 0; t < threads; ++t) {
        workers.emplace_back([&]() {
            try {
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= end) break;
                    function(i);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(failure_mutex);
                if (!failure) failure = std::current_exception();
                next.store(end);
            }
        });
    }
    for (std::thread &worker : workers) worker.join();
    if (failure) std::rethrow_exception(failure);
}

struct ScoredId {
    double score;
    uint64_t id;
};

struct BetterFirst {
    bool operator()(const ScoredId &left, const ScoredId &right) const {
        if (left.score != right.score) return left.score > right.score;
        return left.id < right.id;
    }
};

using TopHeap = std::priority_queue<ScoredId, std::vector<ScoredId>, BetterFirst>;

bool better(const ScoredId &a, const ScoredId &b) {
    return a.score > b.score || (a.score == b.score && a.id < b.id);
}

void addTop(TopHeap &heap, size_t width, const ScoredId &candidate) {
    if (heap.size() < width) heap.push(candidate);
    else if (better(candidate, heap.top())) {
        heap.pop();
        heap.push(candidate);
    }
}

double dot(const float *a, const float *b, size_t dim) {
    double result = 0.0;
    for (size_t d = 0; d < dim; ++d) result += static_cast<double>(a[d]) * b[d];
    return result;
}

std::vector<int64_t> updateGroundTruth(
        std::vector<TopHeap> &heaps,
        const std::vector<float> &queries,
        const std::vector<float> &base,
        const std::vector<float> &timestamps,
        const std::vector<uint64_t> &arrival,
        size_t begin,
        size_t end,
        const Options &o) {
    const double lambda = std::log(2.0) / o.half_life;
#ifdef _OPENMP
    omp_set_num_threads(static_cast<int>(o.gt_threads));
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (std::ptrdiff_t qi_signed = 0;
            qi_signed < static_cast<std::ptrdiff_t>(o.queries); ++qi_signed) {
        const size_t qi = static_cast<size_t>(qi_signed);
        TopHeap &heap = heaps[qi];
        const float *query = queries.data() + qi * o.dim;
        for (size_t position = begin; position < end; ++position) {
            const uint64_t id = arrival[position];
            const double freshness = std::exp(-lambda *
                (o.query_time - static_cast<double>(timestamps[id])));
            const double semantic = dot(query, base.data() + id * o.dim, o.dim);
            const double score = o.mode == "multiplicative"
                ? semantic * freshness
                : o.alpha * semantic + (1.0 - o.alpha) * freshness;
            addTop(heap, o.gt_width, ScoredId{score, id});
        }
    }

    std::vector<int64_t> ids(o.queries * o.gt_width);
    for (size_t qi = 0; qi < o.queries; ++qi) {
        TopHeap copy = heaps[qi];
        std::vector<ScoredId> ordered;
        while (!copy.empty()) {
            ordered.push_back(copy.top());
            copy.pop();
        }
        std::sort(ordered.begin(), ordered.end(), better);
        if (ordered.size() != o.gt_width) throw std::logic_error("checkpoint GT has wrong width");
        for (size_t j = 0; j < o.gt_width; ++j) {
            ids[qi * o.gt_width + j] = static_cast<int64_t>(ordered[j].id);
        }
    }
    return ids;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    const double part = position - lower;
    return values[lower] * (1.0 - part) + values[upper] * part;
}

double mean(const std::vector<double> &values) {
    double sum = 0.0;
    for (double value : values) sum += value;
    return values.empty() ? 0.0 : sum / values.size();
}

double gini(std::vector<size_t> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    long double sum = 0.0;
    long double weighted = 0.0;
    const long double n = values.size();
    for (size_t i = 0; i < values.size(); ++i) {
        sum += values[i];
        weighted += (2.0L * (i + 1) - n - 1.0L) * values[i];
    }
    return sum == 0.0 ? 0.0 : static_cast<double>(weighted / (n * sum));
}

std::string jsonEscape(const std::string &value) {
    std::string escaped;
    for (char c : value) {
        if (c == '\\' || c == '"') escaped += '\\';
        if (c == '\n') escaped += "\\n";
        else if (c == '\r') escaped += "\\r";
        else if (c == '\t') escaped += "\\t";
        else escaped += c;
    }
    return escaped;
}

struct GraphStats {
    double avg_outdegree = 0.0;
    double indegree_p95 = 0.0;
    double indegree_p99 = 0.0;
    size_t indegree_max = 0;
    double indegree_gini = 0.0;
};

GraphStats graphStats(const TANGOIndex &index) {
    const hnswlib::HierarchicalNSW<float> &raw = index.rawIndex();
    const size_t n = raw.getCurrentElementCount();
    std::vector<size_t> indegree(n, 0);
    size_t edges = 0;
    for (size_t i = 0; i < n; ++i) {
        hnswlib::linklistsizeint *links = raw.get_linklist_at_level(
            static_cast<hnswlib::tableint>(i), 0);
        const size_t degree = raw.getListCount(links);
        const hnswlib::tableint *neighbors =
            reinterpret_cast<const hnswlib::tableint *>(links + 1);
        edges += degree;
        for (size_t j = 0; j < degree; ++j) {
            if (neighbors[j] >= n) throw std::runtime_error("invalid layer-0 neighbor");
            ++indegree[neighbors[j]];
        }
    }
    std::vector<double> converted(indegree.begin(), indegree.end());
    GraphStats result;
    result.avg_outdegree = n == 0 ? 0.0 : static_cast<double>(edges) / n;
    result.indegree_p95 = percentile(converted, 0.95);
    result.indegree_p99 = percentile(converted, 0.99);
    result.indegree_max = indegree.empty() ? 0 : *std::max_element(indegree.begin(), indegree.end());
    result.indegree_gini = gini(indegree);
    return result;
}

TangoConfig makeConfig(const Options &o) {
    TangoConfig config;
    config.dim = o.dim;
    config.mode = o.mode == "multiplicative" ? TangoMode::Multiplicative : TangoMode::Additive;
    config.data_data_mode = o.data_data_mode == "semantic"
        ? TangoDataDataMode::Semantic : TangoDataDataMode::TimeLift;
    config.half_life = o.half_life;
    config.lambda = std::log(2.0) / o.half_life;
    config.alpha = o.alpha;
    config.kappa_base = o.kappa_base;
    config.kappa_nav = o.kappa_nav;
    config.schedule = std::fabs(o.kappa_base - o.kappa_nav) <= 1e-15
        ? KappaSchedule::Fixed : KappaSchedule::TwoTier;
    config.normalize_input = false;
    config.validate();
    return config;
}

void writeManifest(const Options &o, const std::filesystem::path &directory,
                   const std::vector<size_t> &boundaries) {
#ifdef _OPENMP
    const size_t effective_gt_threads = o.gt_threads;
#else
    const size_t effective_gt_threads = 1;
#endif
    std::ofstream output(directory / "run_manifest.json");
    if (!output) throw std::runtime_error("cannot create dynamic run manifest");
    output << std::setprecision(17)
           << "{\n  \"schema\": \"chronos_tango_dynamic_v4\",\n"
           << "  \"status\": \"completed\",\n"
           << "  \"base_file\": \"" << jsonEscape(o.base_path) << "\",\n"
           << "  \"timestamp_file\": \"" << jsonEscape(o.timestamps_path) << "\",\n"
           << "  \"query_file\": \"" << jsonEscape(o.query_path) << "\",\n"
           << "  \"arrival_order\": \"arrival_order.u64bin\",\n"
           << "  \"arrival_order_contract\": \"stable ascending (timestamp,row_id); larger timestamp is newer\",\n"
           << "  \"results\": \"dynamic_results.csv\",\n"
           << "  \"ground_truth_directory\": \"groundtruth\",\n"
           << "  \"ground_truth_contract\": \"exact TDVS over active rows; int64 original row IDs\",\n"
           << "  \"n\": " << o.n << ", \"queries\": " << o.queries
           << ", \"dim\": " << o.dim << ",\n"
           << "  \"gt_width\": " << o.gt_width << ", \"recall_k_values\": [";
    for (size_t i = 0; i < o.k_values.size(); ++i) {
        if (i != 0) output << ',';
        output << o.k_values[i];
    }
    output << "],\n"
           << "  \"score_mode\": \"" << o.mode << "\", \"alpha\": " << o.alpha
           << ", \"half_life\": " << o.half_life
           << ", \"query_time\": " << o.query_time << ",\n"
           << "  \"data_data_mode\": \"" << o.data_data_mode
           << "\", \"M\": " << o.M << ", \"ef_construction\": " << o.ef_construction
           << ", \"kappa_base\": " << o.kappa_base << ", \"kappa_nav\": " << o.kappa_nav
           << ", \"random_seed\": " << o.random_seed
           << ", \"decay_fastpath\": \"" << o.fastpath << "\",\n"
           << "  \"initial_percent\": " << o.initial_percent
           << ", \"batch_percent\": " << o.batch_percent
           << ", \"max_search_expansions\": " << o.max_search_expansions << ",\n"
           << "  \"stop_recall\": ";
    if (std::isfinite(o.stop_recall)) output << o.stop_recall;
    else output << "null";
    output << ", \"minimum_ef_points\": " << o.min_ef_points << ",\n"
           << "  \"ef_values\": [";
    for (size_t i = 0; i < o.ef_list.size(); ++i) {
        if (i != 0) output << ',';
        output << o.ef_list[i];
    }
    output << "],\n"
           << "  \"timed_query_scope\": \"serial graph search and heap-to-ID materialization; query packing and recall excluded\",\n"
           << "  \"normalization\": \"base and query semantic rows L2-normalized before build/GT/timing\",\n"
           << "  \"initial_build_threads\": " << o.build_threads
           << ", \"gt_threads\": " << effective_gt_threads
           << ", \"online_insert_threads\": 1, \"query_threads\": 1, \"warmup_queries\": 0"
           << ", \"query_repeats\": " << o.repeats << ",\n"
           << "  \"checkpoint_active_counts\": [";
    for (size_t i = 0; i < boundaries.size(); ++i) {
        if (i != 0) output << ',';
        output << boundaries[i];
    }
    output << "]\n}\n";
}

int run(int argc, char **argv) {
    const Options o = parseOptions(argc, argv);
    const std::filesystem::path output_dir(o.output_dir);
    if (std::filesystem::exists(output_dir / "dynamic_results.csv") ||
            std::filesystem::exists(output_dir / "run_manifest.json")) {
        throw std::runtime_error("output directory already contains a dynamic run");
    }
    std::filesystem::create_directories(output_dir / "groundtruth");

    std::vector<float> base = loadBinary<float>(o.base_path, o.n * o.dim, "base");
    std::vector<float> timestamps = loadBinary<float>(o.timestamps_path, o.n, "timestamps");
    std::vector<float> queries = loadBinary<float>(o.query_path, o.queries * o.dim, "queries");
#ifdef _OPENMP
    omp_set_num_threads(static_cast<int>(std::max(o.build_threads, o.gt_threads)));
#endif
    normalizeRows(base, o.n, o.dim);
    normalizeRows(queries, o.queries, o.dim);

    const TangoConfig config = makeConfig(o);

    std::vector<uint64_t> arrival(o.n);
    for (size_t i = 0; i < o.n; ++i) {
        if (!std::isfinite(timestamps[i]) ||
                timestamps[i] > o.query_time + config.future_time_epsilon) {
            throw std::runtime_error("timestamp is non-finite or later than query-time");
        }
        arrival[i] = i;
    }
    std::stable_sort(arrival.begin(), arrival.end(), [&](uint64_t left, uint64_t right) {
        if (timestamps[left] != timestamps[right]) return timestamps[left] < timestamps[right];
        return left < right;
    });
    writeBinary(output_dir / "arrival_order.u64bin", arrival);

    const size_t checkpoint_count = (100 - o.initial_percent) / o.batch_percent;
    const size_t initial_gcd = std::gcd<size_t>(100, o.initial_percent);
    const size_t batch_gcd = std::gcd<size_t>(100, o.batch_percent);
    const size_t initial_size = (o.n / (100 / initial_gcd)) *
        (o.initial_percent / initial_gcd);
    const size_t batch_size = (o.n / (100 / batch_gcd)) *
        (o.batch_percent / batch_gcd);
    std::vector<size_t> boundaries(checkpoint_count + 1);
    boundaries[0] = initial_size;
    for (size_t checkpoint = 1; checkpoint <= checkpoint_count; ++checkpoint) {
        boundaries[checkpoint] = boundaries[0] + checkpoint * batch_size;
    }
    if (boundaries.back() != o.n) {
        throw std::logic_error("dynamic checkpoint schedule does not end at n");
    }

    const size_t query_record_size = hnswlib::tangoRecordSize(o.dim);
    if (o.queries > std::numeric_limits<size_t>::max() / query_record_size) {
        throw std::overflow_error("packed query storage size overflow");
    }
    std::vector<uint8_t> packed_queries(o.queries * query_record_size);
    for (size_t qi = 0; qi < o.queries; ++qi) {
        uint8_t *record = packed_queries.data() + qi * query_record_size;
        std::memcpy(record, queries.data() + qi * o.dim, o.dim * sizeof(float));
        hnswlib::writeTangoTimestamp(
            record, hnswlib::tangoTimestampOffset(o.dim), o.query_time);
    }

    TangoDecayFastPathOptions fastpath;
    fastpath.mode = hnswlib::parseTangoDecayFastPath(o.fastpath);
    fastpath.reference_time = o.query_time;
    TANGOIndex index(config, o.n, o.M, o.ef_construction, o.random_seed, fastpath);
    index.setMaxSearchExpansions(o.max_search_expansions);

    std::ofstream results(output_dir / "dynamic_results.csv");
    if (!results) throw std::runtime_error("cannot create dynamic results CSV");
    results << "checkpoint,active_count,active_fraction,inserted_count,insertion_threads,"
               "insert_seconds,insert_throughput,gt_seconds,query_threads,warmup_queries,repeats,"
               "k,ef,recall,mean_ms,p50_ms,p95_ms,p99_ms,avg_outdegree,indegree_p95,"
               "indegree_p99,indegree_max,indegree_gini\n";

    std::vector<TopHeap> gt_heaps(o.queries);
    size_t previous = 0;
    for (size_t checkpoint = 0; checkpoint < boundaries.size(); ++checkpoint) {
        const size_t active = boundaries[checkpoint];
        const Clock::time_point insert_start = Clock::now();
        if (checkpoint == 0) {
            const uint64_t first = arrival[0];
            index.addPoint(base.data() + first * o.dim, timestamps[first], first);
            parallelFor(1, active, o.build_threads, [&](size_t position) {
                const uint64_t id = arrival[position];
                index.addPoint(base.data() + id * o.dim, timestamps[id], id);
            });
        } else {
            for (size_t position = previous; position < active; ++position) {
                const uint64_t id = arrival[position];
                index.addPoint(base.data() + id * o.dim, timestamps[id], id);
            }
        }
        const double insert_seconds =
            std::chrono::duration<double>(Clock::now() - insert_start).count();
        const size_t inserted = active - previous;

        const Clock::time_point gt_start = Clock::now();
        const std::vector<int64_t> gt = updateGroundTruth(
            gt_heaps, queries, base, timestamps, arrival, previous, active, o);
        const double gt_seconds = std::chrono::duration<double>(Clock::now() - gt_start).count();
        std::ostringstream gt_name;
        gt_name << "checkpoint_" << std::setw(2) << std::setfill('0') << checkpoint
                << ".k" << o.gt_width << ".i64bin";
        writeBinary(output_dir / "groundtruth" / gt_name.str(), gt);

        const GraphStats graph = graphStats(index);
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        for (size_t k : o.k_values) {
            size_t measured_ef_points = 0;
            for (size_t ef : o.ef_list) {
                if (ef < k) continue;
                index.setEf(std::min(ef, active));
                std::vector<double> latencies;
                latencies.reserve(o.queries * o.repeats);
                std::vector<labeltype> result_ids(k);
                uint64_t correct = 0;
                for (size_t repeat = 0; repeat < o.repeats; ++repeat) {
                    for (size_t qi = 0; qi < o.queries; ++qi) {
                        const Clock::time_point query_start = Clock::now();
                        hnswlib::TangoQueryContext query_context =
                            index.makeQueryContext(o.query_time);
                        std::priority_queue<std::pair<float, labeltype> > found =
                            index.rawIndex().searchKnnWithContext(
                                packed_queries.data() + qi * query_record_size,
                                k, &query_context);
                        if (found.size() != k) {
                            throw std::runtime_error("index returned fewer than k results");
                        }
                        for (size_t position = k; position > 0; --position) {
                            result_ids[position - 1] = found.top().second;
                            found.pop();
                        }
                        latencies.push_back(std::chrono::duration<double, std::milli>(
                            Clock::now() - query_start).count());
                        std::unordered_set<labeltype> expected;
                        for (size_t j = 0; j < k; ++j) {
                            expected.insert(static_cast<labeltype>(gt[qi * o.gt_width + j]));
                        }
                        for (size_t position = 0; position < k; ++position) {
                            if (expected.count(result_ids[position]) != 0) ++correct;
                        }
                    }
                }
                const double recall = static_cast<double>(correct) /
                    static_cast<double>(o.queries * o.repeats * k);
                results << std::setprecision(12) << checkpoint << ',' << active << ','
                        << static_cast<double>(active) / o.n << ',' << inserted << ','
                        << (checkpoint == 0 ? o.build_threads : 1) << ',' << insert_seconds << ','
                        << (insert_seconds > 0.0 ? inserted / insert_seconds : 0.0) << ','
                        << gt_seconds << ",1,0," << o.repeats << ',' << k << ',' << ef << ','
                        << recall << ',' << mean(latencies) << ','
                        << percentile(latencies, 0.50) << ','
                        << percentile(latencies, 0.95) << ','
                        << percentile(latencies, 0.99) << ','
                        << graph.avg_outdegree << ',' << graph.indegree_p95 << ','
                        << graph.indegree_p99 << ',' << graph.indegree_max << ','
                        << graph.indegree_gini << '\n';
                ++measured_ef_points;
                if (std::isfinite(o.stop_recall) && recall >= o.stop_recall &&
                        measured_ef_points >= o.min_ef_points) {
                    break;
                }
            }
        }
        results.flush();
        std::cout << "checkpoint=" << checkpoint << " active=" << active
                  << " inserted=" << inserted << " insert_seconds=" << insert_seconds
                  << " gt_seconds=" << gt_seconds
                  << " recall_curves=" << o.k_values.size() << '\n';
        previous = active;
    }

    if (!o.final_index_prefix.empty()) index.save(o.final_index_prefix);
    writeManifest(o, output_dir, boundaries);
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception &error) {
        std::cerr << "tango_dynamic: " << error.what() << '\n';
        return 2;
    }
}
