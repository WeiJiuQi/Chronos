#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "evaluation.h"
#include "index_mag.h"

namespace {

constexpr const char* kMagCommit = "cbabb2b7d06d0a2e7f171219686aa49b72edf4ea";
constexpr const char* kMetadataMagic = "chronos_mag_upstream_v1";

// Defaults from the MAG authors' public script. The upstream ANMS search uses
// a fixed five-element Euclidean candidate pool and runs it to convergence.
constexpr unsigned kDefaultBuildL = 60;
constexpr unsigned kDefaultR = 48;
constexpr unsigned kDefaultC = 300;
constexpr unsigned kDefaultRip = 20;
constexpr unsigned kDefaultM = 64;
constexpr unsigned kDefaultThreshold = 8;
constexpr size_t kUpstreamEuclideanPoolSize = 5;

std::string MetadataPath(const std::string& index_path) {
  return index_path + ".mag.meta";
}

void WriteMetadata(const std::string& index_path, const tdvs_mips::FloatMatrix& base,
                   const tdvs_mips::KnngInfo& knng, unsigned build_l,
                   unsigned r, unsigned c, unsigned r_ip, unsigned m,
                   unsigned threshold, unsigned seed, unsigned build_threads) {
  tdvs_mips::WriteIndexMetadata(MetadataPath(index_path), kMetadataMagic, {
      {"upstream_commit", kMagCommit},
      {"source_fingerprint", TDVS_SOURCE_FINGERPRINT},
      {"base_rows", std::to_string(base.rows)},
      {"dim", std::to_string(base.dim)},
      {"stored_stride", std::to_string(base.stride)},
      {"knng_degree", std::to_string(knng.degree)},
      {"build_l", std::to_string(build_l)},
      {"r", std::to_string(r)},
      {"c", std::to_string(c)},
      {"r_ip", std::to_string(r_ip)},
      {"m", std::to_string(m)},
      {"threshold", std::to_string(threshold)},
      {"seed", std::to_string(seed)},
      {"build_threads", std::to_string(build_threads)},
  });
}

void ValidateMetadata(const std::string& index_path, size_t rows, size_t dim,
                      size_t stride) {
  const auto fields = tdvs_mips::ReadIndexMetadata(
      MetadataPath(index_path), kMetadataMagic);
  tdvs_mips::RequireIndexMetadataValue(fields, "upstream_commit", kMagCommit);
  tdvs_mips::RequireIndexMetadataValue(
      fields, "source_fingerprint", TDVS_SOURCE_FINGERPRINT);
  tdvs_mips::RequireIndexMetadataSize(fields, "base_rows", rows);
  tdvs_mips::RequireIndexMetadataSize(fields, "dim", dim);
  tdvs_mips::RequireIndexMetadataSize(fields, "stored_stride", stride);
}

void ValidateLoadedMagIndex(const MAG::IndexMAG& index, size_t expected_rows) {
  if (index.final_graph_.size() != expected_rows) {
    throw std::runtime_error(
        "MAG index row count does not match the supplied base matrix");
  }
  if (expected_rows == 0 || index.ep_ >= expected_rows) {
    throw std::runtime_error("MAG index entry point is out of range");
  }
  for (const auto& neighbors : index.final_graph_) {
    for (unsigned id : neighbors) {
      if (id >= expected_rows) {
        throw std::runtime_error("MAG index contains an out-of-range neighbor ID");
      }
    }
  }
}

void PrintHelp() {
  std::cout
      << "MAG/ANMS runner for transformed TDVS MIPS data\n\n"
      << "Build:\n"
      << "  tdvs_mag --mode build --base BASE.fbin --knng BASE.knng --index OUT.mag\\\n"
      << "           --dim 1536 [--build-threads 64 --build-l 60 --r 48 --c 300\\\n"
      << "           --r-ip 20 --m 64 --threshold 8 --seed 42\\\n"
      << "           --knng-manifest PATH --knng-build-seconds SEC]\n\n"
      << "Search (always serial for comparable latency):\n"
      << "  tdvs_mag --mode search --base BASE.fbin --query QUERY.fbin --gt GT.bin\\\n"
      << "           --index OUT.mag --dim 1536 --gt-width 100 --k 100\\\n"
      << "           --search-l 100,150,200,300,500,700 [--query-count 1000\\\n"
      << "           --normalize-query true --repeats 3\\\n"
      << "           --seed 42 --output-csv mag.csv]\n\n"
      << "  Search uses the authors' fixed L_NN=5 Euclidean pool to convergence,\n"
      << "  then switches to the inner-product stage. Only L_search is swept.\n\n"
      << "Load-only footprint (run as a fresh process):\n"
      << "  tdvs_mag --mode footprint --base BASE.fbin --index OUT.mag\\\n"
      << "           --dim 1536 [--base-count 999000 --output-csv mag_footprint.csv]\n";
}

int RunBuild(const tdvs_mips::Args& args) {
  const size_t dim = args.RequireSize("dim");
  const std::string base_path = args.Require("base");
  const std::string knng_path = args.Require("knng");
  const std::string index_path = args.Require("index");
  auto base = tdvs_mips::LoadRawFloatMatrix(base_path, dim, args.GetSize("base-count", 0));
  tdvs_mips::PrintInputSummary(base);
  const auto knng = tdvs_mips::ValidateKnng(knng_path, base.rows);
  std::cerr << "Validated kNN graph: N=" << knng.rows << ", K=" << knng.degree
            << ", bytes=" << knng.bytes << '\n';

  const unsigned build_threads = args.GetUnsigned(
      "build-threads", tdvs_mips::kDefaultBuildThreads);
  tdvs_mips::ConfigureOpenMP(build_threads);
  const unsigned seed = args.GetUnsigned("seed", 42);
  std::srand(seed);

  const tdvs_mips::KnngBuildTiming knng_timing =
      tdvs_mips::ResolveKnngBuildTiming(args, knng_path);

  const unsigned build_l = args.GetUnsigned("build-l", kDefaultBuildL);
  const unsigned r = args.GetUnsigned("r", kDefaultR);
  const unsigned c = args.GetUnsigned("c", kDefaultC);
  const unsigned r_ip = args.GetUnsigned("r-ip", kDefaultRip);
  const unsigned m = args.GetUnsigned("m", kDefaultM);
  const unsigned threshold =
      args.GetUnsigned("threshold", kDefaultThreshold);
  constexpr unsigned n_try = 1;
  MAG::Parameters parameters;
  parameters.Set<unsigned>("L", build_l);
  parameters.Set<unsigned>("R", r);
  parameters.Set<unsigned>("C", c);
  parameters.Set<unsigned>("R_IP", r_ip);
  parameters.Set<unsigned>("M", m);
  parameters.Set<unsigned>("threshold", threshold);
  parameters.Set<unsigned>("n_try", n_try);
  parameters.Set<std::string>("nn_graph_path", knng_path);

  MAG::IndexMAG index(base.stride, base.rows);
  const auto start = tdvs_mips::SteadyClock::now();
  index.Build(base.rows, base.values.data(), parameters);
  const auto build_end = tdvs_mips::SteadyClock::now();
  ValidateLoadedMagIndex(index, base.rows);
  const auto validation_end = tdvs_mips::SteadyClock::now();
  index.Save(index_path.c_str());
  WriteMetadata(index_path, base, knng, build_l, r, c, r_ip, m, threshold,
                seed, build_threads);
  const auto save_end = tdvs_mips::SteadyClock::now();
  const double preprocess_seconds = 0.0;
  const double core_build_seconds =
      std::chrono::duration<double>(build_end - start).count();
  const double validation_seconds =
      std::chrono::duration<double>(validation_end - build_end).count();
  const double serialize_seconds =
      std::chrono::duration<double>(save_end - validation_end).count();
  const double serving_index_pipeline_seconds =
      preprocess_seconds + core_build_seconds + serialize_seconds;
  std::cout << "method=MAG\n"
            << "method_commit=" << kMagCommit << '\n'
            << "source_fingerprint=" << TDVS_SOURCE_FINGERPRINT << '\n'
            << "base_rows=" << base.rows << '\n'
            << "dim=" << base.dim << '\n'
            << "stored_stride=" << base.stride << '\n'
            << "knng_bytes=" << knng.bytes << '\n'
            << "build_threads=" << build_threads << '\n'
            << "build_l=" << build_l << '\n'
            << "r=" << r << '\n'
            << "c=" << c << '\n'
            << "r_ip=" << r_ip << '\n'
            << "m=" << m << '\n'
            << "threshold=" << threshold << '\n'
            << "n_try=" << n_try << '\n'
            << "seed=" << seed << '\n'
            << "parallel_build_thread_safety=upstream_openmp_unmodified\n"
            << "parallel_build_reproducible="
            << (build_threads == 1 ? "true" : "false") << '\n'
            << "core_build_excludes_knng=true\n"
            << "knng_build_time_included_in_core=false\n"
            << "knng_graph_build_seconds=";
  if (knng_timing.known && knng_timing.source != "cli_override") {
    std::cout << knng_timing.graph_build_seconds;
  } else {
    std::cout << "unknown";
  }
  std::cout << '\n' << "knng_graph_extract_seconds=";
  if (knng_timing.known && knng_timing.source != "cli_override") {
    std::cout << knng_timing.graph_extract_seconds;
  } else {
    std::cout << "unknown";
  }
  std::cout << '\n' << "knng_graph_write_seconds=";
  if (knng_timing.known && knng_timing.source != "cli_override") {
    std::cout << knng_timing.graph_write_seconds;
  } else {
    std::cout << "unknown";
  }
  std::cout << '\n'
            << "knng_build_seconds=";
  if (knng_timing.known) {
    std::cout << knng_timing.total_seconds;
  } else {
    std::cout << "unknown";
  }
  std::cout << '\n'
            << "knng_build_time_source=" << knng_timing.source << '\n'
            << "knng_manifest_path="
            << (knng_timing.manifest_path.empty() ? "none"
                                                  : knng_timing.manifest_path)
            << '\n'
            << "preprocess_seconds=" << preprocess_seconds << '\n'
            << "core_build_seconds=" << core_build_seconds << '\n'
            << "mag_build_seconds=" << core_build_seconds << '\n'
            << "post_build_validation_seconds=" << validation_seconds << '\n'
            << "serialize_seconds=" << serialize_seconds << '\n'
            << "serving_index_pipeline_seconds="
            << serving_index_pipeline_seconds << '\n';
  if (knng_timing.known) {
    std::cout << "end_to_end_index_seconds="
              << serving_index_pipeline_seconds + knng_timing.total_seconds
              << '\n';
  }
  const uint64_t serialized_index_bytes = tdvs_mips::CheckedAddBytes(
      tdvs_mips::FileSize(index_path), tdvs_mips::FileSize(MetadataPath(index_path)),
      "MAG serialized index");
  std::cout << "serialized_index_bytes=" << serialized_index_bytes << '\n'
            << "index_bytes=" << serialized_index_bytes << '\n'
            << "metadata_path=" << MetadataPath(index_path) << '\n';
  return 0;
}

int RunFootprint(const tdvs_mips::Args& args) {
  const size_t dim = args.RequireSize("dim");
  const std::string base_path = args.Require("base");
  const std::string index_path = args.Require("index");
  const uint64_t rss_baseline = tdvs_mips::CurrentRSSBytes();

  auto base = tdvs_mips::LoadRawFloatMatrix(
      base_path, dim, args.GetSize("base-count", 0));
  ValidateMetadata(index_path, base.rows, base.dim, base.stride);
  MAG::IndexMAG index(base.stride, base.rows);
  index.Load(index_path.c_str());
  ValidateLoadedMagIndex(index, base.rows);
  index.entry_point_candidate(base.values.data());
  tdvs_mips::FootprintRecord record;
  record.method = "MAG";
  record.method_commit = kMagCommit;
  record.base_rows = base.rows;
  record.dim = base.dim;
  record.stored_stride = base.stride;
  record.index_contains_base = false;
  record.serialized_index_bytes = tdvs_mips::CheckedAddBytes(
      tdvs_mips::FileSize(index_path), tdvs_mips::FileSize(MetadataPath(index_path)),
      "MAG serialized index");
  record.external_base_file_bytes = tdvs_mips::FileSize(base_path);
  record.external_base_allocation_bytes = tdvs_mips::CheckedMulBytes(
      base.values.capacity(), sizeof(float), "MAG external base allocation");
  record.runtime_auxiliary_known_bytes = tdvs_mips::CheckedMulBytes(
      index.entries.capacity(), sizeof(std::pair<float, unsigned>),
      "MAG upstream entry-point candidates");
  record.rss_baseline_bytes = rss_baseline;
  record.rss_after_load_bytes = tdvs_mips::CurrentRSSBytes();
  record.peak_rss_bytes = tdvs_mips::PeakRSSBytes();
  tdvs_mips::EmitFootprint(args.Get("output-csv", ""), record);
  return 0;
}

int RunSearch(const tdvs_mips::Args& args) {
  if (args.Has("warmup-queries")) {
    throw std::invalid_argument(
        "--warmup-queries has been removed; every query is measured");
  }
  if (args.Has("switch-m")) {
    throw std::invalid_argument(
        "--switch-m is not part of the MAG authors' public implementation");
  }
  const size_t dim = args.RequireSize("dim");
  const std::string base_path = args.Require("base");
  const std::string query_path = args.Require("query");
  const std::string gt_path = args.Require("gt");
  const std::string index_path = args.Require("index");
  auto base = tdvs_mips::LoadRawFloatMatrix(base_path, dim, args.GetSize("base-count", 0));
  auto queries = tdvs_mips::LoadRawFloatMatrix(query_path, dim,
                                                args.GetSize("query-count", 0));
  if (queries.stride != base.stride) throw std::logic_error("Stride mismatch");
  if (args.GetBool("normalize-query", true)) tdvs_mips::NormalizeRows(queries);
  tdvs_mips::PrintInputSummary(base, &queries);
  ValidateMetadata(index_path, base.rows, base.dim, base.stride);

  const size_t gt_width = args.GetSize("gt-width", 100);
  auto gt = tdvs_mips::LoadGroundTruth(gt_path, queries.rows, gt_width,
                                        args.Get("gt-type", "int64"), base.rows);
  const size_t k = args.GetSize("k", 100);
  const auto search_values = tdvs_mips::ParseSizeList(args.Require("search-l"));
  for (size_t value : search_values) {
    if (value < k || value < kUpstreamEuclideanPoolSize ||
        value >= base.rows || value > std::numeric_limits<unsigned>::max()) {
      throw std::invalid_argument(
          "MAG requires max(k, 5) <= each search_L < base-count");
    }
  }

  MAG::IndexMAG index(base.stride, base.rows);
  const auto load_start = tdvs_mips::SteadyClock::now();
  index.Load(index_path.c_str());
  const auto load_end = tdvs_mips::SteadyClock::now();
  ValidateLoadedMagIndex(index, base.rows);
  const auto setup_start = tdvs_mips::SteadyClock::now();
  index.entry_point_candidate(base.values.data());
  const auto setup_end = tdvs_mips::SteadyClock::now();
  std::cerr << "MAG graph load: "
            << std::chrono::duration<double>(load_end - load_start).count() << " s\n"
            << "MAG upstream entry-point preparation: "
            << std::chrono::duration<double>(setup_end - setup_start).count()
            << " s\n";

  tdvs_mips::EvaluationConfig config;
  config.k = k;
  config.repeats = args.GetSize("repeats", 3);
  config.seed = args.GetUnsigned("seed", 42);
  config.search_values = search_values;
  config.method = "MAG";
  config.method_commit = kMagCommit;
  config.index_path = index_path;
  config.timed_scope = "serial_search_and_topk_materialization";
  config.distance_count_definition = "upstream_MAG_dis_cal_counter";
  std::srand(config.seed);

  auto records = tdvs_mips::RunSearchEvaluation(
      base, queries, gt, config,
      [&](const float* query, size_t search_l, unsigned* result) {
        MAG::Parameters parameters;
        parameters.Set<unsigned>("L_search", static_cast<unsigned>(search_l));
        const auto search_start = tdvs_mips::SteadyClock::now();
        const int computations = index.Search_NN_IP(
            query, base.values.data(), k, parameters, result);
        const auto search_end = tdvs_mips::SteadyClock::now();
        return tdvs_mips::SearchMeasurement{
            computations,
            std::chrono::duration<double>(search_end - search_start).count()};
      });
  tdvs_mips::WriteCsv(args.Get("output-csv", ""), records);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    tdvs_mips::Args args(argc, argv);
    if (args.Has("help") || argc == 1) {
      PrintHelp();
      return 0;
    }
    const std::string mode = args.Require("mode");
    if (mode == "build") return RunBuild(args);
    if (mode == "search") return RunSearch(args);
    if (mode == "footprint") return RunFootprint(args);
    throw std::invalid_argument("--mode must be build, search, or footprint");
  } catch (const std::exception& error) {
    std::cerr << "tdvs_mag: " << error.what() << '\n';
    return 2;
  }
}
