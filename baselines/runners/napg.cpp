#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "evaluation.h"
#include "hnswlib.h"
#include "napg.h"

namespace {

constexpr const char* kMethodName = "NAPG (clean-room KDD 2021)";
constexpr const char* kMethodReference =
    "Tan et al., KDD 2021, DOI 10.1145/3447548.3467412";
constexpr const char* kMethodProvenance =
    "clean-room-paper-implementation:10.1145/3447548.3467412";
constexpr const char* kMetadataMagic = "NAPG_META_V1";

std::string MetadataPath(const std::string& index_path) {
  return index_path + ".napg.meta";
}

std::string JoinDoubles(const std::vector<double>& values) {
  std::ostringstream output;
  output << std::setprecision(17);
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) output << ',';
    output << values[i];
  }
  return output.str();
}

std::vector<double> ParseDoubles(const std::string& text,
                                 const std::string& field) {
  std::vector<double> values;
  std::istringstream input(text);
  std::string token;
  while (std::getline(input, token, ',')) {
    if (token.empty()) {
      throw std::runtime_error("Empty value in NAPG metadata field " + field);
    }
    size_t consumed = 0;
    const double value = std::stod(token, &consumed);
    if (consumed != token.size() || !std::isfinite(value)) {
      throw std::runtime_error("Invalid value in NAPG metadata field " + field);
    }
    values.push_back(value);
  }
  return values;
}

size_t ParseMetadataSize(const std::map<std::string, std::string>& fields,
                         const std::string& key) {
  const auto found = fields.find(key);
  if (found == fields.end()) {
    throw std::runtime_error("Missing NAPG metadata field: " + key);
  }
  try {
    return tdvs_mips::ParseDecimalSize(
        found->second, "NAPG metadata field " + key);
  } catch (const std::invalid_argument&) {
    throw std::runtime_error("Invalid NAPG metadata field: " + key);
  }
}

struct Metadata {
  size_t base_rows = 0;
  size_t dim = 0;
  size_t stored_stride = 0;
  size_t norm_ranges = 0;
  size_t samples_per_range = 0;
  size_t sample_top_n = 0;
  size_t effective_top_n = 0;
  size_t m = 0;
  size_t ef_construction = 0;
  size_t build_threads = 0;
  size_t random_seed = 0;
  uint64_t sampling_seed = 0;
  std::vector<double> boundaries;
  std::vector<double> alphas;
};

void WriteMetadata(const std::string& path, const Metadata& metadata,
                   const chronos_napg::EstimationResult& estimate) {
  std::ofstream output(path, std::ios::trunc);
  if (!output) throw std::runtime_error("Cannot write NAPG metadata: " + path);
  output << kMetadataMagic << '\n'
         << "method_reference=" << kMethodReference << '\n'
         << "source_fingerprint=" << TDVS_SOURCE_FINGERPRINT << '\n'
         << "base_rows=" << metadata.base_rows << '\n'
         << "dim=" << metadata.dim << '\n'
         << "stored_stride=" << metadata.stored_stride << '\n'
         << "norm_ranges=" << metadata.norm_ranges << '\n'
         << "samples_per_range=" << metadata.samples_per_range << '\n'
         << "sample_top_n=" << metadata.sample_top_n << '\n'
         << "effective_top_n=" << metadata.effective_top_n << '\n'
         << "M=" << metadata.m << '\n'
         << "efConstruction=" << metadata.ef_construction << '\n'
         << "build_threads=" << metadata.build_threads << '\n'
         << "random_seed=" << metadata.random_seed << '\n'
         << "sampling_seed=" << metadata.sampling_seed << '\n'
         << "upper_norm_boundaries=" << JoinDoubles(metadata.boundaries) << '\n'
         << "alphas=" << JoinDoubles(metadata.alphas) << '\n'
         << "source_candidate_inner_products="
         << estimate.source_candidate_inner_products << '\n'
         << "candidate_pair_inner_products="
         << estimate.candidate_pair_inner_products << '\n';
  output << std::setprecision(17);
  for (size_t range = 0; range < estimate.ranges.size(); ++range) {
    const auto& stats = estimate.ranges[range];
    output << "range_" << range << "_rank_begin=" << stats.begin_rank << '\n'
           << "range_" << range << "_rank_end=" << stats.end_rank << '\n'
           << "range_" << range << "_samples=" << stats.samples_used << '\n'
           << "range_" << range << "_min_norm=" << stats.min_norm << '\n'
           << "range_" << range << "_max_norm=" << stats.max_norm << '\n'
           << "range_" << range << "_avg_source_candidate_ip="
           << stats.average_source_candidate_ip << '\n'
           << "range_" << range << "_avg_candidate_pair_ip="
           << stats.average_candidate_pair_ip << '\n'
           << "range_" << range << "_alpha=" << stats.alpha << '\n';
  }
  output.flush();
  if (!output) throw std::runtime_error("Failed to write NAPG metadata: " + path);
}

Metadata ReadMetadata(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open NAPG metadata: " + path);
  std::string line;
  if (!std::getline(input, line) || line != kMetadataMagic) {
    throw std::runtime_error("Invalid NAPG metadata header: " + path);
  }
  std::map<std::string, std::string> fields;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const size_t equals = line.find('=');
    if (equals == std::string::npos || equals == 0) {
      throw std::runtime_error("Malformed NAPG metadata line in " + path);
    }
    fields[line.substr(0, equals)] = line.substr(equals + 1);
  }
  Metadata metadata;
  metadata.base_rows = ParseMetadataSize(fields, "base_rows");
  metadata.dim = ParseMetadataSize(fields, "dim");
  metadata.stored_stride = ParseMetadataSize(fields, "stored_stride");
  metadata.norm_ranges = ParseMetadataSize(fields, "norm_ranges");
  metadata.samples_per_range = ParseMetadataSize(fields, "samples_per_range");
  metadata.sample_top_n = ParseMetadataSize(fields, "sample_top_n");
  metadata.effective_top_n = ParseMetadataSize(fields, "effective_top_n");
  metadata.m = ParseMetadataSize(fields, "M");
  metadata.ef_construction = ParseMetadataSize(fields, "efConstruction");
  metadata.build_threads = ParseMetadataSize(fields, "build_threads");
  metadata.random_seed = ParseMetadataSize(fields, "random_seed");
  metadata.sampling_seed = ParseMetadataSize(fields, "sampling_seed");
  const auto boundaries = fields.find("upper_norm_boundaries");
  const auto alphas = fields.find("alphas");
  if (boundaries == fields.end() || alphas == fields.end()) {
    throw std::runtime_error("NAPG metadata is missing model parameters");
  }
  if (!boundaries->second.empty()) {
    metadata.boundaries =
        ParseDoubles(boundaries->second, "upper_norm_boundaries");
  }
  metadata.alphas = ParseDoubles(alphas->second, "alphas");
  if (metadata.norm_ranges == 0 ||
      metadata.alphas.size() != metadata.norm_ranges ||
      metadata.boundaries.size() + 1 != metadata.norm_ranges) {
    throw std::runtime_error("NAPG metadata model shape is inconsistent");
  }
  return metadata;
}

void ValidateMetadata(const Metadata& metadata, size_t rows, size_t dim,
                      size_t stride) {
  if (metadata.base_rows != rows || metadata.dim != dim ||
      metadata.stored_stride != stride) {
    throw std::runtime_error(
        "NAPG metadata does not match --base-count/--dim or aligned stride");
  }
}

void PrintHelp() {
  std::cout
      << "NAPG clean-room baseline from Tan et al., KDD 2021\n\n"
      << "Build (the transformed MIPS base must NOT be normalized):\n"
      << "  tdvs_napg --mode build --base BASE.fbin --index OUT.napg\\\n"
      << "      --dim 1536 [--base-count 999000 --m 16 --ef-construction 100\\\n"
      << "      --norm-ranges 5 --samples-per-range 100 --sample-top-n 100\\\n"
      << "      --sampling-seed 42 --random-seed 100 --build-threads 64]\n\n"
      << "Search (serial query processing; every query is timed):\n"
      << "  tdvs_napg --mode search --query QUERY.fbin --gt GT.bin\\\n"
      << "      --index OUT.napg --base-count 999000 --dim 1536\\\n"
      << "      --query-count 1000 --gt-width 100 --gt-type int64 --k 100\\\n"
      << "      --ef-list 100,150,200,300,500,700 [--normalize-query true\\\n"
      << "      --max-search-expansions 0 --repeats 3 --seed 42\\\n"
      << "      --output-csv napg.csv]\n\n"
      << "Load-only footprint (run as a fresh process):\n"
      << "  tdvs_napg --mode footprint --index OUT.napg --dim 1536\\\n"
      << "      --base-count 999000 [--output-csv napg_footprint.csv]\n";
}

int RunBuild(const tdvs_mips::Args& args) {
  const size_t dim = args.RequireSize("dim");
  const std::string base_path = args.Require("base");
  const std::string index_path = args.Require("index");
  const unsigned build_threads = args.GetUnsigned(
      "build-threads", tdvs_mips::kDefaultBuildThreads);
  auto base = tdvs_mips::LoadRawFloatMatrix(
      base_path, dim, args.GetSize("base-count", 0));
  tdvs_mips::PrintInputSummary(base);

  const size_t m = args.GetSize("m", 16);
  const size_t ef_construction = args.GetSize("ef-construction", 100);
  const size_t random_seed = args.GetSize("random-seed", 100);
  if (m < 2) throw std::invalid_argument("--m must be at least 2");
  if (ef_construction < m) {
    throw std::invalid_argument("--ef-construction must be at least --m");
  }
  if (m > std::numeric_limits<unsigned>::max() ||
      ef_construction > std::numeric_limits<unsigned>::max() ||
      random_seed > std::numeric_limits<unsigned>::max()) {
    throw std::invalid_argument("NAPG HNSW parameters exceed the supported range");
  }

  chronos_napg::EstimationConfig estimate_config;
  estimate_config.norm_ranges = args.GetSize("norm-ranges", 5);
  estimate_config.samples_per_range = args.GetSize("samples-per-range", 100);
  estimate_config.top_n = args.GetSize("sample-top-n", 100);
  estimate_config.seed = static_cast<uint64_t>(args.GetSize("sampling-seed", 42));
  estimate_config.threads = build_threads;
  tdvs_mips::ConfigureOpenMP(build_threads);

  const auto preprocess_start = tdvs_mips::SteadyClock::now();
  const auto estimate = chronos_napg::EstimateAdjustingFactors(
      base.values.data(), base.rows, base.dim, base.stride, estimate_config);
  const auto preprocess_end = tdvs_mips::SteadyClock::now();

  // The norm of each transformed vector carries the MIPS/TDVS signal.  NAPG
  // estimates norm-range factors and builds directly on these unnormalized
  // vectors, exactly as specified by the paper.
  hnswlib::InnerProductSpace space(base.stride);
  hnswlib::HierarchicalNSW<float> index(
      &space, base.rows, m, ef_construction, random_seed);
  chronos_napg::Runtime runtime(base.rows, base.dim,
                                estimate.upper_norm_boundaries,
                                estimate.alphas, estimate.range_by_row);
  index.setIdAwareDistanceFunctions(
      nullptr, nullptr, chronos_napg::Runtime::PrepareCallback, &runtime);
  index.setNeighborSelectionRejectFunction(
      chronos_napg::Runtime::RejectCallback, &runtime);

  const auto build_start = tdvs_mips::SteadyClock::now();
  index.addPoint(base.Row(0), static_cast<hnswlib::labeltype>(0));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(build_threads) if(build_threads > 1)
#endif
  for (long long row = 1; row < static_cast<long long>(base.rows); ++row) {
    index.addPoint(base.Row(static_cast<size_t>(row)),
                   static_cast<hnswlib::labeltype>(row));
  }
  const auto build_end = tdvs_mips::SteadyClock::now();

  Metadata metadata;
  metadata.base_rows = base.rows;
  metadata.dim = base.dim;
  metadata.stored_stride = base.stride;
  metadata.norm_ranges = estimate_config.norm_ranges;
  metadata.samples_per_range = estimate_config.samples_per_range;
  metadata.sample_top_n = estimate_config.top_n;
  metadata.effective_top_n = estimate.effective_top_n;
  metadata.m = m;
  metadata.ef_construction = ef_construction;
  metadata.build_threads = build_threads;
  metadata.random_seed = random_seed;
  metadata.sampling_seed = estimate_config.seed;
  metadata.boundaries = estimate.upper_norm_boundaries;
  metadata.alphas = estimate.alphas;

  index.saveIndex(index_path);
  const std::string metadata_path = MetadataPath(index_path);
  WriteMetadata(metadata_path, metadata, estimate);
  const auto save_end = tdvs_mips::SteadyClock::now();

  const double preprocess_seconds =
      std::chrono::duration<double>(preprocess_end - preprocess_start).count();
  const double core_build_seconds =
      std::chrono::duration<double>(build_end - build_start).count();
  const double serialize_seconds =
      std::chrono::duration<double>(save_end - build_end).count();
  const double pipeline_seconds =
      preprocess_seconds + core_build_seconds + serialize_seconds;
  const uint64_t serialized_bytes = tdvs_mips::CheckedAddBytes(
      tdvs_mips::FileSize(index_path), tdvs_mips::FileSize(metadata_path),
      "NAPG serialized index");

  std::cout << "method=" << kMethodName << '\n'
            << "method_commit=" << kMethodProvenance << '\n'
            << "method_reference=" << kMethodReference << '\n'
            << "source_fingerprint=" << TDVS_SOURCE_FINGERPRINT << '\n'
            << "base_rows=" << base.rows << '\n'
            << "dim=" << base.dim << '\n'
            << "stored_stride=" << base.stride << '\n'
            << "M=" << m << '\n'
            << "efConstruction=" << ef_construction << '\n'
            << "random_seed=" << random_seed << '\n'
            << "build_threads=" << build_threads << '\n'
            << "parallel_build_thread_safety=current_hnswlib_synchronized\n"
            << "parallel_build_reproducible="
            << (build_threads == 1 ? "true" : "false") << '\n'
            << "norm_ranges=" << estimate_config.norm_ranges << '\n'
            << "samples_per_range=" << estimate_config.samples_per_range << '\n'
            << "sample_top_n=" << estimate_config.top_n << '\n'
            << "effective_top_n=" << estimate.effective_top_n << '\n'
            << "sampling_seed=" << estimate_config.seed << '\n'
            << "norm_range_boundaries="
            << JoinDoubles(estimate.upper_norm_boundaries) << '\n'
            << "adjusting_factors=" << JoinDoubles(estimate.alphas) << '\n'
            << "edge_selection_comparisons="
            << runtime.edgeSelectionComparisons() << '\n'
            << "preprocess_seconds=" << preprocess_seconds << '\n'
            << "core_build_seconds=" << core_build_seconds << '\n'
            << "build_seconds=" << core_build_seconds << '\n'
            << "serialize_seconds=" << serialize_seconds << '\n'
            << "serving_index_pipeline_seconds=" << pipeline_seconds << '\n'
            << "metadata_path=" << metadata_path << '\n'
            << "serialized_index_bytes=" << serialized_bytes << '\n'
            << "index_bytes=" << serialized_bytes << '\n';
  return 0;
}

int RunFootprint(const tdvs_mips::Args& args) {
  const size_t dim = args.RequireSize("dim");
  const size_t base_count = args.RequireSize("base-count");
  const std::string index_path = args.Require("index");
  if (dim == 0 || base_count == 0) {
    throw std::invalid_argument("--dim and --base-count must be positive");
  }
  const size_t stride = (dim + 7U) & ~size_t(7U);
  const std::string metadata_path = MetadataPath(index_path);
  {
    const Metadata metadata = ReadMetadata(metadata_path);
    ValidateMetadata(metadata, base_count, dim, stride);
  }

  const uint64_t rss_baseline = tdvs_mips::CurrentRSSBytes();
  hnswlib::InnerProductSpace space(stride);
  hnswlib::HierarchicalNSW<float> index(&space, index_path, false);
  if (index.getCurrentElementCount() != base_count ||
      index.data_size_ != stride * sizeof(float)) {
    throw std::runtime_error("Loaded NAPG index does not match requested shape");
  }

  tdvs_mips::FootprintRecord record;
  record.method = kMethodName;
  record.method_commit = kMethodProvenance;
  record.base_rows = base_count;
  record.dim = dim;
  record.stored_stride = stride;
  record.index_contains_base = true;
  record.serialized_index_bytes = tdvs_mips::CheckedAddBytes(
      tdvs_mips::FileSize(index_path), tdvs_mips::FileSize(metadata_path),
      "NAPG serialized index");
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
  const size_t dim = args.RequireSize("dim");
  const size_t base_count = args.RequireSize("base-count");
  const std::string query_path = args.Require("query");
  const std::string gt_path = args.Require("gt");
  const std::string index_path = args.Require("index");
  if (base_count == 0) throw std::invalid_argument("--base-count must be positive");

  auto queries = tdvs_mips::LoadRawFloatMatrix(
      query_path, dim, args.GetSize("query-count", 0));
  if (args.GetBool("normalize-query", true)) tdvs_mips::NormalizeRows(queries);
  tdvs_mips::FloatMatrix base_shape;
  base_shape.rows = base_count;
  base_shape.dim = dim;
  base_shape.stride = queries.stride;
  tdvs_mips::PrintInputSummary(base_shape, &queries);

  const size_t gt_width = args.GetSize("gt-width", 100);
  const auto gt = tdvs_mips::LoadGroundTruth(
      gt_path, queries.rows, gt_width, args.Get("gt-type", "int64"), base_count);
  const size_t k = args.GetSize("k", 100);
  if (k > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("--k exceeds the supported integer range");
  }
  const std::string ef_text =
      args.Has("ef-list") ? args.Require("ef-list") : args.Require("ef-search");
  const auto ef_values = tdvs_mips::ParseSizeList(ef_text);
  const std::string metadata_path = MetadataPath(index_path);
  const Metadata metadata = ReadMetadata(metadata_path);
  ValidateMetadata(metadata, base_count, dim, queries.stride);

  hnswlib::InnerProductSpace space(queries.stride);
  const auto load_start = tdvs_mips::SteadyClock::now();
  hnswlib::HierarchicalNSW<float> index(&space, index_path, false);
  const auto load_end = tdvs_mips::SteadyClock::now();
  if (index.getCurrentElementCount() != base_count ||
      index.getMaxElements() < base_count ||
      index.data_size_ != queries.stride * sizeof(float)) {
    throw std::runtime_error("Loaded NAPG index does not match requested shape");
  }
  const size_t max_search_expansions = args.GetSize("max-search-expansions", 0);
  index.setMaxSearchExpansions(max_search_expansions);
  std::cerr << "NAPG index load: "
            << std::chrono::duration<double>(load_end - load_start).count()
            << " s\n";

  tdvs_mips::EvaluationConfig config;
  config.k = k;
  config.repeats = args.GetSize("repeats", 3);
  config.seed = args.GetUnsigned("seed", 42);
  config.search_values = ef_values;
  config.method = std::string(kMethodName) + "; max_expansions=" +
                  std::to_string(max_search_expansions);
  config.method_commit = kMethodProvenance;
  config.index_path = index_path;
  config.search_param_name = "efSearch";
  config.index_contains_base = true;
  config.timed_scope = "serial_search_and_topk_materialization";
  config.distance_count_definition = "hnswlib_metric_distance_computations";

  auto records = tdvs_mips::RunSearchEvaluation(
      base_shape, queries, gt, config,
      [&](const float* query, size_t ef_search, unsigned* result) {
        index.setEf(ef_search);
        index.metric_distance_computations.store(0, std::memory_order_relaxed);
        const auto search_start = tdvs_mips::SteadyClock::now();
        auto heap = index.searchKnn(query, k);
        if (heap.size() != k) {
          throw std::runtime_error("NAPG returned fewer than k results");
        }
        for (size_t position = k; position > 0; --position) {
          const hnswlib::labeltype label = heap.top().second;
          heap.pop();
          if (label >= base_count ||
              label > static_cast<hnswlib::labeltype>(
                          std::numeric_limits<unsigned>::max())) {
            throw std::runtime_error("NAPG returned an out-of-range label");
          }
          result[position - 1] = static_cast<unsigned>(label);
        }
        const auto search_end = tdvs_mips::SteadyClock::now();
        const long computations =
            index.metric_distance_computations.load(std::memory_order_relaxed);
        if (computations < 0 ||
            computations > static_cast<long>(std::numeric_limits<int>::max())) {
          throw std::runtime_error("NAPG distance count exceeds int range");
        }
        return tdvs_mips::SearchMeasurement{
            static_cast<int>(computations),
            std::chrono::duration<double>(search_end - search_start).count()};
      });

  const uint64_t metadata_bytes = tdvs_mips::FileSize(metadata_path);
  for (auto& record : records) {
    record.graph_index_bytes = tdvs_mips::CheckedAddBytes(
        record.graph_index_bytes, metadata_bytes, "NAPG query index");
  }
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
    std::cerr << "tdvs_napg: " << error.what() << '\n';
    return 2;
  }
}
