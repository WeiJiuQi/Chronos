#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "evaluation.h"
#include "hnswlib.h"

namespace {

// Latest commit on stanis-morozov/ip-nsw master. The four vendored core
// headers retain upstream behavior and are covered by TDVS_SOURCE_FINGERPRINT.
constexpr const char* kIpNswUpstreamCommit =
    "884d9a8e9562e173fecd8450d12af502edd25149";
constexpr const char* kMetadataMagic = "chronos_ip_nsw_upstream_v1";

std::string MetadataPath(const std::string& index_path) {
  return index_path + ".ip_nsw.meta";
}

size_t ParseMetadataSize(const std::map<std::string, std::string>& fields,
                         const std::string& key) {
  const auto found = fields.find(key);
  if (found == fields.end()) {
    throw std::runtime_error("Missing ip-NSW metadata field: " + key);
  }
  try {
    return tdvs_mips::ParseDecimalSize(
        found->second, "ip-NSW metadata field " + key);
  } catch (const std::invalid_argument&) {
    throw std::runtime_error("Invalid ip-NSW metadata field: " + key);
  }
}

std::string ParseMetadataString(
    const std::map<std::string, std::string>& fields, const std::string& key) {
  const auto found = fields.find(key);
  if (found == fields.end() || found->second.empty()) {
    throw std::runtime_error("Missing ip-NSW metadata field: " + key);
  }
  return found->second;
}

struct Metadata {
  std::string upstream_commit;
  size_t base_rows = 0;
  size_t dim = 0;
  size_t stored_stride = 0;
  size_t m = 0;
  size_t ef_construction = 0;
  size_t random_seed = 0;
  size_t build_threads = 0;
};

void WriteMetadata(const std::string& path, const Metadata& metadata) {
  std::ofstream output(path, std::ios::trunc);
  if (!output) throw std::runtime_error("Cannot write ip-NSW metadata: " + path);
  output << kMetadataMagic << '\n'
         << "upstream_commit=" << metadata.upstream_commit << '\n'
         << "base_rows=" << metadata.base_rows << '\n'
         << "dim=" << metadata.dim << '\n'
         << "stored_stride=" << metadata.stored_stride << '\n'
         << "M=" << metadata.m << '\n'
         << "efConstruction=" << metadata.ef_construction << '\n'
         << "random_seed=" << metadata.random_seed << '\n'
         << "build_threads=" << metadata.build_threads << '\n';
  output.flush();
  if (!output) throw std::runtime_error("Failed to write ip-NSW metadata: " + path);
}

Metadata ReadMetadata(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open ip-NSW metadata: " + path);
  std::string line;
  if (!std::getline(input, line) || line != kMetadataMagic) {
    throw std::runtime_error("Invalid ip-NSW metadata header: " + path);
  }
  std::map<std::string, std::string> fields;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const size_t equals = line.find('=');
    if (equals == std::string::npos || equals == 0) {
      throw std::runtime_error("Malformed ip-NSW metadata line in " + path);
    }
    fields[line.substr(0, equals)] = line.substr(equals + 1);
  }
  Metadata metadata;
  metadata.upstream_commit = ParseMetadataString(fields, "upstream_commit");
  metadata.base_rows = ParseMetadataSize(fields, "base_rows");
  metadata.dim = ParseMetadataSize(fields, "dim");
  metadata.stored_stride = ParseMetadataSize(fields, "stored_stride");
  metadata.m = ParseMetadataSize(fields, "M");
  metadata.ef_construction = ParseMetadataSize(fields, "efConstruction");
  metadata.random_seed = ParseMetadataSize(fields, "random_seed");
  metadata.build_threads = ParseMetadataSize(fields, "build_threads");
  return metadata;
}

void ValidateMetadata(const Metadata& metadata, size_t rows, size_t dim,
                      size_t stride) {
  if (metadata.upstream_commit != kIpNswUpstreamCommit) {
    throw std::runtime_error(
        "ip-NSW index was not built by the pinned stanis-morozov upstream core");
  }
  if (metadata.base_rows != rows || metadata.dim != dim ||
      metadata.stored_stride != stride) {
    throw std::runtime_error(
        "ip-NSW metadata does not match --base-count/--dim or aligned stride");
  }
  if (metadata.m < 2 || metadata.ef_construction < metadata.m ||
      metadata.random_seed != 100) {
    throw std::runtime_error("ip-NSW metadata contains invalid build parameters");
  }
}

void PrintHelp() {
  std::cout
      << "ip-NSW runner using stanis-morozov/ip-nsw (commit 884d9a8)\n\n"
      << "Build (the transformed base must NOT be normalized):\n"
      << "  tdvs_ip_nsw --mode build --base BASE.fbin --index OUT.ip_nsw\\\n"
      << "      --dim 1536 [--base-count 999000 --m 32 --ef-construction 1024\\\n"
      << "      --random-seed 100 --build-threads 64]\n"
      << "  Pass --build-threads explicitly for controlled multi-method runs.\n\n"
      << "Search (the serialized index embeds the base; latency is always serial):\n"
      << "  tdvs_ip_nsw --mode search --query QUERY.fbin --gt GT.bin\\\n"
      << "      --index OUT.ip_nsw --base-count 999000 --dim 1536\\\n"
      << "      --query-count 1000 --gt-width 100 --gt-type int64 --k 100\\\n"
      << "      --ef-list 100,150,200,300,500,700 [--normalize-query true\\\n"
      << "      --repeats 3 --seed 42\\\n"
      << "      --output-csv ip_nsw.csv]\n\n"
      << "Load-only footprint (run as a fresh process):\n"
      << "  tdvs_ip_nsw --mode footprint --index OUT.ip_nsw --dim 1536\\\n"
      << "      --base-count 999000 [--output-csv ip_nsw_footprint.csv]\n";
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
  if (base.rows == 0) throw std::invalid_argument("ip-NSW base must not be empty");

  const size_t m = args.GetSize("m", 32);
  const size_t ef_construction = args.GetSize("ef-construction", 1024);
  const size_t random_seed = args.GetSize("random-seed", 100);
  if (m < 2) throw std::invalid_argument("--m must be at least 2");
  if (ef_construction < m) {
    throw std::invalid_argument("--ef-construction must be at least --m");
  }
  if (m > std::numeric_limits<unsigned>::max() ||
      ef_construction > std::numeric_limits<unsigned>::max() ||
      random_seed > std::numeric_limits<unsigned>::max()) {
    throw std::invalid_argument("ip-NSW parameters exceed the supported integer range");
  }
  if (random_seed != 100) {
    throw std::invalid_argument(
        "The upstream ip-NSW core fixes its construction seed at 100; "
        "--random-seed must be 100");
  }
  tdvs_mips::ConfigureOpenMP(build_threads);

  // The transformed vector norm encodes decay, so the base is deliberately
  // left unnormalized.  Padding values are zero and do not change dot products.
  hnswlib::L2Space space(base.stride);
  hnswlib::HierarchicalNSW<float> index(
      &space, base.rows, m, ef_construction);

  const auto build_start = tdvs_mips::SteadyClock::now();
  index.addPoint(base.Row(0), static_cast<labeltype>(0));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(build_threads) if(build_threads > 1)
#endif
  for (long long row = 1; row < static_cast<long long>(base.rows); ++row) {
    index.addPoint(base.Row(static_cast<size_t>(row)),
                   static_cast<labeltype>(row));
  }
  const auto build_end = tdvs_mips::SteadyClock::now();
  index.SaveIndex(index_path);
  const std::string metadata_path = MetadataPath(index_path);
  WriteMetadata(metadata_path,
                Metadata{kIpNswUpstreamCommit, base.rows, base.dim, base.stride,
                         m, ef_construction, random_seed, build_threads});
  const auto save_end = tdvs_mips::SteadyClock::now();

  const double preprocess_seconds = 0.0;
  const double core_build_seconds =
      std::chrono::duration<double>(build_end - build_start).count();
  const double serialize_seconds =
      std::chrono::duration<double>(save_end - build_end).count();
  const double serving_index_pipeline_seconds =
      preprocess_seconds + core_build_seconds + serialize_seconds;

  std::cout << "method=ip-NSW (stanis-morozov upstream implementation)\n"
            << "method_commit=" << kIpNswUpstreamCommit << '\n'
            << "source_fingerprint=" << TDVS_SOURCE_FINGERPRINT << '\n'
            << "base_rows=" << base.rows << '\n'
            << "dim=" << base.dim << '\n'
            << "stored_stride=" << base.stride << '\n'
            << "M=" << m << '\n'
            << "efConstruction=" << ef_construction << '\n'
            << "random_seed=" << random_seed << '\n'
            << "build_threads=" << build_threads << '\n'
            << "parallel_build_thread_safety=upstream_openmp_best_effort\n"
            << "parallel_build_reproducible=" << (build_threads == 1 ? "true" : "false") << '\n'
            << "preprocess_seconds=" << preprocess_seconds << '\n'
            << "core_build_seconds=" << core_build_seconds << '\n'
            // Legacy field retained for existing result parsers. For ip-NSW
            // there is no required preprocessing, so it equals core build time.
            << "build_seconds=" << core_build_seconds << '\n'
            << "serialize_seconds=" << serialize_seconds << '\n'
            << "serving_index_pipeline_seconds="
            << serving_index_pipeline_seconds << '\n';
  const uint64_t serialized_index_bytes = tdvs_mips::CheckedAddBytes(
      tdvs_mips::FileSize(index_path), tdvs_mips::FileSize(metadata_path),
      "ip-NSW serialized index");
  std::cout << "serialized_index_bytes=" << serialized_index_bytes << '\n'
            << "index_bytes=" << serialized_index_bytes << '\n'
            << "metadata_path=" << metadata_path << '\n';
  return 0;
}

int RunFootprint(const tdvs_mips::Args& args) {
  const size_t dim = args.RequireSize("dim");
  const size_t base_count = args.RequireSize("base-count");
  const std::string index_path = args.Require("index");
  if (dim == 0 || base_count == 0) {
    throw std::invalid_argument("--dim and --base-count must be positive");
  }
  const size_t stored_stride = (dim + 7U) & ~size_t(7U);
  const std::string metadata_path = MetadataPath(index_path);
  const Metadata metadata = ReadMetadata(metadata_path);
  ValidateMetadata(metadata, base_count, dim, stored_stride);
  const uint64_t rss_baseline = tdvs_mips::CurrentRSSBytes();

  (void)tdvs_mips::FileSize(index_path);
  hnswlib::L2Space space(stored_stride);
  hnswlib::HierarchicalNSW<float> index(&space, index_path, false);
  std::cout.flush();
  if (index.cur_element_count != base_count) {
    throw std::runtime_error("--base-count does not match the loaded ip-NSW index");
  }
  if (index.label_offset_ < index.offsetData_ ||
      index.label_offset_ - index.offsetData_ != stored_stride * sizeof(float)) {
    throw std::runtime_error("--dim does not match the loaded ip-NSW index");
  }

  tdvs_mips::FootprintRecord record;
  record.method = "ip-NSW (stanis-morozov upstream implementation)";
  record.method_commit = kIpNswUpstreamCommit;
  record.base_rows = base_count;
  record.dim = dim;
  record.stored_stride = stored_stride;
  record.index_contains_base = true;
  record.serialized_index_bytes = tdvs_mips::CheckedAddBytes(
      tdvs_mips::FileSize(index_path), tdvs_mips::FileSize(metadata_path),
      "ip-NSW serialized index");
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
  auto gt = tdvs_mips::LoadGroundTruth(
      gt_path, queries.rows, gt_width, args.Get("gt-type", "int64"), base_count);
  const size_t k = args.GetSize("k", 100);
  if (k > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("--k exceeds the upstream integer range");
  }
  const std::string ef_text =
      args.Has("ef-list") ? args.Require("ef-list") : args.Require("ef-search");
  const auto ef_values = tdvs_mips::ParseSizeList(ef_text);
  const std::string metadata_path = MetadataPath(index_path);
  const Metadata metadata = ReadMetadata(metadata_path);
  ValidateMetadata(metadata, base_count, dim, queries.stride);

  (void)tdvs_mips::FileSize(index_path);
  hnswlib::L2Space space(queries.stride);
  const auto load_start = tdvs_mips::SteadyClock::now();
  hnswlib::HierarchicalNSW<float> index(&space, index_path, false);
  const auto load_end = tdvs_mips::SteadyClock::now();
  std::cout.flush();
  if (index.cur_element_count != base_count || index.maxelements_ < base_count) {
    throw std::runtime_error(
        "Loaded ip-NSW index contains N=" +
        std::to_string(index.cur_element_count) + " (capacity " +
        std::to_string(index.maxelements_) + "); expected " +
        std::to_string(base_count));
  }
  if (index.label_offset_ < index.offsetData_ ||
      index.label_offset_ - index.offsetData_ != queries.stride * sizeof(float)) {
    throw std::runtime_error("Loaded ip-NSW index dimension does not match --dim");
  }
  const size_t max_search_expansions = args.GetSize("max-search-expansions", 0);
  if (max_search_expansions != 0) {
    throw std::invalid_argument(
        "The original upstream ip-NSW implementation does not support "
        "--max-search-expansions; use 0");
  }
  std::cerr << "ip-NSW index load: "
            << std::chrono::duration<double>(load_end - load_start).count()
            << " s\n";

  tdvs_mips::EvaluationConfig config;
  config.k = k;
  config.repeats = args.GetSize("repeats", 3);
  config.seed = args.GetUnsigned("seed", 42);
  config.search_values = ef_values;
  config.method = "ip-NSW (stanis-morozov upstream implementation)";
  config.method_commit = kIpNswUpstreamCommit;
  config.index_path = index_path;
  config.search_param_name = "efSearch";
  config.index_contains_base = true;
  config.timed_scope = "serial_search_and_topk_materialization";
  config.distance_count_definition = "upstream_dist_calc_counter";

  auto records = tdvs_mips::RunSearchEvaluation(
      base_shape, queries, gt, config,
      [&](const float* query, size_t ef_search, unsigned* result) {
        index.setEf(ef_search);
        index.dist_calc = 0;
        const auto search_start = tdvs_mips::SteadyClock::now();
        auto heap = index.searchKnn(
            const_cast<float*>(query), static_cast<int>(k));
        if (heap.size() != k) {
          throw std::runtime_error("ip-NSW returned fewer than k results");
        }

        // The upstream heap pops the worst retained item first. Fill from the
        // back so the runner returns IDs in best-first order without another
        // allocation or a separate reverse pass.
        for (size_t position = k; position > 0; --position) {
          const labeltype label = heap.top().second;
          heap.pop();
          if (label >= base_count ||
              label > static_cast<labeltype>(std::numeric_limits<unsigned>::max())) {
            throw std::runtime_error("ip-NSW returned an out-of-range label");
          }
          result[position - 1] = static_cast<unsigned>(label);
        }
        const auto search_end = tdvs_mips::SteadyClock::now();
        const size_t computations = index.dist_calc;
        if (computations > static_cast<size_t>(std::numeric_limits<int>::max())) {
          throw std::runtime_error("ip-NSW distance count exceeds int range");
        }
        return tdvs_mips::SearchMeasurement{
            static_cast<int>(computations),
            std::chrono::duration<double>(search_end - search_start).count()};
      });
  const uint64_t metadata_bytes = tdvs_mips::FileSize(metadata_path);
  for (auto& record : records) {
    record.graph_index_bytes = tdvs_mips::CheckedAddBytes(
        record.graph_index_bytes, metadata_bytes, "ip-NSW query index");
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
    std::cerr << "tdvs_ip_nsw: " << error.what() << '\n';
    return 2;
  }
}
