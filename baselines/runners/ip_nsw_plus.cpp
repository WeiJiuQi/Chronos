#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "evaluation.h"
#include "hnswlib.h"

namespace {

constexpr const char* kIpNswPlusCommit =
    "3a68666dfaef7fa7691889c811aedafb587d8f22";

void PrintHelp() {
  std::cout
      << "ip-NSW+ (GraphMIPS) runner for transformed TDVS MIPS data\n\n"
      << "Build (transformed base is deliberately NOT normalized):\n"
      << "  tdvs_ip_nsw_plus --mode build --base BASE.fbin --index OUT.index\\\n"
      << "      --dim 1536 [--base-count 999000 --m 32 --ef-construction 1024\\\n"
      << "      --cos-m 10 --cos-ef-construction 100 --build-threads 64]\n"
      << "  Parallel construction uses a thread-safety backport from current\n"
      << "  hnswlib; scheduling still makes multi-threaded indexes non-bitwise-\n"
      << "  reproducible. Use --build-threads 1 for strict reproducibility.\n\n"
      << "Search (serialized index embeds the transformed base):\n"
      << "  tdvs_ip_nsw_plus --mode search --query QUERY.fbin --gt GT.bin\\\n"
      << "      --index OUT.index --dim 1536 --gt-width 100 --k 100\\\n"
      << "      --ef-list 100,150,200,300,500,700 [--base-count 999000\\\n"
      << "      --cos-ef-search 1 --query-count 1000 --normalize-query true\\\n"
      << "      --repeats 3 --method-label LABEL\\\n"
      << "      --output-csv results.csv]\n\n"
      << "Load-only footprint (run as a fresh process):\n"
      << "  tdvs_ip_nsw_plus --mode footprint --index OUT.index --dim 1536\\\n"
      << "      [--base-count 999000 --method-label LABEL\\\n"
      << "      --output-csv ip_nsw_plus_footprint.csv]\n";
}

float RowNorm(const float* row, size_t stride) {
  double squared = 0.0;
  for (size_t j = 0; j < stride; ++j) {
    const double value = row[j];
    squared += value * value;
  }
  const double norm = std::sqrt(squared);
  if (!(norm > 0.0) || !std::isfinite(norm) ||
      norm > std::numeric_limits<float>::max()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return static_cast<float>(norm);
}

int RunBuild(const tdvs_mips::Args& args) {
  const size_t dim = args.RequireSize("dim");
  const std::string base_path = args.Require("base");
  const std::string index_path = args.Require("index");
  const unsigned build_threads = args.GetUnsigned(
      "build-threads", tdvs_mips::kDefaultBuildThreads);
  if (build_threads == 0) {
    throw std::invalid_argument("--build-threads must be positive");
  }
  auto base = tdvs_mips::LoadRawFloatMatrix(
      base_path, dim, args.GetSize("base-count", 0));
  tdvs_mips::PrintInputSummary(base);

  const size_t m = args.GetSize("m", 32);
  const size_t ef_construction = args.GetSize("ef-construction", 1024);
  const size_t cos_m = args.GetSize("cos-m", 10);
  const size_t cos_ef_construction = args.GetSize("cos-ef-construction", 100);
  if (m == 0 || cos_m == 0 || ef_construction < m ||
      cos_ef_construction < cos_m) {
    throw std::invalid_argument(
        "Require m>0, cos-m>0, ef-construction>=m, and "
        "cos-ef-construction>=cos-m");
  }
  tdvs_mips::ConfigureOpenMP(build_threads);

  hnswlib::L2Space space(base.stride);  // Upstream name; distance is -inner product.
  hnswlib::HierarchicalNSW<float> index(
      &space, base.rows, m, cos_m, ef_construction, cos_ef_construction);
  index.elementNorms.resize(base.rows);

  const auto norm_start = tdvs_mips::SteadyClock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(build_threads > 1)
#endif
  for (size_t i = 0; i < base.rows; ++i) {
    index.elementNorms[i] = RowNorm(base.Row(i), base.stride);
  }
  if (std::any_of(index.elementNorms.begin(), index.elementNorms.end(),
                  [](float norm) { return !(norm > 0.0f) || !std::isfinite(norm); })) {
    throw std::runtime_error(
        "ip-NSW+ requires every transformed base norm to be finite and positive");
  }
  const auto norm_end = tdvs_mips::SteadyClock::now();

  // Insert the initial entry point serially, then use the thread-safe GraphMIPS
  // construction path for the remaining objects.  As with current hnswlib,
  // insertion interleaving can change the resulting graph even though the
  // implementation is data-race-free.
  const auto build_start = tdvs_mips::SteadyClock::now();
  index.addPoint(base.Row(0), static_cast<labeltype>(0));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) if(build_threads > 1)
#endif
  for (size_t i = 1; i < base.rows; ++i) {
    index.addPoint(base.Row(i), static_cast<labeltype>(i));
  }
  const auto build_end = tdvs_mips::SteadyClock::now();
  index.SaveIndex(index_path);
  const auto save_end = tdvs_mips::SteadyClock::now();

  const double preprocess_seconds =
      std::chrono::duration<double>(norm_end - norm_start).count();
  const double core_build_seconds =
      std::chrono::duration<double>(build_end - build_start).count();
  const double serialize_seconds =
      std::chrono::duration<double>(save_end - build_end).count();
  const double serving_index_pipeline_seconds =
      preprocess_seconds + core_build_seconds + serialize_seconds;

  std::cout << "method=ip-NSW+ (GraphMIPS)\n"
            << "method_commit=" << kIpNswPlusCommit << '\n'
            << "source_fingerprint=" << TDVS_SOURCE_FINGERPRINT << '\n'
            << "base_rows=" << base.rows << '\n'
            << "dim=" << base.dim << '\n'
            << "stored_stride=" << base.stride << '\n'
            << "random_seed=100\n"
            << "m=" << m << '\n'
            << "ef_construction=" << ef_construction << '\n'
            << "cos_m=" << cos_m << '\n'
            << "cos_ef_construction=" << cos_ef_construction << '\n'
            << "build_threads=" << build_threads << '\n'
            << "parallel_build_thread_safety=thread_safety_backport_validated\n"
            << "thread_safety_backport=current_hnswlib_rng_and_adjacency_locks\n"
            << "parallel_build_reproducible="
            << (build_threads == 1 ? "true" : "false") << '\n'
            << "preprocess_seconds=" << preprocess_seconds << '\n'
            // Legacy GraphMIPS-specific name retained for existing parsers.
            << "norm_seconds=" << preprocess_seconds << '\n'
            << "core_build_seconds=" << core_build_seconds << '\n'
            // Legacy generic name retained for existing result parsers.
            << "build_seconds=" << core_build_seconds << '\n'
            << "serialize_seconds=" << serialize_seconds << '\n'
            << "serving_index_pipeline_seconds="
            << serving_index_pipeline_seconds << '\n';
  const uint64_t serialized_index_bytes = tdvs_mips::FileSize(index_path);
  std::cout << "serialized_index_bytes=" << serialized_index_bytes << '\n'
            << "index_bytes=" << serialized_index_bytes << '\n';
  return 0;
}

int RunFootprint(const tdvs_mips::Args& args) {
  const size_t dim = args.RequireSize("dim");
  if (dim == 0) throw std::invalid_argument("--dim must be positive");
  const size_t stored_stride = (dim + 7U) & ~size_t(7U);
  const std::string index_path = args.Require("index");
  const uint64_t rss_baseline = tdvs_mips::CurrentRSSBytes();

  hnswlib::L2Space space(stored_stride);
  hnswlib::HierarchicalNSW<float> index(&space, index_path, false);
  if (index.cur_element_count == 0 ||
      index.cur_element_count != index.maxelements_) {
    throw std::runtime_error("Expected a non-empty, fully populated ip-NSW+ index");
  }
  if (args.Has("base-count") &&
      args.RequireSize("base-count") != index.cur_element_count) {
    throw std::runtime_error("--base-count does not match the loaded ip-NSW+ index");
  }
  const size_t serialized_data_bytes = index.label_offset_ - index.offsetData_;
  if (serialized_data_bytes != stored_stride * sizeof(float)) {
    throw std::runtime_error("--dim does not match the loaded ip-NSW+ index");
  }
  if (index.elementNorms.size() != index.maxelements_) {
    throw std::runtime_error("ip-NSW+ index has an invalid norm table");
  }

  tdvs_mips::FootprintRecord record;
  record.method = args.Get("method-label", "ip-NSW+ (GraphMIPS)");
  record.method_commit = kIpNswPlusCommit;
  record.base_rows = index.cur_element_count;
  record.dim = dim;
  record.stored_stride = stored_stride;
  record.index_contains_base = true;
  record.serialized_index_bytes = tdvs_mips::FileSize(index_path);
  record.runtime_auxiliary_known_bytes = tdvs_mips::CheckedMulBytes(
      index.elementNorms.capacity(), sizeof(float), "ip-NSW+ norm table");
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
  const std::string query_path = args.Require("query");
  const std::string gt_path = args.Require("gt");
  const std::string index_path = args.Require("index");
  auto queries = tdvs_mips::LoadRawFloatMatrix(
      query_path, dim, args.GetSize("query-count", 0));
  if (args.GetBool("normalize-query", true)) tdvs_mips::NormalizeRows(queries);

  hnswlib::L2Space space(queries.stride);
  const auto load_start = tdvs_mips::SteadyClock::now();
  hnswlib::HierarchicalNSW<float> index(&space, index_path, false);
  const auto load_end = tdvs_mips::SteadyClock::now();
  if (index.cur_element_count == 0 || index.cur_element_count != index.maxelements_) {
    throw std::runtime_error("Expected a non-empty, fully populated ip-NSW+ index");
  }
  if (args.Has("base-count") &&
      args.RequireSize("base-count") != index.cur_element_count) {
    throw std::runtime_error("--base-count does not match the loaded ip-NSW+ index");
  }
  const size_t serialized_data_bytes = index.label_offset_ - index.offsetData_;
  if (serialized_data_bytes != queries.stride * sizeof(float)) {
    throw std::runtime_error("Query dimension does not match the ip-NSW+ index");
  }
  if (index.elementNorms.size() != index.maxelements_) {
    throw std::runtime_error("ip-NSW+ index has an invalid norm table");
  }
  for (float norm : index.elementNorms) {
    if (!(norm > 0.0f) || !std::isfinite(norm)) {
      throw std::runtime_error("ip-NSW+ index contains an invalid vector norm");
    }
  }

  tdvs_mips::FloatMatrix base_shape;
  base_shape.rows = index.cur_element_count;
  base_shape.dim = dim;
  base_shape.stride = queries.stride;
  tdvs_mips::PrintInputSummary(base_shape, &queries);
  std::cerr << "ip-NSW+ load seconds: "
            << std::chrono::duration<double>(load_end - load_start).count() << '\n';

  const size_t gt_width = args.GetSize("gt-width", 100);
  auto gt = tdvs_mips::LoadGroundTruth(
      gt_path, queries.rows, gt_width, args.Get("gt-type", "int64"), base_shape.rows);
  const size_t k = args.GetSize("k", 100);
  const std::string ef_text =
      args.Has("ef-list") ? args.Require("ef-list") : args.Require("ef-search");
  const auto ef_values = tdvs_mips::ParseSizeList(ef_text);
  const size_t cos_ef_search = args.GetSize("cos-ef-search", 1);
  if (cos_ef_search == 0) {
    throw std::invalid_argument("--cos-ef-search must be positive");
  }
  index.setCosEf(cos_ef_search);

  tdvs_mips::EvaluationConfig config;
  config.k = k;
  config.repeats = args.GetSize("repeats", 3);
  config.seed = args.GetUnsigned("seed", 42);
  config.search_values = ef_values;
  config.method = args.Get(
      "method-label", "ip-NSW+ (GraphMIPS; cos_ef_search=" +
                          std::to_string(cos_ef_search) + ")");
  config.method_commit = kIpNswPlusCommit;
  config.index_path = index_path;
  config.search_param_name = "efSearch";
  config.index_contains_base = true;
  config.timed_scope = "serial_search_and_topk_materialization";
  config.distance_count_definition =
      "upstream_dist_calc_counter_mixed_angular_and_ip_dot_products";

  auto records = tdvs_mips::RunSearchEvaluation(
      base_shape, queries, gt, config,
      [&](const float* query, size_t ef_search, unsigned* result) {
        index.setEf(ef_search);
        index.dist_calc = 0;
        const auto search_start = tdvs_mips::SteadyClock::now();
        auto heap = index.searchKnn(const_cast<float*>(query), static_cast<int>(k));
        if (heap.size() != k) {
          throw std::runtime_error("ip-NSW+ returned fewer than k results");
        }
        // The heap pops the worst retained item first.  Fill the caller's
        // preallocated output from the back, matching ip-NSW and avoiding a
        // method-specific allocation inside the timed interval.
        for (size_t position = k; position > 0; --position) {
          const labeltype label = heap.top().second;
          heap.pop();
          if (label >= base_shape.rows) {
            throw std::runtime_error("ip-NSW+ returned an out-of-range label");
          }
          result[position - 1] = static_cast<unsigned>(label);
        }
        const auto search_end = tdvs_mips::SteadyClock::now();
        if (index.dist_calc > static_cast<size_t>(std::numeric_limits<int>::max())) {
          throw std::overflow_error("ip-NSW+ distance counter exceeds int range");
        }
        return tdvs_mips::SearchMeasurement{
            static_cast<int>(index.dist_calc),
            std::chrono::duration<double>(search_end - search_start).count()};
      });
  tdvs_mips::WriteCsv(args.Get("output-csv", ""), records);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    tdvs_mips::Args args(argc, argv);
    if (argc == 1 || args.Has("help")) {
      PrintHelp();
      return 0;
    }
    const std::string mode = args.Require("mode");
    if (mode == "build") return RunBuild(args);
    if (mode == "search") return RunSearch(args);
    if (mode == "footprint") return RunFootprint(args);
    throw std::invalid_argument("--mode must be build, search, or footprint");
  } catch (const std::exception& error) {
    std::cerr << "tdvs_ip_nsw_plus: " << error.what() << '\n';
    return 2;
  }
}
