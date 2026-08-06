#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "evaluation.h"
#include "mips/index_mips.h"
#include "ssg/index_random.h"

namespace {

constexpr const char* kPspCommit = "d3eb2e09059828a883ce96c9f6a9b98a39e824ed";
constexpr const char* kMetadataMagic = "chronos_psp_upstream_v1";

std::string MetadataPath(const std::string& index_path) {
  return index_path + ".psp.meta";
}

void WriteMetadata(const std::string& index_path, const tdvs_mips::FloatMatrix& base,
                   const tdvs_mips::KnngInfo& knng, unsigned build_l,
                   unsigned r, double angle, unsigned m, unsigned seed,
                   unsigned build_threads) {
  tdvs_mips::WriteIndexMetadata(MetadataPath(index_path), kMetadataMagic, {
      {"upstream_commit", kPspCommit},
      {"source_fingerprint", TDVS_SOURCE_FINGERPRINT},
      {"base_rows", std::to_string(base.rows)},
      {"dim", std::to_string(base.dim)},
      {"stored_stride", std::to_string(base.stride)},
      {"knng_degree", std::to_string(knng.degree)},
      {"build_l", std::to_string(build_l)},
      {"r", std::to_string(r)},
      {"angle", std::to_string(angle)},
      {"m", std::to_string(m)},
      {"seed", std::to_string(seed)},
      {"build_threads", std::to_string(build_threads)},
      {"query_function", "Search_Mips_IP_Cal_with_No_SN"},
  });
}

void ValidateMetadata(const std::string& index_path, size_t rows, size_t dim,
                      size_t stride) {
  const auto fields = tdvs_mips::ReadIndexMetadata(
      MetadataPath(index_path), kMetadataMagic);
  tdvs_mips::RequireIndexMetadataValue(fields, "upstream_commit", kPspCommit);
  tdvs_mips::RequireIndexMetadataValue(
      fields, "source_fingerprint", TDVS_SOURCE_FINGERPRINT);
  tdvs_mips::RequireIndexMetadataValue(
      fields, "query_function", "Search_Mips_IP_Cal_with_No_SN");
  tdvs_mips::RequireIndexMetadataSize(fields, "base_rows", rows);
  tdvs_mips::RequireIndexMetadataSize(fields, "dim", dim);
  tdvs_mips::RequireIndexMetadataSize(fields, "stored_stride", stride);
}

class CheckedIndexMips : public efanna2e::IndexMips {
 public:
  using efanna2e::IndexMips::IndexMips;

  void ValidateGraph(size_t expected_rows) const {
    if (final_graph_.size() != expected_rows) {
      throw std::runtime_error(
          "PSP index row count does not match the supplied base matrix");
    }
    for (const auto& neighbors : final_graph_) {
      for (unsigned id : neighbors) {
        if (id >= expected_rows) {
          throw std::runtime_error(
              "PSP index contains an out-of-range neighbor ID");
        }
      }
    }
  }
};

void PrintHelp() {
  std::cout
      << "PSP runner for transformed TDVS MIPS data\n\n"
      << "Build:\n"
      << "  tdvs_psp --mode build --base BASE.fbin --knng BASE.knng --index OUT.psp\\\n"
      << "           --dim 1536 [--knng-k 400 --build-threads 64\\\n"
      << "           --build-l 800 --r 40 --angle 60 --m 5 --seed 42\\\n"
      << "           --knng-build-seconds SEC]\n\n"
      << "Search (always serial for comparable latency):\n"
      << "  tdvs_psp --mode search --base BASE.fbin --query QUERY.fbin --gt GT.bin\\\n"
      << "           --index OUT.psp --dim 1536 --gt-width 100 --k 100\\\n"
      << "           --search-l 100,150,200,300,500,700 [--query-count 1000\\\n"
      << "           --normalize-query true --repeats 3\\\n"
      << "           --seed 42 --output-csv psp.csv]\n"
      << "  Search calls the No-SN function enabled by the authors' public test.\n"
      << "  L_search is the only query-time sweep parameter.\n\n"
      << "Load-only footprint (run as a fresh process):\n"
      << "  tdvs_psp --mode footprint --base BASE.fbin --index OUT.psp\\\n"
      << "           --dim 1536 [--base-count 999000 --output-csv psp_footprint.csv]\n";
}

int RunBuild(const tdvs_mips::Args& args) {
  if (args.Has("sn") || args.Has("sn-clusters") ||
      args.Has("sn-navigation-points") || args.Has("sn-training-points") ||
      args.Has("sn-iterations") || args.Has("sn-sampling") ||
      args.Has("sn-gaussian-target-z") || args.Has("sn-gaussian-bandwidth")) {
    throw std::invalid_argument(
        "SN construction is not provided by the PSP authors' public implementation");
  }
  const size_t dim = args.RequireSize("dim");
  const std::string base_path = args.Require("base");
  const std::string knng_path = args.Require("knng");
  const std::string index_path = args.Require("index");
  auto base = tdvs_mips::LoadRawFloatMatrix(base_path, dim, args.GetSize("base-count", 0));
  tdvs_mips::PrintInputSummary(base);
  const auto knng = tdvs_mips::ValidateKnng(knng_path, base.rows);
  const unsigned expected_knng_k = args.GetUnsigned("knng-k", 400);
  if (knng.degree != expected_knng_k) {
    throw std::invalid_argument(
        "PSP paper-budget build expects kNNG K=" +
        std::to_string(expected_knng_k) + "; file contains K=" +
        std::to_string(knng.degree) +
        ". Pass --knng-k explicitly only for an ablation.");
  }
  std::cerr << "Validated kNN graph: N=" << knng.rows << ", K=" << knng.degree
            << ", bytes=" << knng.bytes << '\n';

  const unsigned build_threads = args.GetUnsigned(
      "build-threads", tdvs_mips::kDefaultBuildThreads);
  tdvs_mips::ConfigureOpenMP(build_threads);
  const unsigned seed = args.GetUnsigned("seed", 42);
  std::srand(seed);

  const tdvs_mips::KnngBuildTiming knng_timing =
      tdvs_mips::ResolveKnngBuildTiming(args, knng_path);

  const unsigned build_l = args.GetUnsigned("build-l", 800);
  const unsigned r = args.GetUnsigned("r", 40);
  const double angle = args.GetDouble("angle", 60.0);
  const unsigned m = args.GetUnsigned("m", 5);
  constexpr unsigned n_try = 10;
  efanna2e::Parameters parameters;
  parameters.Set<unsigned>("L", build_l);
  parameters.Set<unsigned>("R", r);
  parameters.Set<float>("A", static_cast<float>(angle));
  parameters.Set<unsigned>("M", m);
  parameters.Set<unsigned>("n_try", n_try);
  parameters.Set<std::string>("nn_graph_path", knng_path);

  efanna2e::IndexRandom initializer(base.stride, base.rows);
  CheckedIndexMips index(base.stride, base.rows, efanna2e::L2, &initializer);
  const auto start = tdvs_mips::SteadyClock::now();
  index.Build(base.rows, base.values.data(), parameters);
  const auto build_end = tdvs_mips::SteadyClock::now();
  index.ValidateGraph(base.rows);
  const auto validation_end = tdvs_mips::SteadyClock::now();
  index.Save(index_path.c_str());
  WriteMetadata(index_path, base, knng, build_l, r, angle, m, seed,
                build_threads);
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
  std::cout << "method=PSP\n"
            << "method_commit=" << kPspCommit << '\n'
            << "source_fingerprint=" << TDVS_SOURCE_FINGERPRINT << '\n'
            << "base_rows=" << base.rows << '\n'
            << "dim=" << base.dim << '\n'
            << "stored_stride=" << base.stride << '\n'
            << "knng_bytes=" << knng.bytes << '\n'
            << "knng_k=" << knng.degree << '\n'
            << "build_threads=" << build_threads << '\n'
            << "build_l=" << build_l << '\n'
            << "r=" << r << '\n'
            << "angle=" << angle << '\n'
            << "m=" << m << '\n'
            << "n_try=" << n_try << '\n'
            << "seed=" << seed << '\n'
            << "query_variant=authors_public_No_SN\n"
            << "core_implementation=authors_public_algorithm_with_thread_control_adapter\n"
            << "parallel_build_reproducible="
            << "false\n"
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
  std::cout << '\n' << "knng_build_seconds=";
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
            << "psp_build_seconds=" << core_build_seconds << '\n'
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
      "PSP serialized index");
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
  efanna2e::IndexRandom initializer(base.stride, base.rows);
  CheckedIndexMips index(base.stride, base.rows, efanna2e::FAST_L2, &initializer);
  index.SaveData(base.values.data());
  // Load reconstructs the per-vector norm table required by PSP search.
  index.Load(index_path.c_str());
  index.ValidateGraph(base.rows);

  tdvs_mips::FootprintRecord record;
  record.method = "PSP";
  record.method_commit = kPspCommit;
  record.base_rows = base.rows;
  record.dim = base.dim;
  record.stored_stride = base.stride;
  record.index_contains_base = false;
  record.serialized_index_bytes = tdvs_mips::CheckedAddBytes(
      tdvs_mips::FileSize(index_path), tdvs_mips::FileSize(MetadataPath(index_path)),
      "PSP serialized index");
  record.external_base_file_bytes = tdvs_mips::FileSize(base_path);
  record.external_base_allocation_bytes = tdvs_mips::CheckedMulBytes(
      base.values.capacity(), sizeof(float), "PSP external base allocation");
  // norms_ is private upstream state, but Load() unconditionally resizes it to N.
  record.runtime_auxiliary_known_bytes = tdvs_mips::CheckedMulBytes(
      base.rows, sizeof(float), "PSP norm table");
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
  if (args.Has("no-sn")) {
    throw std::invalid_argument(
        "--no-sn is retired; the PSP authors' public No-SN path is always used");
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
    if (value < k || value >= base.rows ||
        value > std::numeric_limits<unsigned>::max()) {
      throw std::invalid_argument(
          "PSP requires k <= each L_search < base-count");
    }
  }

  efanna2e::IndexRandom initializer(base.stride, base.rows);
  CheckedIndexMips index(base.stride, base.rows, efanna2e::FAST_L2, &initializer);
  const auto load_start = tdvs_mips::SteadyClock::now();
  index.SaveData(base.values.data());
  index.Load(index_path.c_str());
  const auto load_end = tdvs_mips::SteadyClock::now();
  index.ValidateGraph(base.rows);
  std::cerr << "PSP graph load + norm initialization: "
            << std::chrono::duration<double>(load_end - load_start).count() << " s\n";
  std::cerr << "PSP search variant: authors' public No-SN path\n";

  tdvs_mips::EvaluationConfig config;
  config.k = k;
  config.repeats = args.GetSize("repeats", 3);
  config.seed = args.GetUnsigned("seed", 42);
  config.search_values = search_values;
  config.method = "PSP";
  config.method_commit = kPspCommit;
  config.index_path = index_path;
  config.timed_scope = "serial_search_and_topk_materialization";
  config.distance_count_definition = "upstream_PSP_dis_cal_counter";
  std::srand(config.seed);

  auto records = tdvs_mips::RunSearchEvaluation(
      base, queries, gt, config,
      [&](const float* query, size_t search_l, unsigned* result) {
        efanna2e::Parameters parameters;
        parameters.Set<unsigned>("L_search", static_cast<unsigned>(search_l));
        const auto search_start = tdvs_mips::SteadyClock::now();
        const int computations = index.Search_Mips_IP_Cal_with_No_SN(
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
    throw std::invalid_argument(
        "--mode must be build, search, or footprint");
  } catch (const std::exception& error) {
    std::cerr << "tdvs_psp: " << error.what() << '\n';
    return 2;
  }
}
