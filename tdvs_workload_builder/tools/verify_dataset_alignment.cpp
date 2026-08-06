#include "alignment.h"
#include "io.h"
#include "time_metadata.h"
#include "validation.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kTimestampRangeEpsilon = 1e-9;

struct Options {
    std::string base;
    std::string query;
    std::string timestamps;
    std::string transformed_base;
    std::string transformed_query;
    std::string source;
    std::string base_original_indices;
    std::string query_original_indices;
    std::string base_source_row_ids;
    std::string query_source_row_ids;
    std::string groundtruth;
    std::string report;
    std::string dataset_name = "unspecified";
    std::string distribution = "unspecified";
    std::string mode = "multiplicative";
    std::size_t dim = 0;
    std::size_t gt_queries = 0;
    std::size_t gt_width = 0;
    std::size_t base_source_rows = 0;
    std::size_t query_source_rows = 0;
    std::size_t chunk_rows = 1024;
    std::size_t progress_every_rows = 100000;
    double half_life = 0.0;
    double query_time = 0.0;
    double alpha = -1.0;
    double absolute_tolerance = 1e-7;
    double relative_tolerance = 1e-6;
    double unit_norm_tolerance = 1e-4;
    int threads = 64;
    bool cosine = true;
    bool require_unit = true;
    bool same_source = false;
};

struct TimestampAudit {
    std::size_t rows = 0;
    std::size_t nonfinite = 0;
    std::size_t out_of_range = 0;
    double minimum = 0.0;
    double maximum = 0.0;

    bool passed() const { return nonfinite == 0 && out_of_range == 0; }
};

void usage(const char* program) {
    std::cout
        << "Usage: " << program << " \\\n"
        << "  --base RAW.fbin --timestamps TIME.f32bin \\\n"
        << "  --transformed-base MIPS.fbin --dim D \\\n"
        << "  --half-life H --query-time T [options]\n\n"
        << "Transform contract:\n"
        << "  --mode multiplicative|additive      default multiplicative\n"
        << "  --query QUERY.fbin                  canonical query audit; required for additive\n"
        << "  --transformed-query MIPS_QUERY.fbin required for additive\n"
        << "  --alpha A                           required for additive\n"
        << "  --metric cosine|ip                  default cosine\n"
        << "  --threads N                         default 64\n"
        << "  --chunk-rows N                      default 1024\n"
        << "  --progress-every-rows N             default 100000; 0 disables progress\n"
        << "  --absolute-tolerance X              default 1e-7\n"
        << "  --relative-tolerance X              default 1e-6\n\n"
        << "Canonical checks:\n"
        << "  --require-unit 0|1                  default 1\n"
        << "  --unit-norm-tolerance X             default 1e-4\n"
        << "  --source SOURCE.fbin --base-original-indices IDS.txt\n"
        << "  --query-original-indices IDS.txt    optional legacy source byte check\n"
        << "  --base-source-row-ids IDS.u64bin --query-source-row-ids IDS.u64bin\n"
        << "  --base-source-rows N --query-source-rows N --same-source 0|1\n"
        << "  --gt GT.bin --gt-queries Q --gt-width K\n"
        << "  --dataset-name NAME --distribution NAME --report FILE\n";
}

std::size_t parse_size(const char* value, const char* name, bool allow_zero = false) {
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (value[consumed] != '\0' || (!allow_zero && parsed == 0) ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string(name) + " must be a valid integer");
    }
    return static_cast<std::size_t>(parsed);
}

double parse_double(const char* value, const char* name, bool allow_zero = false) {
    std::size_t consumed = 0;
    const double parsed = std::stod(value, &consumed);
    if (value[consumed] != '\0' || !std::isfinite(parsed) ||
        (!allow_zero && parsed <= 0.0) || (allow_zero && parsed < 0.0)) {
        throw std::invalid_argument(std::string(name) + " has an invalid numeric value");
    }
    return parsed;
}

bool parse_bool(const char* value, const char* name) {
    const std::string text(value);
    if (text == "1" || text == "true") return true;
    if (text == "0" || text == "false") return false;
    throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (flag == "--base") options.base = value("--base");
        else if (flag == "--query") options.query = value("--query");
        else if (flag == "--timestamps") options.timestamps = value("--timestamps");
        else if (flag == "--transformed-base") options.transformed_base = value("--transformed-base");
        else if (flag == "--transformed-query") options.transformed_query = value("--transformed-query");
        else if (flag == "--source") options.source = value("--source");
        else if (flag == "--base-original-indices") {
            options.base_original_indices = value("--base-original-indices");
        } else if (flag == "--query-original-indices") {
            options.query_original_indices = value("--query-original-indices");
        } else if (flag == "--base-source-row-ids") {
            options.base_source_row_ids = value("--base-source-row-ids");
        } else if (flag == "--query-source-row-ids") {
            options.query_source_row_ids = value("--query-source-row-ids");
        } else if (flag == "--gt") options.groundtruth = value("--gt");
        else if (flag == "--report") options.report = value("--report");
        else if (flag == "--dataset-name") options.dataset_name = value("--dataset-name");
        else if (flag == "--distribution") options.distribution = value("--distribution");
        else if (flag == "--mode") options.mode = value("--mode");
        else if (flag == "--dim") options.dim = parse_size(value("--dim"), "dim");
        else if (flag == "--gt-queries") options.gt_queries = parse_size(value("--gt-queries"), "gt-queries");
        else if (flag == "--gt-width") options.gt_width = parse_size(value("--gt-width"), "gt-width");
        else if (flag == "--base-source-rows") {
            options.base_source_rows = parse_size(value("--base-source-rows"), "base-source-rows");
        } else if (flag == "--query-source-rows") {
            options.query_source_rows = parse_size(value("--query-source-rows"), "query-source-rows");
        } else if (flag == "--chunk-rows") options.chunk_rows = parse_size(value("--chunk-rows"), "chunk-rows");
        else if (flag == "--progress-every-rows") {
            options.progress_every_rows = parse_size(
                value("--progress-every-rows"), "progress-every-rows", true);
        } else if (flag == "--threads") {
            options.threads = static_cast<int>(parse_size(value("--threads"), "threads"));
        } else if (flag == "--half-life") options.half_life = parse_double(value("--half-life"), "half-life");
        else if (flag == "--query-time") options.query_time = parse_double(value("--query-time"), "query-time");
        else if (flag == "--alpha") options.alpha = parse_double(value("--alpha"), "alpha", true);
        else if (flag == "--absolute-tolerance") {
            options.absolute_tolerance = parse_double(value("--absolute-tolerance"), "absolute-tolerance", true);
        } else if (flag == "--relative-tolerance") {
            options.relative_tolerance = parse_double(value("--relative-tolerance"), "relative-tolerance", true);
        } else if (flag == "--unit-norm-tolerance") {
            options.unit_norm_tolerance = parse_double(value("--unit-norm-tolerance"), "unit-norm-tolerance", true);
        } else if (flag == "--require-unit") options.require_unit = parse_bool(value("--require-unit"), "require-unit");
        else if (flag == "--same-source") options.same_source = parse_bool(value("--same-source"), "same-source");
        else if (flag == "--metric") {
            const std::string metric = value("--metric");
            if (metric == "cosine") options.cosine = true;
            else if (metric == "ip") options.cosine = false;
            else throw std::invalid_argument("--metric must be cosine or ip");
        } else if (flag == "--help" || flag == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + flag);
        }
    }
    if (options.base.empty() || options.timestamps.empty() || options.transformed_base.empty() ||
        options.dim == 0 || options.half_life <= 0.0 || options.query_time <= 0.0) {
        throw std::invalid_argument(
            "--base, --timestamps, --transformed-base, --dim, --half-life, and --query-time are required");
    }
    if (options.mode != "multiplicative" && options.mode != "additive") {
        throw std::invalid_argument("--mode must be multiplicative or additive");
    }
    if (options.mode == "additive" &&
        (options.query.empty() || options.transformed_query.empty() || options.alpha < 0.0 || options.alpha > 1.0)) {
        throw std::invalid_argument("additive mode requires --query, --transformed-query, and --alpha in [0,1]");
    }
    if (options.source.empty() != options.base_original_indices.empty()) {
        throw std::invalid_argument("--source and --base-original-indices must be supplied together");
    }
    if (!options.query_original_indices.empty() && (options.query.empty() || options.source.empty())) {
        throw std::invalid_argument("--query-original-indices requires --query and --source");
    }
    const bool any_gt = !options.groundtruth.empty() || options.gt_queries != 0 || options.gt_width != 0;
    const bool all_gt = !options.groundtruth.empty() && options.gt_queries != 0 && options.gt_width != 0;
    if (any_gt != all_gt) throw std::invalid_argument("--gt, --gt-queries, and --gt-width must be supplied together");
    const bool any_split = !options.base_source_row_ids.empty() || !options.query_source_row_ids.empty() ||
        options.base_source_rows != 0 || options.query_source_rows != 0;
    const bool all_split = !options.base_source_row_ids.empty() && !options.query_source_row_ids.empty() &&
        options.base_source_rows != 0 && options.query_source_rows != 0 && !options.query.empty();
    if (any_split != all_split) {
        throw std::invalid_argument(
            "binary source split requires both ID files, both source row counts, and --query");
    }
    return options;
}

TimestampAudit audit_timestamps(
        const std::string& path,
        std::size_t expected_rows,
        double query_time) {
    std::vector<float> timestamps;
    if (!chronos::read_timestamps_binary(path, timestamps) || timestamps.size() != expected_rows) {
        throw std::runtime_error("timestamp count does not match canonical base rows");
    }
    TimestampAudit report;
    report.rows = timestamps.size();
    report.minimum = std::numeric_limits<double>::infinity();
    report.maximum = -std::numeric_limits<double>::infinity();
    for (float value : timestamps) {
        if (!std::isfinite(value)) {
            ++report.nonfinite;
            continue;
        }
        report.minimum = std::min(report.minimum, static_cast<double>(value));
        report.maximum = std::max(report.maximum, static_cast<double>(value));
        if (value < -kTimestampRangeEpsilon ||
            value > query_time + kTimestampRangeEpsilon) {
            ++report.out_of_range;
        }
    }
    if (!std::isfinite(report.minimum)) report.minimum = 0.0;
    if (!std::isfinite(report.maximum)) report.maximum = 0.0;
    return report;
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (char c : value) {
        if (c == '\\') output << "\\\\";
        else if (c == '"') output << "\\\"";
        else if (c == '\n') output << "\\n";
        else if (c == '\r') output << "\\r";
        else if (c == '\t') output << "\\t";
        else output << c;
    }
    return output.str();
}

std::string index_or_null(std::size_t value) {
    return value == static_cast<std::size_t>(-1) ? "null" : std::to_string(value);
}

void write_report(
        const Options& options,
        const chronos::VectorRowsReport& base_audit,
        const chronos::VectorRowsReport* query_audit,
        const TimestampAudit& timestamp_audit,
        const chronos::TransformAlignmentReport* multiplicative,
        const chronos::AdditiveTransformAlignmentReport* additive,
        const chronos::MappedRowsReport* base_map,
        const chronos::MappedRowsReport* query_map,
        const chronos::SourceRowSplitReport* source_split,
        const chronos::GroundTruthStructureReport* groundtruth,
        bool passed) {
    if (options.report.empty()) return;
    const std::filesystem::path report_path(options.report);
    if (!report_path.parent_path().empty()) {
        std::error_code error;
        std::filesystem::create_directories(report_path.parent_path(), error);
        if (error) throw std::runtime_error("cannot create report directory: " + error.message());
    }
    std::ofstream output(options.report.c_str(), std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create report: " + options.report);
    auto emit_vector = [&](const char* name, const chronos::VectorRowsReport* audit, bool trailing) {
        output << "  \"" << name << "\": ";
        if (!audit) output << "null";
        else output << "{\"passed\": " << (audit->valid(options.require_unit) ? "true" : "false")
                    << ", \"rows\": " << audit->rows
                    << ", \"coordinates\": " << audit->coordinates
                    << ", \"nonfinite_coordinates\": " << audit->nonfinite_coordinates
                    << ", \"zero_norm_rows\": " << audit->zero_norm_rows
                    << ", \"non_unit_rows\": " << audit->non_unit_rows
                    << ", \"min_norm\": " << audit->min_norm
                    << ", \"max_norm\": " << audit->max_norm << '}';
        output << (trailing ? ",\n" : "\n");
    };
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"chronos_dataset_alignment_v2\",\n"
           << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
           << "  \"dataset_name\": \"" << json_escape(options.dataset_name) << "\",\n"
           << "  \"distribution\": \"" << json_escape(options.distribution) << "\",\n"
           << "  \"score_mode\": \"" << options.mode << "\",\n"
           << "  \"configuration\": {\"dim\": " << options.dim
           << ", \"half_life_days\": " << options.half_life
           << ", \"query_time\": " << options.query_time
           << ", \"alpha\": " << (options.mode == "additive" ? std::to_string(options.alpha) : "null")
           << ", \"metric\": \"" << (options.cosine ? "cosine" : "ip")
           << "\", \"threads\": " << options.threads
           << ", \"require_unit\": " << (options.require_unit ? "true" : "false") << "},\n";
    emit_vector("canonical_base", &base_audit, true);
    emit_vector("canonical_query", query_audit, true);
    output << "  \"timestamps\": {\"passed\": " << (timestamp_audit.passed() ? "true" : "false")
           << ", \"rows\": " << timestamp_audit.rows
           << ", \"nonfinite\": " << timestamp_audit.nonfinite
           << ", \"out_of_range\": " << timestamp_audit.out_of_range
           << ", \"min\": " << timestamp_audit.minimum
           << ", \"max\": " << timestamp_audit.maximum << "},\n"
           << "  \"transform_alignment\": {\n";
    if (multiplicative) {
        output << "    \"passed\": " << (multiplicative->passed() ? "true" : "false")
               << ", \"rows\": " << multiplicative->rows
               << ", \"coordinates\": " << multiplicative->coordinates
               << ", \"mismatched_rows\": " << multiplicative->mismatched_rows
               << ", \"mismatched_coordinates\": " << multiplicative->mismatched_coordinates
               << ", \"first_mismatch_row\": " << index_or_null(multiplicative->first_mismatch_row)
               << ", \"first_mismatch_coordinate\": " << index_or_null(multiplicative->first_mismatch_coordinate)
               << ", \"max_absolute_error\": " << multiplicative->max_absolute_error
               << ", \"max_relative_error\": " << multiplicative->max_relative_error << "\n";
    } else {
        output << "    \"passed\": " << (additive->passed() ? "true" : "false")
               << ", \"base_rows\": " << additive->base_rows
               << ", \"query_rows\": " << additive->query_rows
               << ", \"coordinates\": " << additive->compared_coordinates
               << ", \"mismatched_coordinates\": " << additive->mismatched_coordinates
               << ", \"first_mismatch_row\": " << index_or_null(additive->first_mismatch_row)
               << ", \"first_mismatch_coordinate\": " << index_or_null(additive->first_mismatch_coordinate)
               << ", \"first_mismatch_is_query\": " << (additive->first_mismatch_is_query ? "true" : "false")
               << ", \"max_absolute_error\": " << additive->max_absolute_error
               << ", \"max_relative_error\": " << additive->max_relative_error << "\n";
    }
    output << "  },\n";
    auto emit_map = [&](const char* name, const chronos::MappedRowsReport* map) {
        output << "  \"" << name << "\": ";
        if (!map) output << "null";
        else output << "{\"passed\": " << (map->passed() ? "true" : "false")
                    << ", \"mapped_rows\": " << map->mapped_rows
                    << ", \"mismatched_rows\": " << map->mismatched_rows << '}';
        output << ",\n";
    };
    emit_map("base_source_mapping", base_map);
    emit_map("query_source_mapping", query_map);
    output << "  \"source_row_split\": ";
    if (!source_split) output << "null";
    else output << "{\"passed\": " << (source_split->passed() ? "true" : "false")
                << ", \"same_source\": " << (source_split->same_source ? "true" : "false")
                << ", \"duplicate_base_ids\": " << source_split->duplicate_base_ids
                << ", \"duplicate_query_ids\": " << source_split->duplicate_query_ids
                << ", \"out_of_range_base_ids\": " << source_split->out_of_range_base_ids
                << ", \"out_of_range_query_ids\": " << source_split->out_of_range_query_ids
                << ", \"overlapping_ids\": " << source_split->overlapping_ids << '}';
    output << ",\n  \"groundtruth_structure\": ";
    if (!groundtruth) output << "null,\n";
    else output << "{\"passed\": " << (groundtruth->passed() ? "true" : "false")
                << ", \"queries\": " << groundtruth->queries
                << ", \"width\": " << groundtruth->width
                << ", \"invalid_ids\": " << groundtruth->invalid_ids
                << ", \"rows_with_duplicate_ids\": " << groundtruth->rows_with_duplicate_ids
                << ", \"min_id\": " << groundtruth->min_id
                << ", \"max_id\": " << groundtruth->max_id << "},\n";
    output << "  \"producer\": {\"tool\": \"verify_dataset_alignment\", "
           << "\"git_revision\": \"" << chronos::source_revision()
           << "\", \"source_tree_dirty\": "
           << (chronos::source_tree_was_dirty_at_configure() ? "true" : "false") << "}\n"
           << "}\n";
    if (!output) throw std::runtime_error("failed writing report: " + options.report);
}

int run(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    const chronos::VectorRowsReport base_audit = chronos::audit_vector_file(
        options.base, static_cast<int>(options.dim), options.require_unit,
        options.unit_norm_tolerance, options.chunk_rows);
    chronos::VectorRowsReport query_audit;
    chronos::VectorRowsReport* query_audit_ptr = nullptr;
    if (!options.query.empty()) {
        query_audit = chronos::audit_vector_file(
            options.query, static_cast<int>(options.dim), options.require_unit,
            options.unit_norm_tolerance, options.chunk_rows);
        query_audit_ptr = &query_audit;
    }
    const TimestampAudit timestamps = audit_timestamps(
        options.timestamps, base_audit.rows, options.query_time);

    chronos::TransformAlignmentOptions transform_options;
    transform_options.dim = options.dim;
    transform_options.half_life_days = options.half_life;
    transform_options.query_time = options.query_time;
    transform_options.cosine = options.cosine;
    transform_options.absolute_tolerance = options.absolute_tolerance;
    transform_options.relative_tolerance = options.relative_tolerance;
    transform_options.chunk_rows = options.chunk_rows;
    transform_options.progress_every_rows = options.progress_every_rows;
    transform_options.threads = options.threads;

    chronos::TransformAlignmentReport multiplicative;
    chronos::AdditiveTransformAlignmentReport additive;
    chronos::TransformAlignmentReport* multiplicative_ptr = nullptr;
    chronos::AdditiveTransformAlignmentReport* additive_ptr = nullptr;
    if (options.mode == "multiplicative") {
        multiplicative = chronos::verify_multiplicative_transform_files(
            options.base, options.timestamps, options.transformed_base, transform_options);
        multiplicative_ptr = &multiplicative;
    } else {
        additive = chronos::verify_additive_transform_files(
            options.base, options.query, options.timestamps, options.transformed_base,
            options.transformed_query, transform_options, options.alpha);
        additive_ptr = &additive;
    }

    chronos::MappedRowsReport base_map;
    chronos::MappedRowsReport query_map;
    chronos::MappedRowsReport* base_map_ptr = nullptr;
    chronos::MappedRowsReport* query_map_ptr = nullptr;
    if (!options.source.empty()) {
        base_map = chronos::verify_mapped_raw_rows(
            options.source, options.base, options.base_original_indices, options.dim, options.chunk_rows);
        base_map_ptr = &base_map;
    }
    if (!options.query_original_indices.empty()) {
        query_map = chronos::verify_mapped_raw_rows(
            options.source, options.query, options.query_original_indices, options.dim, options.chunk_rows);
        query_map_ptr = &query_map;
    }

    chronos::SourceRowSplitReport source_split;
    chronos::SourceRowSplitReport* source_split_ptr = nullptr;
    if (!options.base_source_row_ids.empty()) {
        source_split = chronos::verify_source_row_id_split(
            options.base_source_row_ids, base_audit.rows, options.base_source_rows,
            options.query_source_row_ids, query_audit.rows, options.query_source_rows,
            options.same_source);
        source_split_ptr = &source_split;
    }

    chronos::GroundTruthStructureReport groundtruth;
    chronos::GroundTruthStructureReport* groundtruth_ptr = nullptr;
    if (!options.groundtruth.empty()) {
        groundtruth = chronos::verify_groundtruth_structure(
            options.groundtruth, options.gt_queries, options.gt_width, base_audit.rows);
        groundtruth_ptr = &groundtruth;
    }

    bool passed = base_audit.valid(options.require_unit) && timestamps.passed();
    if (query_audit_ptr) passed = passed && query_audit_ptr->valid(options.require_unit);
    if (multiplicative_ptr) passed = passed && multiplicative_ptr->passed();
    if (additive_ptr) passed = passed && additive_ptr->passed();
    if (base_map_ptr) passed = passed && base_map_ptr->passed();
    if (query_map_ptr) passed = passed && query_map_ptr->passed();
    if (source_split_ptr) passed = passed && source_split_ptr->passed();
    if (groundtruth_ptr) passed = passed && groundtruth_ptr->passed();
    write_report(options, base_audit, query_audit_ptr, timestamps, multiplicative_ptr,
                 additive_ptr, base_map_ptr, query_map_ptr, source_split_ptr,
                 groundtruth_ptr, passed);

    const std::size_t mismatch_coordinates = multiplicative_ptr
        ? multiplicative_ptr->mismatched_coordinates : additive_ptr->mismatched_coordinates;
    std::cout << std::setprecision(12)
              << "alignment_passed=" << (passed ? 1 : 0)
              << " mode=" << options.mode
              << " base_rows=" << base_audit.rows
              << " query_rows=" << (query_audit_ptr ? query_audit.rows : 0)
              << " mismatched_coordinates=" << mismatch_coordinates
              << " timestamps_valid=" << (timestamps.passed() ? 1 : 0) << '\n';
    if (!options.report.empty()) std::cout << "report=" << options.report << '\n';
    return passed ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "verify_dataset_alignment: " << error.what() << '\n';
        return 1;
    }
}
