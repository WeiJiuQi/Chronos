/**
 * Creates the canonical Chronos semantic dataset.
 *
 * Inputs may be headerless float32, fvecs, or standard headered fbin. Output
 * vectors are always headerless float32. Cosine datasets are L2-normalized on
 * disk by default, and source row IDs are retained in uint64 and text form.
 */

#include "tdvs.h"
#include "io.h"
#include "sampling.h"
#include "validation.h"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string base_path;
    std::string query_path;
    std::string out_dir;
    std::string dataset_name = "unspecified";
    int base_dim = 0;
    int query_dim = 0;
    chronos::InputVectorFormat base_format = chronos::InputVectorFormat::RawFloat32;
    chronos::InputVectorFormat query_format = chronos::InputVectorFormat::RawFloat32;
    std::size_t sample_base = 0;
    std::size_t sample_query = 0;
    chronos::SampleMode sample_mode = chronos::SampleMode::Prefix;
    chronos::SampleMode query_sample_mode = chronos::SampleMode::Prefix;
    bool query_sample_mode_set = false;
    std::uint64_t seed = 42;
    int topk = 0;
    int threads = 64;
    bool cosine = true;
    int normalize_output = -1;
    int compute_control_gt = -1;
    bool strict_external_disjoint = true;
};

void usage() {
    std::cout
        << "Usage: prepare_vector_subset \\\n"
        << "  --base PATH --out_dir DIR --base_dim D [options]\n\n"
        << "Input and sampling:\n"
        << "  --base_format raw|fvecs|fbin   base format (default raw)\n"
        << "  --query PATH                   independent real-query source\n"
        << "  --query_format raw|fvecs|fbin query format (default base format)\n"
        << "  --query_dim D                  defaults to base_dim\n"
        << "  --sample_base N               0/omitted keeps all base rows\n"
        << "  --sample_query Q              sample Q queries; required without --query\n"
        << "  --sample_mode prefix|random   deterministic base row selection (default prefix)\n"
        << "  --query_sample_mode prefix|random\n"
        << "                                query selection; defaults to --sample_mode\n"
        << "  --seed U64                    default 42; query selection uses seed+1\n"
        << "  --input_format FORMAT         legacy alias setting both input formats\n\n"
        << "Canonical output and validation:\n"
        << "  --metric cosine|ip            default cosine\n"
        << "  --normalize_output 0|1        default 1 for cosine, 0 for ip\n"
        << "  --strict_external_disjoint 0|1 fail on exact base/query row overlap (default 1)\n"
        << "  --dataset_name NAME\n"
        << "  --threads T                   default 64\n"
        << "  --topk K                      optional conventional control GT width\n"
        << "  --compute_control_gt 0|1      default 1 iff --topk is supplied\n\n"
        << "Outputs: base.fbin, query.fbin, source_row_ids.u64bin files, legacy text\n"
        << "row maps, optional conventional GT, and manifest_dataset.json.\n"
        << "When --query is omitted, base and query source rows are strictly disjoint.\n";
}

bool parse_format(const std::string& value, chronos::InputVectorFormat& format) {
    if (value == "raw") format = chronos::InputVectorFormat::RawFloat32;
    else if (value == "fvecs") format = chronos::InputVectorFormat::Fvecs;
    else if (value == "fbin") format = chronos::InputVectorFormat::Fbin;
    else return false;
    return true;
}

bool parse_bool(const char* value, const char* name) {
    const std::string text(value);
    if (text == "1" || text == "true") return true;
    if (text == "0" || text == "false") return false;
    throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (char c : value) {
        switch (c) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default: output << c; break;
        }
    }
    return output.str();
}

std::string relative_to(const std::string& path, const std::string& directory) {
    std::error_code error;
    const std::filesystem::path relative = std::filesystem::relative(path, directory, error);
    return error || relative.empty() ? std::filesystem::path(path).filename().string()
                                     : relative.generic_string();
}

void write_text_indices(const std::string& path, const std::vector<std::size_t>& indices) {
    std::ostringstream output;
    for (std::size_t index : indices) output << index << '\n';
    if (!chronos::write_text_file(path, output.str())) {
        throw std::runtime_error("failed writing source-row text map: " + path);
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    bool query_format_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (flag == "--base") options.base_path = value("--base");
        else if (flag == "--query") options.query_path = value("--query");
        else if (flag == "--out_dir") options.out_dir = value("--out_dir");
        else if (flag == "--dataset_name") options.dataset_name = value("--dataset_name");
        else if (flag == "--base_dim") options.base_dim = std::stoi(value("--base_dim"));
        else if (flag == "--query_dim") options.query_dim = std::stoi(value("--query_dim"));
        else if (flag == "--sample_base") options.sample_base = std::stoull(value("--sample_base"));
        else if (flag == "--sample_query") options.sample_query = std::stoull(value("--sample_query"));
        else if (flag == "--seed") options.seed = std::stoull(value("--seed"));
        else if (flag == "--topk") options.topk = std::stoi(value("--topk"));
        else if (flag == "--threads") options.threads = std::stoi(value("--threads"));
        else if (flag == "--normalize_output") {
            options.normalize_output = parse_bool(value("--normalize_output"), "--normalize_output") ? 1 : 0;
        } else if (flag == "--compute_control_gt") {
            options.compute_control_gt = parse_bool(value("--compute_control_gt"), "--compute_control_gt") ? 1 : 0;
        } else if (flag == "--strict_external_disjoint") {
            options.strict_external_disjoint = parse_bool(
                value("--strict_external_disjoint"), "--strict_external_disjoint");
        } else if (flag == "--metric") {
            const std::string metric = value("--metric");
            if (metric == "cosine") options.cosine = true;
            else if (metric == "ip") options.cosine = false;
            else throw std::invalid_argument("--metric must be cosine or ip");
        } else if (flag == "--sample_mode") {
            const std::string mode = value("--sample_mode");
            if (mode == "prefix") options.sample_mode = chronos::SampleMode::Prefix;
            else if (mode == "random") options.sample_mode = chronos::SampleMode::Random;
            else throw std::invalid_argument("--sample_mode must be prefix or random");
        } else if (flag == "--query_sample_mode") {
            const std::string mode = value("--query_sample_mode");
            if (mode == "prefix") options.query_sample_mode = chronos::SampleMode::Prefix;
            else if (mode == "random") options.query_sample_mode = chronos::SampleMode::Random;
            else throw std::invalid_argument("--query_sample_mode must be prefix or random");
            options.query_sample_mode_set = true;
        } else if (flag == "--base_format") {
            if (!parse_format(value("--base_format"), options.base_format)) {
                throw std::invalid_argument("--base_format must be raw, fvecs, or fbin");
            }
        } else if (flag == "--query_format") {
            if (!parse_format(value("--query_format"), options.query_format)) {
                throw std::invalid_argument("--query_format must be raw, fvecs, or fbin");
            }
            query_format_set = true;
        } else if (flag == "--input_format") {
            chronos::InputVectorFormat format;
            if (!parse_format(value("--input_format"), format)) {
                throw std::invalid_argument("--input_format must be raw, fvecs, or fbin");
            }
            options.base_format = format;
            options.query_format = format;
            query_format_set = true;
        } else if (flag == "--help" || flag == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + flag);
        }
    }
    if (!query_format_set) options.query_format = options.base_format;
    if (!options.query_sample_mode_set) options.query_sample_mode = options.sample_mode;
    if (options.query_dim == 0) options.query_dim = options.base_dim;
    if (options.normalize_output < 0) options.normalize_output = options.cosine ? 1 : 0;
    if (options.compute_control_gt < 0) options.compute_control_gt = options.topk > 0 ? 1 : 0;
    if (options.base_path.empty() || options.out_dir.empty() || options.base_dim <= 0 ||
        options.query_dim <= 0 || options.threads <= 0) {
        throw std::invalid_argument("--base, --out_dir, and positive dimensions/threads are required");
    }
    if (options.base_dim != options.query_dim) {
        throw std::invalid_argument("base and query dimensions must match");
    }
    if (options.query_path.empty() && options.sample_query == 0) {
        throw std::invalid_argument("--sample_query is required when --query is omitted");
    }
    if (options.compute_control_gt && options.topk <= 0) {
        throw std::invalid_argument("--compute_control_gt 1 requires a positive --topk");
    }
    return options;
}

int run(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    std::error_code directory_error;
    std::filesystem::create_directories(options.out_dir, directory_error);
    if (directory_error) throw std::runtime_error("cannot create output directory: " + directory_error.message());

    const std::size_t total_base = chronos::count_vectors_in_file(
        options.base_path, options.base_dim, options.base_format);
    if (total_base == 0) throw std::runtime_error("base is empty or does not match its declared format/dimension");
    const std::size_t requested_base = options.sample_base == 0 ? total_base : options.sample_base;
    if (requested_base > total_base) throw std::runtime_error("--sample_base exceeds the source base row count");

    std::vector<std::size_t> base_ids = chronos::build_sample_indices(
        total_base, requested_base, options.sample_mode, options.seed);
    std::vector<std::size_t> query_ids;
    std::vector<float> base;
    std::vector<float> query;
    const bool external_query = !options.query_path.empty();

    std::cerr << "[prepare_vector_subset] reading " << base_ids.size() << " / " << total_base
              << " base rows from " << chronos::input_vector_format_name(options.base_format) << '\n';
    if (!chronos::read_vectors_sorted_indices(
            options.base_path, options.base_dim, options.base_format, base_ids, base)) {
        throw std::runtime_error("failed reading selected base rows");
    }

    std::size_t total_query_source = total_base;
    if (external_query) {
        total_query_source = chronos::count_vectors_in_file(
            options.query_path, options.query_dim, options.query_format);
        if (total_query_source == 0) {
            throw std::runtime_error("query is empty or does not match its declared format/dimension");
        }
        const std::size_t requested_query = options.sample_query == 0
            ? total_query_source : options.sample_query;
        if (requested_query > total_query_source) {
            throw std::runtime_error("--sample_query exceeds the source query row count");
        }
        query_ids = chronos::build_sample_indices(
            total_query_source, requested_query, options.query_sample_mode, options.seed + 1);
        std::cerr << "[prepare_vector_subset] reading " << query_ids.size() << " / "
                  << total_query_source << " external query rows from "
                  << chronos::input_vector_format_name(options.query_format) << '\n';
        if (!chronos::read_vectors_sorted_indices(
                options.query_path, options.query_dim, options.query_format, query_ids, query)) {
            throw std::runtime_error("failed reading selected query rows");
        }
    } else {
        query_ids = chronos::build_disjoint_query_indices(
            base_ids, total_base, options.sample_query, options.query_sample_mode, options.seed + 1);
        if (query_ids.size() != options.sample_query) {
            throw std::runtime_error("cannot draw the requested query rows disjoint from the base source rows");
        }
        if (!chronos::read_vectors_sorted_indices(
                options.base_path, options.base_dim, options.base_format, query_ids, query)) {
            throw std::runtime_error("failed reading disjoint query rows");
        }
    }

    const std::size_t base_rows = base_ids.size();
    const std::size_t query_rows = query_ids.size();
    const chronos::VectorRowsReport raw_base_audit = chronos::audit_vector_rows(
        base.data(), base_rows, options.base_dim, false, 0.0, false);
    const chronos::VectorRowsReport raw_query_audit = chronos::audit_vector_rows(
        query.data(), query_rows, options.query_dim, false, 0.0, false);
    if (!raw_base_audit.valid(false) || !raw_query_audit.valid(false)) {
        throw std::runtime_error("source selection contains non-finite coordinates or zero-norm rows");
    }

    if (options.normalize_output) {
        chronos::l2_normalize_rows_inplace(base.data(), base_rows, options.base_dim, options.threads);
        chronos::l2_normalize_rows_inplace(query.data(), query_rows, options.query_dim, options.threads);
    }
    const bool require_unit = options.normalize_output != 0;
    const double norm_tolerance = 1e-4;
    const chronos::VectorRowsReport base_audit = chronos::audit_vector_rows(
        base.data(), base_rows, options.base_dim, require_unit, norm_tolerance, true);
    const chronos::VectorRowsReport query_audit = chronos::audit_vector_rows(
        query.data(), query_rows, options.query_dim, require_unit, norm_tolerance, true);
    if (!base_audit.valid(require_unit) || !query_audit.valid(require_unit)) {
        throw std::runtime_error("canonical vectors failed finite/unit-norm validation");
    }
    const std::size_t exact_overlap = chronos::count_exact_query_base_overlaps(
        base.data(), base_rows, query.data(), query_rows, options.base_dim);
    if (exact_overlap != 0 && (external_query ? options.strict_external_disjoint : true)) {
        throw std::runtime_error(
            "canonical base/query contain " + std::to_string(exact_overlap) +
            " byte-identical query rows; disable --strict_external_disjoint only for an intentional external-query workload");
    }

    const std::string base_out = options.out_dir + "/base.fbin";
    const std::string query_out = options.out_dir + "/query.fbin";
    const std::string base_ids_out = options.out_dir + "/base_source_row_ids.u64bin";
    const std::string query_ids_out = options.out_dir + "/query_source_row_ids.u64bin";
    if (!chronos::write_raw_vectors(base_out, base.data(), base_rows, options.base_dim) ||
        !chronos::write_raw_vectors(query_out, query.data(), query_rows, options.query_dim) ||
        !chronos::write_source_row_ids(base_ids_out, base_ids) ||
        !chronos::write_source_row_ids(query_ids_out, query_ids)) {
        throw std::runtime_error("failed writing canonical vectors or binary source-row maps");
    }
    write_text_indices(options.out_dir + "/base_original_indices.txt", base_ids);
    write_text_indices(options.out_dir + "/query_original_indices.txt", query_ids);

    std::string gt_path;
    if (options.compute_control_gt) {
        if (static_cast<std::size_t>(options.topk) > base_rows) {
            throw std::runtime_error("control --topk exceeds the canonical base row count");
        }
        std::vector<chronos::GroundTruthId> gt;
        const int progress_every = std::max(1, static_cast<int>(query_rows / 40));
        chronos::brute_force_ip_topk(
            query.data(), query_rows, base.data(), base_rows, options.base_dim,
            options.topk, options.threads, gt, nullptr, progress_every);
        std::ostringstream name;
        name << options.out_dir << '/' << (options.cosine ? "gt_cosine.k" : "gt_ip.k")
             << options.topk << ".bin";
        gt_path = name.str();
        if (!chronos::write_groundtruth(gt_path, gt.data(), query_rows, options.topk)) {
            throw std::runtime_error("failed writing conventional control ground truth");
        }
    }

    const std::string manifest_path = options.out_dir + "/manifest_dataset.json";
    std::ostringstream manifest;
    manifest << std::setprecision(17)
             << "{\n"
             << "  \"schema\": \"chronos_canonical_dataset_v1\",\n"
             << "  \"dataset_name\": \"" << json_escape(options.dataset_name) << "\",\n"
             << "  \"semantic_metric\": \"" << (options.cosine ? "cosine" : "inner_product") << "\",\n"
             << "  \"canonical_vectors_l2_normalized\": " << (options.normalize_output ? "true" : "false") << ",\n"
             << "  \"dimension\": " << options.base_dim << ",\n"
             << "  \"base_rows\": " << base_rows << ",\n"
             << "  \"query_rows\": " << query_rows << ",\n"
             << "  \"seed\": " << options.seed << ",\n"
             << "  \"sampling_mode\": \""
             << (options.sample_mode == chronos::SampleMode::Prefix ? "prefix" : "random") << "\",\n"
             << "  \"base_sampling_mode\": \""
             << (options.sample_mode == chronos::SampleMode::Prefix ? "prefix" : "random") << "\",\n"
             << "  \"query_sampling_mode\": \""
             << (options.query_sample_mode == chronos::SampleMode::Prefix ? "prefix" : "random") << "\",\n"
             << "  \"base_query_source_split\": \""
             << (external_query ? "independent_sources" : "strictly_disjoint_source_rows") << "\",\n"
             << "  \"inputs\": {\n"
             << "    \"base\": {\"path\": \"" << json_escape(relative_to(options.base_path, options.out_dir))
             << "\", \"format\": \"" << chronos::input_vector_format_name(options.base_format)
             << "\", \"source_rows\": " << total_base
             << ", \"bytes\": " << chronos::file_size_bytes(options.base_path) << "},\n"
             << "    \"query\": ";
    if (external_query) {
        manifest << "{\"path\": \"" << json_escape(relative_to(options.query_path, options.out_dir))
                 << "\", \"format\": \"" << chronos::input_vector_format_name(options.query_format)
                 << "\", \"source_rows\": " << total_query_source
                 << ", \"bytes\": " << chronos::file_size_bytes(options.query_path) << "}\n";
    } else {
        manifest << "{\"path\": \"" << json_escape(relative_to(options.base_path, options.out_dir))
                 << "\", \"format\": \"" << chronos::input_vector_format_name(options.base_format)
                 << "\", \"source_rows\": " << total_base << ", \"same_as_base_source\": true}\n";
    }
    manifest << "  },\n"
             << "  \"outputs\": {\n"
             << "    \"base\": \"base.fbin\",\n"
             << "    \"query\": \"query.fbin\",\n"
             << "    \"base_source_row_ids\": \"base_source_row_ids.u64bin\",\n"
             << "    \"query_source_row_ids\": \"query_source_row_ids.u64bin\",\n"
             << "    \"control_groundtruth\": "
             << (gt_path.empty() ? "null" : "\"" + json_escape(relative_to(gt_path, options.out_dir)) + "\"") << "\n"
             << "  },\n"
             << "  \"validation\": {\n"
             << "    \"unit_norm_tolerance\": " << norm_tolerance << ",\n"
             << "    \"base_nonfinite_coordinates\": " << base_audit.nonfinite_coordinates << ",\n"
             << "    \"query_nonfinite_coordinates\": " << query_audit.nonfinite_coordinates << ",\n"
             << "    \"base_zero_norm_rows\": " << base_audit.zero_norm_rows << ",\n"
             << "    \"query_zero_norm_rows\": " << query_audit.zero_norm_rows << ",\n"
             << "    \"base_non_unit_rows\": " << base_audit.non_unit_rows << ",\n"
             << "    \"query_non_unit_rows\": " << query_audit.non_unit_rows << ",\n"
             << "    \"base_exact_duplicate_rows\": " << base_audit.exact_duplicate_rows << ",\n"
             << "    \"query_exact_duplicate_rows\": " << query_audit.exact_duplicate_rows << ",\n"
             << "    \"exact_query_rows_present_in_base\": " << exact_overlap << "\n"
             << "  },\n"
             << "  \"producer\": {\"tool\": \"prepare_vector_subset\", \"threads\": "
             << options.threads << ", \"git_revision\": \"" << chronos::source_revision()
             << "\", \"source_tree_dirty\": "
             << (chronos::source_tree_was_dirty_at_configure() ? "true" : "false") << "}\n"
             << "}\n";
    if (!chronos::write_text_file(manifest_path, manifest.str())) {
        throw std::runtime_error("failed writing canonical dataset manifest");
    }

    std::ostringstream legacy;
    legacy << "task: prepare_vector_subset\n"
           << "dataset_name: " << options.dataset_name << '\n'
           << "canonical_manifest: manifest_dataset.json\n"
           << "base_rows: " << base_rows << '\n'
           << "query_rows: " << query_rows << '\n'
           << "dim: " << options.base_dim << '\n'
           << "stored_vectors: " << (options.normalize_output ? "l2_normalized" : "raw") << '\n'
           << "source_row_id_format: uint64 headerless; text compatibility maps also retained\n";
    chronos::write_text_file(options.out_dir + "/manifest_ip.txt", legacy.str());

    std::cout << "Wrote canonical dataset to " << options.out_dir
              << " (base=" << base_rows << ", query=" << query_rows
              << ", dim=" << options.base_dim << ")\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "prepare_vector_subset: " << error.what() << '\n';
        return 1;
    }
}
