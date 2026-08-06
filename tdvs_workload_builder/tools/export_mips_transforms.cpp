/**
 * Exports exact inner-product representations for the supported TDVS scores.
 * Multiplicative mode scales each base vector by its decay. Additive mode
 * writes paired dim+1 base and query vectors whose inner product is the
 * additive score. The exported vectors must be searched with inner product.
 */

#include "tdvs.h"
#include "io.h"
#include "time_metadata.h"
#include "validation.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static void usage() {
    std::cerr
        << "export_mips_transforms - export vectors for IP retrieval (see README).\n\n"
        << "Multiplicative (default):\n"
        << "  --tdvs_mode multiplicative \\\n"
        << "  --base B --timestamps T --dim D --half_life_days H --max_timestamp_days TREF \\\n"
        << "  --base_out PATH  (or legacy: --out PATH) \\\n"
        << "  --distribution topic-independent|topic-correlated --metric ip|cosine\n\n"
        << "Additive (requires query + both outputs + alpha):\n"
        << "  --tdvs_mode additive \\\n"
        << "  --base B --query Q --timestamps T --dim D --half_life_days H --max_timestamp_days TREF \\\n"
        << "  --alpha A \\\n"
        << "  --base_out PATH --query_out PATH \\\n"
        << "  --distribution topic-independent|topic-correlated --metric ip|cosine\n\n"
        << "  --dataset_name NAME: optional provenance label (default unspecified).\n"
        << "  --tdvs_mode: default multiplicative if omitted. --metric is required.\n"
        << "  Multiplicative mode rejects --query and --query_out; reuse the canonical query.\n"
        << "  --threads defaults to 64.\n"
        << "  --manifest defaults to BASE_OUT.manifest.json.\n";
}

static bool validate_timestamps_vs_max(const std::vector<float>& ts, double max_ts) {
    // Keep this contract identical to TANGO's future-time validation.
    constexpr double kEps = 1e-9;
    for (float v : ts) {
        double t = static_cast<double>(v);
        if (!std::isfinite(t) || t < -kEps || t > max_ts + kEps) return false;
    }
    return true;
}

static std::string json_escape(const std::string& value) {
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

static std::string relative_to(const std::string& path, const std::string& directory) {
    std::error_code error;
    const std::filesystem::path relative = std::filesystem::relative(path, directory, error);
    return error || relative.empty() ? std::filesystem::path(path).filename().string()
                                     : relative.generic_string();
}

int main(int argc, char** argv) {
    std::string base_path, query_path, timestamps_path, distribution, dataset_name = "unspecified";
    std::string base_out, query_out, legacy_out;
    std::string manifest_out;
    std::string metric_str;
    std::string tdvs_mode = "multiplicative";
    int dim = 0;
    double hl = 0.0;
    double max_timestamp_days = 0.0;
    double alpha = 0.0;
    bool alpha_set = false;
    int threads = 64;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* n) {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << n << "\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--base")
            base_path = need("--base");
        else if (a == "--query")
            query_path = need("--query");
        else if (a == "--timestamps")
            timestamps_path = need("--timestamps");
        else if (a == "--distribution")
            distribution = need("--distribution");
        else if (a == "--dataset_name")
            dataset_name = need("--dataset_name");
        else if (a == "--dim")
            dim = std::atoi(need("--dim"));
        else if (a == "--half_life_days")
            hl = std::stod(need("--half_life_days"));
        else if (a == "--max_timestamp_days")
            max_timestamp_days = std::stod(need("--max_timestamp_days"));
        else if (a == "--base_out")
            base_out = need("--base_out");
        else if (a == "--query_out")
            query_out = need("--query_out");
        else if (a == "--out")
            legacy_out = need("--out");
        else if (a == "--manifest")
            manifest_out = need("--manifest");
        else if (a == "--threads")
            threads = std::atoi(need("--threads"));
        else if (a == "--metric")
            metric_str = need("--metric");
        else if (a == "--tdvs_mode")
            tdvs_mode = need("--tdvs_mode");
        else if (a == "--alpha") {
            alpha = std::stod(need("--alpha"));
            alpha_set = true;
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            std::cerr << "Unknown arg: " << a << "\n";
            usage();
            return 1;
        }
    }

    const bool additive = (tdvs_mode == "additive");
    const bool multiplicative = (tdvs_mode == "multiplicative");
    if (!additive && !multiplicative) {
        std::cerr << "Bad --tdvs_mode (use multiplicative or additive)\n";
        return 1;
    }

    if (base_out.empty() && !legacy_out.empty()) base_out = legacy_out;
    if (base_path.empty() || timestamps_path.empty() || distribution.empty() || dim <= 0 ||
        hl <= 0.0 || max_timestamp_days <= 0.0 || base_out.empty() || threads <= 0) {
        usage();
        return 1;
    }
    if (distribution != "topic-independent" && distribution != "topic-correlated") {
        std::cerr << "--distribution must be topic-independent or topic-correlated\n";
        return 1;
    }
    if (metric_str.empty()) {
        std::cerr << "--metric is required (use ip or cosine)\n";
        return 1;
    }
    if (manifest_out.empty()) manifest_out = base_out + ".manifest.json";

    const bool cosine = (metric_str == "cosine");
    if (!cosine && metric_str != "ip") {
        std::cerr << "Bad --metric\n";
        return 1;
    }

    if (additive) {
        if (query_path.empty() || query_out.empty() || !alpha_set) {
            std::cerr << "additive mode requires --query, --alpha, --base_out, and --query_out\n";
            return 1;
        }
        if (alpha < 0.0 || alpha > 1.0) {
            std::cerr << "--alpha must be in [0,1]\n";
            return 1;
        }
    } else if (!query_path.empty() || !query_out.empty()) {
        std::cerr << "multiplicative mode rejects --query and --query_out; reuse the canonical query\n";
        return 1;
    } else if (alpha_set) {
        std::cerr << "Warning: --alpha ignored for multiplicative mode\n";
    }

    std::size_t row = static_cast<std::size_t>(dim) * sizeof(float);
    std::size_t bsz = chronos::file_size_bytes(base_path);
    if (bsz == 0 || bsz % row != 0) {
        std::cerr << "Bad base file\n";
        return 1;
    }
    std::size_t nb = bsz / row;

    std::cerr << "[export_mips_transforms] mode=" << tdvs_mode << " metric=" << metric_str
              << " half_life_days=" << hl << " max_timestamp_days=" << max_timestamp_days << "\n";
    std::cerr.flush();

    std::vector<float> base(nb * static_cast<std::size_t>(dim));
    FILE* fp = std::fopen(base_path.c_str(), "rb");
    if (!fp || std::fread(base.data(), sizeof(float), base.size(), fp) != base.size()) {
        std::cerr << "Read base failed\n";
        if (fp) std::fclose(fp);
        return 1;
    }
    std::fclose(fp);

    if (cosine) {
        const chronos::VectorRowsReport audit = chronos::audit_vector_rows(
            base.data(), nb, dim, false, 0.0, false);
        if (!audit.valid(false)) {
            std::cerr << "base contains non-finite coordinates or zero-norm rows\n";
            return 1;
        }
    }

    std::vector<float> timestamps;
    if (!chronos::read_timestamps_binary(timestamps_path, timestamps) || timestamps.size() != nb) {
        std::cerr << "timestamps length must match num_base\n";
        return 1;
    }
    if (!validate_timestamps_vs_max(timestamps, max_timestamp_days)) {
        std::cerr << "timestamp_days out of [0, max_timestamp_days] for given --max_timestamp_days\n";
        return 1;
    }

    const double lambda = chronos::half_life_to_lambda(hl);

    std::size_t nq = 0;
    if (multiplicative) {
        std::vector<float> out;
        chronos::build_multiplicative_transformed_base(base.data(), nb, dim, timestamps.data(),
                                                       max_timestamp_days, hl, cosine, out, threads);
        if (out.empty()) {
            std::cerr << "multiplicative export build failed\n";
            return 1;
        }
        if (!chronos::write_raw_vectors(base_out, out.data(), nb, dim)) {
            std::cerr << "Write failed\n";
            return 1;
        }
    } else {
        std::size_t qsz = chronos::file_size_bytes(query_path);
        if (qsz == 0 || qsz % row != 0) {
            std::cerr << "Bad query file\n";
            return 1;
        }
        nq = qsz / row;
        std::vector<float> queries(nq * static_cast<std::size_t>(dim));
        fp = std::fopen(query_path.c_str(), "rb");
        if (!fp || std::fread(queries.data(), sizeof(float), queries.size(), fp) != queries.size()) {
            std::cerr << "Read query failed\n";
            if (fp) std::fclose(fp);
            return 1;
        }
        std::fclose(fp);
        if (cosine) {
            const chronos::VectorRowsReport audit = chronos::audit_vector_rows(
                queries.data(), nq, dim, false, 0.0, false);
            if (!audit.valid(false)) {
                std::cerr << "query contains non-finite coordinates or zero-norm rows\n";
                return 1;
            }
        }

        std::vector<float> aug_base, aug_query;
        chronos::build_additive_transformed_base(
            base.data(), nb, dim, timestamps.data(), max_timestamp_days, lambda,
            alpha, cosine, aug_base, threads);
        chronos::build_additive_transformed_queries(
            queries.data(), nq, dim, alpha, cosine, aug_query, threads);
        if (aug_base.empty() || aug_query.empty()) {
            std::cerr << "additive transform build failed\n";
            return 1;
        }
        if (!chronos::verify_additive_transformed_ip_equivalence(
                base.data(), queries.data(), aug_base.data(), aug_query.data(), nb, nq, dim,
                timestamps.data(), max_timestamp_days, lambda, alpha, cosine, 8)) {
            std::cerr << "additive IP equivalence check failed\n";
            return 1;
        }
        const int dim_out = dim + 1;
        if (!chronos::write_raw_vectors(base_out, aug_base.data(), nb, dim_out) ||
            !chronos::write_raw_vectors(query_out, aug_query.data(), nq, dim_out)) {
            std::cerr << "Write additive output failed\n";
            return 1;
        }
        std::cerr << "[export_mips_transforms] additive IP equivalence check passed (sampled)\n";
    }

    const std::filesystem::path manifest_path(manifest_out);
    const std::string manifest_directory = manifest_path.parent_path().empty()
        ? "." : manifest_path.parent_path().string();
    std::error_code directory_error;
    if (!manifest_path.parent_path().empty()) {
        std::filesystem::create_directories(manifest_path.parent_path(), directory_error);
        if (directory_error) {
            std::cerr << "Cannot create manifest directory: " << directory_error.message() << '\n';
            return 1;
        }
    }
    std::ostringstream manifest;
    manifest << std::setprecision(17)
             << "{\n"
             << "  \"schema\": \"chronos_mips_transform_v1\",\n"
             << "  \"dataset_name\": \"" << json_escape(dataset_name) << "\",\n"
             << "  \"distribution\": \"" << distribution << "\",\n"
             << "  \"score_mode\": \"" << tdvs_mode << "\",\n"
             << "  \"semantic_metric\": \"" << metric_str << "\",\n"
             << "  \"base_rows\": " << nb << ", \"query_rows\": " << nq
             << ", \"input_dimension\": " << dim
             << ", \"output_dimension\": " << (additive ? dim + 1 : dim) << ",\n"
             << "  \"half_life_days\": " << hl
             << ", \"query_time_days\": " << max_timestamp_days << ",\n"
             << "  \"alpha\": ";
    if (additive) manifest << alpha;
    else manifest << "null";
    manifest << ",\n"
             << "  \"inputs\": {\"base\": \"" << json_escape(relative_to(base_path, manifest_directory))
             << "\", \"query\": "
             << (query_path.empty() ? "null" : "\"" + json_escape(relative_to(query_path, manifest_directory)) + "\"")
             << ", \"timestamps\": \"" << json_escape(relative_to(timestamps_path, manifest_directory)) << "\"},\n"
             << "  \"outputs\": {\"base\": \"" << json_escape(relative_to(base_out, manifest_directory))
             << "\", \"query\": "
             << (query_out.empty() ? "null" : "\"" + json_escape(relative_to(query_out, manifest_directory)) + "\"")
             << "},\n"
             << "  \"row_order_preserved\": true,\n"
             << "  \"retrieval_metric\": \"inner_product\",\n"
             << "  \"post_transform_l2_normalization_allowed\": false,\n"
             << "  \"producer\": {\"tool\": \"export_mips_transforms\", \"threads\": "
             << threads << ", \"git_revision\": \"" << chronos::source_revision()
             << "\", \"source_tree_dirty\": "
             << (chronos::source_tree_was_dirty_at_configure() ? "true" : "false") << "}\n"
             << "}\n";
    if (!chronos::write_text_file(manifest_out, manifest.str())) {
        std::cerr << "Write transform manifest failed\n";
        return 1;
    }
    std::cout << "Wrote " << base_out;
    if (additive) std::cout << " and " << query_out;
    std::cout << " (manifest " << manifest_out << ")\n";
    return 0;
}
