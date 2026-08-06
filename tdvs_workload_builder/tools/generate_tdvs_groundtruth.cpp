/**
 * Computes exact TDVS top-k for one or more half-lives from a static base,
 * query set, and timestamp metadata. With --export_weighted 1, the tool also
 * exports the exact multiplicative or additive MIPS representation.
 */

#include "tdvs.h"
#include "io.h"
#include "time_metadata.h"
#include "validation.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

static void usage() {
    std::cerr << "Usage: generate_tdvs_groundtruth \\\n"
              << "  --base PATH --query PATH --timestamps PATH --dim D \\\n"
              << "  --dataset_name NAME \\\n"
              << "  --distribution topic-independent|topic-correlated \\\n"
              << "  --max_timestamp_days TREF \\\n"
              << "  --tdvs_mode multiplicative|additive \\\n"
              << "  [--alpha 0..1]   (required for additive; ignored for multiplicative) \\\n"
              << "  --half_life_days CSV (e.g. 7,30,90,180) \\\n"
              << "  --topk K --out_dir DIR [--threads T] \\\n"
              << "  [--export_weighted 0|1] --metric ip|cosine\n"
              << "\n"
              << "  --tdvs_mode, --half_life_days, and --metric are required.\n"
              << "  --dataset_name: recorded in manifest only (provenance); does not affect scores.\n"
              << "  --threads: defaults to 64.\n"
              << "  --export_weighted 1:\n"
              << "    multiplicative: writes base_weighted_multiplicative*.fbin (dim unchanged)\n"
              << "    additive: writes base_transformed_additive*.fbin + query_transformed_additive*.fbin "
                 "(dim+1; use inner product only)\n";
}

static std::vector<double> parse_csv_doubles(const std::string& s) {
    std::vector<double> out;
    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t j = s.find(',', i);
        std::string tok = (j == std::string::npos) ? s.substr(i) : s.substr(i, j - i);
        if (!tok.empty()) out.push_back(std::stod(tok));
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return out;
}

static std::string mode_slug(chronos::ChronosScoreMode m) {
    return m == chronos::ChronosScoreMode::Multiplicative ? "multiplicative" : "additive";
}

static std::string alpha_file_tag(double alpha) {
    if (alpha == 0.0) alpha = 0.0;  // Canonicalize negative zero.

    std::ostringstream concise;
    concise << std::fixed << std::setprecision(6) << alpha;
    std::string value = concise.str();
    while (value.size() > 1 && value.back() == '0' &&
           value.find('.') != std::string::npos) {
        value.pop_back();
    }
    if (!value.empty() && value.back() == '.') value.pop_back();
    try {
        if (std::stod(value) == alpha) return "a" + value;
    } catch (const std::exception&) {
        // Fall through to the unambiguous round-trip representation.
    }

    std::ostringstream round_trip;
    round_trip << std::setprecision(std::numeric_limits<double>::max_digits10)
               << std::defaultfloat << alpha;
    value = round_trip.str();
    for (char& character : value) {
        if (character == '.') character = 'p';
        else if (character == '-') character = 'm';
        else if (character == '+') character = 'p';
    }
    return "a" + value;
}

static std::string half_life_file_tag(double half_life) {
    std::ostringstream output;
    output << std::setprecision(15) << std::defaultfloat << half_life;
    std::string tag = output.str();
    for (char& c : tag) {
        if (c == '.') c = 'p';
        else if (c == '-') c = 'm';
    }
    tag.erase(std::remove(tag.begin(), tag.end(), '+'), tag.end());
    return tag;
}

static bool validate_timestamps_vs_max(const std::vector<float>& ts, double max_ts) {
    // Match TANGO's future-time acceptance threshold so a generated workload
    // cannot pass validation here and then be rejected by the index/query CLI.
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
    std::string base_path, query_path, timestamps_path, out_dir, dataset_name, distribution;
    std::string hl_csv;
    std::string mode_str;
    int dim = 0, topk = 0, threads = 64;
    bool export_weighted = false;
    std::string metric_str;
    double alpha = 0.0;
    bool alpha_set = false;
    double max_timestamp_days = 0.0;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto need = [&](const char* name) -> const char* {
                if (i + 1 >= argc) {
                    throw std::invalid_argument(std::string("missing value for ") + name);
                }
                return argv[++i];
            };
            if (a == "--base")
                base_path = need("--base");
            else if (a == "--query")
                query_path = need("--query");
            else if (a == "--timestamps")
                timestamps_path = need("--timestamps");
            else if (a == "--dim")
                dim = std::atoi(need("--dim"));
            else if (a == "--dataset_name")
                dataset_name = need("--dataset_name");
            else if (a == "--distribution")
                distribution = need("--distribution");
            else if (a == "--max_timestamp_days")
                max_timestamp_days = std::stod(need("--max_timestamp_days"));
            else if (a == "--half_life_days")
                hl_csv = need("--half_life_days");
            else if (a == "--topk")
                topk = std::atoi(need("--topk"));
            else if (a == "--threads")
                threads = std::atoi(need("--threads"));
            else if (a == "--out_dir")
                out_dir = need("--out_dir");
            else if (a == "--export_weighted")
                export_weighted = (std::atoi(need("--export_weighted")) != 0);
            else if (a == "--metric")
                metric_str = need("--metric");
            else if (a == "--tdvs_mode")
                mode_str = need("--tdvs_mode");
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
    } catch (const std::exception& error) {
        std::cerr << "Argument error: " << error.what() << '\n';
        usage();
        return 1;
    }

    if (base_path.empty() || query_path.empty() || timestamps_path.empty() || out_dir.empty() ||
        dataset_name.empty() || distribution.empty() || hl_csv.empty() || mode_str.empty() ||
        metric_str.empty() || dim <= 0 || topk <= 0 ||
        threads <= 0 || max_timestamp_days <= 0.0) {
        usage();
        return 1;
    }
    if (distribution != "topic-independent" && distribution != "topic-correlated") {
        std::cerr << "--distribution must be topic-independent or topic-correlated\n";
        return 1;
    }

    chronos::ChronosScoreMode mode = chronos::ChronosScoreMode::Multiplicative;
    if (mode_str == "multiplicative") {
        mode = chronos::ChronosScoreMode::Multiplicative;
    } else if (mode_str == "additive") {
        mode = chronos::ChronosScoreMode::Additive;
    } else {
        std::cerr << "Bad --tdvs_mode (use multiplicative or additive)\n";
        return 1;
    }

    if (mode == chronos::ChronosScoreMode::Additive) {
        if (!alpha_set) {
            std::cerr << "additive mode requires --alpha in [0,1]\n";
            return 1;
        }
        if (alpha < 0.0 || alpha > 1.0) {
            std::cerr << "--alpha must be in [0,1] for additive mode\n";
            return 1;
        }
    } else if (alpha_set) {
        std::cerr << "Warning: --alpha is ignored for multiplicative mode\n";
    }

    bool metric_cosine = false;
    if (metric_str == "ip")
        metric_cosine = false;
    else if (metric_str == "cosine")
        metric_cosine = true;
    else {
        std::cerr << "Bad --metric (use ip or cosine)\n";
        return 1;
    }

    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
        std::cerr << "mkdir: " << ec.message() << "\n";
        return 1;
    }

    std::cerr << "[generate_tdvs_groundtruth] loading base/query/timestamps; metric="
              << (metric_cosine ? "cosine" : "ip") << " mode=" << mode_slug(mode) << "\n";
    std::cerr.flush();

    std::size_t base_bytes = chronos::file_size_bytes(base_path);
    std::size_t q_bytes = chronos::file_size_bytes(query_path);
    if (base_bytes == 0 || q_bytes == 0) {
        std::cerr << "Empty base/query\n";
        return 1;
    }
    std::size_t row = static_cast<std::size_t>(dim) * sizeof(float);
    if (base_bytes % row != 0 || q_bytes % row != 0) {
        std::cerr << "Bad file size vs dim\n";
        return 1;
    }
    std::size_t nb = base_bytes / row;
    std::size_t nq = q_bytes / row;
    if (static_cast<std::size_t>(topk) > nb) {
        std::cerr << "--topk cannot exceed the base row count\n";
        return 1;
    }
    std::cerr << "[generate_tdvs_groundtruth] n_base=" << nb << " n_query=" << nq << " dim=" << dim
              << "\n";
    std::cerr.flush();

    std::vector<float> base_flat(nb * static_cast<std::size_t>(dim));
    std::vector<float> query_flat(nq * static_cast<std::size_t>(dim));
    std::cerr << "[generate_tdvs_groundtruth] reading base.fbin...\n";
    std::cerr.flush();
    {
        FILE* fp = std::fopen(base_path.c_str(), "rb");
        if (!fp || std::fread(base_flat.data(), sizeof(float), base_flat.size(), fp) != base_flat.size()) {
            std::cerr << "Read base failed\n";
            if (fp) std::fclose(fp);
            return 1;
        }
        std::fclose(fp);
    }
    std::cerr << "[generate_tdvs_groundtruth] reading query.fbin...\n";
    std::cerr.flush();
    {
        FILE* fp = std::fopen(query_path.c_str(), "rb");
        if (!fp || std::fread(query_flat.data(), sizeof(float), query_flat.size(), fp) !=
                       query_flat.size()) {
            std::cerr << "Read query failed\n";
            if (fp) std::fclose(fp);
            return 1;
        }
        std::fclose(fp);
    }

    std::cerr << "[generate_tdvs_groundtruth] reading timestamp_days.f32bin...\n";
    std::cerr.flush();
    std::vector<float> timestamps;
    if (!chronos::read_timestamps_binary(timestamps_path, timestamps) || timestamps.size() != nb) {
        std::cerr << "timestamps file length must match num_base (" << nb << ")\n";
        return 1;
    }
    if (!validate_timestamps_vs_max(timestamps, max_timestamp_days)) {
        std::cerr << "timestamp_days values must lie in [0, max_timestamp_days] with max_timestamp_days="
                  << max_timestamp_days << " (do not infer max from the file; pass the metadata value)\n";
        return 1;
    }

    auto half_lives = parse_csv_doubles(hl_csv);
    if (half_lives.empty()) {
        std::cerr << "No half_life_days\n";
        return 1;
    }

    std::vector<float> work_q, work_b;
    const float* qptr = query_flat.data();
    const float* bptr = base_flat.data();
    if (metric_cosine) {
        std::cerr << "[generate_tdvs_groundtruth] L2-normalizing copies for cosine...\n";
        std::cerr.flush();
        work_q = query_flat;
        work_b = base_flat;
        const chronos::VectorRowsReport base_audit = chronos::audit_vector_rows(
            work_b.data(), nb, dim, false, 0.0, false);
        const chronos::VectorRowsReport query_audit = chronos::audit_vector_rows(
            work_q.data(), nq, dim, false, 0.0, false);
        if (!base_audit.valid(false) || !query_audit.valid(false)) {
            std::cerr << "base/query contain non-finite coordinates or zero-norm rows\n";
            return 1;
        }
        chronos::l2_normalize_rows_inplace(work_q.data(), nq, dim, threads);
        chronos::l2_normalize_rows_inplace(work_b.data(), nb, dim, threads);
        qptr = work_q.data();
        bptr = work_b.data();
    }

    int progress_every = 1;
    if (nq >= 40)
        progress_every = static_cast<int>(nq / 40);
    if (progress_every < 1)
        progress_every = 1;

    const double alpha_call = (mode == chronos::ChronosScoreMode::Additive) ? alpha : 0.0;

    for (double hl : half_lives) {
        if (!std::isfinite(hl) || hl <= 0.0) {
            std::cerr << "half-life values must be finite and positive\n";
            return 1;
        }
        std::cerr << "[generate_tdvs_groundtruth] half_life_days=" << hl
                  << ": brute-force top-" << topk << " (" << nq << " queries x " << nb
                  << " base, threads=" << threads << "; progress every ~" << progress_every
                  << " queries on stderr)\n";
        std::cerr.flush();
        const std::string hl_tag = half_life_file_tag(hl);
        std::vector<chronos::GroundTruthId> gt;
        chronos::brute_force_chronos_topk(qptr, nq, bptr, nb, dim, timestamps.data(), max_timestamp_days, hl,
                                          mode, alpha_call, topk, threads, gt, nullptr, progress_every);
        std::cerr << "[generate_tdvs_groundtruth] half_life_days=" << hl
                  << ": groundtruth done; writing outputs...\n";
        std::cerr.flush();

        std::ostringstream gt_path;
        gt_path << out_dir << "/gt_chronos_" << mode_slug(mode);
        if (mode == chronos::ChronosScoreMode::Additive) gt_path << "." << alpha_file_tag(alpha);
        gt_path << (metric_cosine ? "_cosine" : "") << ".hl" << hl_tag << ".k" << topk
                << ".bin";
        if (!chronos::write_groundtruth(gt_path.str(), gt.data(), nq, topk)) {
            std::cerr << "Write gt failed\n";
            return 1;
        }

        std::string transformed_base_path;
        std::string transformed_query_path;
        if (export_weighted) {
            double lambda = chronos::half_life_to_lambda(hl);
            if (mode == chronos::ChronosScoreMode::Multiplicative) {
                std::cerr << "[generate_tdvs_groundtruth] exporting weighted base (multiplicative) for hl=" << hl
                          << "...\n";
                std::cerr.flush();
                std::vector<float> wb;
                chronos::build_multiplicative_transformed_base(base_flat.data(), nb, dim, timestamps.data(),
                                                               max_timestamp_days, hl, metric_cosine, wb,
                                                               threads);
                std::ostringstream wb_path;
                wb_path << out_dir << "/base_weighted_multiplicative" << (metric_cosine ? "_cosine" : "") << ".hl"
                        << hl_tag << ".fbin";
                transformed_base_path = wb_path.str();
                if (!chronos::write_raw_vectors(transformed_base_path, wb.data(), nb, dim)) {
                    std::cerr << "Write weighted base failed\n";
                    return 1;
                }
            } else {
                std::cerr << "[generate_tdvs_groundtruth] exporting additive IP-augmented base+query for hl="
                          << hl << " (dim_out=" << (dim + 1) << ")...\n";
                std::cerr.flush();
                std::vector<float> aug_base, aug_query;
                chronos::build_additive_transformed_base(base_flat.data(), nb, dim, timestamps.data(),
                                                         max_timestamp_days, lambda, alpha, metric_cosine,
                                                         aug_base, threads);
                chronos::build_additive_transformed_queries(query_flat.data(), nq, dim, alpha, metric_cosine,
                                                            aug_query, threads);
                if (aug_base.empty() || aug_query.empty()) {
                    std::cerr << "Additive transform build failed\n";
                    return 1;
                }
                if (!chronos::verify_additive_transformed_ip_equivalence(
                        base_flat.data(), query_flat.data(), aug_base.data(), aug_query.data(), nb, nq, dim,
                        timestamps.data(), max_timestamp_days, lambda, alpha, metric_cosine, 8)) {
                    std::cerr << "Additive augmented dataset failed internal IP equivalence check\n";
                    return 1;
                }
                std::cerr << "[generate_tdvs_groundtruth] additive IP equivalence check passed (sampled)\n";
                std::cerr.flush();
                const int dim_out = dim + 1;
                std::string atag = alpha_file_tag(alpha);
                std::ostringstream bpath, qpath;
                bpath << out_dir << "/base_transformed_additive.hl" << hl_tag << "." << atag
                      << (metric_cosine ? "_cosine" : "") << ".fbin";
                qpath << out_dir << "/query_transformed_additive.hl" << hl_tag << "." << atag
                      << (metric_cosine ? "_cosine" : "") << ".fbin";
                transformed_base_path = bpath.str();
                transformed_query_path = qpath.str();
                if (!chronos::write_raw_vectors(transformed_base_path, aug_base.data(), nb, dim_out)) {
                    std::cerr << "Write transformed base failed\n";
                    return 1;
                }
                if (!chronos::write_raw_vectors(transformed_query_path, aug_query.data(), nq, dim_out)) {
                    std::cerr << "Write transformed query failed\n";
                    return 1;
                }
            }
        }

        std::ostringstream man;
        man << "task: generate_tdvs_groundtruth\n";
        man << "dataset_name: " << dataset_name << "  (manifest / provenance only; not used in scoring)\n";
        man << "distribution: " << distribution << "\n";
        man << "tdvs_mode: " << mode_slug(mode) << "\n";
        if (mode == chronos::ChronosScoreMode::Additive) man << "alpha: " << alpha << "\n";
        man << "n_base: " << nb << "\n";
        man << "n_query: " << nq << "\n";
        man << "dim: " << dim << "\n";
        man << "half_life_days: " << hl << "\n";
        man << "metric: " << (metric_cosine ? "cosine" : "inner_product") << "\n";
        man << "max_timestamp_days: " << max_timestamp_days << "\n";
        man << "t_ref: max_timestamp_days (must match metadata generation)\n";
        man << "bounded_time_definition: t_i_bounded = timestamp_days[i] - t_ref\n";
        man << "lambda: ln(2) / half_life_days\n";
        man << "decay_i: exp(lambda * t_i_bounded)\n";
        if (metric_cosine) {
            man << "semantic: dot(L2_norm(q), L2_norm(x)); on_disk_vectors raw; GT on normalized copies\n";
        } else {
            man << "semantic: inner_product(q, x)\n";
        }
        if (mode == chronos::ChronosScoreMode::Multiplicative) {
            man << "score: semantic * decay_i\n";
        } else {
            man << "score: alpha * semantic + (1 - alpha) * decay_i\n";
        }
        man << "topk: " << topk << "\n";
        man << "threads: " << threads << "\n";
        man << "path_base: manifest_directory\n";
        man << "base_file: " << relative_to(base_path, out_dir) << "\n";
        man << "query_file: " << relative_to(query_path, out_dir) << "\n";
        man << "timestamps_file: " << relative_to(timestamps_path, out_dir) << "\n";
        man << "groundtruth_file: " << relative_to(gt_path.str(), out_dir) << "\n";
        man << "groundtruth_format: Chronos evaluation (int64 ids only)\n";
        man << "groundtruth_order: score descending, then row ID ascending\n";
        if (export_weighted) {
            if (mode == chronos::ChronosScoreMode::Multiplicative) {
                man << "export_kind: multiplicative_weighted_base\n";
                man << "weighted_base_semantics: weight_i = decay_i; out_i = weight_i * x_i "
                       "(cosine: weight_i * L2_norm(x_i))\n";
                man << "weighted_base_file: base_weighted_multiplicative"
                    << (metric_cosine ? "_cosine" : "") << ".hl" << hl_tag << ".fbin\n";
                if (metric_cosine) {
                    man << "hnswlib: use space='ip' with this file as base; query must be L2-normalized\n";
                    man << "hnswlib: do NOT use space='cosine' (re-normalizes and removes time weights)\n";
                }
            } else {
                man << "export_kind: additive_ip_augmented_dataset\n";
                man << "dim_out: " << (dim + 1) << "  (original dim + 1)\n";
                man << "retrieval_metric_on_exported_files: inner_product_only (do not use cosine space)\n";
                man << "additive_transform_equivalence: dot(q', x'_i) == alpha*semantic + (1-alpha)*decay_i "
                       "(exact up to float arithmetic; verified by sampled check in tool)\n";
                man << "semantic_on_disk_chronos_metric: "
                    << (metric_cosine ? "cosine (L2 row-normalize before sqrt(alpha) block)" : "inner_product")
                    << "\n";
                man << "x'_i: [ sqrt(alpha)*x_part , sqrt(1-alpha)*decay_i ]; "
                       "q': [ sqrt(alpha)*q_part , sqrt(1-alpha) ]\n";
                man << "x_part/q_part: raw vectors for ip; L2-normalized per row for cosine Chronos semantic\n";
                man << "decay_i: exp(lambda * (timestamp_days[i] - t_ref))\n";
                man << "transformed_base_file: base_transformed_additive.hl" << hl_tag << "."
                    << alpha_file_tag(alpha) << (metric_cosine ? "_cosine" : "") << ".fbin\n";
                man << "transformed_query_file: query_transformed_additive.hl" << hl_tag
                    << "." << alpha_file_tag(alpha) << (metric_cosine ? "_cosine" : "") << ".fbin\n";
            }
        }
        std::ostringstream man_path;
        man_path << out_dir << "/manifest_chronos_" << mode_slug(mode);
        if (mode == chronos::ChronosScoreMode::Additive) man_path << "." << alpha_file_tag(alpha);
        man_path << ".hl" << hl_tag << ".txt";
        chronos::write_text_file(man_path.str(), man.str());

        std::ostringstream json_path;
        json_path << out_dir << "/manifest_chronos_" << mode_slug(mode);
        if (mode == chronos::ChronosScoreMode::Additive) json_path << "." << alpha_file_tag(alpha);
        json_path << ".hl" << hl_tag << ".json";
        std::ostringstream json;
        json << std::setprecision(17)
             << "{\n"
             << "  \"schema\": \"chronos_tdvs_artifact_v1\",\n"
             << "  \"dataset_name\": \"" << json_escape(dataset_name) << "\",\n"
             << "  \"distribution\": \"" << distribution << "\",\n"
             << "  \"score_mode\": \"" << mode_slug(mode) << "\",\n"
             << "  \"semantic_metric\": \"" << (metric_cosine ? "cosine" : "inner_product") << "\",\n"
             << "  \"half_life_days\": " << hl << ",\n"
             << "  \"query_time_days\": " << max_timestamp_days << ",\n"
             << "  \"timestamp_semantics\": \"creation_time_days; larger_is_newer\",\n";
        if (mode == chronos::ChronosScoreMode::Additive) {
            json << "  \"alpha\": " << alpha << ",\n";
        } else {
            json << "  \"alpha\": null,\n";
        }
        json << "  \"base_rows\": " << nb << ", \"query_rows\": " << nq
             << ", \"dimension\": " << dim << ", \"groundtruth_width\": " << topk << ",\n"
             << "  \"inputs\": {\n"
             << "    \"base\": \"" << json_escape(relative_to(base_path, out_dir)) << "\",\n"
             << "    \"query\": \"" << json_escape(relative_to(query_path, out_dir)) << "\",\n"
             << "    \"timestamps\": \"" << json_escape(relative_to(timestamps_path, out_dir)) << "\"\n"
             << "  },\n"
             << "  \"outputs\": {\n"
             << "    \"groundtruth\": \"" << json_escape(relative_to(gt_path.str(), out_dir)) << "\",\n"
             << "    \"groundtruth_format\": \"headerless_int64_ids\",\n"
             << "    \"groundtruth_order\": \"score_desc_then_row_id_asc\",\n"
             << "    \"transformed_base\": "
             << (transformed_base_path.empty() ? "null" : "\"" + json_escape(relative_to(transformed_base_path, out_dir)) + "\"") << ",\n"
             << "    \"transformed_query\": "
             << (transformed_query_path.empty() ? "null" : "\"" + json_escape(relative_to(transformed_query_path, out_dir)) + "\"") << "\n"
             << "  },\n"
             << "  \"producer\": {\"tool\": \"generate_tdvs_groundtruth\", \"threads\": "
             << threads << ", \"git_revision\": \"" << chronos::source_revision()
             << "\", \"source_tree_dirty\": "
             << (chronos::source_tree_was_dirty_at_configure() ? "true" : "false") << "}\n"
             << "}\n";
        if (!chronos::write_text_file(json_path.str(), json.str())) {
            std::cerr << "Write JSON manifest failed\n";
            return 1;
        }

        std::cout << "Wrote " << gt_path.str() << "\n";
    }

    return 0;
}
