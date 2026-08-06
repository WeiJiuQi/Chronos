/** Generates the two publication workloads for Chronos timestamps. */

#include "io.h"
#include "time_metadata.h"
#include "validation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string base_path;
    std::string out_root;
    std::string dataset_name;
    std::string distribution = "all";
    int dim = 0;
    int threads = 64;
    int num_clusters = 256;
    int kmeans_iters = 15;
    std::size_t training_sample_rows = 256000;
    std::uint64_t seed = 42;
    float horizon = 365.0f;
    float cluster_std = 0.0f;
};

struct TimestampStats {
    double minimum = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double standard_deviation = 0.0;
    double p01 = 0.0;
    double p10 = 0.0;
    double p25 = 0.0;
    double p50 = 0.0;
    double p75 = 0.0;
    double p90 = 0.0;
    double p99 = 0.0;
    std::size_t boundary_values = 0;
};

void usage() {
    std::cout
        << "Usage: generate_time_metadata \\\n"
        << "  --base_subset PATH --dim D --dataset_name NAME --out_root DIR [options]\n\n"
        << "  --distribution topic-independent|topic-correlated|all  default all\n"
        << "  --max_timestamp_days H                         default 365\n"
        << "  --seed U64                                     default 42\n"
        << "  --threads T                                    default 64\n"
        << "  --num_clusters K                               default 256\n"
        << "  --training_sample_rows N                       default 256000\n"
        << "  --kmeans_iters N                               default 15\n"
        << "  --cluster_timestamp_std_days S                 default H/12\n"
        << "  --cluster_noise_std S                          legacy alias\n\n"
        << "topic-independent samples timestamp~Uniform(0,H). topic-correlated uses\n"
        << "spherical k-means and timestamp~TruncatedNormal(cluster_time,S,[0,H]).\n"
        << "Larger timestamp values always mean newer objects.\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (flag == "--base_subset") options.base_path = value("--base_subset");
        else if (flag == "--out_root") options.out_root = value("--out_root");
        else if (flag == "--dataset_name") options.dataset_name = value("--dataset_name");
        else if (flag == "--distribution") options.distribution = value("--distribution");
        else if (flag == "--dim") options.dim = std::stoi(value("--dim"));
        else if (flag == "--threads") options.threads = std::stoi(value("--threads"));
        else if (flag == "--num_clusters") options.num_clusters = std::stoi(value("--num_clusters"));
        else if (flag == "--kmeans_iters") options.kmeans_iters = std::stoi(value("--kmeans_iters"));
        else if (flag == "--training_sample_rows") {
            options.training_sample_rows = std::stoull(value("--training_sample_rows"));
        } else if (flag == "--seed") options.seed = std::stoull(value("--seed"));
        else if (flag == "--max_timestamp_days") options.horizon = std::stof(value("--max_timestamp_days"));
        else if (flag == "--cluster_timestamp_std_days" || flag == "--cluster_noise_std") {
            options.cluster_std = std::stof(value(flag.c_str()));
        } else if (flag == "--help" || flag == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + flag);
        }
    }
    if (options.base_path.empty() || options.out_root.empty() || options.dataset_name.empty() ||
        options.dim <= 0 || options.threads <= 0 || options.horizon <= 0.0f) {
        throw std::invalid_argument("base, output root, dataset name, dimension, threads, and horizon are required");
    }
    if (options.distribution != "all" && options.distribution != "topic-independent" &&
        options.distribution != "topic-correlated") {
        throw std::invalid_argument("--distribution must be topic-independent, topic-correlated, or all");
    }
    if (options.num_clusters < 2 || options.kmeans_iters < 1 || options.training_sample_rows == 0 ||
        options.cluster_std < 0.0f) {
        throw std::invalid_argument("invalid cluster parameters");
    }
    return options;
}

double quantile(const std::vector<float>& sorted, double probability) {
    if (sorted.empty()) return 0.0;
    const double position = probability * static_cast<double>(sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - lower;
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

TimestampStats summarize(const std::vector<float>& timestamps, float horizon) {
    TimestampStats stats;
    if (timestamps.empty()) return stats;
    std::vector<float> sorted = timestamps;
    std::sort(sorted.begin(), sorted.end());
    stats.minimum = sorted.front();
    stats.maximum = sorted.back();
    double sum = 0.0;
    for (float value : timestamps) {
        sum += value;
        if (value == 0.0f || value == horizon) ++stats.boundary_values;
    }
    stats.mean = sum / timestamps.size();
    double squared = 0.0;
    for (float value : timestamps) {
        const double delta = value - stats.mean;
        squared += delta * delta;
    }
    stats.standard_deviation = std::sqrt(squared / timestamps.size());
    stats.p01 = quantile(sorted, 0.01);
    stats.p10 = quantile(sorted, 0.10);
    stats.p25 = quantile(sorted, 0.25);
    stats.p50 = quantile(sorted, 0.50);
    stats.p75 = quantile(sorted, 0.75);
    stats.p90 = quantile(sorted, 0.90);
    stats.p99 = quantile(sorted, 0.99);
    return stats;
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

std::string relative_to(const std::string& path, const std::string& directory) {
    std::error_code error;
    const std::filesystem::path relative = std::filesystem::relative(path, directory, error);
    return error || relative.empty() ? std::filesystem::path(path).filename().string()
                                     : relative.generic_string();
}

void write_distribution(
        const Options& options,
        const chronos::TimeMetadataResult& result,
        chronos::TimeDistribution distribution,
        std::size_t n_base) {
    const std::string name = chronos::time_distribution_name(distribution);
    const std::string directory = options.out_root + '/' + name;
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) throw std::runtime_error("cannot create metadata directory: " + error.message());

    const std::string timestamp_path = directory + "/timestamp_days.f32bin";
    if (!chronos::write_timestamps_binary(timestamp_path, result.timestamps)) {
        throw std::runtime_error("failed writing timestamps: " + timestamp_path);
    }
    if (distribution == chronos::TimeDistribution::TopicCorrelated) {
        if (!chronos::write_cluster_ids_binary(directory + "/cluster_ids.i32bin", result.cluster_ids) ||
            !chronos::write_raw_vectors(directory + "/cluster_timestamp_centers.f32bin",
                                        result.cluster_timestamp_centers.data(),
                                        result.cluster_timestamp_centers.size(), 1)) {
            throw std::runtime_error("failed writing cluster metadata");
        }
    }

    const TimestampStats stats = summarize(result.timestamps, options.horizon);
    std::vector<std::size_t> cluster_sizes;
    double mean_within_cluster_std = 0.0;
    if (!result.cluster_ids.empty()) {
        cluster_sizes.assign(result.cluster_timestamp_centers.size(), 0);
        std::vector<double> sums(cluster_sizes.size(), 0.0);
        std::vector<double> squared_sums(cluster_sizes.size(), 0.0);
        for (std::size_t row = 0; row < result.cluster_ids.size(); ++row) {
            const std::size_t cluster = static_cast<std::size_t>(result.cluster_ids[row]);
            ++cluster_sizes[cluster];
            sums[cluster] += result.timestamps[row];
            squared_sums[cluster] += static_cast<double>(result.timestamps[row]) * result.timestamps[row];
        }
        std::size_t nonempty_clusters = 0;
        for (std::size_t cluster = 0; cluster < cluster_sizes.size(); ++cluster) {
            if (cluster_sizes[cluster] == 0) continue;
            ++nonempty_clusters;
            const double mean = sums[cluster] / cluster_sizes[cluster];
            mean_within_cluster_std += std::sqrt(std::max(
                0.0, squared_sums[cluster] / cluster_sizes[cluster] - mean * mean));
        }
        if (nonempty_clusters != 0) {
            mean_within_cluster_std /= nonempty_clusters;
        }
        std::sort(cluster_sizes.begin(), cluster_sizes.end());
    }

    std::ostringstream manifest;
    manifest << std::setprecision(17)
             << "{\n"
             << "  \"schema\": \"chronos_time_metadata_v1\",\n"
             << "  \"dataset_name\": \"" << json_escape(options.dataset_name) << "\",\n"
             << "  \"distribution\": \"" << name << "\",\n"
             << "  \"timestamp_semantics\": \"creation_time_days; larger_is_newer\",\n"
             << "  \"horizon_days\": " << options.horizon << ",\n"
             << "  \"base_rows\": " << n_base << ",\n"
             << "  \"dimension\": " << options.dim << ",\n"
             << "  \"seed\": " << options.seed << ",\n"
             << "  \"input_base\": {\"path\": \""
             << json_escape(relative_to(options.base_path, directory)) << "\", \"bytes\": "
             << chronos::file_size_bytes(options.base_path) << "},\n"
             << "  \"outputs\": {\"timestamps\": \"timestamp_days.f32bin\"";
    if (!result.cluster_ids.empty()) {
        manifest << ", \"cluster_ids\": \"cluster_ids.i32bin\", "
                 << "\"cluster_timestamp_centers\": \"cluster_timestamp_centers.f32bin\"";
    }
    manifest << "},\n"
             << "  \"generation\": {";
    if (distribution == chronos::TimeDistribution::TopicIndependent) {
        manifest << "\"model\": \"iid_uniform\", \"formula\": \"timestamp_i ~ Uniform(0,H)\"";
    } else {
        const double sigma = options.cluster_std > 0.0f ? options.cluster_std : options.horizon / 12.0;
        manifest << "\"model\": \"semantic_topic_correlated\", "
                 << "\"clustering\": \"spherical_kmeans\", "
                 << "\"initialization\": \"kmeans++\", "
                 << "\"num_clusters\": " << result.cluster_timestamp_centers.size() << ", "
                 << "\"training_sample_rows\": " << result.training_rows << ", "
                 << "\"kmeans_iterations\": " << options.kmeans_iters << ", "
                 << "\"cluster_centers_in_time\": \"iid_uniform_0_H\", "
                 << "\"within_cluster_distribution\": \"truncated_normal_0_H\", "
                 << "\"within_cluster_std_days\": " << sigma;
    }
    manifest << "},\n"
             << "  \"statistics\": {\n"
             << "    \"min\": " << stats.minimum << ", \"max\": " << stats.maximum
             << ", \"mean\": " << stats.mean << ", \"std\": " << stats.standard_deviation << ",\n"
             << "    \"p01\": " << stats.p01 << ", \"p10\": " << stats.p10
             << ", \"p25\": " << stats.p25 << ", \"p50\": " << stats.p50
             << ", \"p75\": " << stats.p75 << ", \"p90\": " << stats.p90
             << ", \"p99\": " << stats.p99 << ",\n"
             << "    \"exact_boundary_values\": " << stats.boundary_values;
    if (!cluster_sizes.empty()) {
        manifest << ",\n    \"cluster_size_min\": " << cluster_sizes.front()
                 << ", \"cluster_size_median\": " << cluster_sizes[cluster_sizes.size() / 2]
                 << ", \"cluster_size_max\": " << cluster_sizes.back()
                 << ", \"mean_within_cluster_timestamp_std_days\": " << mean_within_cluster_std;
    }
    manifest << "\n  },\n"
             << "  \"producer\": {\"tool\": \"generate_time_metadata\", \"threads\": "
             << options.threads << ", \"git_revision\": \"" << chronos::source_revision()
             << "\", \"source_tree_dirty\": "
             << (chronos::source_tree_was_dirty_at_configure() ? "true" : "false") << "}\n"
             << "}\n";
    if (!chronos::write_text_file(directory + "/manifest_meta.json", manifest.str())) {
        throw std::runtime_error("failed writing time metadata JSON manifest");
    }

    std::ostringstream legacy;
    legacy << "task: generate_time_metadata\n"
           << "canonical_manifest: manifest_meta.json\n"
           << "dataset_name: " << options.dataset_name << '\n'
           << "distribution: " << name << '\n'
           << "n_base_vectors: " << n_base << '\n'
           << "dim: " << options.dim << '\n'
           << "timestamp_semantics: creation timestamp in [0,H]; larger = newer\n"
           << "max_timestamp_days: " << options.horizon << '\n';
    chronos::write_text_file(directory + "/manifest_meta.txt", legacy.str());
    std::cout << "Wrote " << timestamp_path << '\n';
}

int run(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    const std::size_t row_bytes = static_cast<std::size_t>(options.dim) * sizeof(float);
    const std::size_t bytes = chronos::file_size_bytes(options.base_path);
    if (bytes == 0 || bytes % row_bytes != 0) {
        throw std::runtime_error("base must be headerless float32 rows matching --dim");
    }
    const std::size_t n_base = bytes / row_bytes;
    std::vector<float> base(n_base * static_cast<std::size_t>(options.dim));
    FILE* input = std::fopen(options.base_path.c_str(), "rb");
    if (!input || std::fread(base.data(), sizeof(float), base.size(), input) != base.size()) {
        if (input) std::fclose(input);
        throw std::runtime_error("failed reading canonical base");
    }
    std::fclose(input);
    const chronos::VectorRowsReport audit = chronos::audit_vector_rows(
        base.data(), n_base, options.dim, true, 1e-4, false);
    if (!audit.valid(true)) {
        throw std::runtime_error(
            "base is not a finite, nonzero unit-vector dataset; run prepare_vector_subset with cosine normalization");
    }

    auto generate = [&](chronos::TimeDistribution distribution) {
        chronos::TimeMetadataOptions metadata;
        metadata.distribution = distribution;
        metadata.seed = options.seed;
        metadata.max_timestamp_days = options.horizon;
        metadata.num_clusters = options.num_clusters;
        metadata.kmeans_iters = options.kmeans_iters;
        metadata.training_sample_rows = options.training_sample_rows;
        metadata.cluster_timestamp_std_days = options.cluster_std;
        metadata.num_threads = options.threads;
        const chronos::TimeMetadataResult result = chronos::generate_time_metadata(
            base.data(), n_base, options.dim, metadata);
        write_distribution(options, result, distribution, n_base);
    };
    if (options.distribution == "all" || options.distribution == "topic-independent") {
        generate(chronos::TimeDistribution::TopicIndependent);
    }
    if (options.distribution == "all" || options.distribution == "topic-correlated") {
        generate(chronos::TimeDistribution::TopicCorrelated);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "generate_time_metadata: " << error.what() << '\n';
        return 1;
    }
}
