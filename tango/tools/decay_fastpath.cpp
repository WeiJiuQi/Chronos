// Performance profiler for the TANGO decay-basis factorization fastpath.
//
// Cache initialization is outside every timed region.
#include "tango.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

volatile double profiling_sink = 0.0;

struct Lcg {
    uint64_t state = 0x123456789abcdefULL;
    size_t next(size_t bound) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<size_t>((state >> 16) % bound);
    }
};

template <class Operation>
double runTimed(size_t iterations, Operation operation, double &checksum) {
    Lcg rng;
    double sum = 0.0;
    const Clock::time_point start = Clock::now();
    for (size_t i = 0; i < iterations; ++i) sum += operation(rng);
    const double elapsed_ns = std::chrono::duration<double, std::nano>(
        Clock::now() - start).count();
    profiling_sink = sum;
    checksum = sum;
    return elapsed_ns / static_cast<double>(iterations);
}

void emit(
        const char *operation,
        size_t dim,
        size_t iterations,
        double ns_per_op,
        double checksum) {
    std::cout << operation << ',' << dim << ',' << iterations << ','
              << std::setprecision(12) << ns_per_op << ',' << checksum << '\n';
}

void printUsage(const char *program) {
    std::cout << "Usage: " << program
              << " [scalar_iterations] [dot_iterations]\n"
              << "Profiles direct and cached TANGO decay evaluation plus "
                 "semantic dot-product kernels.\n";
}

}  // namespace

int main(int argc, char **argv) {
    if (argc > 1 &&
        (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        printUsage(argv[0]);
        return 0;
    }
    const size_t scalar_iterations = argc > 1
        ? static_cast<size_t>(std::strtoull(argv[1], NULL, 10))
        : 2000000;
    const size_t dot_iterations = argc > 2
        ? static_cast<size_t>(std::strtoull(argv[2], NULL, 10))
        : 20000;
    if (scalar_iterations == 0 || dot_iterations == 0) {
        std::cerr << "iterations must be positive\n";
        return 1;
    }

    const size_t cache_size = 65536;
    const double half_life = 30.0;
    const double lambda = std::log(2.0) / half_life;
    const double reference_time = 365.0;
    const double query_time = 365.0;
    const double query_scale = std::exp(
        -lambda * (query_time - reference_time));
    std::vector<double> timestamps(cache_size);
    std::vector<double> basis(cache_size);
    std::vector<double> inverse(cache_size);
    std::vector<double> query_times(cache_size);
    for (size_t i = 0; i < cache_size; ++i) {
        timestamps[i] = reference_time - 365.0 *
            static_cast<double>(i) / static_cast<double>(cache_size - 1);
        basis[i] = std::exp(lambda * (timestamps[i] - reference_time));
        inverse[i] = 1.0 / basis[i];
        query_times[i] = reference_time + 365.0 *
            static_cast<double>(i) / static_cast<double>(cache_size - 1);
    }

    std::cout << "operation,dim,iterations,ns_per_op,checksum\n";
    double checksum = 0.0;
    double ns_per_op = runTimed(scalar_iterations, [&](Lcg &rng) {
        const size_t i = rng.next(cache_size);
        return std::exp(-lambda * (query_time - timestamps[i]));
    }, checksum);
    emit("qd_direct_exp", 0, scalar_iterations, ns_per_op, checksum);
    ns_per_op = runTimed(scalar_iterations, [&](Lcg &rng) {
        return query_scale * basis[rng.next(cache_size)];
    }, checksum);
    emit("qd_basis_multiply", 0, scalar_iterations, ns_per_op, checksum);
    ns_per_op = runTimed(scalar_iterations, [&](Lcg &rng) {
        return std::exp(
            -lambda * (query_times[rng.next(cache_size)] - reference_time));
    }, checksum);
    emit("query_scale_exp", 0, scalar_iterations, ns_per_op, checksum);
    ns_per_op = runTimed(scalar_iterations, [&](Lcg &rng) {
        const size_t i = rng.next(cache_size);
        const size_t j = rng.next(cache_size);
        return std::exp(-lambda * std::fabs(timestamps[i] - timestamps[j]));
    }, checksum);
    emit("dd_direct_exp", 0, scalar_iterations, ns_per_op, checksum);
    ns_per_op = runTimed(scalar_iterations, [&](Lcg &rng) {
        const size_t i = rng.next(cache_size);
        const size_t j = rng.next(cache_size);
        return std::min(basis[i], basis[j]) / std::max(basis[i], basis[j]);
    }, checksum);
    emit("dd_basis_division", 0, scalar_iterations, ns_per_op, checksum);
    ns_per_op = runTimed(scalar_iterations, [&](Lcg &rng) {
        const size_t i = rng.next(cache_size);
        const size_t j = rng.next(cache_size);
        return basis[i] < basis[j]
            ? basis[i] * inverse[j]
            : basis[j] * inverse[i];
    }, checksum);
    emit("dd_basis_inverse", 0, scalar_iterations, ns_per_op, checksum);

    const size_t dims[] = {1024, 1536, 3072};
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    for (size_t di = 0; di < sizeof(dims) / sizeof(dims[0]); ++di) {
        const size_t dim = dims[di];
        // Large enough to exceed the private caches on typical target CPUs;
        // block selection therefore exercises random vector access too.
        const size_t blocks = 1024;
        std::vector<float> left(blocks * dim);
        std::vector<float> right(blocks * dim);
        for (size_t i = 0; i < left.size(); ++i) {
            left[i] = uniform(generator);
            right[i] = uniform(generator);
        }
        hnswlib::InnerProductSpace semantic_space(dim);
        hnswlib::DISTFUNC<float> semantic_distance =
            semantic_space.get_dist_func();
        void *semantic_distance_param = semantic_space.get_dist_func_param();
        ns_per_op = runTimed(dot_iterations, [&](Lcg &rng) {
            const size_t block = rng.next(blocks);
            return 1.0 - static_cast<double>(semantic_distance(
                left.data() + block * dim,
                right.data() + block * dim,
                semantic_distance_param));
        }, checksum);
        emit("semantic_dot", dim, dot_iterations, ns_per_op, checksum);
        ns_per_op = runTimed(dot_iterations, [&](Lcg &rng) {
            const size_t block = rng.next(blocks);
            const size_t item = rng.next(cache_size);
            const double semantic = 1.0 - static_cast<double>(semantic_distance(
                left.data() + block * dim,
                right.data() + block * dim,
                semantic_distance_param));
            return semantic *
                std::exp(-lambda * (query_time - timestamps[item]));
        }, checksum);
        emit("dot_plus_qd_direct", dim, dot_iterations, ns_per_op, checksum);
        ns_per_op = runTimed(dot_iterations, [&](Lcg &rng) {
            const size_t block = rng.next(blocks);
            const size_t item = rng.next(cache_size);
            const double semantic = 1.0 - static_cast<double>(semantic_distance(
                left.data() + block * dim,
                right.data() + block * dim,
                semantic_distance_param));
            return semantic *
                query_scale * basis[item];
        }, checksum);
        emit("dot_plus_qd_fast", dim, dot_iterations, ns_per_op, checksum);
    }
    return profiling_sink == -1.0 ? 1 : 0;
}
