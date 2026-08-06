#pragma once

#include <random>

namespace efanna2e {

void GenRandom(std::mt19937& rng, unsigned* addr, unsigned size, unsigned N);

float* load_data(const char* filename, unsigned& num, unsigned& dim);

float* data_align(float* data_ori, unsigned point_num, unsigned& dim);

}  // namespace efanna2e

