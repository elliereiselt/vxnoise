#include "NoiseAVX2.hpp"

vx::noise::NoiseAVX2::NoiseAVX2(int seed, float defaultScale)
        : _avx2Seed(_mm256_set1_epi32(seed)), _avx2DefaultScale(_mm256_set1_ps(defaultScale)) {

}
