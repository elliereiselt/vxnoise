#ifndef VXNOISE_AVX2_HPP
#define VXNOISE_AVX2_HPP

// AVX2
#include <immintrin.h>
#include "simd.hpp"

namespace vx {
    namespace simd {
        namespace avx2 {
            /// Calculate absolute value
            inline __m256 VX_VECTORCALL abs_ps(__m256 x) {
                return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
            }

            /// Calculate dot product
            inline __m256 VX_VECTORCALL dot_ps(__m256 x0, __m256 y0, __m256 x1, __m256 y1) {
                return _mm256_add_ps(_mm256_mul_ps(x0, x1), _mm256_mul_ps(y0, y1));
            }

            /// Calculate dot product
            inline __m256 VX_VECTORCALL dot_ps(__m256 x0, __m256 y0, __m256 z0, __m256 x1, __m256 y1, __m256 z1) {
                return _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(x0, x1), _mm256_mul_ps(y0, y1)), _mm256_mul_ps(z0, z1));
            }

            /**
             * Generate a partial Jenkins hash
             *
             * CREDIT: https://arxiv.org/pdf/1903.12270.pdf
             *
             * @param seed
             * @param key
             * @return
             */
            inline __m256i VX_VECTORCALL partialJenkinsHash(__m256i seed, __m256i key) {
                __m256i hash = seed;
                hash = _mm256_add_epi32(hash, key);
                hash = _mm256_add_epi32(hash, _mm256_slli_epi32(hash, 10));
                hash = _mm256_xor_si256(hash, _mm256_srai_epi32(hash, 6));
                hash = _mm256_add_epi32(hash, _mm256_slli_epi32(hash, 3));
                hash = _mm256_xor_si256(hash, _mm256_srai_epi32(hash, 11));
                hash = _mm256_add_epi32(hash, _mm256_slli_epi32(hash, 15));
                return hash;
            }
        }
    }
}

#endif //VXNOISE_AVX2_HPP
