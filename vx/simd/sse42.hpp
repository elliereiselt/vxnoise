#ifndef VXNOISE_SSE42_HPP
#define VXNOISE_SSE42_HPP

// SSE4.2
#include <smmintrin.h>
#include "simd.hpp"

namespace vx {
    namespace simd {
        namespace sse42 {
            inline __m128 VX_VECTORCALL abs_ps(__m128 x) {
                return _mm_andnot_ps(_mm_set1_ps(-0.0f), x);
            }

            inline __m128 VX_VECTORCALL dot_ps(__m128 x0, __m128 y0, __m128 x1, __m128 y1) {
                return _mm_add_ps(_mm_mul_ps(x0, x1), _mm_mul_ps(y0, y1));
            }

            inline __m128 VX_VECTORCALL dot_ps(__m128 x0, __m128 y0, __m128 z0, __m128 x1, __m128 y1, __m128 z1) {
                return _mm_add_ps(_mm_add_ps(_mm_mul_ps(x0, x1), _mm_mul_ps(y0, y1)), _mm_mul_ps(z0, z1));
            }

            // CREDIT: https://arxiv.org/pdf/1903.12270.pdf
            inline __m128i VX_VECTORCALL partialJenkinsHash(__m128i seed, __m128i key) {
                __m128i hash = seed;
                hash = _mm_add_epi32(hash, key);
                hash = _mm_add_epi32(hash, _mm_slli_epi32(hash, 10));
                hash = _mm_xor_si128(hash, _mm_srai_epi32(hash, 6));
                hash = _mm_add_epi32(hash, _mm_slli_epi32(hash, 3));
                hash = _mm_xor_si128(hash, _mm_srai_epi32(hash, 11));
                hash = _mm_add_epi32(hash, _mm_slli_epi32(hash, 15));
                return hash;
            }
        }
    }
}

#endif //VXNOISE_SSE42_HPP
