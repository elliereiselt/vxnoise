#ifndef VXNOISE_NOISESSE42_HPP
#define VXNOISE_NOISESSE42_HPP

#include "../../simd/sse42.hpp"

namespace vx {
    namespace noise {
        class NoiseSSE42 {
        public:
            __m128 VX_VECTORCALL sse42Get2d(__m128 x, __m128 y) {
                return sse42Get2d(x, y, _sse42DefaultScale);
            }

            __m128 VX_VECTORCALL sse42Get3d(__m128 x, __m128 y, __m128 z) {
                return sse42Get3d(x, y, z, _sse42DefaultScale);
            }

            virtual __m128 VX_VECTORCALL sse42Get2d(__m128 x, __m128 y, __m128 scale) = 0;
            virtual __m128 VX_VECTORCALL sse42Get3d(__m128 x, __m128 y, __m128 z, __m128 scale) = 0;

        protected:
            alignas(16) __m128i _sse42Seed;
            alignas(16) __m128 _sse42DefaultScale;

            explicit NoiseSSE42(int seed, float defaultScale)
                    : _sse42Seed(_mm_set1_epi32(seed)), _sse42DefaultScale(_mm_set1_ps(defaultScale)) {}

        };
    }
}

#endif //VXNOISE_NOISESSE42_HPP
