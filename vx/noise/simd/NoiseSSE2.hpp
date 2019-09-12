#ifndef VXNOISE_NOISESSE2_HPP
#define VXNOISE_NOISESSE2_HPP

#include "../../simd/sse2.hpp"

namespace vx {
    namespace noise {
        class NoiseSSE2 {
        public:
            __m128 VX_VECTORCALL sse2Get2d(__m128 x, __m128 y) {
                return sse2Get2d(x, y, _sse2DefaultScale);
            }

            __m128 VX_VECTORCALL sse2Get3d(__m128 x, __m128 y, __m128 z) {
                return sse2Get3d(x, y, z, _sse2DefaultScale);
            }

            virtual __m128 VX_VECTORCALL sse2Get2d(__m128 x, __m128 y, __m128 scale) = 0;
            virtual __m128 VX_VECTORCALL sse2Get3d(__m128 x, __m128 y, __m128 z, __m128 scale) = 0;

        protected:
            alignas(16) __m128i _sse2Seed;
            alignas(16) __m128 _sse2DefaultScale;

            explicit NoiseSSE2(int seed, float defaultScale)
                    : _sse2Seed(_mm_set1_epi32(seed)), _sse2DefaultScale(_mm_set1_ps(defaultScale)) {}

        };
    }
}

#endif //VXNOISE_NOISESSE2_HPP
