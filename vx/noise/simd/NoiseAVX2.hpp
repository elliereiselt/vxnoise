#ifndef VXNOISE_NOISEAVX2_HPP
#define VXNOISE_NOISEAVX2_HPP

#include "../../simd/avx2.hpp"

namespace vx {
    namespace noise {
        class NoiseAVX2 {
        public:
            __m256 VX_VECTORCALL avx2Get2d(__m256 x, __m256 y) {
                return avx2Get2d(x, y, _avx2DefaultScale);
            }

            __m256 VX_VECTORCALL avx2Get3d(__m256 x, __m256 y, __m256 z) {
                return avx2Get3d(x, y, z, _avx2DefaultScale);
            }

            virtual __m256 VX_VECTORCALL avx2Get2d(__m256 x, __m256 y, __m256 scale) = 0;
            virtual __m256 VX_VECTORCALL avx2Get3d(__m256 x, __m256 y, __m256 z, __m256 scale) = 0;

        protected:
            alignas(32) __m256i _avx2Seed;
            alignas(32) __m256 _avx2DefaultScale;

            NoiseAVX2(int seed, float defaultScale);

        };
    }
}

#endif //VXNOISE_NOISEAVX2_HPP
