#ifndef VXNOISE_SIMPLEXFRACTALSSE2_HPP
#define VXNOISE_SIMPLEXFRACTALSSE2_HPP

#include "../SimplexFractal.hpp"
#include "../simd/NoiseSSE2.hpp"

namespace vx {
    namespace noise {
        class SimplexFractalSSE2 : public SimplexFractal, public NoiseSSE2 {
            friend SimplexFractal;

        public:
            __m128 VX_VECTORCALL sse2Get2d(__m128 x, __m128 y, __m128 scale) override;
            __m128 VX_VECTORCALL sse2Get3d(__m128 x, __m128 y, __m128 z, __m128 scale) override;

            float get2d(float x, float y, float scale) override;
            float get3d(float x, float y, float z, float scale) override;

            aligned_array_2d<float> noise(float offsetX, float offsetY,
                                          std::size_t sizeX, std::size_t sizeY, float step, float scale) override;
            aligned_array_3d<float> noise(float offsetX, float offsetY, float offsetZ,
                                          std::size_t sizeX, std::size_t sizeY, std::size_t sizeZ,
                                          float step, float scale) override;

            void* operator new(size_t s) {
                return _mm_malloc(s, 16);
            }

            void operator delete(void* ptr) {
                _mm_free(ptr);
            }

        protected:
            __m128 _sse2Persistence;
            __m128 _sse2Lacunarity;

            SimplexFractalSSE2(int seed, float defaultScale, std::uint32_t octaves, float persistence,
                               float lacunarity)
                    : SimplexFractal(seed, defaultScale, octaves, persistence, lacunarity),
                      NoiseSSE2(seed, defaultScale), _sse2Persistence(_mm_set1_ps(persistence)),
                      _sse2Lacunarity(_mm_set1_ps(lacunarity)) {}

        };
    }
}

#endif //VXNOISE_SIMPLEXFRACTALSSE2_HPP
