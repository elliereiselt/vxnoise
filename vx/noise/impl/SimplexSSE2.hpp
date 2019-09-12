#ifndef VXNOISE_SIMPLEXSSE2_HPP
#define VXNOISE_SIMPLEXSSE2_HPP

#include "../simd/NoiseSSE2.hpp"
#include "../Simplex.hpp"

namespace vx {
    namespace noise {
        __m128 VX_VECTORCALL sse2SimplexGet2d(__m128i seed, __m128 x, __m128 y);
        __m128 VX_VECTORCALL sse2SimplexGet3d(__m128i seed, __m128 x, __m128 y, __m128 z);

        class SimplexSSE2 : public Simplex, public NoiseSSE2 {
            friend Simplex;

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
            explicit SimplexSSE2(int seed, float defaultScale)
                    : Simplex(seed, defaultScale), NoiseSSE2(seed, defaultScale) {}

        };
    }
}

#endif //VXNOISE_SIMPLEXSSE2_HPP
