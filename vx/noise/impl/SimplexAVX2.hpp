#ifndef VXNOISE_SIMPLEXAVX2_HPP
#define VXNOISE_SIMPLEXAVX2_HPP

#include "../simd/NoiseAVX2.hpp"
#include "../Simplex.hpp"

namespace vx {
    namespace noise {
        __m256 VX_VECTORCALL avx2SimplexGet2d(__m256i seed, __m256 x, __m256 y);
        __m256 VX_VECTORCALL avx2SimplexGet3d(__m256i seed, __m256 x, __m256 y, __m256 z);

        class SimplexAVX2 : public Simplex, public NoiseAVX2 {
            friend Simplex;

        public:
            __m256 VX_VECTORCALL avx2Get2d(__m256 x, __m256 y, __m256 scale) override;
            __m256 VX_VECTORCALL avx2Get3d(__m256 x, __m256 y, __m256 z, __m256 scale) override;

            float get2d(float x, float y, float scale) override;
            float get3d(float x, float y, float z, float scale) override;

            aligned_array_2d<float> noise(float offsetX, float offsetY, std::size_t sizeX, std::size_t sizeY,
                                          float step, float scale) override;
            aligned_array_3d<float> noise(float offsetX, float offsetY, float offsetZ,
                                          std::size_t sizeX, std::size_t sizeY, std::size_t sizeZ,
                                          float step, float scale) override;

            void* operator new(size_t s) {
                return _mm_malloc(s, 32);
            }

            void operator delete(void* ptr) {
                _mm_free(ptr);
            }

        protected:
            explicit SimplexAVX2(int seed, float defaultScale)
                    : Simplex(seed, defaultScale), NoiseAVX2(seed, defaultScale) {}

        };
    }
}

#endif //VXNOISE_SIMPLEXAVX2_HPP
