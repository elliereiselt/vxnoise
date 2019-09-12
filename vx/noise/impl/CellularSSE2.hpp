#ifndef VXNOISE_CELLULARSSE2_HPP
#define VXNOISE_CELLULARSSE2_HPP

#include "../Cellular.hpp"
#include "../simd/NoiseSSE2.hpp"

namespace vx {
    namespace noise {
        class CellularSSE2 : public Cellular, public NoiseSSE2 {
            friend Cellular;

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
            NoiseSSE2 *_noiseLookupSSE2;

            explicit CellularSSE2(int seed, float defaultScale, Noise *noiseLookup)
                    : Cellular(seed, defaultScale, noiseLookup), NoiseSSE2(seed, defaultScale),
                      _noiseLookupSSE2(dynamic_cast<NoiseSSE2*>(noiseLookup)) {}

        };
    }
}

#endif //VXNOISE_CELLULARSSE2_HPP
