#ifndef VXNOISE_CELLULARSSE42_HPP
#define VXNOISE_CELLULARSSE42_HPP

#include "../Cellular.hpp"
#include "../simd/NoiseSSE42.hpp"

namespace vx {
    namespace noise {
        class CellularSSE42 : public Cellular, public NoiseSSE42 {
            friend Cellular;

        public:
            __m128 VX_VECTORCALL sse42Get2d(__m128 x, __m128 y, __m128 scale) override;
            __m128 VX_VECTORCALL sse42Get3d(__m128 x, __m128 y, __m128 z, __m128 scale) override;

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
            NoiseSSE42 *_noiseLookupSSE42;

            explicit CellularSSE42(int seed, float defaultScale, Noise *noiseLookup)
                    : Cellular(seed, defaultScale, noiseLookup), NoiseSSE42(seed, defaultScale),
                      _noiseLookupSSE42(dynamic_cast<NoiseSSE42*>(noiseLookup)) {}

        };
    }
}

#endif //VXNOISE_CELLULARSSE42_HPP
