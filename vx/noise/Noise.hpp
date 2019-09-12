#ifndef VXNOISE_NOISE_HPP
#define VXNOISE_NOISE_HPP

#include "../aligned_array_2d.hpp"
#include "../aligned_array_3d.hpp"

namespace vx {
    namespace noise {
        class Noise {
        public:
            float get2d(float x, float y) {
                return get2d(x, y, _defaultScale);
            }

            float get3d(float x, float y, float z) {
                return get3d(x, y, z, _defaultScale);
            }

            aligned_array_2d<float> noise(float offsetX, float offsetY,
                                          std::size_t sizeX, std::size_t sizeY, float step) {
                return noise(offsetX, offsetY, sizeX, sizeY, step, _defaultScale);
            }

            aligned_array_3d<float> noise(float offsetX, float offsetY, float offsetZ,
                                          std::size_t sizeX, std::size_t sizeY, std::size_t sizeZ, float step) {
                return noise(offsetX, offsetY, offsetZ, sizeX, sizeY, sizeZ, step, _defaultScale);
            }

            virtual float get2d(float x, float y, float scale) = 0;
            virtual float get3d(float x, float y, float z, float scale) = 0;

            virtual aligned_array_2d<float> noise(float offsetX, float offsetY,
                                                  std::size_t sizeX, std::size_t sizeY, float step, float scale) = 0;
            virtual aligned_array_3d<float> noise(float offsetX, float offsetY, float offsetZ,
                                                  std::size_t sizeX, std::size_t sizeY, std::size_t sizeZ, float step,
                                                  float scale) = 0;

        protected:
            int _seed;
            float _defaultScale;

            explicit Noise(int seed, float defaultScale)
                    : _seed(seed), _defaultScale(defaultScale) {}

        };
    }
}

#endif //VXNOISE_NOISE_HPP
