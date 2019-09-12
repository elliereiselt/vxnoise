#ifndef VXNOISE_SIMPLEXFRACTALNORMAL_HPP
#define VXNOISE_SIMPLEXFRACTALNORMAL_HPP

#include "../SimplexFractal.hpp"

namespace vx {
    namespace noise {
        class SimplexFractalNormal : public SimplexFractal {
            friend SimplexFractal;

        public:
            float get2d(float x, float y, float scale) override;
            float get3d(float x, float y, float z, float scale) override;

            aligned_array_2d<float> noise(float offsetX, float offsetY,
                                          std::size_t sizeX, std::size_t sizeY, float step, float scale) override;
            aligned_array_3d<float> noise(float offsetX, float offsetY, float offsetZ,
                                          std::size_t sizeX, std::size_t sizeY, std::size_t sizeZ,
                                          float step, float scale) override;

        protected:
            SimplexFractalNormal(int seed, float defaultScale, std::uint32_t octaves, float persistence,
                                 float lacunarity)
                    : SimplexFractal(seed, defaultScale, octaves, persistence, lacunarity) {}

        };
    }
}

#endif //VXNOISE_SIMPLEXFRACTALNORMAL_HPP
