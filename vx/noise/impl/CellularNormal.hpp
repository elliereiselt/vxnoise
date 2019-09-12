#ifndef VXNOISE_CELLULARNORMAL_HPP
#define VXNOISE_CELLULARNORMAL_HPP

#include "../Cellular.hpp"

namespace vx {
    namespace noise {
        class CellularNormal : public Cellular {
            friend Cellular;

        public:
            float get2d(float x, float y, float scale) override;
            float get3d(float x, float y, float z, float scale) override;

            aligned_array_2d<float> noise(float offsetX, float offsetY,
                                          std::size_t sizeX, std::size_t sizeY, float step, float scale) override;
            aligned_array_3d<float> noise(float offsetX, float offsetY, float offsetZ,
                                          std::size_t sizeX, std::size_t sizeY, std::size_t sizeZ, float step,
                                          float scale) override;

        protected:
            CellularNormal(int seed, float defaultScale, Noise *noiseLookup)
                    : Cellular(seed, defaultScale, noiseLookup) {}

        };
    }
}

#endif //VXNOISE_CELLULARNORMAL_HPP
