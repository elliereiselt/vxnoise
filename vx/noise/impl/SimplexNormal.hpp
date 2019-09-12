#ifndef VXNOISE_SIMPLEXNORMAL_HPP
#define VXNOISE_SIMPLEXNORMAL_HPP

#include "../Simplex.hpp"

namespace vx {
    namespace noise {
        float normalSimplexGet2d(int seed, float x, float y);
        float normalSimplexGet3d(int seed, float x, float y, float z);

        class SimplexNormal : public Simplex {
            friend Simplex;

        public:
            float get2d(float x, float y, float scale) override;
            float get3d(float x, float y, float z, float scale) override;

            aligned_array_2d<float> noise(float offsetX, float offsetY, std::size_t sizeX, std::size_t sizeY,
                                          float step, float scale) override;
            aligned_array_3d<float> noise(float offsetX, float offsetY, float offsetZ,
                                          std::size_t sizeX, std::size_t sizeY, std::size_t sizeZ,
                                          float step, float scale) override;

        protected:
            explicit SimplexNormal(int seed, float defaultScale)
                    : Simplex(seed, defaultScale) {}

        };
    }
}

#endif //VXNOISE_SIMPLEXNORMAL_HPP
