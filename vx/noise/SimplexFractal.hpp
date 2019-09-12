#ifndef VXNOISE_SIMPLEXFRACTAL_HPP
#define VXNOISE_SIMPLEXFRACTAL_HPP

#include "../simd/simd_level.hpp"
#include "NoiseFractal.hpp"

namespace vx {
    namespace noise {
        class SimplexFractal : public NoiseFractal {
        public:
            static SimplexFractal *create(int seed, float defaultScale, std::uint32_t octaves, float persistence,
                                          float lacunarity);
            static SimplexFractal *create(vx::simd::simd_level simdLevel, int seed, float defaultScale,
                                          std::uint32_t octaves, float persistence, float lacunarity);

        protected:
            SimplexFractal(int seed, float defaultScale, std::uint32_t octaves, float persistence, float lacunarity)
                    : NoiseFractal(seed, defaultScale, octaves, persistence, lacunarity) {}

        };
    }
}

#endif //VXNOISE_SIMPLEXFRACTAL_HPP
