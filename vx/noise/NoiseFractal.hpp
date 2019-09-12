#ifndef VXNOISE_NOISEFRACTAL_HPP
#define VXNOISE_NOISEFRACTAL_HPP

#include "Noise.hpp"

namespace vx {
    namespace noise {
        class NoiseFractal : public Noise {
        public:

        protected:
            std::uint32_t _octaves;
            float _persistence;
            float _lacunarity;

            NoiseFractal(int seed, float defaultScale, std::uint32_t octaves, float persistence, float lacunarity)
                    : Noise(seed, defaultScale), _octaves(octaves), _persistence(persistence), _lacunarity(lacunarity) {}

        };
    }
}

#endif //VXNOISE_NOISEFRACTAL_HPP
