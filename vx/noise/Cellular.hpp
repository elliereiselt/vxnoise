#ifndef VXNOISE_CELLULAR_HPP
#define VXNOISE_CELLULAR_HPP

#include "../simd/simd_level.hpp"
#include "Noise.hpp"

namespace vx {
    namespace noise {
        class Cellular : public Noise {
        public:
            static Cellular *create(int seed, float defaultScale, Noise *noiseLookup);
            static Cellular *create(vx::simd::simd_level simdLevel, int seed, float defaultScale, Noise *noiseLookup);

        protected:
            Noise *_noiseLookup;

            Cellular(int seed, float defaultScale, Noise *noiseLookup)
                    : Noise(seed, defaultScale), _noiseLookup(noiseLookup) {}

        };
    }
}

#endif //VXNOISE_CELLULAR_HPP
