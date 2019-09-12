#ifndef VXNOISE_SIMPLEX_HPP
#define VXNOISE_SIMPLEX_HPP

#include "../simd/simd_level.hpp"
#include "Noise.hpp"

#ifdef VX_NOISE_INTERNAL_DEFS
#define F2 0.366025403f // F2 = 0.5*(sqrt(3.0)-1.0)
#define G2 0.211324865f // G2 = (3.0-Math.sqrt(3.0))/6.0
// Simple skewing factors for the 3D case
#define F3 0.333333333f
#define G3 0.166666667f
#endif

namespace vx {
    namespace noise {
        class Simplex : public Noise {
        public:
            static Simplex *create(int seed, float defaultScale);
            static Simplex *create(vx::simd::simd_level simdLevel, int seed, float defaultScale);

        protected:
            explicit Simplex(int seed, float defaultScale)
                    : Noise(seed, defaultScale) {}

        };
    }
}

#endif //VXNOISE_SIMPLEX_HPP
