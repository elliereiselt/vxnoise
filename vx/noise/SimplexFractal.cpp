#include "SimplexFractal.hpp"
#include "../simd/simd.hpp"
#include "impl/SimplexFractalNormal.hpp"
#include "impl/SimplexFractalSSE2.hpp"
#include "impl/SimplexFractalSSE42.hpp"
#include "impl/SimplexFractalAVX2.hpp"

vx::noise::SimplexFractal *vx::noise::SimplexFractal::create(int seed, float defaultScale, std::uint32_t octaves,
                                                             float persistence, float lacunarity) {
    return create(vx::simd::simd_level::Auto, seed, defaultScale, octaves, persistence, lacunarity);
}

vx::noise::SimplexFractal *vx::noise::SimplexFractal::create(vx::simd::simd_level simdLevel, int seed,
                                                             float defaultScale, std::uint32_t octaves,
                                                             float persistence, float lacunarity) {
    bool checkSupport = true;

    if (simdLevel == vx::simd::simd_level::Auto) {
        checkSupport = false;
        simdLevel = vx::simd::getFastestSupportedLevel();
    }

    switch (simdLevel) {
        case vx::simd::simd_level::NEON:
            throw std::runtime_error("SIMD level 'NEON' not yet supported!");
        case vx::simd::simd_level::AVX2:
            if (checkSupport && !vx::simd::checkLevelIsSupported(vx::simd::simd_level::AVX2)) throw std::runtime_error("Computer does not support SIMD level AVX2!");
            return new vx::noise::SimplexFractalAVX2(seed, defaultScale, octaves, persistence, lacunarity);
        case vx::simd::simd_level::SSE42:
            if (checkSupport && !vx::simd::checkLevelIsSupported(vx::simd::simd_level::SSE42)) throw std::runtime_error("Computer does not support SIMD level SSE42!");
            return new vx::noise::SimplexFractalSSE42(seed, defaultScale, octaves, persistence, lacunarity);
        case vx::simd::simd_level::SSE2:
            if (checkSupport && !vx::simd::checkLevelIsSupported(vx::simd::simd_level::SSE2)) throw std::runtime_error("Computer does not support SIMD level SSE2!");
            return new vx::noise::SimplexFractalSSE2(seed, defaultScale, octaves, persistence, lacunarity);
        default: // Fallback
            return new vx::noise::SimplexFractalNormal(seed, defaultScale, octaves, persistence, lacunarity);
    }
}
