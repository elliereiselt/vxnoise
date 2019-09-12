#include "Cellular.hpp"
#include "../simd/simd.hpp"
#include "impl/CellularSSE2.hpp"
#include "impl/CellularNormal.hpp"
#include "impl/CellularSSE42.hpp"
#include "impl/CellularAVX2.hpp"

vx::noise::Cellular *vx::noise::Cellular::create(int seed, float defaultScale, vx::noise::Noise *noiseLookup) {
    return create(vx::simd::simd_level::Auto, seed, defaultScale, noiseLookup);
}

vx::noise::Cellular *vx::noise::Cellular::create(vx::simd::simd_level simdLevel,
                                                 int seed, float defaultScale, vx::noise::Noise *noiseLookup) {
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
            return new vx::noise::CellularAVX2(seed, defaultScale, noiseLookup);
        case vx::simd::simd_level::SSE42:
            if (checkSupport && !vx::simd::checkLevelIsSupported(vx::simd::simd_level::SSE42)) throw std::runtime_error("Computer does not support SIMD level SSE42!");
            return new vx::noise::CellularSSE42(seed, defaultScale, noiseLookup);
        case vx::simd::simd_level::SSE2:
            if (checkSupport && !vx::simd::checkLevelIsSupported(vx::simd::simd_level::SSE2)) throw std::runtime_error("Computer does not support SIMD level SSE2!");
            return new vx::noise::CellularSSE2(seed, defaultScale, noiseLookup);
        default: // Fallback
            return new vx::noise::CellularNormal(seed, defaultScale, noiseLookup);
    }
}
