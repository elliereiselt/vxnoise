#include "Simplex.hpp"
#include "impl/SimplexSSE2.hpp"
#include "impl/SimplexNormal.hpp"
#include "impl/SimplexAVX2.hpp"
#include "impl/SimplexSSE42.hpp"

vx::noise::Simplex *vx::noise::Simplex::create(int seed, float defaultScale) {
    return create(vx::simd::simd_level::Auto, seed, defaultScale);
}

vx::noise::Simplex *vx::noise::Simplex::create(vx::simd::simd_level simdLevel, int seed, float defaultScale) {
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
            return new vx::noise::SimplexAVX2(seed, defaultScale);
        case vx::simd::simd_level::SSE42:
            if (checkSupport && !vx::simd::checkLevelIsSupported(vx::simd::simd_level::SSE42)) throw std::runtime_error("Computer does not support SIMD level SSE42!");
            return new vx::noise::SimplexSSE42(seed, defaultScale);
        case vx::simd::simd_level::SSE2:
            if (checkSupport && !vx::simd::checkLevelIsSupported(vx::simd::simd_level::SSE2)) throw std::runtime_error("Computer does not support SIMD level SSE2!");
            return new vx::noise::SimplexSSE2(seed, defaultScale);
        default: // Fallback
            return new vx::noise::SimplexNormal(seed, defaultScale);
    }
}
