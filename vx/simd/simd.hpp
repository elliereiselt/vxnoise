#ifndef VXNOISE_SIMD_HPP
#define VXNOISE_SIMD_HPP

#ifdef _MSC_VER
#define VX_VECTORCALL __vectorcall
#else
#define VX_VECTORCALL
#endif

#include "simd_level.hpp"

namespace vx {
    namespace simd {
        simd_level getFastestSupportedLevel();
        bool checkLevelIsSupported(simd_level checkLevel);
    }
}

#endif //VXNOISE_SIMD_HPP
