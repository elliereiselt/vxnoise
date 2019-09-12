#ifndef VXNOISE_SIMD_LEVEL_HPP
#define VXNOISE_SIMD_LEVEL_HPP

namespace vx {
    namespace simd {
        enum class simd_level {
            Auto = 0,
            Fallback,
            // TODO: Support ARM NEON
            NEON,
            SSE2,
            SSE42,
            AVX2
        };
    }
}

#endif //VXNOISE_SIMD_LEVEL_HPP
