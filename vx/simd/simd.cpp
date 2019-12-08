#include <cstdint>
#include <cpuid.h>
#include "simd.hpp"

// Based on https://gist.github.com/hi2p-perim/7855506
#ifdef _MSC_VER
#include <intrin.h>

void get_cpuid(int32_t out[4], int32_t x) {
    __cpuid(out, x);
}
#else // TODO: Does this work on MAC and iOS?
void get_cpuid(int32_t out[4], int32_t x) {
    __cpuid_count(x, 0, out[0], out[1], out[2], out[3]);
}

unsigned long long _xgetbv(unsigned int index) {
	unsigned int eax, edx;
	__asm__ __volatile__(
		"xgetbv;"
		: "=a" (eax), "=d"(edx)
		: "c" (index)
	);
	return ((unsigned long long)edx << 32) | eax;
}
#endif

vx::simd::simd_level vx::simd::getFastestSupportedLevel() {
    int cpuInfo[4];
    get_cpuid(cpuInfo, 0);
    int funcNums = cpuInfo[0];

    if (funcNums < 1) {
        return vx::simd::simd_level::Fallback;
    }

    get_cpuid(cpuInfo, 1);

    // If SSE2 isn't supported then use Fallback...
    if ((cpuInfo[3] & (1 << 25)) == 0) {
        return vx::simd::simd_level::Fallback;
    }

    // If SSE42 isn't supported then use SSE2...
    if ((cpuInfo[2] & (1 << 20)) == 0) {
        return vx::simd::simd_level::SSE2;
    }

    // If AVX isn't supported by the CPU (1 << 28) or the OS doesn't support it (1 << 27) then return SSE42...
    if ((cpuInfo[2] & (1 << 28)) != 0 && (cpuInfo[2] & (1 << 27)) != 0) {
        // _XCR_XFEATURE_ENABLED_MASK = 0
        unsigned long long xcrFeatureMask = _xgetbv(0);

        if ((xcrFeatureMask & 0x6) == 0) {
            return vx::simd::simd_level::SSE42;
        }
    } else {
        return vx::simd::simd_level::SSE42;
    }

    if (funcNums < 7) {
        return vx::simd::simd_level::SSE42;
    }

    // FMA3
    bool fma3Support = (cpuInfo[2] & 1 << 12) == 0;

    get_cpuid(cpuInfo, 7);

    // Check AVX2 support
    if ((cpuInfo[1] & 1 << 5) != 0) {
        if (!fma3Support) {
            return vx::simd::simd_level::AVX2;
        }
    } else {
        return vx::simd::simd_level::SSE42;
    }

    // TODO: Support FMA
    // If we reach this point then AVX2 is supported.
    return vx::simd::simd_level::AVX2;
}

bool vx::simd::checkLevelIsSupported(vx::simd::simd_level checkLevel) {
    int cpuInfo[4];
    get_cpuid(cpuInfo, 0);
    int funcNums = cpuInfo[0];

    if (funcNums < 1) {
        return checkLevel == vx::simd::simd_level::Fallback;
    }

    get_cpuid(cpuInfo, 1);

    // If SSE2 isn't supported then use Fallback...
    if ((cpuInfo[3] & (1 << 25)) != 0) {
        if (checkLevel == vx::simd::simd_level::SSE2) {
            return true;
        }
    }

    // If SSE42 isn't supported then use SSE2...
    if ((cpuInfo[2] & (1 << 20)) != 0) {
        if (checkLevel == vx::simd::simd_level::SSE42) {
            return true;
        }
    }

    // If AVX2 isn't supported by the CPU (1 << 28) or the OS doesn't support it (1 << 27) then return SSE42...
    if ((cpuInfo[2] & (1 << 28)) != 0 && (cpuInfo[2] & (1 << 27)) != 0) {
        // _XCR_XFEATURE_ENABLED_MASK = 0
        unsigned long long xcrFeatureMask = _xgetbv(0);

        if ((xcrFeatureMask & 0x6) == 0) {
            if (checkLevel == vx::simd::simd_level::SSE42) {
                return true;
            }
        }
    } else {
        if (checkLevel == vx::simd::simd_level::SSE42) {
            return true;
        }
    }

    if (funcNums < 7) {
        if (checkLevel == vx::simd::simd_level::SSE42) {
            return true;
        }
    }

    // FMA3
    bool fma3Support = (cpuInfo[2] & 1 << 12) == 0;

    get_cpuid(cpuInfo, 7);

    // Check AVX2 support
    if ((cpuInfo[1] & 1 << 5) != 0) {
        if (!fma3Support) {
            if (checkLevel == vx::simd::simd_level::AVX2) {
                return true;
            }
        }
    } else {
        if (checkLevel == vx::simd::simd_level::SSE42) {
            return true;
        }
    }

    // TODO: Support FMA
    // If we reach this point then check if checkLevel is AVX2, this probably isn't right.
    return checkLevel == vx::simd::simd_level::AVX2;
}
