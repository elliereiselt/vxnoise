#ifndef __AVX__
#ifdef _MSC_VER
#error Missing argument "/arch:AVX(2)"
#else
#error Missing argument "-march=core-avx2"
#endif
#endif

// AVX2
#include "../../simd/avx2.hpp"
#include "CellularAVX2.hpp"
#include <limits>

static __m256 VX_VECTORCALL hashCellX(__m256i seed, __m256 cellX, __m256 cellY) {
    __m256 rx = vx::simd::avx2::dot_ps(cellX, cellY, _mm256_set1_ps(127.1f), _mm256_set1_ps(311.7f));
    __m256i hashX = vx::simd::avx2::partialJenkinsHash(seed, _mm256_cvttps_epi32(rx));
    __m256 pointX = _mm256_div_ps(_mm256_cvtepi32_ps(hashX), _mm256_set1_ps(1000000.0f));
    __m256 fractX = _mm256_sub_ps(pointX, _mm256_floor_ps(pointX));

    return _mm256_add_ps(cellX, fractX);
}

static __m256 VX_VECTORCALL hashCellY(__m256i seed, __m256 cellX, __m256 cellY) {
    __m256 ry = vx::simd::avx2::dot_ps(cellX, cellY, _mm256_set1_ps(269.5f), _mm256_set1_ps(183.3f));
    __m256i hashY = vx::simd::avx2::partialJenkinsHash(seed, _mm256_cvttps_epi32(ry));
    __m256 pointY = _mm256_div_ps(_mm256_cvtepi32_ps(hashY), _mm256_set1_ps(1000000.0f));
    __m256 fractY = _mm256_sub_ps(pointY, _mm256_floor_ps(pointY));

    return _mm256_add_ps(cellY, fractY);
}

static __m256 hashCellX(__m256i seed, __m256 cellX, __m256 cellY, __m256 cellZ) {
    // Gonna be honest with ya here. Don't remember where the original x and y constants came from, the z constants were made up from keyboard smashing. Probably not good.
    __m256 rx = vx::simd::avx2::dot_ps(cellX, cellY, cellZ,
                                        _mm256_set1_ps(127.1f), _mm256_set1_ps(311.7f), _mm256_set1_ps(231.4f));
    __m256i hashX = vx::simd::avx2::partialJenkinsHash(seed, _mm256_cvttps_epi32(rx));

    __m256 pointX = _mm256_div_ps(_mm256_cvtepi32_ps(hashX), _mm256_set1_ps(1000000.0f));
    __m256 fractX = _mm256_sub_ps(pointX, _mm256_floor_ps(pointX));

    return _mm256_add_ps(cellX, fractX);
}

static __m256 hashCellY(__m256i seed, __m256 cellX, __m256 cellY, __m256 cellZ) {
    // Gonna be honest with ya here. Don't remember where the original x and y constants came from, the z constants were made up from keyboard smashing. Probably not good.
    __m256 ry = vx::simd::avx2::dot_ps(cellX, cellY, cellZ,
                                        _mm256_set1_ps(269.5f), _mm256_set1_ps(183.3f), _mm256_set1_ps(352.6f));
    __m256i hashY = vx::simd::avx2::partialJenkinsHash(seed, _mm256_cvttps_epi32(ry));
    __m256 pointY = _mm256_div_ps(_mm256_cvtepi32_ps(hashY), _mm256_set1_ps(1000000.0f));
    __m256 fractY = _mm256_sub_ps(pointY, _mm256_floor_ps(pointY));

    return _mm256_add_ps(cellY, fractY);
}

static __m256 hashCellZ(__m256i seed, __m256 cellX, __m256 cellY, __m256 cellZ) {
    // Gonna be honest with ya here. Don't remember where the original x and y constants came from, the z constants were made up from keyboard smashing. Probably not good.
    __m256 rz = vx::simd::avx2::dot_ps(cellX, cellY, cellZ,
                                        _mm256_set1_ps(419.2f), _mm256_set1_ps(371.9f), _mm256_set1_ps(523.7f));
    __m256i hashZ = vx::simd::avx2::partialJenkinsHash(seed, _mm256_cvttps_epi32(rz));
    __m256 pointZ = _mm256_div_ps(_mm256_cvtepi32_ps(hashZ), _mm256_set1_ps(1000000.0f));
    __m256 fractZ = _mm256_sub_ps(pointZ, _mm256_floor_ps(pointZ));

    return _mm256_add_ps(cellZ, fractZ);
}

__m256 vx::noise::CellularAVX2::avx2Get2d(__m256 x, __m256 y, __m256 scale) {
    x = _mm256_div_ps(x, scale);
    y = _mm256_div_ps(y, scale);

    __m256 fx = _mm256_div_ps(x, _mm256_set1_ps(16));
    __m256 fy = _mm256_div_ps(y, _mm256_set1_ps(16));
    __m256 ix = _mm256_floor_ps(fx);
    __m256 iy = _mm256_floor_ps(fy);

    __m256 lowestDistance = _mm256_set1_ps((std::numeric_limits<float>::max)());
    __m256 resultX = _mm256_setzero_ps();
    __m256 resultY = _mm256_setzero_ps();

    __m256 i = _mm256_set1_ps(-1);

    for (int ignoreI = -1; ignoreI < 2; ++ignoreI) {
        __m256 j = _mm256_set1_ps(-1);

        for (int ignoreJ = -1; ignoreJ < 2; ++ignoreJ) {
            __m256 cellX = hashCellX(_avx2Seed, _mm256_add_ps(ix, i), _mm256_add_ps(iy, j));
            __m256 cellY = hashCellY(_avx2Seed, _mm256_add_ps(ix, i), _mm256_add_ps(iy, j));

            __m256 euclideanDistance =
                    _mm256_sqrt_ps(
                            _mm256_add_ps(
                                    // (cellY - fy)^2
                                    _mm256_mul_ps(_mm256_sub_ps(cellY, fy), _mm256_sub_ps(cellY, fy)),
                                    // (cellZ - fz)^2
                                    _mm256_mul_ps(_mm256_sub_ps(cellX, fx), _mm256_sub_ps(cellX, fx))));
            __m256 manhattanDistance = _mm256_add_ps(vx::simd::avx2::abs_ps(_mm256_sub_ps(cellY, fy)), vx::simd::avx2::abs_ps(_mm256_sub_ps(cellX, fx)));
            __m256 naturalDistance = _mm256_add_ps(euclideanDistance, manhattanDistance);

            __m256 distLTlowestDist = _mm256_cmp_ps(naturalDistance, lowestDistance, _CMP_LT_OQ);

            lowestDistance = _mm256_blendv_ps(lowestDistance, naturalDistance, distLTlowestDist);
            resultX = _mm256_blendv_ps(resultX, cellX, distLTlowestDist);
            resultY = _mm256_blendv_ps(resultY, cellY, distLTlowestDist);

            j = _mm256_add_ps(j, _mm256_set1_ps(1));
        }

        i = _mm256_add_ps(i, _mm256_set1_ps(1));
    }

    if (_noiseLookupAVX2 != nullptr) {
        return _noiseLookupAVX2->avx2Get2d(resultX, resultY);
    } else {
        alignas(32) float pResultX[8];
        alignas(32) float pResultY[8];

        _mm256_store_ps(pResultX, resultX);
        _mm256_store_ps(pResultY, resultY);

        alignas(32) float pResult[8];

        for (int idx = 0; idx < 8; ++idx) {
            pResult[idx] = _noiseLookup->get2d(pResultX[idx], pResultY[idx]);
        }

        return _mm256_load_ps(pResult);
    }
}

float vx::noise::CellularAVX2::get2d(float x, float y, float scale) {
    __m256 sse2Result = avx2Get2d(_mm256_set1_ps(x), _mm256_set1_ps(y), _mm256_set1_ps(scale));
    alignas(32) float pResult[8];
    _mm256_store_ps(pResult, sse2Result);
    return pResult[0];
}

__m256 vx::noise::CellularAVX2::avx2Get3d(__m256 x, __m256 y, __m256 z, __m256 scale) {
    x = _mm256_div_ps(x, scale);
    y = _mm256_div_ps(y, scale);
    z = _mm256_div_ps(z, scale);

    __m256 fx = _mm256_div_ps(x, _mm256_set1_ps(16));
    __m256 fy = _mm256_div_ps(y, _mm256_set1_ps(16));
    __m256 fz = _mm256_div_ps(z, _mm256_set1_ps(16));
    __m256 ix = _mm256_floor_ps(fx);
    __m256 iy = _mm256_floor_ps(fy);
    __m256 iz = _mm256_floor_ps(fz);

    __m256 lowestDistance = _mm256_set1_ps((std::numeric_limits<float>::max)());
    __m256 resultX = _mm256_setzero_ps();
    __m256 resultY = _mm256_setzero_ps();
    __m256 resultZ = _mm256_setzero_ps();

    __m256 i = _mm256_set1_ps(-1);

    for (int ignoreI = -1; ignoreI < 2; ++ignoreI) {
        __m256 j = _mm256_set1_ps(-1);

        for (int ignoreJ = -1; ignoreJ < 2; ++ignoreJ) {
            __m256 k = _mm256_set1_ps(-1);

            for (int ignoreK = -1; ignoreK < 2; ++ignoreK) {
                __m256 cellX = hashCellX(_avx2Seed, _mm256_add_ps(ix, i), _mm256_add_ps(iy, j), _mm256_add_ps(iz, k));
                __m256 cellY = hashCellY(_avx2Seed, _mm256_add_ps(ix, i), _mm256_add_ps(iy, j), _mm256_add_ps(iz, k));
                __m256 cellZ = hashCellZ(_avx2Seed, _mm256_add_ps(ix, i), _mm256_add_ps(iy, j), _mm256_add_ps(iz, k));

                __m256 euclideanDistance =
                        _mm256_sqrt_ps(
                                _mm256_add_ps(
                                        _mm256_add_ps(
                                                // (cellZ - fz)^2
                                                _mm256_mul_ps(_mm256_sub_ps(cellZ, fz), _mm256_sub_ps(cellZ, fz)),
                                                // (cellY - fy)^2
                                                _mm256_mul_ps(_mm256_sub_ps(cellY, fy), _mm256_sub_ps(cellY, fy))),
                                        // (cellX - fx)^2
                                        _mm256_mul_ps(_mm256_sub_ps(cellX, fx), _mm256_sub_ps(cellX, fx))));
                __m256 manhattanDistance =
                        _mm256_add_ps(
                                _mm256_add_ps(
                                        vx::simd::avx2::abs_ps(_mm256_sub_ps(cellZ, fz)),
                                        vx::simd::avx2::abs_ps(_mm256_sub_ps(cellY, fy))),
                                vx::simd::avx2::abs_ps(_mm256_sub_ps(cellX, fx)));
                __m256 naturalDistance = _mm256_add_ps(euclideanDistance, manhattanDistance);

                __m256 distLTlowestDist = _mm256_cmp_ps(naturalDistance, lowestDistance, _CMP_LT_OQ);

                lowestDistance = _mm256_blendv_ps(lowestDistance, naturalDistance, distLTlowestDist);
                resultX = _mm256_blendv_ps(resultX, cellX, distLTlowestDist);
                resultY = _mm256_blendv_ps(resultY, cellY, distLTlowestDist);
                resultZ = _mm256_blendv_ps(resultZ, cellZ, distLTlowestDist);

                k = _mm256_add_ps(k, _mm256_set1_ps(1));
            }

            j = _mm256_add_ps(j, _mm256_set1_ps(1));
        }

        i = _mm256_add_ps(i, _mm256_set1_ps(1));
    }

    if (_noiseLookupAVX2 != nullptr) {
        return _noiseLookupAVX2->avx2Get3d(resultX, resultY, resultZ);
    } else {
        alignas(32) float pResultX[8];
        alignas(32) float pResultY[8];
        alignas(32) float pResultZ[8];

        _mm256_store_ps(pResultX, resultX);
        _mm256_store_ps(pResultY, resultY);
        _mm256_store_ps(pResultZ, resultZ);

        alignas(32) float pResult[8];

        for (int idx = 0; idx < 8; ++idx) {
            pResult[idx] = _noiseLookup->get3d(pResultX[idx], pResultY[idx], pResultZ[idx]);
        }

        return _mm256_load_ps(pResult);
    }
}

float vx::noise::CellularAVX2::get3d(float x, float y, float z, float scale) {
    __m256 sse2Result = avx2Get3d(_mm256_set1_ps(x), _mm256_set1_ps(y), _mm256_set1_ps(z), _mm256_set1_ps(scale));
    alignas(32) float pResult[8];
    _mm256_store_ps(pResult, sse2Result);
    return pResult[0];
}

vx::aligned_array_2d<float> vx::noise::CellularAVX2::noise(float offsetX, float offsetY,
                                                            std::size_t sizeX, std::size_t sizeY, float step,
                                                            float scale) {
    // I believe we have to call this at the beginning of the function, right? Since none of the arguments should be stored in YMM?
    _mm256_zeroall();

    vx::aligned_array_2d<float> result(sizeX, sizeY, 32);
    float* rawPtr = result.ptr();

    std::size_t maxIndex = sizeX * sizeY;
    __m256 vecOffsetX = _mm256_set1_ps(offsetX);
    __m256 vecOffsetY = _mm256_set1_ps(offsetY);

    float x = 0, y = 0;
    __m256 vecScale = _mm256_set1_ps(scale);
    __m256 vecStep = _mm256_set1_ps(step);

    std::size_t index = 0;

	if (maxIndex >= 8) {
		for (; index < maxIndex - 7; index += 8) {
			float x0 = x, y0 = y;

			float x1 = x0, y1 = y0 + 1;
			if (y1 >= sizeY) y1 = 0, x1 += 1;

			float x2 = x1, y2 = y1 + 1;
			if (y2 >= sizeY) y2 = 0, x2 += 1;

			float x3 = x2, y3 = y2 + 1;
			if (y3 >= sizeY) y3 = 0, x3 += 1;

			float x4 = x3, y4 = y3 + 1;
			if (y4 >= sizeY) y4 = 0, x4 += 1;

			float x5 = x4, y5 = y4 + 1;
			if (y5 >= sizeY) y5 = 0, x5 += 1;

			float x6 = x5, y6 = y5 + 1;
			if (y6 >= sizeY) y6 = 0, x6 += 1;

			float x7 = x6, y7 = y6 + 1;
			if (y7 >= sizeY) y7 = 0, x7 += 1;

			__m256 vecX = _mm256_set_ps(x7, x6, x5, x4, x3, x2, x1, x0);
			__m256 vecY = _mm256_set_ps(y7, y6, y5, y4, y3, y2, y1, y0);

			__m256 vecNoise = avx2Get2d(_mm256_add_ps(vecOffsetX, _mm256_mul_ps(vecX, vecStep)),
                                        _mm256_add_ps(vecOffsetY, _mm256_mul_ps(vecY, vecStep)), vecScale);

			_mm256_store_ps(&rawPtr[index], vecNoise);

			x = x7, y = y7 + 1;
			if (y >= sizeY) y = 0, x += 1;
		}
	}

    if ((sizeX * sizeY) % 8 != 0) {
        float x0 = x, y0 = y;

        float x1 = x0, y1 = y0 + 1;
        if (y1 >= sizeY) y1 = 0, x1 += 1;

        float x2 = x1, y2 = y1 + 1;
        if (y2 >= sizeY) y2 = 0, x2 += 1;

        float x3 = x2, y3 = y2 + 1;
        if (y3 >= sizeY) y3 = 0, x3 += 1;

        float x4 = x3, y4 = y3 + 1;
        if (y4 >= sizeY) y4 = 0, x4 += 1;

        float x5 = x4, y5 = y4 + 1;
        if (y5 >= sizeY) y5 = 0, x5 += 1;

        float x6 = x5, y6 = y5 + 1;
        if (y6 >= sizeY) y6 = 0, x6 += 1;

        float x7 = x6, y7 = y6 + 1;
        if (y7 >= sizeY) y7 = 0, x7 += 1;

        __m256 vecX = _mm256_set_ps(x7, x6, x5, x4, x3, x2, x1, x0);
        __m256 vecY = _mm256_set_ps(y7, y6, y5, y4, y3, y2, y1, y0);

        __m256 vecNoise = avx2Get2d(_mm256_add_ps(vecOffsetX, _mm256_mul_ps(vecX, vecStep)),
                                    _mm256_add_ps(vecOffsetY, _mm256_mul_ps(vecY, vecStep)), vecScale);

        alignas(32) float pResult[8];

        _mm256_store_ps(pResult, vecNoise);

        if ((sizeX * sizeY) % 8 == 1) {
            rawPtr[index] = pResult[0];
        } else if ((sizeX * sizeY) % 8 == 2) {
            rawPtr[index] = pResult[0];
            rawPtr[index + 1] = pResult[1];
        } else if ((sizeX * sizeY) % 8 == 3) {
            rawPtr[index] = pResult[0];
            rawPtr[index + 1] = pResult[1];
            rawPtr[index + 2] = pResult[2];
        } else if ((sizeX * sizeY) % 8 == 4) {
            rawPtr[index] = pResult[0];
            rawPtr[index + 1] = pResult[1];
            rawPtr[index + 2] = pResult[2];
            rawPtr[index + 3] = pResult[3];
        } else if ((sizeX * sizeY) % 8 == 5) {
            rawPtr[index] = pResult[0];
            rawPtr[index + 1] = pResult[1];
            rawPtr[index + 2] = pResult[2];
            rawPtr[index + 3] = pResult[3];
            rawPtr[index + 4] = pResult[4];
        } else if ((sizeX * sizeY) % 8 == 6) {
            rawPtr[index] = pResult[0];
            rawPtr[index + 1] = pResult[1];
            rawPtr[index + 2] = pResult[2];
            rawPtr[index + 3] = pResult[3];
            rawPtr[index + 4] = pResult[4];
            rawPtr[index + 5] = pResult[5];
        } else if ((sizeX * sizeY) % 8 == 7) {
            rawPtr[index] = pResult[0];
            rawPtr[index + 1] = pResult[1];
            rawPtr[index + 2] = pResult[2];
            rawPtr[index + 3] = pResult[3];
            rawPtr[index + 4] = pResult[4];
            rawPtr[index + 5] = pResult[5];
            rawPtr[index + 6] = pResult[6];
        }
    }

    _mm256_zeroall();
    return result;
}

vx::aligned_array_3d<float> vx::noise::CellularAVX2::noise(float offsetX, float offsetY, float offsetZ,
                                                           std::size_t sizeX, std::size_t sizeY, std::size_t sizeZ,
                                                           float step, float scale) {
    // I believe we have to call this at the beginning of the function, right? Since none of the arguments should be stored in YMM?
    _mm256_zeroall();

    vx::aligned_array_3d<float> result(sizeX, sizeY, sizeZ, 32);
    float* rawPtr = result.ptr();

    std::size_t maxIndex = sizeX * sizeY * sizeZ;
    __m256 vecOffsetX = _mm256_set1_ps(offsetX);
    __m256 vecOffsetY = _mm256_set1_ps(offsetY);
    __m256 vecOffsetZ = _mm256_set1_ps(offsetZ);

    float x = 0, y = 0, z = 0;
    __m256 vecScale = _mm256_set1_ps(scale);
    __m256 vecStep = _mm256_set1_ps(step);

    std::size_t index = 0;

	if (maxIndex >= 8) {
		for (; index < maxIndex - 7; index += 8) {
			float x0 = x, y0 = y, z0 = z;

			float x1 = x0, y1 = y0, z1 = z0 + 1;
			if (z1 >= sizeZ) z1 = 0, y1 += 1;
			if (y1 >= sizeY) y1 = 0, x1 += 1;

			float x2 = x1, y2 = y1, z2 = z1 + 1;
			if (z2 >= sizeZ) z2 = 0, y2 += 1;
			if (y2 >= sizeY) y2 = 0, x2 += 1;

			float x3 = x2, y3 = y2, z3 = z2 + 1;
			if (z3 >= sizeZ) z3 = 0, y3 += 1;
			if (y3 >= sizeY) y3 = 0, x3 += 1;

			float x4 = x3, y4 = y3, z4 = z3 + 1;
			if (z4 >= sizeZ) z4 = 0, y4 += 1;
			if (y4 >= sizeY) y4 = 0, x4 += 1;

			float x5 = x4, y5 = y4, z5 = z4 + 1;
			if (z5 >= sizeZ) z5 = 0, y5 += 1;
			if (y5 >= sizeY) y5 = 0, x5 += 1;

			float x6 = x5, y6 = y5, z6 = z5 + 1;
			if (z6 >= sizeZ) z6 = 0, y6 += 1;
			if (y6 >= sizeY) y6 = 0, x6 += 1;

			float x7 = x6, y7 = y6, z7 = z6 + 1;
			if (z7 >= sizeZ) z7 = 0, y7 += 1;
			if (y7 >= sizeY) y7 = 0, x7 += 1;

			__m256 vecX = _mm256_set_ps(x7, x6, x5, x4, x3, x2, x1, x0);
			__m256 vecY = _mm256_set_ps(y7, y6, y5, y4, y3, y2, y1, y0);
			__m256 vecZ = _mm256_set_ps(z7, z6, z5, z4, z3, z2, z1, z0);

			__m256 vecNoise = avx2Get3d(_mm256_add_ps(vecOffsetX, _mm256_mul_ps(vecX, vecStep)),
                                        _mm256_add_ps(vecOffsetY, _mm256_mul_ps(vecY, vecStep)),
                                        _mm256_add_ps(vecOffsetZ, _mm256_mul_ps(vecZ, vecStep)), vecScale);

			_mm256_store_ps(&rawPtr[index], vecNoise);

			x = x7, y = y7, z = z7 + 1;
			if (z >= sizeZ) z = 0, y += 1;
			if (y >= sizeY) y = 0, x += 1;
		}
	}

    if (maxIndex % 8 != 0) {
        float x0 = x, y0 = y, z0 = z;

        float x1 = x0, y1 = y0, z1 = z0 + 1;
        if (z1 >= sizeZ) z1 = 0, y1 += 1;
        if (y1 >= sizeY) y1 = 0, x1 += 1;

        float x2 = x1, y2 = y1, z2 = z1 + 1;
        if (z2 >= sizeZ) z2 = 0, y2 += 1;
        if (y2 >= sizeY) y2 = 0, x2 += 1;

        float x3 = x2, y3 = y2, z3 = z2 + 1;
        if (z3 >= sizeZ) z3 = 0, y3 += 1;
        if (y3 >= sizeY) y3 = 0, x3 += 1;

        float x4 = x3, y4 = y3, z4 = z3 + 1;
        if (z4 >= sizeZ) z4 = 0, y4 += 1;
        if (y4 >= sizeY) y4 = 0, x4 += 1;

        float x5 = x4, y5 = y4, z5 = z4 + 1;
        if (z5 >= sizeZ) z5 = 0, y5 += 1;
        if (y5 >= sizeY) y5 = 0, x5 += 1;

        float x6 = x5, y6 = y5, z6 = z5 + 1;
        if (z6 >= sizeZ) z6 = 0, y6 += 1;
        if (y6 >= sizeY) y6 = 0, x6 += 1;

        float x7 = x6, y7 = y6, z7 = z6 + 1;
        if (z7 >= sizeZ) z7 = 0, y7 += 1;
        if (y7 >= sizeY) y7 = 0, x7 += 1;

        __m256 vecX = _mm256_set_ps(x7, x6, x5, x4, x3, x2, x1, x0);
        __m256 vecY = _mm256_set_ps(y7, y6, y5, y4, y3, y2, y1, y0);
        __m256 vecZ = _mm256_set_ps(z7, z6, z5, z4, z3, z2, z1, z0);

        __m256 vecNoise = avx2Get3d(_mm256_add_ps(vecOffsetX, _mm256_mul_ps(vecX, vecStep)),
                                    _mm256_add_ps(vecOffsetY, _mm256_mul_ps(vecY, vecStep)),
                                    _mm256_add_ps(vecOffsetZ, _mm256_mul_ps(vecZ, vecStep)), vecScale);

        alignas(32) float pResult[8];

        _mm256_store_ps(pResult, vecNoise);

        if ((sizeX * sizeY) % 8 == 1) {
            rawPtr[index] = pResult[0];
        } else if ((sizeX * sizeY) % 8 == 2) {
            rawPtr[index] = pResult[0];
            rawPtr[index + 1] = pResult[1];
        } else if ((sizeX * sizeY) % 8 == 3) {
            rawPtr[index] = pResult[0];
            rawPtr[index + 1] = pResult[1];
            rawPtr[index + 2] = pResult[2];
        } else if ((sizeX * sizeY) % 8 == 4) {
            rawPtr[index] = pResult[0];
            rawPtr[index + 1] = pResult[1];
            rawPtr[index + 2] = pResult[2];
            rawPtr[index + 3] = pResult[3];
        } else if ((sizeX * sizeY) % 8 == 5) {
            rawPtr[index] = pResult[0];
            rawPtr[index + 1] = pResult[1];
            rawPtr[index + 2] = pResult[2];
            rawPtr[index + 3] = pResult[3];
            rawPtr[index + 4] = pResult[4];
        } else if ((sizeX * sizeY) % 8 == 6) {
            rawPtr[index] = pResult[0];
            rawPtr[index + 1] = pResult[1];
            rawPtr[index + 2] = pResult[2];
            rawPtr[index + 3] = pResult[3];
            rawPtr[index + 4] = pResult[4];
            rawPtr[index + 5] = pResult[5];
        } else if ((sizeX * sizeY) % 8 == 7) {
            rawPtr[index] = pResult[0];
            rawPtr[index + 1] = pResult[1];
            rawPtr[index + 2] = pResult[2];
            rawPtr[index + 3] = pResult[3];
            rawPtr[index + 4] = pResult[4];
            rawPtr[index + 5] = pResult[5];
            rawPtr[index + 6] = pResult[6];
        }
    }

    _mm256_zeroall();
    return result;
}
