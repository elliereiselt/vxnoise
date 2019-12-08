// SSE2
#include "../../simd/sse2.hpp"
#include "CellularSSE2.hpp"
#include <limits>

static __m128 VX_VECTORCALL hashCellX(__m128i seed, __m128 cellX, __m128 cellY) {
    __m128 rx = vx::simd::sse2::dot_ps(cellX, cellY, _mm_set1_ps(127.1f), _mm_set1_ps(311.7f));
    __m128i hashX = vx::simd::sse2::partialJenkinsHash(seed, _mm_cvttps_epi32(rx));
    __m128 pointX = _mm_div_ps(_mm_cvtepi32_ps(hashX), _mm_set1_ps(1000000.0f));
    __m128 fractX = _mm_sub_ps(pointX, vx::simd::sse2::floor_ps(pointX));

    return _mm_add_ps(cellX, fractX);
}

static __m128 VX_VECTORCALL hashCellY(__m128i seed, __m128 cellX, __m128 cellY) {
    __m128 ry = vx::simd::sse2::dot_ps(cellX, cellY, _mm_set1_ps(269.5f), _mm_set1_ps(183.3f));
    __m128i hashY = vx::simd::sse2::partialJenkinsHash(seed, _mm_cvttps_epi32(ry));
    __m128 pointY = _mm_div_ps(_mm_cvtepi32_ps(hashY), _mm_set1_ps(1000000.0f));
    __m128 fractY = _mm_sub_ps(pointY, vx::simd::sse2::floor_ps(pointY));

    return _mm_add_ps(cellY, fractY);
}

static __m128 hashCellX(__m128i seed, __m128 cellX, __m128 cellY, __m128 cellZ) {
    // Gonna be honest with ya here. Don't remember where the original x and y constants came from, the z constants were made up from keyboard smashing. Probably not good.
    __m128 rx = vx::simd::sse2::dot_ps(cellX, cellY, cellZ,
                                       _mm_set1_ps(127.1f), _mm_set1_ps(311.7f), _mm_set1_ps(231.4f));
    __m128i hashX = vx::simd::sse2::partialJenkinsHash(seed, _mm_cvttps_epi32(rx));

    __m128 pointX = _mm_div_ps(_mm_cvtepi32_ps(hashX), _mm_set1_ps(1000000.0f));
    __m128 fractX = _mm_sub_ps(pointX, vx::simd::sse2::floor_ps(pointX));

    return _mm_add_ps(cellX, fractX);
}

static __m128 hashCellY(__m128i seed, __m128 cellX, __m128 cellY, __m128 cellZ) {
    // Gonna be honest with ya here. Don't remember where the original x and y constants came from, the z constants were made up from keyboard smashing. Probably not good.
    __m128 ry = vx::simd::sse2::dot_ps(cellX, cellY, cellZ,
                                       _mm_set1_ps(269.5f), _mm_set1_ps(183.3f), _mm_set1_ps(352.6f));
    __m128i hashY = vx::simd::sse2::partialJenkinsHash(seed, _mm_cvttps_epi32(ry));
    __m128 pointY = _mm_div_ps(_mm_cvtepi32_ps(hashY), _mm_set1_ps(1000000.0f));
    __m128 fractY = _mm_sub_ps(pointY, vx::simd::sse2::floor_ps(pointY));

    return _mm_add_ps(cellY, fractY);
}

static __m128 hashCellZ(__m128i seed, __m128 cellX, __m128 cellY, __m128 cellZ) {
    // Gonna be honest with ya here. Don't remember where the original x and y constants came from, the z constants were made up from keyboard smashing. Probably not good.
    __m128 rz = vx::simd::sse2::dot_ps(cellX, cellY, cellZ,
                                       _mm_set1_ps(419.2f), _mm_set1_ps(371.9f), _mm_set1_ps(523.7f));
    __m128i hashZ = vx::simd::sse2::partialJenkinsHash(seed, _mm_cvttps_epi32(rz));
    __m128 pointZ = _mm_div_ps(_mm_cvtepi32_ps(hashZ), _mm_set1_ps(1000000.0f));
    __m128 fractZ = _mm_sub_ps(pointZ, vx::simd::sse2::floor_ps(pointZ));

    return _mm_add_ps(cellZ, fractZ);
}

__m128 vx::noise::CellularSSE2::sse2Get2d(__m128 x, __m128 y, __m128 scale) {
    x = _mm_div_ps(x, scale);
    y = _mm_div_ps(y, scale);

    __m128 fx = _mm_div_ps(x, _mm_set1_ps(16));
    __m128 fy = _mm_div_ps(y, _mm_set1_ps(16));
    __m128 ix = vx::simd::sse2::floor_ps(fx);
    __m128 iy = vx::simd::sse2::floor_ps(fy);

    __m128 lowestDistance = _mm_set1_ps((std::numeric_limits<float>::max)());
    __m128 resultX = _mm_setzero_ps();
    __m128 resultY = _mm_setzero_ps();

    __m128 i = _mm_set1_ps(-1);

    for (int ignoreI = -1; ignoreI < 2; ++ignoreI) {
        __m128 j = _mm_set1_ps(-1);

        for (int ignoreJ = -1; ignoreJ < 2; ++ignoreJ) {
            __m128 cellX = hashCellX(_sse2Seed, _mm_add_ps(ix, i), _mm_add_ps(iy, j));
            __m128 cellY = hashCellY(_sse2Seed, _mm_add_ps(ix, i), _mm_add_ps(iy, j));

            __m128 euclideanDistance =
                    _mm_sqrt_ps(
                            _mm_add_ps(
                                    // (cellY - fy)^2
                                    _mm_mul_ps(_mm_sub_ps(cellY, fy), _mm_sub_ps(cellY, fy)),
                                    // (cellZ - fz)^2
                                    _mm_mul_ps(_mm_sub_ps(cellX, fx), _mm_sub_ps(cellX, fx))));
            __m128 manhattanDistance = _mm_add_ps(vx::simd::sse2::abs_ps(_mm_sub_ps(cellY, fy)), vx::simd::sse2::abs_ps(_mm_sub_ps(cellX, fx)));
            __m128 naturalDistance = _mm_add_ps(euclideanDistance, manhattanDistance);

            __m128 distLTlowestDist = _mm_cmplt_ps(naturalDistance, lowestDistance);

            lowestDistance = _mm_or_ps(_mm_and_ps(distLTlowestDist, naturalDistance), _mm_andnot_ps(distLTlowestDist, lowestDistance));
            resultX = _mm_or_ps(_mm_and_ps(distLTlowestDist, cellX), _mm_andnot_ps(distLTlowestDist, resultX));
            resultY = _mm_or_ps(_mm_and_ps(distLTlowestDist, cellY), _mm_andnot_ps(distLTlowestDist, resultY));

            j = _mm_add_ps(j, _mm_set1_ps(1));
        }

        i = _mm_add_ps(i, _mm_set1_ps(1));
    }

    if (_noiseLookupSSE2 != nullptr) {
        return _noiseLookupSSE2->sse2Get2d(resultX, resultY);
    } else {
        alignas(16) float pResultX[4];
        alignas(16) float pResultY[4];

        _mm_store_ps(pResultX, resultX);
        _mm_store_ps(pResultY, resultY);

        alignas(16) float pResult[4];

        for (int idx = 0; idx < 4; ++idx) {
            pResult[idx] = _noiseLookup->get2d(pResultX[idx], pResultY[idx]);
        }

        return _mm_load_ps(pResult);
    }
}

float vx::noise::CellularSSE2::get2d(float x, float y, float scale) {
    __m128 sse2Result = sse2Get2d(_mm_set1_ps(x), _mm_set1_ps(y), _mm_set1_ps(scale));
    alignas(16) float pResult[4];
    _mm_store_ps(pResult, sse2Result);
    return pResult[0];
}

__m128 vx::noise::CellularSSE2::sse2Get3d(__m128 x, __m128 y, __m128 z, __m128 scale) {
    x = _mm_div_ps(x, scale);
    y = _mm_div_ps(y, scale);
    z = _mm_div_ps(z, scale);

    __m128 fx = _mm_div_ps(x, _mm_set1_ps(16));
    __m128 fy = _mm_div_ps(y, _mm_set1_ps(16));
    __m128 fz = _mm_div_ps(z, _mm_set1_ps(16));
    __m128 ix = vx::simd::sse2::floor_ps(fx);
    __m128 iy = vx::simd::sse2::floor_ps(fy);
    __m128 iz = vx::simd::sse2::floor_ps(fz);

    __m128 lowestDistance = _mm_set1_ps((std::numeric_limits<float>::max)());
    __m128 resultX = _mm_setzero_ps();
    __m128 resultY = _mm_setzero_ps();
    __m128 resultZ = _mm_setzero_ps();

    __m128 i = _mm_set1_ps(-1);

    for (int ignoreI = -1; ignoreI < 2; ++ignoreI) {
        __m128 j = _mm_set1_ps(-1);

        for (int ignoreJ = -1; ignoreJ < 2; ++ignoreJ) {
            __m128 k = _mm_set1_ps(-1);

            for (int ignoreK = -1; ignoreK < 2; ++ignoreK) {
                __m128 cellX = hashCellX(_sse2Seed, _mm_add_ps(ix, i), _mm_add_ps(iy, j), _mm_add_ps(iz, k));
                __m128 cellY = hashCellY(_sse2Seed, _mm_add_ps(ix, i), _mm_add_ps(iy, j), _mm_add_ps(iz, k));
                __m128 cellZ = hashCellZ(_sse2Seed, _mm_add_ps(ix, i), _mm_add_ps(iy, j), _mm_add_ps(iz, k));

                __m128 euclideanDistance =
                        _mm_sqrt_ps(
                                _mm_add_ps(
                                        _mm_add_ps(
                                                // (cellZ - fz)^2
                                                _mm_mul_ps(_mm_sub_ps(cellZ, fz), _mm_sub_ps(cellZ, fz)),
                                                // (cellY - fy)^2
                                                _mm_mul_ps(_mm_sub_ps(cellY, fy), _mm_sub_ps(cellY, fy))),
                                        // (cellX - fx)^2
                                        _mm_mul_ps(_mm_sub_ps(cellX, fx), _mm_sub_ps(cellX, fx))));
                __m128 manhattanDistance =
                        _mm_add_ps(
                                _mm_add_ps(
                                        vx::simd::sse2::abs_ps(_mm_sub_ps(cellZ, fz)),
                                        vx::simd::sse2::abs_ps(_mm_sub_ps(cellY, fy))),
                                vx::simd::sse2::abs_ps(_mm_sub_ps(cellX, fx)));
                __m128 naturalDistance = _mm_add_ps(euclideanDistance, manhattanDistance);

                __m128 distLTlowestDist = _mm_cmplt_ps(naturalDistance, lowestDistance);

                lowestDistance = vx::simd::sse2::blendv_ps(lowestDistance, naturalDistance, distLTlowestDist);
                resultX = vx::simd::sse2::blendv_ps(resultX, cellX, distLTlowestDist);
                resultY = vx::simd::sse2::blendv_ps(resultY, cellY, distLTlowestDist);
                resultZ = vx::simd::sse2::blendv_ps(resultZ, cellZ, distLTlowestDist);

                k = _mm_add_ps(k, _mm_set1_ps(1));
            }

            j = _mm_add_ps(j, _mm_set1_ps(1));
        }

        i = _mm_add_ps(i, _mm_set1_ps(1));
    }

    if (_noiseLookupSSE2 != nullptr) {
        return _noiseLookupSSE2->sse2Get3d(resultX, resultY, resultZ);
    } else {
        alignas(16) float pResultX[4];
        alignas(16) float pResultY[4];
        alignas(16) float pResultZ[4];

        _mm_store_ps(pResultX, resultX);
        _mm_store_ps(pResultY, resultY);
        _mm_store_ps(pResultZ, resultZ);

        alignas(16) float pResult[4];

        for (int idx = 0; idx < 4; ++idx) {
            pResult[idx] = _noiseLookup->get3d(pResultX[idx], pResultY[idx], pResultZ[idx]);
        }

        return _mm_load_ps(pResult);
    }
}

float vx::noise::CellularSSE2::get3d(float x, float y, float z, float scale) {
    __m128 sse2Result = sse2Get3d(_mm_set1_ps(x), _mm_set1_ps(y), _mm_set1_ps(z), _mm_set1_ps(scale));
    alignas(16) float pResult[4];
    _mm_store_ps(pResult, sse2Result);
    return pResult[0];
}

vx::aligned_array_2d<float> vx::noise::CellularSSE2::noise(float offsetX, float offsetY,
                                                           std::size_t sizeX, std::size_t sizeY, float step,
                                                           float scale) {
    vx::aligned_array_2d<float> result(sizeX, sizeY, 16);
    float* rawPtr = result.ptr();

    std::size_t maxIndex = sizeX * sizeY;
    __m128 vecOffsetX = _mm_set1_ps(offsetX);
    __m128 vecOffsetY = _mm_set1_ps(offsetY);

    float x = 0, y = 0;
    __m128 vecScale = _mm_set1_ps(scale);
    __m128 vecStep = _mm_set1_ps(step);

    std::size_t index = 0;

	if (maxIndex >= 4) {
		for (; index < maxIndex - 3; index += 4) {
			float x0 = x, y0 = y;

			float x1 = x0, y1 = y0 + 1;
			if (y1 >= sizeY) y1 = 0, x1 += 1;

			float x2 = x1, y2 = y1 + 1;
			if (y2 >= sizeY) y2 = 0, x2 += 1;

			float x3 = x2, y3 = y2 + 1;
			if (y3 >= sizeY) y3 = 0, x3 += 1;

			__m128 vecX = _mm_set_ps(x3, x2, x1, x0);
			__m128 vecY = _mm_set_ps(y3, y2, y1, y0);

			__m128 vecNoise = sse2Get2d(_mm_add_ps(vecOffsetX, _mm_mul_ps(vecX, vecStep)),
                                        _mm_add_ps(vecOffsetY, _mm_mul_ps(vecY, vecStep)), vecScale);

			_mm_store_ps(&rawPtr[index], vecNoise);

			x = x3, y = y3 + 1;
			if (y >= sizeY) y = 0, x += 1;
		}
	}

    if (maxIndex % 4 != 0) {
        float x0 = x, y0 = y;

        float x1 = x0, y1 = y0 + 1;
        if (y1 >= sizeY) y1 = 0, x1 += 1;

        float x2 = x1, y2 = y1 + 1;
        if (y2 >= sizeY) y2 = 0, x2 += 1;

        float x3 = x2, y3 = y2 + 1;
        if (y3 >= sizeY) y3 = 0, x3 += 1;

        __m128 vecX = _mm_set_ps(x3, x2, x1, x0);
        __m128 vecY = _mm_set_ps(y3, y2, y1, y0);

        __m128 vecNoise = sse2Get2d(_mm_add_ps(vecOffsetX, _mm_mul_ps(vecX, vecStep)),
                                    _mm_add_ps(vecOffsetY, _mm_mul_ps(vecY, vecStep)), vecScale);

        alignas(16) float pResult[4];

        _mm_store_ps(pResult, vecNoise);

        if (maxIndex % 4 == 1) {
            rawPtr[index] = pResult[0];
        } else if (maxIndex % 4 == 2) {
            rawPtr[index] = pResult[0];
            rawPtr[index + 1] = pResult[1];
        } else if (maxIndex % 4 == 3) {
            rawPtr[index] = pResult[0];
            rawPtr[index + 1] = pResult[1];
            rawPtr[index + 2] = pResult[2];
        }
    }

    return result;
}

vx::aligned_array_3d<float> vx::noise::CellularSSE2::noise(float offsetX, float offsetY, float offsetZ,
                                                           std::size_t sizeX, std::size_t sizeY, std::size_t sizeZ,
                                                           float step, float scale) {
    vx::aligned_array_3d<float> result(sizeX, sizeY, sizeZ, 16);
    float* rawPtr = result.ptr();

    std::size_t maxIndex = sizeX * sizeY * sizeZ;
    __m128 vecOffsetX = _mm_set1_ps(offsetX);
    __m128 vecOffsetY = _mm_set1_ps(offsetY);
    __m128 vecOffsetZ = _mm_set1_ps(offsetZ);

    float x = 0, y = 0, z = 0;
    __m128 vecScale = _mm_set1_ps(scale);
    __m128 vecStep = _mm_set1_ps(step);

    std::size_t index = 0;

	if (maxIndex >= 4) {
		for (; index < maxIndex - 3; index += 4) {
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

			__m128 vecX = _mm_set_ps(x3, x2, x1, x0);
			__m128 vecY = _mm_set_ps(y3, y2, y1, y0);
			__m128 vecZ = _mm_set_ps(z3, z2, z1, z0);

			__m128 vecNoise = sse2Get3d(_mm_add_ps(vecOffsetX, _mm_mul_ps(vecX, vecStep)),
                                        _mm_add_ps(vecOffsetY, _mm_mul_ps(vecY, vecStep)),
                                        _mm_add_ps(vecOffsetZ, _mm_mul_ps(vecZ, vecStep)), vecScale);

			_mm_store_ps(&rawPtr[index], vecNoise);

			x = x3, y = y3, z = z3 + 1;
			if (z >= sizeZ) z = 0, y += 1;
			if (y >= sizeY) y = 0, x += 1;
		}
	}

    if (maxIndex % 4 != 0) {
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

        __m128 vecX = _mm_set_ps(x3, x2, x1, x0);
        __m128 vecY = _mm_set_ps(y3, y2, y1, y0);
        __m128 vecZ = _mm_set_ps(z3, z2, z1, z0);

        __m128 vecNoise = sse2Get3d(_mm_add_ps(vecOffsetX, _mm_mul_ps(vecX, vecStep)),
                                    _mm_add_ps(vecOffsetY, _mm_mul_ps(vecY, vecStep)),
                                    _mm_add_ps(vecOffsetZ, _mm_mul_ps(vecZ, vecStep)), vecScale);

        alignas(16) float pResult[4];

        _mm_store_ps(pResult, vecNoise);

        if (maxIndex % 4 == 1) {
            rawPtr[index] = pResult[0];
        } else if (maxIndex % 4 == 2) {
            rawPtr[index] = pResult[0];
            rawPtr[index + 1] = pResult[1];
        } else if (maxIndex % 4 == 3) {
            rawPtr[index] = pResult[0];
            rawPtr[index + 1] = pResult[1];
            rawPtr[index + 2] = pResult[2];
        }
    }

    return result;
}
