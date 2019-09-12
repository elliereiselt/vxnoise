#include "SimplexFractalSSE42.hpp"
#include "SimplexSSE42.hpp"

__m128 VX_VECTORCALL vx::noise::SimplexFractalSSE42::sse42Get2d(__m128 x, __m128 y, __m128 scale) {
    __m128 frequency = _mm_set1_ps(1);
    __m128 amplitude = _mm_set1_ps(1);
    __m128 resultNoise = _mm_setzero_ps();

    for (std::uint32_t octave = 0; octave < _octaves; ++octave) {
        __m128 noise = vx::noise::sse42SimplexGet2d(_sse42Seed,
                                                   _mm_mul_ps(_mm_div_ps(x, scale), frequency),
                                                   _mm_mul_ps(_mm_div_ps(y, scale), frequency));

        resultNoise = _mm_add_ps(resultNoise, _mm_mul_ps(noise, amplitude));

        amplitude = _mm_mul_ps(amplitude, _sse42Persistence);
        frequency = _mm_mul_ps(frequency, _sse42Lacunarity);
    }

    return resultNoise;
}

__m128 VX_VECTORCALL vx::noise::SimplexFractalSSE42::sse42Get3d(__m128 x, __m128 y, __m128 z, __m128 scale) {
    __m128 frequency = _mm_set1_ps(1);
    __m128 amplitude = _mm_set1_ps(1);
    __m128 resultNoise = _mm_setzero_ps();

    for (std::uint32_t octave = 0; octave < _octaves; ++octave) {
        __m128 noise = vx::noise::sse42SimplexGet3d(_sse42Seed,
                                                   _mm_mul_ps(_mm_div_ps(x, scale), frequency),
                                                   _mm_mul_ps(_mm_div_ps(y, scale), frequency),
                                                   _mm_mul_ps(_mm_div_ps(z, scale), frequency));

        resultNoise = _mm_add_ps(resultNoise, _mm_mul_ps(noise, amplitude));

        amplitude = _mm_mul_ps(amplitude, _sse42Persistence);
        frequency = _mm_mul_ps(frequency, _sse42Lacunarity);
    }

    return resultNoise;
}

float vx::noise::SimplexFractalSSE42::get2d(float x, float y, float scale) {
    __m128 sse2Result = sse42Get2d(_mm_set1_ps(x), _mm_set1_ps(y), _mm_set1_ps(scale));
    alignas(16) float pResult[4];
    _mm_store_ps(pResult, sse2Result);
    return pResult[0];
}

float vx::noise::SimplexFractalSSE42::get3d(float x, float y, float z, float scale) {
    __m128 sse2Result = sse42Get3d(_mm_set1_ps(x), _mm_set1_ps(y), _mm_set1_ps(z), _mm_set1_ps(scale));
    alignas(16) float pResult[4];
    _mm_store_ps(pResult, sse2Result);
    return pResult[0];
}

vx::aligned_array_2d<float> vx::noise::SimplexFractalSSE42::noise(float offsetX, float offsetY,
                                                                  std::size_t sizeX, std::size_t sizeY,
                                                                  float step, float scale) {
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

			__m128 vecNoise = sse42Get2d(_mm_add_ps(vecOffsetX, _mm_mul_ps(vecX, vecStep)),
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

        __m128 vecNoise = sse42Get2d(_mm_add_ps(vecOffsetX, _mm_mul_ps(vecX, vecStep)),
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

vx::aligned_array_3d<float> vx::noise::SimplexFractalSSE42::noise(float offsetX, float offsetY, float offsetZ,
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

			__m128 vecNoise = sse42Get3d(_mm_add_ps(vecOffsetX, _mm_mul_ps(vecX, vecStep)),
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

        __m128 vecNoise = sse42Get3d(_mm_add_ps(vecOffsetX, _mm_mul_ps(vecX, vecStep)),
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
