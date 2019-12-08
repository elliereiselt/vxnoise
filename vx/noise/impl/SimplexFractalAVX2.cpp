#ifndef __AVX__
#ifdef _MSC_VER
#error Missing argument "/arch:AVX(2)"
#else
#error Missing argument "-march=core-avx2"
#endif
#endif

#include "SimplexAVX2.hpp"
#include "SimplexFractalAVX2.hpp"

vx::noise::SimplexFractalAVX2::SimplexFractalAVX2(int seed, float defaultScale, std::uint32_t octaves,
                                                  float persistence, float lacunarity)
        : SimplexFractal(seed, defaultScale, octaves, persistence, lacunarity),
          NoiseAVX2(seed, defaultScale), _avx2Persistence(_mm256_set1_ps(persistence)),
          _avx2Lacunarity(_mm256_set1_ps(lacunarity)) {

}

__m256 VX_VECTORCALL vx::noise::SimplexFractalAVX2::avx2Get2d(__m256 x, __m256 y, __m256 scale) {
    __m256 frequency = _mm256_set1_ps(1);
    __m256 amplitude = _mm256_set1_ps(1);
    __m256 resultNoise = _mm256_setzero_ps();

    for (std::uint32_t octave = 0; octave < _octaves; ++octave) {
        __m256 noise = vx::noise::avx2SimplexGet2d(_avx2Seed,
                                                    _mm256_mul_ps(_mm256_div_ps(x, scale), frequency),
                                                    _mm256_mul_ps(_mm256_div_ps(y, scale), frequency));

        resultNoise = _mm256_add_ps(resultNoise, _mm256_mul_ps(noise, amplitude));

        amplitude = _mm256_mul_ps(amplitude, _avx2Persistence);
        frequency = _mm256_mul_ps(frequency, _avx2Lacunarity);
    }

    return resultNoise;
}

__m256 VX_VECTORCALL vx::noise::SimplexFractalAVX2::avx2Get3d(__m256 x, __m256 y, __m256 z, __m256 scale) {
    __m256 frequency = _mm256_set1_ps(1);
    __m256 amplitude = _mm256_set1_ps(1);
    __m256 resultNoise = _mm256_setzero_ps();

    for (std::uint32_t octave = 0; octave < _octaves; ++octave) {
        __m256 noise = vx::noise::avx2SimplexGet3d(_avx2Seed,
                                                    _mm256_mul_ps(_mm256_div_ps(x, scale), frequency),
                                                    _mm256_mul_ps(_mm256_div_ps(y, scale), frequency),
                                                    _mm256_mul_ps(_mm256_div_ps(z, scale), frequency));

        resultNoise = _mm256_add_ps(resultNoise, _mm256_mul_ps(noise, amplitude));

        amplitude = _mm256_mul_ps(amplitude, _avx2Persistence);
        frequency = _mm256_mul_ps(frequency, _avx2Lacunarity);
    }

    return resultNoise;
}

float vx::noise::SimplexFractalAVX2::get2d(float x, float y, float scale) {
    __m256 avx2Result = avx2Get2d(_mm256_set1_ps(x), _mm256_set1_ps(y), _mm256_set1_ps(scale));
    alignas(32) float pResult[8];
    _mm256_store_ps(pResult, avx2Result);
    return pResult[0];
}

float vx::noise::SimplexFractalAVX2::get3d(float x, float y, float z, float scale) {
    __m256 avx2Result = avx2Get3d(_mm256_set1_ps(x), _mm256_set1_ps(y), _mm256_set1_ps(z), _mm256_set1_ps(scale));
    alignas(32) float pResult[8];
    _mm256_store_ps(pResult, avx2Result);
    return pResult[0];
}

vx::aligned_array_2d<float> vx::noise::SimplexFractalAVX2::noise(float offsetX, float offsetY,
                                                                 std::size_t sizeX, std::size_t sizeY,
                                                                 float step, float scale) {
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

vx::aligned_array_3d<float> vx::noise::SimplexFractalAVX2::noise(float offsetX, float offsetY, float offsetZ,
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
