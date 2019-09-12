#include "SimplexFractalNormal.hpp"
#include "SimplexNormal.hpp"

float vx::noise::SimplexFractalNormal::get2d(float x, float y, float scale) {
    float frequency = 1;
    float amplitude = 1;
    float resultNoise = 0;

    for (std::uint32_t octave = 0; octave < _octaves; ++octave) {
        float noise = vx::noise::normalSimplexGet2d(_seed,
                                                    x / scale * frequency,
                                                    y / scale * frequency);

        resultNoise += noise * amplitude;

        amplitude *= _persistence;
        frequency *= _lacunarity;
    }

    return resultNoise;
}

float vx::noise::SimplexFractalNormal::get3d(float x, float y, float z, float scale) {
    float frequency = 1;
    float amplitude = 1;
    float resultNoise = 0;

    for (std::uint32_t octave = 0; octave < _octaves; ++octave) {
        float noise = vx::noise::normalSimplexGet3d(_seed,
                                                    x / scale * frequency,
                                                    y / scale * frequency,
                                                    z / scale * frequency);

        resultNoise += noise * amplitude;

        amplitude *= _persistence;
        frequency *= _lacunarity;
    }

    return resultNoise;
}

vx::aligned_array_2d<float> vx::noise::SimplexFractalNormal::noise(float offsetX, float offsetY,
                                                                   std::size_t sizeX, std::size_t sizeY,
                                                                   float step, float scale) {
    vx::aligned_array_2d<float> result(sizeX, sizeY, alignof(float));

    for (std::size_t x = 0; x < sizeX; ++x) {
        for (std::size_t y = 0; y < sizeY; ++y) {
            result[{x, y}] = get2d(offsetX + static_cast<float>(x) * step,
                                   offsetY + static_cast<float>(y) * step, scale);
        }
    }

    return result;
}

vx::aligned_array_3d<float> vx::noise::SimplexFractalNormal::noise(float offsetX, float offsetY, float offsetZ,
                                                                   std::size_t sizeX, std::size_t sizeY, std::size_t sizeZ,
                                                                   float step, float scale) {
    vx::aligned_array_3d<float> result(sizeX, sizeY, sizeZ, alignof(float));

    for (std::size_t x = 0; x < sizeX; ++x) {
        for (std::size_t y = 0; y < sizeY; ++y) {
            for (std::size_t z = 0; z < sizeZ; ++z) {
                result[{x, y, z}] = get3d(offsetX + static_cast<float>(x) * step,
                                          offsetY + static_cast<float>(y) * step,
                                          offsetZ + static_cast<float>(z) * step, scale);
            }
        }
    }

    return result;
}
