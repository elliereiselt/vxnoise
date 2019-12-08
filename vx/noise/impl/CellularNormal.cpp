#include <limits>
#include <cmath>
#include "../../simd/none.hpp"
#include "CellularNormal.hpp"

static float hashCellX(int seed, int cellX, int cellY) {
    float rx = vx::simd::none::dot(static_cast<float>(cellX), static_cast<float>(cellY), 127.1f, 311.7f);
    int hashX = vx::simd::none::partialJenkinsHash(seed, static_cast<int>(rx));
    float pointX = static_cast<float>(hashX) / 1000000.0f;
    float fractX = pointX - static_cast<float>(vx::simd::none::floor(pointX));

    return static_cast<float>(cellX) + fractX;
}

static float hashCellY(int seed, int cellX, int cellY) {
    float ry = vx::simd::none::dot(static_cast<float>(cellX), static_cast<float>(cellY), 269.5f, 183.3f);
    int hashY = vx::simd::none::partialJenkinsHash(seed, static_cast<int>(ry));
    float pointY = static_cast<float>(hashY) / 1000000.0f;
    float fractY = pointY - static_cast<float>(vx::simd::none::floor(pointY));

    return static_cast<float>(cellY) + fractY;
}

static float hashCellX(int seed, int cellX, int cellY, int cellZ) {
    // Gonna be honest with ya here. Don't remember where the original x and y constants came from, the z constants were made up from keyboard smashing. Probably not good.
    float rx = vx::simd::none::dot(static_cast<float>(cellX), static_cast<float>(cellY), static_cast<float>(cellZ),
                                   127.1f, 311.7f, 231.4f);
    int hashX = vx::simd::none::partialJenkinsHash(seed, static_cast<int>(rx));

    float pointX = static_cast<float>(hashX) / 1000000.0f;
    float fractX = pointX - static_cast<float>(vx::simd::none::floor(pointX));

    return static_cast<float>(cellX) + fractX;
}

static float hashCellY(int seed, int cellX, int cellY, int cellZ) {
    // Gonna be honest with ya here. Don't remember where the original x and y constants came from, the z constants were made up from keyboard smashing. Probably not good.
    float ry = vx::simd::none::dot(static_cast<float>(cellX), static_cast<float>(cellY), static_cast<float>(cellZ),
                                   269.5f, 183.3f, 352.6f);
    int hashY = vx::simd::none::partialJenkinsHash(seed, static_cast<int>(ry));
    float pointY = static_cast<float>(hashY) / 1000000.0f;
    float fractY = pointY - static_cast<float>(vx::simd::none::floor(pointY));

    return static_cast<float>(cellY) + fractY;
}

static float hashCellZ(int seed, int cellX, int cellY, int cellZ) {
    // Gonna be honest with ya here. Don't remember where the original x and y constants came from, the z constants were made up from keyboard smashing. Probably not good.
    float rz = vx::simd::none::dot(static_cast<float>(cellX), static_cast<float>(cellY), static_cast<float>(cellZ),
                                   419.2f, 371.9f, 523.7f);
    int hashZ = vx::simd::none::partialJenkinsHash(seed, static_cast<int>(rz));
    float pointZ = static_cast<float>(hashZ) / 1000000.0f;
    float fractZ = pointZ - static_cast<float>(vx::simd::none::floor(pointZ));

    return static_cast<float>(cellZ) + fractZ;
}

float vx::noise::CellularNormal::get2d(float x, float y, float scale) {
    x /= scale;
    y /= scale;

    float fx = x / 16.0f;
    float fy = y / 16.0f;
    int ix = vx::simd::none::floor(fx);
    int iy = vx::simd::none::floor(fy);

    float lowestDistance = (std::numeric_limits<float>::max)();
    float resultX = 0;
    float resultY = 0;

    for (int i = -1; i < 2; ++i) {
        for (int j = -1; j < 2; ++j) {
            float cellX = hashCellX(_seed, (ix + i), (iy + j));
            float cellY = hashCellY(_seed, (ix + i), (iy + j));

            float euclideanDistance = sqrtf((cellY - fy) * (cellY - fy) + (cellX - fx) * (cellX - fx));
            float manhattanDistance = fabsf(cellY - fy) + fabsf(cellX - fx);
            float naturalDistance = euclideanDistance + manhattanDistance;

            if (naturalDistance < lowestDistance) {
                lowestDistance = naturalDistance;
                resultX = cellX;
                resultY = cellY;
            }
        }
    }

    return _noiseLookup->get2d(resultX, resultY);
}

float vx::noise::CellularNormal::get3d(float x, float y, float z, float scale) {
    x /= scale;
    y /= scale;
    z /= scale;

    float fx = x / 16.0f;
    float fy = y / 16.0f;
    float fz = z / 16.0f;
    int ix = vx::simd::none::floor(fx);
    int iy = vx::simd::none::floor(fy);
    int iz = vx::simd::none::floor(fz);

    float lowestDistance = (std::numeric_limits<float>::max)();
    float resultX = 0;
    float resultY = 0;
    float resultZ = 0;

    for (int i = -1; i < 2; ++i) {
        for (int j = -1; j < 2; ++j) {
            for (int k = -1; k < 2; ++k) {
                float cellX = hashCellX(_seed, (ix + i), (iy + j), (iz + k));
                float cellY = hashCellY(_seed, (ix + i), (iy + j), (iz + k));
                float cellZ = hashCellZ(_seed, (ix + i), (iy + j), (iz + k));

                float euclideanDistance = sqrtf((cellZ - fz) * (cellZ - fz) + (cellY - fy) * (cellY - fy) + (cellX - fx) * (cellX - fx));
                float manhattanDistance = fabsf(cellZ - fz) + fabsf(cellY - fy) + fabsf(cellX - fx);
                float naturalDistance = euclideanDistance + manhattanDistance;

                if (naturalDistance < lowestDistance) {
                    lowestDistance = naturalDistance;
                    resultX = cellX;
                    resultY = cellY;
                    resultZ = cellZ;
                }
            }
        }
    }

    return _noiseLookup->get3d(resultX, resultY, resultZ);
}

vx::aligned_array_2d<float> vx::noise::CellularNormal::noise(float offsetX, float offsetY,
                                                             std::size_t sizeX, std::size_t sizeY, float step,
                                                             float scale) {
    vx::aligned_array_2d<float> result(sizeX, sizeY, alignof(float));

    for (std::size_t x = 0; x < sizeX; ++x) {
        for (std::size_t y = 0; y < sizeY; ++y) {
            result[{x, y}] = get2d(offsetX + static_cast<float>(x) * step,
                                   offsetY + static_cast<float>(y) * step, scale);
        }
    }

    return result;
}

vx::aligned_array_3d<float> vx::noise::CellularNormal::noise(float offsetX, float offsetY, float offsetZ,
                                                             std::size_t sizeX, std::size_t sizeY, std::size_t sizeZ,
                                                             float step, float scale) {
    vx::aligned_array_3d<float> result(sizeX, sizeY, sizeZ, alignof(float));

    for (std::size_t x = 0; x < sizeX; ++x) {
        for (std::size_t y = 0; y < sizeY; ++y) {
            for (std::size_t z = 0; z < sizeZ; ++z) {
                result[{x, y, z}] = get3d(offsetX + static_cast<float>(x) * step,
                                          offsetY + static_cast<float>(y) * step,
                                          offsetZ + static_cast<float>(z) * step,
                                          scale);
            }
        }
    }

    return result;
}
