#define VX_NOISE_INTERNAL_DEFS
#include "../../simd/none.hpp"
#include "SimplexNormal.hpp"

static float grad(int hash, float x, float y) {
    int h = hash & 7;      // Convert low 3 bits of hash code
    float u = h < 4 ? x : y;  // into 8 simple gradient directions,
    float v = h < 4 ? y : x;  // and compute the dot product with (x,y).
    return ((h & 1) ? -u : u) + ((h & 2) ? -2.0f*v : 2.0f*v);
}

static float grad(int hash, float x, float y, float z) {
    int h = hash & 15;     // Convert low 4 bits of hash code into 12 simple
    float u = h < 8 ? x : y; // gradient directions, and compute dot product.
    float v = h < 4 ? y : h == 12 || h == 14 ? x : z; // Fix repeats at h = 12 to 15
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

float vx::noise::normalSimplexGet2d(int seed, float x, float y) {
    float n0, n1, n2; // Noise contributions from the three corners

// Skew the input space to determine which simplex cell we're in
    float s = (x + y)*F2; // Hairy factor for 2D
    float xs = x + s;
    float ys = y + s;
    int i = vx::simd::none::floor(xs);
    int j = vx::simd::none::floor(ys);

    float t = (float)(i + j)*G2;
    float X0 = static_cast<float>(i) - t; // Unskew the cell origin back to (x,y) space
    float Y0 = static_cast<float>(j) - t;
    float x0 = x - X0; // The x,y distances from the cell origin
    float y0 = y - Y0;

    // For the 2D case, the simplex shape is an equilateral triangle.
    // Determine which simplex we are in.
    int i1, j1; // Offsets for second (middle) corner of simplex in (i,j) coords
    if (x0 > y0) { i1 = 1; j1 = 0; } // lower triangle, XY order: (0,0)->(1,0)->(1,1)
    else { i1 = 0; j1 = 1; }      // upper triangle, YX order: (0,0)->(0,1)->(1,1)

    // A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
    // a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
    // c = (3-sqrt(3))/6

    float x1 = x0 - static_cast<float>(i1) + G2; // Offsets for middle corner in (x,y) unskewed coords
    float y1 = y0 - static_cast<float>(j1) + G2;
    float x2 = x0 - 1.0f + 2.0f * G2; // Offsets for last corner in (x,y) unskewed coords
    float y2 = y0 - 1.0f + 2.0f * G2;

    // Wrap the integer indices at 256, to avoid indexing perm[] out of bounds
    int ii = i;
    int jj = j;

    // Calculate the contribution from the three corners
    float t0 = 0.5f - x0 * x0 - y0 * y0;
    if (t0 < 0.0f) n0 = 0.0f;
    else {
        int jjh = vx::simd::none::partialJenkinsHash(seed, jj);
        float g0 = grad(vx::simd::none::partialJenkinsHash(seed, ii + jjh), x0, y0);

        t0 *= t0;
        t0 *= t0;
        n0 = t0 * g0;
    }

    float t1 = 0.5f - x1 * x1 - y1 * y1;
    if (t1 < 0.0f) n1 = 0.0f;
    else {
        float g1 = grad(vx::simd::none::partialJenkinsHash(seed, ii + i1 + vx::simd::none::partialJenkinsHash(seed, jj + j1)), x1, y1);

        t1 *= t1;
        t1 *= t1;
        n1 = t1 * g1;
    }

    float t2 = 0.5f - x2 * x2 - y2 * y2;
    if (t2 < 0.0f) n2 = 0.0f;
    else {
        float g2 = grad(vx::simd::none::partialJenkinsHash(seed, ii + 1 + vx::simd::none::partialJenkinsHash(seed, jj + 1)), x2, y2);

        t2 *= t2;
        t2 *= t2;
        n2 = t2 * g2;
    }

    // Add contributions from each corner to get the final noise value.
    // The result is scaled to return values in the interval [-1,1].
    return 40.0f * (n0 + n1 + n2); // TODO: The scale factor is preliminary!
}

float vx::noise::normalSimplexGet3d(int seed, float x, float y, float z) {
    float n0, n1, n2, n3; // Noise contributions from the four corners

    // Skew the input space to determine which simplex cell we're in
    float s = (x + y + z)*F3; // Very nice and simple skew factor for 3D
    float xs = x + s;
    float ys = y + s;
    float zs = z + s;
    int i = vx::simd::none::floor(xs);
    int j = vx::simd::none::floor(ys);
    int k = vx::simd::none::floor(zs);

    float t = (float)(i + j + k)*G3;
    float X0 = static_cast<float>(i) - t; // Unskew the cell origin back to (x,y,z) space
    float Y0 = static_cast<float>(j) - t;
    float Z0 = static_cast<float>(k) - t;
    float x0 = x - X0; // The x,y,z distances from the cell origin
    float y0 = y - Y0;
    float z0 = z - Z0;

    // For the 3D case, the simplex shape is a slightly irregular tetrahedron.
    // Determine which simplex we are in.
    int i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
    int i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords

/* This code would benefit from a backport from the GLSL version! */
    if (x0 >= y0) {
        if (y0 >= z0)
        {
            i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
        } // X Y Z order
        else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; } // X Z Y order
        else { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; } // Z X Y order
    }
    else { // x0<y0
        if (y0 < z0) { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; } // Z Y X order
        else if (x0 < z0) { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; } // Y Z X order
        else { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; } // Y X Z order
    }

    // A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
    // a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
    // a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
    // c = 1/6.

    float x1 = x0 - static_cast<float>(i1) + G3; // Offsets for second corner in (x,y,z) coords
    float y1 = y0 - static_cast<float>(j1) + G3;
    float z1 = z0 - static_cast<float>(k1) + G3;
    float x2 = x0 - static_cast<float>(i2) + 2.0f*G3; // Offsets for third corner in (x,y,z) coords
    float y2 = y0 - static_cast<float>(j2) + 2.0f*G3;
    float z2 = z0 - static_cast<float>(k2) + 2.0f*G3;
    float x3 = x0 - 1.0f + 3.0f*G3; // Offsets for last corner in (x,y,z) coords
    float y3 = y0 - 1.0f + 3.0f*G3;
    float z3 = z0 - 1.0f + 3.0f*G3;

    // Calculate the contribution from the four corners
    float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0;
    if (t0 < 0.0f) n0 = 0.0f;
    else {
        t0 *= t0;
        n0 = t0 * t0 * grad(vx::simd::none::partialJenkinsHash(seed, i + vx::simd::none::partialJenkinsHash(seed, j + vx::simd::none::partialJenkinsHash(seed, k))), x0, y0, z0);
    }

    float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1;
    if (t1 < 0.0f) n1 = 0.0f;
    else {
        t1 *= t1;
        n1 = t1 * t1 * grad(vx::simd::none::partialJenkinsHash(seed, i + i1 + vx::simd::none::partialJenkinsHash(seed, j + j1 + vx::simd::none::partialJenkinsHash(seed, k + k1))), x1, y1, z1);
    }

    float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2;
    if (t2 < 0.0f) n2 = 0.0f;
    else {
        t2 *= t2;
        n2 = t2 * t2 * grad(vx::simd::none::partialJenkinsHash(seed, i + i2 + vx::simd::none::partialJenkinsHash(seed, j + j2 + vx::simd::none::partialJenkinsHash(seed, k + k2))), x2, y2, z2);
    }

    float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3;
    if (t3 < 0.0f) n3 = 0.0f;
    else {
        t3 *= t3;
        n3 = t3 * t3 * grad(vx::simd::none::partialJenkinsHash(seed, i + 1 + vx::simd::none::partialJenkinsHash(seed, j + 1 + vx::simd::none::partialJenkinsHash(seed, k + 1))), x3, y3, z3);
    }

    // Add contributions from each corner to get the final noise value.
    // The result is scaled to stay just inside [-1,1]
    return 32.0f * (n0 + n1 + n2 + n3); // TODO: The scale factor is preliminary!
}

float vx::noise::SimplexNormal::get2d(float x, float y, float scale) {
    return normalSimplexGet2d(_seed, x / scale, y / scale);
}

float vx::noise::SimplexNormal::get3d(float x, float y, float z, float scale) {
    return normalSimplexGet3d(_seed, x / scale, y / scale, z / scale);
}

vx::aligned_array_2d<float> vx::noise::SimplexNormal::noise(float offsetX, float offsetY,
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

vx::aligned_array_3d<float> vx::noise::SimplexNormal::noise(float offsetX, float offsetY, float offsetZ,
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
