#ifndef __AVX__
#ifdef _MSC_VER
#error Missing argument "/arch:AVX(2)"
#else
#error Missing argument "-march=core-avx2"
#endif
#endif

#define VX_NOISE_INTERNAL_DEFS
#include "SimplexAVX2.hpp"

static __m256 VX_VECTORCALL avx2_grad(__m256i hash, __m256 x, __m256 y) {
    __m256i h = _mm256_and_si256(hash, _mm256_set1_epi32(7));
    __m256 hlt4 = _mm256_cmp_ps(_mm256_cvtepi32_ps(h), _mm256_set1_ps(4), _CMP_LT_OQ);
    __m256 u = _mm256_blendv_ps(y, x, hlt4);
    __m256 v = _mm256_blendv_ps(x, y, hlt4);
    __m256 h1 = _mm256_cvtepi32_ps(_mm256_cmpeq_epi32(_mm256_and_si256(h, _mm256_set1_epi32(1)), _mm256_set1_epi32(1)));
    __m256 h2 = _mm256_cvtepi32_ps(_mm256_cmpeq_epi32(_mm256_and_si256(h, _mm256_set1_epi32(2)), _mm256_set1_epi32(2)));

    h1 = _mm256_add_ps(_mm256_mul_ps(h1, _mm256_set1_ps(2)), _mm256_set1_ps(1));
    h2 = _mm256_add_ps(_mm256_mul_ps(h2, _mm256_set1_ps(4)), _mm256_set1_ps(2));

    __m256 lValue = _mm256_mul_ps(u, h1);
    __m256 rValue = _mm256_mul_ps(v, h2);

    return _mm256_add_ps(lValue, rValue);
}

static __m256 VX_VECTORCALL avx2_grad(__m256i hash, __m256 x, __m256 y, __m256 z) {
    __m256i h = _mm256_and_si256(hash, _mm256_set1_epi32(15));     // Convert low 4 bits of hash code into 12 simple
    __m256 hlt8 = _mm256_cmp_ps(_mm256_cvtepi32_ps(h), _mm256_set1_ps(8), _CMP_LT_OQ);
    __m256 hlt4 = _mm256_cmp_ps(_mm256_cvtepi32_ps(h), _mm256_set1_ps(4), _CMP_LT_OQ);
    __m256 heq12or14 = _mm256_castsi256_ps(_mm256_or_si256(_mm256_cmpeq_epi32(h, _mm256_set1_epi32(12)), _mm256_cmpeq_epi32(h, _mm256_set1_epi32(14))));
    __m256 u = _mm256_blendv_ps(y, x, hlt8);
    __m256 v = _mm256_blendv_ps(_mm256_blendv_ps(z, x, heq12or14), y, hlt4);

    // We want to convert instead of cast here to abuse the fact that the returned mask will be all 1 on true (producing a -1 value)
    __m256 h1 = _mm256_cvtepi32_ps(_mm256_cmpeq_epi32(_mm256_and_si256(h, _mm256_set1_epi32(1)), _mm256_set1_epi32(1)));
    __m256 h2 = _mm256_cvtepi32_ps(_mm256_cmpeq_epi32(_mm256_and_si256(h, _mm256_set1_epi32(2)), _mm256_set1_epi32(2)));

    // We then change the range of h1 and h2 from [-1, 0] to [-1, 1]
    h1 = _mm256_add_ps(_mm256_mul_ps(h1, _mm256_set1_ps(2)), _mm256_set1_ps(1));
    h2 = _mm256_add_ps(_mm256_mul_ps(h2, _mm256_set1_ps(2)), _mm256_set1_ps(1));

    // Then multiply to apply the negation.
    __m256 lValue = _mm256_mul_ps(u, h1);
    __m256 rValue = _mm256_mul_ps(v, h2);

    return _mm256_add_ps(lValue, rValue);
}

__m256 VX_VECTORCALL vx::noise::avx2SimplexGet2d(__m256i seed, __m256 x, __m256 y) {
    const __m256 sse42_F2 = _mm256_set1_ps(F2);
    const __m256 sse42_G2 = _mm256_set1_ps(G2);

    __m256 n0, n1, n2; // Noise contributions from the three corners

    // Skew the input space to determine which simplex cell we're in
    __m256 s = _mm256_mul_ps(_mm256_add_ps(x, y), sse42_F2); // Hairy factor for 2D
    __m256 xs = _mm256_add_ps(x, s);
    __m256 ys = _mm256_add_ps(y, s);
    // There's no point (that I know of) to convert to an int here like in the original. It would just be a wasted operation
    __m256 i = _mm256_floor_ps(xs);
    __m256 j = _mm256_floor_ps(ys);

    __m256 t = _mm256_mul_ps(_mm256_add_ps(i, j), sse42_G2);
    __m256 X0 = _mm256_sub_ps(i, t); // Unskew the cell origin back to (x,y) space
    __m256 Y0 = _mm256_sub_ps(j, t);
    __m256 x0 = _mm256_sub_ps(x, X0); // The x,y distances from the cell origin
    __m256 y0 = _mm256_sub_ps(y, Y0);

    // For the 2D case, the simplex shape is an equilateral triangle.
    // Determine which simplex we are in.
    // Offsets for second (middle) corner of simplex in (i,j) coords
    __m256 i1 = _mm256_and_ps(_mm256_cmp_ps(x0, y0, _CMP_GT_OQ), _mm256_set1_ps(1)); // lower triangle, XY order: (0,0)->(1,0)->(1,1)
    __m256 j1 = _mm256_and_ps(_mm256_cmp_ps(x0, y0, _CMP_LE_OQ), _mm256_set1_ps(1)); // upper triangle, YX order: (0,0)->(0,1)->(1,1)

    // A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
    // a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
    // c = (3-sqrt(3))/6

    __m256 x1 = _mm256_add_ps(_mm256_sub_ps(x0, i1), sse42_G2); // Offsets for middle corner in (x,y) unskewed coords
    __m256 y1 = _mm256_add_ps(_mm256_sub_ps(y0, j1), sse42_G2);
    __m256 x2 = _mm256_add_ps(_mm256_sub_ps(x0, _mm256_set1_ps(1.0f)), _mm256_mul_ps(_mm256_set1_ps(2.0f), sse42_G2)); // Offsets for last corner in (x,y) unskewed coords
    __m256 y2 = _mm256_add_ps(_mm256_sub_ps(y0, _mm256_set1_ps(1.0f)), _mm256_mul_ps(_mm256_set1_ps(2.0f), sse42_G2));

    __m256i ii = _mm256_cvttps_epi32(i);
    __m256i jj = _mm256_cvttps_epi32(j);

    // Calculate the contribution from the three corners
    __m256 t0 = _mm256_sub_ps(_mm256_sub_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(x0, x0)), _mm256_mul_ps(y0, y0));
    __m256 t1 = _mm256_sub_ps(_mm256_sub_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(x1, x1)), _mm256_mul_ps(y1, y1));
    __m256 t2 = _mm256_sub_ps(_mm256_sub_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(x2, x2)), _mm256_mul_ps(y2, y2));

    __m256 t0gt0 = _mm256_cmp_ps(t0, _mm256_set1_ps(0.0f), _CMP_GE_OQ);
    __m256 t1gt0 = _mm256_cmp_ps(t1, _mm256_set1_ps(0.0f), _CMP_GE_OQ);
    __m256 t2gt0 = _mm256_cmp_ps(t2, _mm256_set1_ps(0.0f), _CMP_GE_OQ);

    // We perform it twice since it is basically done twice in the original algorithm
    t0 = _mm256_mul_ps(t0, t0);
    t1 = _mm256_mul_ps(t1, t1);
    t2 = _mm256_mul_ps(t2, t2);
    t0 = _mm256_mul_ps(t0, t0);
    t1 = _mm256_mul_ps(t1, t1);
    t2 = _mm256_mul_ps(t2, t2);

    __m256i i1_int = _mm256_cvttps_epi32(i1);
    __m256i j1_int = _mm256_cvttps_epi32(j1);

    __m256i ii1 = _mm256_add_epi32(ii, i1_int);
    __m256i jj1 = _mm256_add_epi32(jj, j1_int);
    __m256i ii2 = _mm256_add_epi32(ii, _mm256_set1_epi32(1));
    __m256i jj2 = _mm256_add_epi32(jj, _mm256_set1_epi32(1));

    __m256i jj0h = vx::simd::avx2::partialJenkinsHash(seed, jj);
    __m256i jj1h = vx::simd::avx2::partialJenkinsHash(seed, jj1);
    __m256i jj2h = vx::simd::avx2::partialJenkinsHash(seed, jj2);

    __m256i ii0h = vx::simd::avx2::partialJenkinsHash(seed, _mm256_add_epi32(ii, jj0h));
    __m256i ii1h = vx::simd::avx2::partialJenkinsHash(seed, _mm256_add_epi32(ii1, jj1h));
    __m256i ii2h = vx::simd::avx2::partialJenkinsHash(seed, _mm256_add_epi32(ii2, jj2h));

    __m256 g0 = avx2_grad(ii0h, x0, y0);
    __m256 g1 = avx2_grad(ii1h, x1, y1);
    __m256 g2 = avx2_grad(ii2h, x2, y2);

    n0 = _mm256_and_ps(t0gt0, _mm256_mul_ps(t0, g0));
    n1 = _mm256_and_ps(t1gt0, _mm256_mul_ps(t1, g1));
    n2 = _mm256_and_ps(t2gt0, _mm256_mul_ps(t2, g2));

    // Add contributions from each corner to get the final noise value.
    // The result is scaled to return values in the interval [-1,1].
    return _mm256_mul_ps(_mm256_set1_ps(40.0f), _mm256_add_ps(_mm256_add_ps(n0, n1), n2)); // TODO: The scale factor is preliminary!
}

__m256 VX_VECTORCALL vx::noise::avx2SimplexGet3d(__m256i seed, __m256 x, __m256 y, __m256 z) {
    const __m256 sse2_F3 = _mm256_set1_ps(F3);
    const __m256 sse2_G3 = _mm256_set1_ps(G3);

    __m256 n0, n1, n2, n3; // Noise contributions from the four corners

    // Skew the input space to determine which simplex cell we're in
    __m256 s = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(x, y), z), sse2_F3); // Very nice and simple skew factor for 3D
    __m256 xs = _mm256_add_ps(x, s);
    __m256 ys = _mm256_add_ps(y, s);
    __m256 zs = _mm256_add_ps(z, s);
    __m256 i = _mm256_floor_ps(xs);
    __m256 j = _mm256_floor_ps(ys);
    __m256 k = _mm256_floor_ps(zs);

    __m256 t = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(i, j), k), sse2_G3);
    __m256 X0 = _mm256_sub_ps(i, t); // Unskew the cell origin back to (x,y,z) space
    __m256 Y0 = _mm256_sub_ps(j, t);
    __m256 Z0 = _mm256_sub_ps(k, t);
    __m256 x0 = _mm256_sub_ps(x, X0); // The x,y,z distances from the cell origin
    __m256 y0 = _mm256_sub_ps(y, Y0);
    __m256 z0 = _mm256_sub_ps(z, Z0);

    // For the 3D case, the simplex shape is a slightly irregular tetrahedron.
    // Determine which simplex we are in.
    __m256 i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
    __m256 i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords

    __m256i x0gey0 = _mm256_castps_si256(_mm256_cmp_ps(x0, y0, _CMP_GE_OQ));
    __m256i y0gez0 = _mm256_castps_si256(_mm256_cmp_ps(y0, z0, _CMP_GE_OQ));
    __m256i x0gez0 = _mm256_castps_si256(_mm256_cmp_ps(x0, z0, _CMP_GE_OQ));

    /* This code would benefit from a backport from the GLSL version! */
    // TODO: Is this right?
    __m256i i1m = (_mm256_and_si256(x0gey0, _mm256_or_si256(y0gez0, x0gez0)));
    __m256i j1m = (_mm256_andnot_si256(x0gey0, y0gez0));
    __m256i k1m = (_mm256_or_si256(_mm256_xor_si256(_mm256_or_si256(x0gey0, y0gez0), _mm256_set1_epi32(static_cast<int>(0xFFFFFFFF))),
                                   _mm256_xor_si256(_mm256_or_si256(y0gez0, x0gez0), _mm256_set1_epi32(static_cast<int>(0xFFFFFFFF)))));
    __m256i i2m = (_mm256_or_si256(x0gey0, _mm256_and_si256(y0gez0, x0gez0)));
    __m256i j2m = (_mm256_or_si256(_mm256_and_si256(x0gey0, y0gez0), _mm256_xor_si256(x0gey0, _mm256_set1_epi32(static_cast<int>(0xFFFFFFFF)))));
    __m256i k2m = (_mm256_or_si256(_mm256_xor_si256(_mm256_or_si256(x0gey0, x0gez0), _mm256_set1_epi32(static_cast<int>(0xFFFFFFFF))), _mm256_xor_si256(y0gez0, _mm256_set1_epi32(static_cast<int>(0xFFFFFFFF)))));

    i1 = _mm256_cvtepi32_ps(_mm256_and_si256(i1m, _mm256_set1_epi32(1)));
    j1 = _mm256_cvtepi32_ps(_mm256_and_si256(j1m, _mm256_set1_epi32(1)));
    k1 = _mm256_cvtepi32_ps(_mm256_and_si256(k1m, _mm256_set1_epi32(1)));
    i2 = _mm256_cvtepi32_ps(_mm256_and_si256(i2m, _mm256_set1_epi32(1)));
    j2 = _mm256_cvtepi32_ps(_mm256_and_si256(j2m, _mm256_set1_epi32(1)));
    k2 = _mm256_cvtepi32_ps(_mm256_and_si256(k2m, _mm256_set1_epi32(1)));

    // A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
    // a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
    // a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
    // c = 1/6.

    __m256 x1 = _mm256_add_ps(_mm256_sub_ps(x0, i1), sse2_G3); // Offsets for second corner in (x,y,z) coords
    __m256 y1 = _mm256_add_ps(_mm256_sub_ps(y0, j1), sse2_G3);
    __m256 z1 = _mm256_add_ps(_mm256_sub_ps(z0, k1), sse2_G3);
    __m256 x2 = _mm256_add_ps(_mm256_sub_ps(x0, i2), _mm256_mul_ps(_mm256_set1_ps(2.0f), sse2_G3)); // Offsets for third corner in (x,y,z) coords
    __m256 y2 = _mm256_add_ps(_mm256_sub_ps(y0, j2), _mm256_mul_ps(_mm256_set1_ps(2.0f), sse2_G3));
    __m256 z2 = _mm256_add_ps(_mm256_sub_ps(z0, k2), _mm256_mul_ps(_mm256_set1_ps(2.0f), sse2_G3));
    __m256 x3 = _mm256_add_ps(_mm256_sub_ps(x0, _mm256_set1_ps(1.0f)), _mm256_mul_ps(_mm256_set1_ps(3.0f), sse2_G3)); // Offsets for last corner in (x,y,z) coords
    __m256 y3 = _mm256_add_ps(_mm256_sub_ps(y0, _mm256_set1_ps(1.0f)), _mm256_mul_ps(_mm256_set1_ps(3.0f), sse2_G3));
    __m256 z3 = _mm256_add_ps(_mm256_sub_ps(z0, _mm256_set1_ps(1.0f)), _mm256_mul_ps(_mm256_set1_ps(3.0f), sse2_G3));

    __m256i ii = _mm256_cvttps_epi32(i);
    __m256i jj = _mm256_cvttps_epi32(j);
    __m256i kk = _mm256_cvttps_epi32(k);

    // Calculate the contribution from the four corners
    __m256 t0 = _mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_set1_ps(0.6f), _mm256_mul_ps(x0, x0)), _mm256_mul_ps(y0, y0)), _mm256_mul_ps(z0, z0));
    __m256 t1 = _mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_set1_ps(0.6f), _mm256_mul_ps(x1, x1)), _mm256_mul_ps(y1, y1)), _mm256_mul_ps(z1, z1));
    __m256 t2 = _mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_set1_ps(0.6f), _mm256_mul_ps(x2, x2)), _mm256_mul_ps(y2, y2)), _mm256_mul_ps(z2, z2));
    __m256 t3 = _mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_set1_ps(0.6f), _mm256_mul_ps(x3, x3)), _mm256_mul_ps(y3, y3)), _mm256_mul_ps(z3, z3));

    __m256 t0gt0 = _mm256_cmp_ps(t0, _mm256_set1_ps(0.0f), _CMP_GE_OQ);
    __m256 t1gt0 = _mm256_cmp_ps(t1, _mm256_set1_ps(0.0f), _CMP_GE_OQ);
    __m256 t2gt0 = _mm256_cmp_ps(t2, _mm256_set1_ps(0.0f), _CMP_GE_OQ);
    __m256 t3gt0 = _mm256_cmp_ps(t3, _mm256_set1_ps(0.0f), _CMP_GE_OQ);

    // We perform it twice since it is basically done twice in the original algorithm
    t0 = _mm256_mul_ps(t0, t0);
    t1 = _mm256_mul_ps(t1, t1);
    t2 = _mm256_mul_ps(t2, t2);
    t3 = _mm256_mul_ps(t3, t3);
    t0 = _mm256_mul_ps(t0, t0);
    t1 = _mm256_mul_ps(t1, t1);
    t2 = _mm256_mul_ps(t2, t2);
    t3 = _mm256_mul_ps(t3, t3);

    __m256i i1_int = _mm256_cvttps_epi32(i1);
    __m256i j1_int = _mm256_cvttps_epi32(j1);
    __m256i k1_int = _mm256_cvttps_epi32(k1);
    __m256i i2_int = _mm256_cvttps_epi32(i2);
    __m256i j2_int = _mm256_cvttps_epi32(j2);
    __m256i k2_int = _mm256_cvttps_epi32(k2);

    __m256i ii1 = _mm256_add_epi32(ii, i1_int);
    __m256i jj1 = _mm256_add_epi32(jj, j1_int);
    __m256i kk1 = _mm256_add_epi32(kk, k1_int);
    __m256i ii2 = _mm256_add_epi32(ii, i2_int);
    __m256i jj2 = _mm256_add_epi32(jj, j2_int);
    __m256i kk2 = _mm256_add_epi32(kk, k2_int);
    __m256i ii3 = _mm256_add_epi32(ii, _mm256_set1_epi32(1));
    __m256i jj3 = _mm256_add_epi32(jj, _mm256_set1_epi32(1));
    __m256i kk3 = _mm256_add_epi32(kk, _mm256_set1_epi32(1));

    __m256i kk0h = vx::simd::avx2::partialJenkinsHash(seed, kk);
    __m256i kk1h = vx::simd::avx2::partialJenkinsHash(seed, kk1);
    __m256i kk2h = vx::simd::avx2::partialJenkinsHash(seed, kk2);
    __m256i kk3h = vx::simd::avx2::partialJenkinsHash(seed, kk3);

    __m256i jj0h = vx::simd::avx2::partialJenkinsHash(seed, _mm256_add_epi32(jj, kk0h));
    __m256i jj1h = vx::simd::avx2::partialJenkinsHash(seed, _mm256_add_epi32(jj1, kk1h));
    __m256i jj2h = vx::simd::avx2::partialJenkinsHash(seed, _mm256_add_epi32(jj2, kk2h));
    __m256i jj3h = vx::simd::avx2::partialJenkinsHash(seed, _mm256_add_epi32(jj3, kk3h));

    __m256i ii0h = vx::simd::avx2::partialJenkinsHash(seed, _mm256_add_epi32(ii, jj0h));
    __m256i ii1h = vx::simd::avx2::partialJenkinsHash(seed, _mm256_add_epi32(ii1, jj1h));
    __m256i ii2h = vx::simd::avx2::partialJenkinsHash(seed, _mm256_add_epi32(ii2, jj2h));
    __m256i ii3h = vx::simd::avx2::partialJenkinsHash(seed, _mm256_add_epi32(ii3, jj3h));

    __m256 g0 = avx2_grad(ii0h, x0, y0, z0);
    __m256 g1 = avx2_grad(ii1h, x1, y1, z1);
    __m256 g2 = avx2_grad(ii2h, x2, y2, z2);
    __m256 g3 = avx2_grad(ii3h, x3, y3, z3);

    n0 = _mm256_and_ps(t0gt0, _mm256_mul_ps(t0, g0));
    n1 = _mm256_and_ps(t1gt0, _mm256_mul_ps(t1, g1));
    n2 = _mm256_and_ps(t2gt0, _mm256_mul_ps(t2, g2));
    n3 = _mm256_and_ps(t3gt0, _mm256_mul_ps(t3, g3));

    // Add contributions from each corner to get the final noise value.
    // The result is scaled to stay just inside [-1,1]
    return _mm256_mul_ps(_mm256_set1_ps(32.0f), _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(n0, n1), n2), n3)); // TODO: The scale factor is preliminary!
}

__m256 VX_VECTORCALL vx::noise::SimplexAVX2::avx2Get2d(__m256 x, __m256 y, __m256 scale) {
    return avx2SimplexGet2d(_avx2Seed, _mm256_div_ps(x, scale), _mm256_div_ps(y, scale));
}

float vx::noise::SimplexAVX2::get2d(float x, float y, float scale) {
    __m256 avx2Result = avx2SimplexGet2d(_avx2Seed,
                                         _mm256_div_ps(_mm256_set1_ps(x), _mm256_set1_ps(scale)),
                                         _mm256_div_ps(_mm256_set1_ps(y), _mm256_set1_ps(scale)));
    alignas(32) float pResult[8];
    _mm256_store_ps(pResult, avx2Result);
    return pResult[0];
}

__m256 VX_VECTORCALL vx::noise::SimplexAVX2::avx2Get3d(__m256 x, __m256 y, __m256 z, __m256 scale) {
    return avx2SimplexGet3d(_avx2Seed, _mm256_div_ps(x, scale), _mm256_div_ps(y, scale), _mm256_div_ps(z, scale));
}

float vx::noise::SimplexAVX2::get3d(float x, float y, float z, float scale) {
    __m256 avx2Result = avx2SimplexGet3d(_avx2Seed,
                                         _mm256_div_ps(_mm256_set1_ps(x), _mm256_set1_ps(scale)),
                                         _mm256_div_ps(_mm256_set1_ps(y), _mm256_set1_ps(scale)),
                                         _mm256_div_ps(_mm256_set1_ps(z), _mm256_set1_ps(scale)));
    alignas(32) float pResult[8];
    _mm256_store_ps(pResult, avx2Result);
    return pResult[0];
}

vx::aligned_array_2d<float> vx::noise::SimplexAVX2::noise(float offsetX, float offsetY,
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

vx::aligned_array_3d<float> vx::noise::SimplexAVX2::noise(float offsetX, float offsetY, float offsetZ,
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
                                        _mm256_add_ps(vecOffsetZ, _mm256_mul_ps(vecZ, vecStep)),
                                        vecScale);

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
                                    _mm256_add_ps(vecOffsetZ, _mm256_mul_ps(vecZ, vecStep)),
                                    vecScale);

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
