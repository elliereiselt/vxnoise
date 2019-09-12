#define VX_NOISE_INTERNAL_DEFS
#include "../../simd/sse42.hpp"
#include "SimplexSSE42.hpp"

static __m128 VX_VECTORCALL sse42_grad(__m128i hash, __m128 x, __m128 y) {
    __m128i h = _mm_and_si128(hash, _mm_set1_epi32(7));
    __m128 hlt4 = _mm_cmplt_ps(_mm_cvtepi32_ps(h), _mm_set1_ps(4));
    __m128 u = _mm_blendv_ps(y, x, hlt4);
    __m128 v = _mm_blendv_ps(x, y, hlt4);

    __m128 h1 = _mm_cvtepi32_ps(_mm_cmpeq_epi32(_mm_and_si128(h, _mm_set1_epi32(1)), _mm_set1_epi32(1)));
    __m128 h2 = _mm_cvtepi32_ps(_mm_cmpeq_epi32(_mm_and_si128(h, _mm_set1_epi32(2)), _mm_set1_epi32(2)));

    h1 = _mm_add_ps(_mm_mul_ps(h1, _mm_set1_ps(2)), _mm_set1_ps(1));
    h2 = _mm_add_ps(_mm_mul_ps(h2, _mm_set1_ps(4)), _mm_set1_ps(2));

    return _mm_add_ps(_mm_mul_ps(u, h1), _mm_mul_ps(v, h2));
}

static __m128 VX_VECTORCALL sse42_grad(__m128i hash, __m128 x, __m128 y, __m128 z) {
    __m128i h = _mm_and_si128(hash, _mm_set1_epi32(15));     // Convert low 4 bits of hash code into 12 simple
    __m128 hlt8 = _mm_castsi128_ps(_mm_cmplt_epi32(h, _mm_set1_epi32(8)));
    __m128 hlt4 = _mm_castsi128_ps(_mm_cmplt_epi32(h, _mm_set1_epi32(4)));
    __m128 heq12or14 = _mm_castsi128_ps(_mm_or_si128(_mm_cmpeq_epi32(h, _mm_set1_epi32(12)), _mm_cmpeq_epi32(h, _mm_set1_epi32(14))));
    __m128 u = _mm_blendv_ps(y, x, hlt8);
    __m128 v = _mm_blendv_ps(_mm_blendv_ps(z, x, heq12or14), y, hlt4);

    // We want to convert instead of cast here to abuse the fact that the returned mask will be all 1 on true (producing a -1 value)
    __m128 h1 = _mm_cvtepi32_ps(_mm_cmpeq_epi32(_mm_and_si128(h, _mm_set1_epi32(1)), _mm_set1_epi32(1)));
    __m128 h2 = _mm_cvtepi32_ps(_mm_cmpeq_epi32(_mm_and_si128(h, _mm_set1_epi32(2)), _mm_set1_epi32(2)));

    // We then change the range of h1 and h2 from [-1, 0] to [-1, 1]
    h1 = _mm_add_ps(_mm_mul_ps(h1, _mm_set1_ps(2)), _mm_set1_ps(1));
    h2 = _mm_add_ps(_mm_mul_ps(h2, _mm_set1_ps(2)), _mm_set1_ps(1));

    // Then multiply to apply the negation.
    return _mm_add_ps(_mm_mul_ps(u, h1), _mm_mul_ps(v, h2));
}

__m128 VX_VECTORCALL vx::noise::sse42SimplexGet2d(__m128i seed, __m128 x, __m128 y) {
    const __m128 sse42_F2 = _mm_set1_ps(F2);
    const __m128 sse42_G2 = _mm_set1_ps(G2);

    __m128 n0, n1, n2; // Noise contributions from the three corners

    // Skew the input space to determine which simplex cell we're in
    __m128 s = _mm_mul_ps(_mm_add_ps(x, y), sse42_F2); // Hairy factor for 2D
    __m128 xs = _mm_add_ps(x, s);
    __m128 ys = _mm_add_ps(y, s);
    // There's no point (that I know of) to convert to an int here like in the original. It would just be a wasted operation
    __m128 i = _mm_floor_ps(xs);
    __m128 j = _mm_floor_ps(ys);

    __m128 t = _mm_mul_ps(_mm_add_ps(i, j), sse42_G2);
    __m128 X0 = _mm_sub_ps(i, t); // Unskew the cell origin back to (x,y) space
    __m128 Y0 = _mm_sub_ps(j, t);
    __m128 x0 = _mm_sub_ps(x, X0); // The x,y distances from the cell origin
    __m128 y0 = _mm_sub_ps(y, Y0);

    // For the 2D case, the simplex shape is an equilateral triangle.
    // Determine which simplex we are in.
    // Offsets for second (middle) corner of simplex in (i,j) coords
    __m128 i1 = _mm_and_ps(_mm_cmpgt_ps(x0, y0), _mm_set1_ps(1)); // lower triangle, XY order: (0,0)->(1,0)->(1,1)
    __m128 j1 = _mm_and_ps(_mm_cmple_ps(x0, y0), _mm_set1_ps(1)); // upper triangle, YX order: (0,0)->(0,1)->(1,1)

    // A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
    // a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
    // c = (3-sqrt(3))/6

    __m128 x1 = _mm_add_ps(_mm_sub_ps(x0, i1), sse42_G2); // Offsets for middle corner in (x,y) unskewed coords
    __m128 y1 = _mm_add_ps(_mm_sub_ps(y0, j1), sse42_G2);
    __m128 x2 = _mm_add_ps(_mm_sub_ps(x0, _mm_set1_ps(1.0f)), _mm_mul_ps(_mm_set1_ps(2.0f), sse42_G2)); // Offsets for last corner in (x,y) unskewed coords
    __m128 y2 = _mm_add_ps(_mm_sub_ps(y0, _mm_set1_ps(1.0f)), _mm_mul_ps(_mm_set1_ps(2.0f), sse42_G2));

    __m128i ii = _mm_cvttps_epi32(i);
    __m128i jj = _mm_cvttps_epi32(j);

    // Calculate the contribution from the three corners
    __m128 t0 = _mm_sub_ps(_mm_sub_ps(_mm_set1_ps(0.5f), _mm_mul_ps(x0, x0)), _mm_mul_ps(y0, y0));
    __m128 t1 = _mm_sub_ps(_mm_sub_ps(_mm_set1_ps(0.5f), _mm_mul_ps(x1, x1)), _mm_mul_ps(y1, y1));
    __m128 t2 = _mm_sub_ps(_mm_sub_ps(_mm_set1_ps(0.5f), _mm_mul_ps(x2, x2)), _mm_mul_ps(y2, y2));

    __m128 t0gt0 = _mm_cmpge_ps(t0, _mm_set1_ps(0.0f));
    __m128 t1gt0 = _mm_cmpge_ps(t1, _mm_set1_ps(0.0f));
    __m128 t2gt0 = _mm_cmpge_ps(t2, _mm_set1_ps(0.0f));

    // We perform it twice since it is basically done twice in the original algorithm
    t0 = _mm_mul_ps(t0, t0);
    t1 = _mm_mul_ps(t1, t1);
    t2 = _mm_mul_ps(t2, t2);
    t0 = _mm_mul_ps(t0, t0);
    t1 = _mm_mul_ps(t1, t1);
    t2 = _mm_mul_ps(t2, t2);

    __m128i i1_int = _mm_cvttps_epi32(i1);
    __m128i j1_int = _mm_cvttps_epi32(j1);

    __m128i ii1 = _mm_add_epi32(ii, i1_int);
    __m128i jj1 = _mm_add_epi32(jj, j1_int);
    __m128i ii2 = _mm_add_epi32(ii, _mm_set1_epi32(1));
    __m128i jj2 = _mm_add_epi32(jj, _mm_set1_epi32(1));

    __m128i jj0h = vx::simd::sse42::partialJenkinsHash(seed, jj);
    __m128i jj1h = vx::simd::sse42::partialJenkinsHash(seed, jj1);
    __m128i jj2h = vx::simd::sse42::partialJenkinsHash(seed, jj2);

    __m128i ii0h = vx::simd::sse42::partialJenkinsHash(seed, _mm_add_epi32(ii, jj0h));
    __m128i ii1h = vx::simd::sse42::partialJenkinsHash(seed, _mm_add_epi32(ii1, jj1h));
    __m128i ii2h = vx::simd::sse42::partialJenkinsHash(seed, _mm_add_epi32(ii2, jj2h));

    __m128 g0 = sse42_grad(ii0h, x0, y0);
    __m128 g1 = sse42_grad(ii1h, x1, y1);
    __m128 g2 = sse42_grad(ii2h, x2, y2);

    n0 = _mm_and_ps(t0gt0, _mm_mul_ps(t0, g0));
    n1 = _mm_and_ps(t1gt0, _mm_mul_ps(t1, g1));
    n2 = _mm_and_ps(t2gt0, _mm_mul_ps(t2, g2));

    // Add contributions from each corner to get the final noise value.
    // The result is scaled to return values in the interval [-1,1].
    return _mm_mul_ps(_mm_set1_ps(40.0f), _mm_add_ps(_mm_add_ps(n0, n1), n2)); // TODO: The scale factor is preliminary!
}

__m128 VX_VECTORCALL vx::noise::sse42SimplexGet3d(__m128i seed, __m128 x, __m128 y, __m128 z) {
    const __m128 sse2_F3 = _mm_set1_ps(F3);
    const __m128 sse2_G3 = _mm_set1_ps(G3);

    __m128 n0, n1, n2, n3; // Noise contributions from the four corners

    // Skew the input space to determine which simplex cell we're in
    __m128 s = _mm_mul_ps(_mm_add_ps(_mm_add_ps(x, y), z), sse2_F3); // Very nice and simple skew factor for 3D
    __m128 xs = _mm_add_ps(x, s);
    __m128 ys = _mm_add_ps(y, s);
    __m128 zs = _mm_add_ps(z, s);
    __m128 i = _mm_floor_ps(xs);
    __m128 j = _mm_floor_ps(ys);
    __m128 k = _mm_floor_ps(zs);

    __m128 t = _mm_mul_ps(_mm_add_ps(_mm_add_ps(i, j), k), sse2_G3);
    __m128 X0 = _mm_sub_ps(i, t); // Unskew the cell origin back to (x,y,z) space
    __m128 Y0 = _mm_sub_ps(j, t);
    __m128 Z0 = _mm_sub_ps(k, t);
    __m128 x0 = _mm_sub_ps(x, X0); // The x,y,z distances from the cell origin
    __m128 y0 = _mm_sub_ps(y, Y0);
    __m128 z0 = _mm_sub_ps(z, Z0);

    // For the 3D case, the simplex shape is a slightly irregular tetrahedron.
    // Determine which simplex we are in.
    __m128 i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
    __m128 i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords

    __m128i x0gey0 = _mm_castps_si128(_mm_cmpge_ps(x0, y0));
    __m128i y0gez0 = _mm_castps_si128(_mm_cmpge_ps(y0, z0));
    __m128i x0gez0 = _mm_castps_si128(_mm_cmpge_ps(x0, z0));

    /* This code would benefit from a backport from the GLSL version! */
    // TODO: Is this right?
    __m128i i1m = (_mm_and_si128(x0gey0, _mm_or_si128(y0gez0, x0gez0)));
    __m128i j1m = (_mm_andnot_si128(x0gey0, y0gez0));
    __m128i k1m = (_mm_or_si128(_mm_xor_si128(_mm_or_si128(x0gey0, y0gez0), _mm_set1_epi32(static_cast<int>(0xFFFFFFFF))),
                                _mm_xor_si128(_mm_or_si128(y0gez0, x0gez0), _mm_set1_epi32(static_cast<int>(0xFFFFFFFF)))));
    __m128i i2m = (_mm_or_si128(x0gey0, _mm_and_si128(y0gez0, x0gez0)));
    __m128i j2m = (_mm_or_si128(_mm_and_si128(x0gey0, y0gez0), _mm_xor_si128(x0gey0, _mm_set1_epi32(static_cast<int>(0xFFFFFFFF)))));
    __m128i k2m = (_mm_or_si128(_mm_xor_si128(_mm_or_si128(x0gey0, x0gez0), _mm_set1_epi32(static_cast<int>(0xFFFFFFFF))), _mm_xor_si128(y0gez0, _mm_set1_epi32(static_cast<int>(0xFFFFFFFF)))));

    i1 = _mm_cvtepi32_ps(_mm_and_si128(i1m, _mm_set1_epi32(1)));
    j1 = _mm_cvtepi32_ps(_mm_and_si128(j1m, _mm_set1_epi32(1)));
    k1 = _mm_cvtepi32_ps(_mm_and_si128(k1m, _mm_set1_epi32(1)));
    i2 = _mm_cvtepi32_ps(_mm_and_si128(i2m, _mm_set1_epi32(1)));
    j2 = _mm_cvtepi32_ps(_mm_and_si128(j2m, _mm_set1_epi32(1)));
    k2 = _mm_cvtepi32_ps(_mm_and_si128(k2m, _mm_set1_epi32(1)));

    // A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
    // a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
    // a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
    // c = 1/6.

    __m128 x1 = _mm_add_ps(_mm_sub_ps(x0, i1), sse2_G3); // Offsets for second corner in (x,y,z) coords
    __m128 y1 = _mm_add_ps(_mm_sub_ps(y0, j1), sse2_G3);
    __m128 z1 = _mm_add_ps(_mm_sub_ps(z0, k1), sse2_G3);
    __m128 x2 = _mm_add_ps(_mm_sub_ps(x0, i2), _mm_mul_ps(_mm_set1_ps(2.0f), sse2_G3)); // Offsets for third corner in (x,y,z) coords
    __m128 y2 = _mm_add_ps(_mm_sub_ps(y0, j2), _mm_mul_ps(_mm_set1_ps(2.0f), sse2_G3));
    __m128 z2 = _mm_add_ps(_mm_sub_ps(z0, k2), _mm_mul_ps(_mm_set1_ps(2.0f), sse2_G3));
    __m128 x3 = _mm_add_ps(_mm_sub_ps(x0, _mm_set1_ps(1.0f)), _mm_mul_ps(_mm_set1_ps(3.0f), sse2_G3)); // Offsets for last corner in (x,y,z) coords
    __m128 y3 = _mm_add_ps(_mm_sub_ps(y0, _mm_set1_ps(1.0f)), _mm_mul_ps(_mm_set1_ps(3.0f), sse2_G3));
    __m128 z3 = _mm_add_ps(_mm_sub_ps(z0, _mm_set1_ps(1.0f)), _mm_mul_ps(_mm_set1_ps(3.0f), sse2_G3));

    __m128i ii = _mm_cvttps_epi32(i);
    __m128i jj = _mm_cvttps_epi32(j);
    __m128i kk = _mm_cvttps_epi32(k);

    // Calculate the contribution from the four corners
    __m128 t0 = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(_mm_set1_ps(0.6f), _mm_mul_ps(x0, x0)), _mm_mul_ps(y0, y0)), _mm_mul_ps(z0, z0));
    __m128 t1 = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(_mm_set1_ps(0.6f), _mm_mul_ps(x1, x1)), _mm_mul_ps(y1, y1)), _mm_mul_ps(z1, z1));
    __m128 t2 = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(_mm_set1_ps(0.6f), _mm_mul_ps(x2, x2)), _mm_mul_ps(y2, y2)), _mm_mul_ps(z2, z2));
    __m128 t3 = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(_mm_set1_ps(0.6f), _mm_mul_ps(x3, x3)), _mm_mul_ps(y3, y3)), _mm_mul_ps(z3, z3));

    __m128 t0gt0 = _mm_cmpge_ps(t0, _mm_set1_ps(0.0f));
    __m128 t1gt0 = _mm_cmpge_ps(t1, _mm_set1_ps(0.0f));
    __m128 t2gt0 = _mm_cmpge_ps(t2, _mm_set1_ps(0.0f));
    __m128 t3gt0 = _mm_cmpge_ps(t3, _mm_set1_ps(0.0f));

    // We perform it twice since it is basically done twice in the original algorithm
    t0 = _mm_mul_ps(t0, t0);
    t1 = _mm_mul_ps(t1, t1);
    t2 = _mm_mul_ps(t2, t2);
    t3 = _mm_mul_ps(t3, t3);
    t0 = _mm_mul_ps(t0, t0);
    t1 = _mm_mul_ps(t1, t1);
    t2 = _mm_mul_ps(t2, t2);
    t3 = _mm_mul_ps(t3, t3);

    __m128i i1_int = _mm_cvttps_epi32(i1);
    __m128i j1_int = _mm_cvttps_epi32(j1);
    __m128i k1_int = _mm_cvttps_epi32(k1);
    __m128i i2_int = _mm_cvttps_epi32(i2);
    __m128i j2_int = _mm_cvttps_epi32(j2);
    __m128i k2_int = _mm_cvttps_epi32(k2);

    __m128i ii1 = _mm_add_epi32(ii, i1_int);
    __m128i jj1 = _mm_add_epi32(jj, j1_int);
    __m128i kk1 = _mm_add_epi32(kk, k1_int);
    __m128i ii2 = _mm_add_epi32(ii, i2_int);
    __m128i jj2 = _mm_add_epi32(jj, j2_int);
    __m128i kk2 = _mm_add_epi32(kk, k2_int);
    __m128i ii3 = _mm_add_epi32(ii, _mm_set1_epi32(1));
    __m128i jj3 = _mm_add_epi32(jj, _mm_set1_epi32(1));
    __m128i kk3 = _mm_add_epi32(kk, _mm_set1_epi32(1));

    __m128i kk0h = vx::simd::sse42::partialJenkinsHash(seed, kk);
    __m128i kk1h = vx::simd::sse42::partialJenkinsHash(seed, kk1);
    __m128i kk2h = vx::simd::sse42::partialJenkinsHash(seed, kk2);
    __m128i kk3h = vx::simd::sse42::partialJenkinsHash(seed, kk3);

    __m128i jj0h = vx::simd::sse42::partialJenkinsHash(seed, _mm_add_epi32(jj, kk0h));
    __m128i jj1h = vx::simd::sse42::partialJenkinsHash(seed, _mm_add_epi32(jj1, kk1h));
    __m128i jj2h = vx::simd::sse42::partialJenkinsHash(seed, _mm_add_epi32(jj2, kk2h));
    __m128i jj3h = vx::simd::sse42::partialJenkinsHash(seed, _mm_add_epi32(jj3, kk3h));

    __m128i ii0h = vx::simd::sse42::partialJenkinsHash(seed, _mm_add_epi32(ii, jj0h));
    __m128i ii1h = vx::simd::sse42::partialJenkinsHash(seed, _mm_add_epi32(ii1, jj1h));
    __m128i ii2h = vx::simd::sse42::partialJenkinsHash(seed, _mm_add_epi32(ii2, jj2h));
    __m128i ii3h = vx::simd::sse42::partialJenkinsHash(seed, _mm_add_epi32(ii3, jj3h));

    __m128 g0 = sse42_grad(ii0h, x0, y0, z0);
    __m128 g1 = sse42_grad(ii1h, x1, y1, z1);
    __m128 g2 = sse42_grad(ii2h, x2, y2, z2);
    __m128 g3 = sse42_grad(ii3h, x3, y3, z3);

    n0 = _mm_and_ps(t0gt0, _mm_mul_ps(t0, g0));
    n1 = _mm_and_ps(t1gt0, _mm_mul_ps(t1, g1));
    n2 = _mm_and_ps(t2gt0, _mm_mul_ps(t2, g2));
    n3 = _mm_and_ps(t3gt0, _mm_mul_ps(t3, g3));

    // Add contributions from each corner to get the final noise value.
    // The result is scaled to stay just inside [-1,1]
    return _mm_mul_ps(_mm_set1_ps(32.0f), _mm_add_ps(_mm_add_ps(_mm_add_ps(n0, n1), n2), n3)); // TODO: The scale factor is preliminary!
}

__m128 VX_VECTORCALL vx::noise::SimplexSSE42::sse42Get2d(__m128 x, __m128 y, __m128 scale) {
    return sse42SimplexGet2d(_sse42Seed, _mm_div_ps(x, scale), _mm_div_ps(y, scale));
}

float vx::noise::SimplexSSE42::get2d(float x, float y, float scale) {
    __m128 sse42Result = sse42SimplexGet2d(_sse42Seed,
                                           _mm_div_ps(_mm_set1_ps(x), _mm_set1_ps(scale)),
                                           _mm_div_ps(_mm_set1_ps(y), _mm_set1_ps(scale)));
    alignas(16) float pResult[4];
    _mm_store_ps(pResult, sse42Result);
    return pResult[0];
}

__m128 VX_VECTORCALL vx::noise::SimplexSSE42::sse42Get3d(__m128 x, __m128 y, __m128 z, __m128 scale) {
    return sse42SimplexGet3d(_sse42Seed, _mm_div_ps(x, scale), _mm_div_ps(y, scale), _mm_div_ps(z, scale));
}

float vx::noise::SimplexSSE42::get3d(float x, float y, float z, float scale) {
    __m128 sse42Result = sse42SimplexGet3d(_sse42Seed,
                                           _mm_div_ps(_mm_set1_ps(x), _mm_set1_ps(scale)),
                                           _mm_div_ps(_mm_set1_ps(y), _mm_set1_ps(scale)),
                                           _mm_div_ps(_mm_set1_ps(z), _mm_set1_ps(scale)));
    alignas(16) float pResult[4];
    _mm_store_ps(pResult, sse42Result);
    return pResult[0];
}

vx::aligned_array_2d<float> vx::noise::SimplexSSE42::noise(float offsetX, float offsetY,
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

vx::aligned_array_3d<float> vx::noise::SimplexSSE42::noise(float offsetX, float offsetY, float offsetZ,
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
