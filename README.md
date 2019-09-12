# vx::noise

vx::noise is a basic implementation of Simplex noise, FBM Fractal Simplex noise, and Cellular noise with SIMD support. The library's main goal was to be an easier to modify replacement to 'FastNoiseSIMD' since 'FastNoiseSIMD' uses macros excessively. The library is quite verbose and can easily be used to learn SIMD intrinsics.

## Implemented Noise
 * Simplex noise using the partial jenkins hashing function from https://arxiv.org/pdf/1903.12270.pdf
 * FBM Simplex noise
 * Cellular noise

## Supported SIMD Levels
 * AVX2
 * SSE42 (can be changed to SSE41)
 * SSE2
 * NEON (planned)

## Getting Started

    // Creating a noise object is the same for all noise types. For Simplex Fractal just call `vx::noise::SimplexFractal::create`
    vx::noise::Simplex* simplex = vx::noise::Simplex::create(12, 1);
    vx::aligned_array_3d<float> noiseValues = simplex->noise(0, 0, 0, 16, 16, 16, 1);
    // Access noise values the same way you would with a normal 3D array
    float indexNoiseValue = noiseValues[0][12][3];

