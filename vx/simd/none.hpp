#ifndef VXNOISE_NONE_HPP
#define VXNOISE_NONE_HPP

#include "simd.hpp"

namespace vx {
    namespace simd {
        namespace none {
            inline int floor(float a) {
                if (a > 0) {
                    return static_cast<int>(a);
                } else {
                    return static_cast<int>(a) - 1;
                }
            }

            inline float dot(float x0, float y0, float x1, float y1) {
                return (x0 * x1) + (y0 * y1);
            }

            inline float dot(float x0, float y0, float z0, float x1, float y1, float z1) {
                return (x0 * x1) + (y0 * y1) + (z0 * z1);
            }

            // CREDIT: https://arxiv.org/pdf/1903.12270.pdf
            inline int partialJenkinsHash(int seed, int key) {
                int hash = seed;
                hash += key;
                hash += (hash << 10);
                hash ^= (hash >> 6);
                hash += (hash << 3);
                hash ^= (hash >> 11);
                hash += (hash << 15);
                return hash;
            }
        }
    }
}

#endif //VXNOISE_NONE_HPP
