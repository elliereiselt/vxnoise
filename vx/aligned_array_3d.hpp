#ifndef VXNOISE_ALIGNED_ARRAY_3D_HPP
#define VXNOISE_ALIGNED_ARRAY_3D_HPP

// TODO: We will have to replace _mm_malloc with an internal solution for ARM
#include <xmmintrin.h>
#include <cstdint>
#include <cstddef>
#include <tuple>
#include <stdexcept>
#include <iostream>

namespace vx {
    template<typename T>
    class aligned_array_3d;

    // Hidden indexer, NOT meant to be used explicitly
    template<typename T>
    struct _const_aligned_array_3d_indexer_offsetY {
        const aligned_array_3d<T>& _array_3d;
        size_t _offsetY;

        _const_aligned_array_3d_indexer_offsetY(const aligned_array_3d<T>& array_3d, size_t offsetY)
                : _array_3d(array_3d), _offsetY(offsetY) {}

        T operator[](size_t z) const {
            if (z >= _array_3d._sizeZ) {
                throw std::runtime_error("Out of range index passed to aligned_array_3d!");
            }

            return _array_3d._data[z + _offsetY];
        }
    };

    // Hidden indexer, NOT meant to be used explicitly
    template<typename T>
    struct _const_aligned_array_3d_indexer_offsetX {
        const aligned_array_3d<T>& _array_3d;
        size_t _offsetX;

        _const_aligned_array_3d_indexer_offsetX(const aligned_array_3d<T>& array_3d, size_t offsetX)
                : _array_3d(array_3d), _offsetX(offsetX) {}

        _const_aligned_array_3d_indexer_offsetY<T> operator[](size_t y) const {
            if (y >= _array_3d._sizeY) {
                throw std::runtime_error("Out of range index passed to aligned_array_3d!");
            }

            return _const_aligned_array_3d_indexer_offsetY<T>(_array_3d, _array_3d._sizeZ * (y + _offsetX));
        }
    };

    // Hidden indexer, NOT meant to be used explicitly
    template<typename T>
    struct _aligned_array_3d_indexer_offsetY {
        aligned_array_3d<T>& _array_3d;
        size_t _offsetY;

        _aligned_array_3d_indexer_offsetY(aligned_array_3d<T>& array_3d, size_t offsetY)
                : _array_3d(array_3d), _offsetY(offsetY) {}

        T &operator[](size_t z) {
            if (z >= _array_3d._sizeZ) {
                throw std::runtime_error("Out of range index passed to aligned_array_3d!");
            }

            return _array_3d._data[z + _offsetY];
        }
    };

    // Hidden indexer, NOT meant to be used explicitly
    template<typename T>
    struct _aligned_array_3d_indexer_offsetX {
        aligned_array_3d<T>& _array_3d;
        size_t _offsetX;

        _aligned_array_3d_indexer_offsetX(aligned_array_3d<T>& array_3d, size_t offsetX)
                : _array_3d(array_3d), _offsetX(offsetX) {}

        _aligned_array_3d_indexer_offsetY<T> operator[](size_t y) {
            if (y >= _array_3d._sizeY) {
                throw std::runtime_error("Out of range index passed to aligned_array_3d!");
            }

            return _aligned_array_3d_indexer_offsetY<T>(_array_3d, _array_3d._sizeZ * (y + _offsetX));
        }
    };

    /**
     * Basic 3D array of type `T` that is properly aligned for each SIMD level
     *
     * @tparam T Index Type
     */
    template<typename T>
    class aligned_array_3d {
        friend _aligned_array_3d_indexer_offsetX<T>;
        friend _aligned_array_3d_indexer_offsetY<T>;
        friend _const_aligned_array_3d_indexer_offsetX<T>;
        friend _const_aligned_array_3d_indexer_offsetY<T>;

    public:
        // Make this class only movable
        // TODO: Give the ability to copy the aligned_array_3d
        aligned_array_3d(aligned_array_3d const &) = delete;
        aligned_array_3d &operator=(aligned_array_3d const &) = delete;
        aligned_array_3d &operator=(aligned_array_3d &&other) noexcept {
            // Move 'other' values to us
            _data = other._data;
            _alignment = other._alignment;
            _sizeX = other._sizeX;
            _sizeY = other._sizeY;
            _sizeZ = other._sizeZ;

            // Zero out 'other' so our data isn't freed by accident
            other._data = nullptr;
            other._alignment = 0;
            other._sizeX = 0;
            other._sizeY = 0;
            other._sizeZ = 0;

            return *this;
        }

        aligned_array_3d(std::size_t size, std::size_t alignment)
                : aligned_array_3d(size, size, size, alignment) {}

        aligned_array_3d(std::size_t sizeX, std::size_t sizeY, std::size_t sizeZ, std::size_t alignment)
                : _sizeX(sizeX), _sizeY(sizeY), _sizeZ(sizeZ), _data(nullptr), _alignment(alignment) {
            // None of the sizes can be zero else data will be null
            if (!(_sizeX == 0 || _sizeY == 0 || _sizeZ == 0)) {
                // Allocate the underlying array using `_mm_malloc` which will properly align the pointer to how
                // the SIMD backend expects
                _data = static_cast<T*>(_mm_malloc(_sizeX * _sizeY * _sizeZ * sizeof(T), _alignment));
            }
        }

        // Move constructor
        aligned_array_3d(aligned_array_3d &&other) noexcept {
            // Move 'other' values to us
            _data = other._data;
            _alignment = other._alignment;
            _sizeX = other._sizeX;
            _sizeY = other._sizeY;
            _sizeZ = other._sizeZ;

            // Zero out 'other' so our data isn't freed by accident
            other._data = nullptr;
            other._alignment = 0;
            other._sizeX = 0;
            other._sizeY = 0;
            other._sizeZ = 0;
        }

        /// Check if the array has any data
        bool isEmpty() const {
            return !_data;
        }

        /// Get the size of an axis, 0th axis by default
        std::size_t size(std::uint_fast8_t axis = 0) const {
            switch (axis) {
                case 0:
                    return _sizeX;
                case 1:
                    return _sizeY;
                case 2:
                    return _sizeZ;
                default:
                    throw std::runtime_error("Invalid axis passed to aligned_array_3d::size()!");
            }
        }

        /// Access data at an exact 3d position
        T operator[](std::tuple<std::size_t, std::size_t, std::size_t> index) const {
            std::size_t x = std::get<0>(index),
                    y = std::get<1>(index),
                    z = std::get<2>(index);

            if (isEmpty()) throw std::runtime_error("aligned_array_3d is empty!");
            if (x >= _sizeX || y >= _sizeY || z >= _sizeZ) {
                throw std::runtime_error("Out of range index passed to aligned_array_3d!");
            }

            return _data[z + _sizeZ * (y + (_sizeY * x))];
        }

        /// Access data at an exact 3d position with mutable access
        T &operator[](std::tuple<std::size_t, std::size_t, std::size_t> index) {
            std::size_t x = std::get<0>(index),
                    y = std::get<1>(index),
                    z = std::get<2>(index);

            if (isEmpty()) throw std::runtime_error("aligned_array_3d is empty!");
            if (x >= _sizeX || y >= _sizeY || z >= _sizeZ) {
                throw std::runtime_error("Out of range index passed to aligned_array_3d!");
            }

            // Is this correct?
            return _data[z + _sizeZ * (y + (_sizeY * x))];
        }

        /// Access the first dimension of the 3d array
        _const_aligned_array_3d_indexer_offsetX<T> operator[](size_t x) const {
            if (x >= _sizeX) {
                throw std::runtime_error("Out of range index passed to aligned_array_3d!");
            }

            return _const_aligned_array_3d_indexer_offsetX<T>(*this, _sizeY * x);
        }

        /// Access the first dimension of the 2d array with mutable access
        _aligned_array_3d_indexer_offsetX<T> operator[](size_t x) {
            if (x >= _sizeX) {
                throw std::runtime_error("Out of range index passed to aligned_array_3d!");
            }

            return _aligned_array_3d_indexer_offsetX<T>(*this, _sizeY * x);
        }

        /// Get the 3d array as a flat pointer
        T* ptr() {
            return _data;
        }

        ~aligned_array_3d() {
            _mm_free(_data);
        }

    private:
        std::size_t _alignment;
        T *_data;
        std::size_t _sizeX;
        std::size_t _sizeY;
        std::size_t _sizeZ;

    };
}

#endif //VXNOISE_ALIGNED_ARRAY_3D_HPP
