#ifndef VXNOISE_ALIGNED_ARRAY_2D_HPP
#define VXNOISE_ALIGNED_ARRAY_2D_HPP

// TODO: We will have to replace _mm_malloc with an internal solution for ARM
#include <xmmintrin.h>
#include <cstdint>
#include <cstddef>
#include <tuple>
#include <stdexcept>
#include <iostream>

namespace vx {
    template<typename T>
    class aligned_array_2d;

    // Hidden indexer, NOT meant to be used explicitly
    template<typename T>
    struct _const_aligned_array_2d_indexer {
        const aligned_array_2d<T>& _array_2d;
        size_t _offsetX;

        _const_aligned_array_2d_indexer(const aligned_array_2d<T>& array_2d, size_t offsetX)
                : _array_2d(array_2d), _offsetX(offsetX) {}

        T operator[](size_t y) const {
            if (y >= _array_2d._sizeY) {
                throw std::runtime_error("Out of range index passed to aligned_array_2d!");
            }

            return _array_2d._data[y + _offsetX];
        }
    };

    // Hidden indexer, NOT meant to be used explicitly
    template<typename T>
    struct _aligned_array_2d_indexer {
        aligned_array_2d<T>& _array_2d;
        size_t _offsetX;

        _aligned_array_2d_indexer(aligned_array_2d<T>& array_2d, size_t offsetX)
                : _array_2d(array_2d), _offsetX(offsetX) {}

        T &operator[](size_t y) {
            if (y >= _array_2d._sizeY) {
                throw std::runtime_error("Out of range index passed to aligned_array_2d!");
            }

            return _array_2d._data[y + _offsetX];
        }
    };

    /**
     * Basic 2D array of type `T` that is properly aligned for each SIMD level
     *
     * @tparam T Index Type
     */
    template<typename T>
    class aligned_array_2d {
        friend _aligned_array_2d_indexer<T>;
        friend _const_aligned_array_2d_indexer<T>;

    public:
        // Make this class only movable
        // TODO: Give the ability to copy the aligned_array_2d
        aligned_array_2d(aligned_array_2d const &) = delete;
        aligned_array_2d &operator=(aligned_array_2d const &) = delete;
        aligned_array_2d &operator=(aligned_array_2d &&other) noexcept {
            // Move 'other' values to us
            _data = other._data;
            _alignment = other._alignment;
            _sizeX = other._sizeX;
            _sizeY = other._sizeY;

            // Zero out 'other' so our data isn't freed by accident
            other._data = nullptr;
            other._alignment = 0;
            other._sizeX = 0;
            other._sizeY = 0;

            return *this;
        }

        aligned_array_2d(std::size_t size, std::size_t alignment)
                : aligned_array_2d(size, size, alignment) {}

        aligned_array_2d(std::size_t sizeX, std::size_t sizeY, std::size_t alignment)
                : _sizeX(sizeX), _sizeY(sizeY), _data(nullptr), _alignment(alignment) {
            // None of the sizes can be zero else data will be null
            if (!(_sizeX == 0 || _sizeY == 0)) {
                // Allocate the underlying array using `_mm_malloc` which will properly align the pointer to how
                // the SIMD backend expects
                _data = static_cast<T*>(_mm_malloc(_sizeX * _sizeY * sizeof(T), _alignment));
            }
        }

        // Move constructor
        aligned_array_2d(aligned_array_2d &&other) noexcept {
            // Move 'other' values to us
            _data = other._data;
            _alignment = other._alignment;
            _sizeX = other._sizeX;
            _sizeY = other._sizeY;

            // Zero out 'other' so our data isn't freed by accident
            other._data = nullptr;
            other._alignment = 0;
            other._sizeX = 0;
            other._sizeY = 0;
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
                default:
                    throw std::runtime_error("Invalid axis passed to aligned_array_2d::size()!");
            }
        }

        /// Access data at an exact 2d position
        T operator[](std::tuple<std::size_t, std::size_t> index) const {
            std::size_t x = std::get<0>(index),
                    y = std::get<1>(index);

            if (isEmpty()) throw std::runtime_error("aligned_array_2d is empty!");
            if (x >= _sizeX || y >= _sizeY) {
                throw std::runtime_error("Out of range index passed to aligned_array_2d!");
            }

            return _data[y + (_sizeY * x)];
        }

        /// Access data at an exact 2d position with mutable access
        T &operator[](std::tuple<std::size_t, std::size_t> index) {
            std::size_t x = std::get<0>(index),
                    y = std::get<1>(index);

            if (isEmpty()) throw std::runtime_error("aligned_array_2d is empty!");
            if (x >= _sizeX || y >= _sizeY) {
                throw std::runtime_error("Out of range index passed to aligned_array_2d!");
            }

            return _data[y + (_sizeY * x)];
        }

        /// Access the first dimension of the 2d array
        _const_aligned_array_2d_indexer<T> operator[](size_t x) const {
            if (x >= _sizeX) {
                throw std::runtime_error("Out of range index passed to aligned_array_2d!");
            }

            return _const_aligned_array_2d_indexer<T>(*this, _sizeY * x);
        }

        /// Access the first dimension of the 2d array with mutable access
        _aligned_array_2d_indexer<T> operator[](size_t x) {
            if (x >= _sizeX) {
                throw std::runtime_error("Out of range index passed to aligned_array_2d!");
            }

            return _aligned_array_2d_indexer<T>(*this, _sizeY * x);
        }

        /// Get the 2d array as a flat pointer
        T* ptr() {
            return _data;
        }

        ~aligned_array_2d() {
            _mm_free(_data);
        }

    private:
        std::size_t _alignment;
        T *_data;
        std::size_t _sizeX;
        std::size_t _sizeY;

    };
}

#endif //VXNOISE_ALIGNED_ARRAY_2D_HPP
