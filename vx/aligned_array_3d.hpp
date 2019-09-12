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

    template<typename T>
    struct _const_aligned_array_3d_indexer_offsetY {
        const aligned_array_3d<T>& _array_2d;
        size_t _offsetY;

        _const_aligned_array_3d_indexer_offsetY(const aligned_array_3d<T>& array_2d, size_t offsetY)
                : _array_2d(array_2d), _offsetY(offsetY) {}

        T operator[](size_t z) const {
            return _array_2d._data[z + _offsetY];
        }
    };

    template<typename T>
    struct _const_aligned_array_3d_indexer_offsetX {
        const aligned_array_3d<T>& _array_2d;
        size_t _offsetX;

        _const_aligned_array_3d_indexer_offsetX(const aligned_array_3d<T>& array_2d, size_t offsetX)
                : _array_2d(array_2d), _offsetX(offsetX) {}

        const _const_aligned_array_3d_indexer_offsetY<T> operator[](size_t y) const {
            return _const_aligned_array_3d_indexer_offsetY<T>(_array_2d, _array_2d._sizeZ * (y + _offsetX));
        }
    };

    template<typename T>
    struct _aligned_array_3d_indexer_offsetY {
        aligned_array_3d<T>& _array_2d;
        size_t _offsetY;

        _aligned_array_3d_indexer_offsetY(aligned_array_3d<T>& array_2d, size_t offsetY)
                : _array_2d(array_2d), _offsetY(offsetY) {}

        T &operator[](size_t z) {
            return _array_2d._data[z + _offsetY];
        }
    };

    template<typename T>
    struct _aligned_array_3d_indexer_offsetX {
        aligned_array_3d<T>& _array_2d;
        size_t _offsetX;

        _aligned_array_3d_indexer_offsetX(aligned_array_3d<T>& array_2d, size_t offsetX)
                : _array_2d(array_2d), _offsetX(offsetX) {}

        _aligned_array_3d_indexer_offsetY<T> operator[](size_t y) {
            return _aligned_array_3d_indexer_offsetY<T>(_array_2d, _array_2d._sizeZ * (y + _offsetX));
        }
    };

    template<typename T>
    class aligned_array_3d {
        friend _aligned_array_3d_indexer_offsetX<T>;
        friend _aligned_array_3d_indexer_offsetY<T>;
        friend _const_aligned_array_3d_indexer_offsetX<T>;
        friend _const_aligned_array_3d_indexer_offsetY<T>;

    public:
        // Make this class only movable
        // TODO: Give developers the option to handle this however they want. If they want to shoot themself in the foot by copying megabytes of data every assignment let them.
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
            // Sorry I like doing weird syntax sometimes. This just sets everything to 0 if that's not easy to understand.
            other._alignment =
            other._sizeX =
            other._sizeY =
            other._sizeZ = 0;

            return *this;
        }

        explicit aligned_array_3d(std::size_t size, std::size_t alignment)
                : aligned_array_3d(size, size, size, alignment) {}

        aligned_array_3d(std::size_t sizeX, std::size_t sizeY, std::size_t sizeZ, std::size_t alignment)
                : _sizeX(sizeX), _sizeY(sizeY), _sizeZ(sizeZ), _data(nullptr), _alignment(alignment) {
            // None of the sizes can be zero else data will be null
            if (!(_sizeX == 0 || _sizeY == 0 || _sizeZ == 0)) {
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
            // Sorry I like doing weird syntax sometimes. This just sets everything to 0 if that's not easy to understand.
            other._alignment =
            other._sizeX =
            other._sizeY =
            other._sizeZ = 0;
        }

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

        T operator[](std::tuple<std::size_t, std::size_t, std::size_t> index) const {
            std::size_t x = std::get<0>(index),
                    y = std::get<1>(index),
                    z = std::get<2>(index);

            // TODO: Consider only enabling this during debug
            if (isEmpty()) throw std::runtime_error("aligned_array_3d is empty!");
            if (x < 0 || y < 0 || z < 0 || x >= _sizeX || y >= _sizeY || z >= _sizeZ) {
                std::cerr << "Error Position: [" << x << ", " << y << ", " << z << "]" << std::endl;
                throw std::runtime_error("Out of range index passed to aligned_array_3d!");
            }

            // Is this correct?
            return _data[z + _sizeZ * (y + (_sizeY * x))];
        }

        T &operator[](std::tuple<std::size_t, std::size_t, std::size_t> index) {
            std::size_t x = std::get<0>(index),
                    y = std::get<1>(index),
                    z = std::get<2>(index);

            // TODO: Consider only enabling this during debug
            if (isEmpty()) throw std::runtime_error("aligned_array_3d is empty!");
            if (x < 0 || y < 0 || z < 0 || x >= _sizeX || y >= _sizeY || z >= _sizeZ) {
                std::cerr << "Error Position: [" << x << ", " << y << ", " << z << "]" << std::endl;
                throw std::runtime_error("Out of range index passed to aligned_array_3d!");
            }

            // Is this correct?
            return _data[z + _sizeZ * (y + (_sizeY * x))];
        }

        const _const_aligned_array_3d_indexer_offsetX<T> operator[](size_t x) const {
            return _const_aligned_array_3d_indexer_offsetX<T>(*this, _sizeY * x);
        }

        _aligned_array_3d_indexer_offsetX<T> operator[](size_t x) {
            return _aligned_array_3d_indexer_offsetX<T>(*this, _sizeY * x);
        }

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
