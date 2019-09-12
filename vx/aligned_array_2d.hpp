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
    class aligned_array_2d {
    public:
        // Make this class only movable
        // TODO: Give developers the option to handle this however they want. If they want to shoot themself in the foot by copying megabytes of data every assignment let them.
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
            // Sorry I like doing weird syntax sometimes. This just sets everything to 0 if that's not easy to understand.
            other._alignment =
            other._sizeX =
            other._sizeY = 0;

            return *this;
        }

        explicit aligned_array_2d(std::size_t size, std::size_t alignment)
                : aligned_array_2d(size, size, alignment) {}

        aligned_array_2d(std::size_t sizeX, std::size_t sizeY, std::size_t alignment)
                : _sizeX(sizeX), _sizeY(sizeY), _data(nullptr), _alignment(alignment) {
            // None of the sizes can be zero else data will be null
            if (!(_sizeX == 0 || _sizeY == 0)) {
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
            // Sorry I like doing weird syntax sometimes. This just sets everything to 0 if that's not easy to understand.
            other._alignment =
            other._sizeX =
            other._sizeY = 0;
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
                default:
                    throw std::runtime_error("Invalid axis passed to aligned_array_2d::size()!");
            }
        }

        T operator[](std::tuple<std::size_t, std::size_t> index) const {
            std::size_t x = std::get<0>(index),
                    y = std::get<1>(index);

            // TODO: Consider only enabling this during debug
            if (isEmpty()) throw std::runtime_error("aligned_array_2d is empty!");
            if (x < 0 || y < 0 || x >= _sizeX || y >= _sizeY) {
                std::cerr << "Error Position: [" << x << ", " << y << "]" << std::endl;
                throw std::runtime_error("Out of range index passed to aligned_array_2d!");
            }

            // Is this correct?
            return _data[y + (_sizeY * x)];
        }

        T &operator[](std::tuple<std::size_t, std::size_t> index) {
            std::size_t x = std::get<0>(index),
                    y = std::get<1>(index);

            // TODO: Consider only enabling this during debug
            if (isEmpty()) throw std::runtime_error("aligned_array_2d is empty!");
            if (x < 0 || y < 0 || x >= _sizeX || y >= _sizeY) {
                std::cerr << "Error Position: [" << x << ", " << y << "]" << std::endl;
                throw std::runtime_error("Out of range index passed to aligned_array_2d!");
            }

            // Is this correct?
            return _data[y + (_sizeY * x)];
        }

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
