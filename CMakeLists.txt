cmake_minimum_required(VERSION 3.14)
project(vxnoise)

set(CMAKE_CXX_STANDARD 11)

if(MSVC)
    set_property(SOURCE vx/noise/impl/CellularAVX2.cpp APPEND PROPERTY COMPILE_FLAGS "/arch:AVX")
    set_property(SOURCE vx/noise/simd/NoiseAVX2.cpp APPEND PROPERTY COMPILE_FLAGS "/arch:AVX")
    set_property(SOURCE vx/noise/impl/SimplexAVX2.cpp APPEND PROPERTY COMPILE_FLAGS "/arch:AVX")
    set_property(SOURCE vx/noise/impl/SimplexFractalAVX2.cpp APPEND PROPERTY COMPILE_FLAGS "/arch:AVX")
else()
    set_property(SOURCE vx/noise/impl/CellularAVX2.cpp APPEND PROPERTY COMPILE_FLAGS "-mavx -mavx2")
    set_property(SOURCE vx/noise/simd/NoiseAVX2.cpp APPEND PROPERTY COMPILE_FLAGS "-mavx -mavx2")
    set_property(SOURCE vx/noise/impl/SimplexAVX2.cpp APPEND PROPERTY COMPILE_FLAGS "-mavx -mavx2")
    set_property(SOURCE vx/noise/impl/SimplexFractalAVX2.cpp APPEND PROPERTY COMPILE_FLAGS "-mavx -mavx2")

    set_property(SOURCE vx/noise/impl/SimplexSSE42.cpp APPEND PROPERTY COMPILE_FLAGS "-msse4.1")
    set_property(SOURCE vx/noise/impl/SimplexSSE42.hpp APPEND PROPERTY COMPILE_FLAGS "-msse4.1")
    set_property(SOURCE vx/noise/impl/SimplexFractalSSE42.cpp APPEND PROPERTY COMPILE_FLAGS "-msse4.1")
    set_property(SOURCE vx/noise/impl/SimplexFractalSSE42.hpp APPEND PROPERTY COMPILE_FLAGS "-msse4.1")
    set_property(SOURCE vx/noise/impl/CellularSSE42.cpp APPEND PROPERTY COMPILE_FLAGS "-msse4.1")
    set_property(SOURCE vx/noise/impl/CellularSSE42.hpp APPEND PROPERTY COMPILE_FLAGS "-msse4.1")
endif()

add_library(vxnoise SHARED
        vx/aligned_array_2d.hpp
        vx/aligned_array_3d.hpp
        vx/simd/avx2.hpp
        vx/simd/simd.cpp
        vx/simd/simd.hpp
        vx/simd/simd_level.hpp
        vx/simd/none.hpp
        vx/simd/sse2.hpp
        vx/simd/sse42.hpp
        vx/noise/Noise.hpp
        vx/noise/NoiseFractal.hpp
        vx/noise/Simplex.hpp
        vx/noise/Simplex.cpp
        vx/noise/SimplexFractal.cpp
        vx/noise/SimplexFractal.hpp
        vx/noise/Cellular.cpp
        vx/noise/Cellular.hpp
        vx/noise/simd/NoiseAVX2.cpp
        vx/noise/simd/NoiseAVX2.hpp
        vx/noise/simd/NoiseSSE2.hpp
        vx/noise/simd/NoiseSSE42.hpp
        vx/noise/impl/SimplexAVX2.cpp
        vx/noise/impl/SimplexAVX2.hpp
        vx/noise/impl/SimplexNormal.cpp
        vx/noise/impl/SimplexNormal.hpp
        vx/noise/impl/SimplexSSE2.cpp
        vx/noise/impl/SimplexSSE2.hpp
        vx/noise/impl/SimplexSSE42.cpp
        vx/noise/impl/SimplexSSE42.hpp
        vx/noise/impl/SimplexFractalAVX2.cpp
        vx/noise/impl/SimplexFractalAVX2.hpp
        vx/noise/impl/SimplexFractalNormal.cpp
        vx/noise/impl/SimplexFractalNormal.hpp
        vx/noise/impl/SimplexFractalSSE2.cpp
        vx/noise/impl/SimplexFractalSSE2.hpp
        vx/noise/impl/SimplexFractalSSE42.cpp
        vx/noise/impl/SimplexFractalSSE42.hpp
        vx/noise/impl/CellularAVX2.cpp
        vx/noise/impl/CellularAVX2.hpp
        vx/noise/impl/CellularNormal.cpp
        vx/noise/impl/CellularNormal.hpp
        vx/noise/impl/CellularSSE2.cpp
        vx/noise/impl/CellularSSE2.hpp
        vx/noise/impl/CellularSSE42.cpp
        vx/noise/impl/CellularSSE42.hpp)
