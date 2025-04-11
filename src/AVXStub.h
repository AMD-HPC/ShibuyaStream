//------------------------------------------------------------------------------
/// \file
/// \brief      HostStreamAVX class declaration and inline routines
/// \date       2020-2024
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

typedef struct { double data[4]; } __m256d;

inline __m256d _mm256_load_pd(const double* ptr)
{
    ERROR("ENABLE_AVX_SUPPORT is OFF");
}

inline void _mm256_store_pd(double* ptr, __m256d val)
{
    ERROR("ENABLE_AVX_SUPPORT is OFF");
}

inline void _mm256_stream_pd(double* ptr, __m256d val)
{
    ERROR("ENABLE_AVX_SUPPORT is OFF");
}

inline __m256d _mm256_set_pd(double d3, double d2, double d1, double d0)
{
    ERROR("ENABLE_AVX_SUPPORT is OFF");
}

inline __m256d _mm256_add_pd(__m256d a, __m256d b)
{
    ERROR("ENABLE_AVX_SUPPORT is OFF");
}

inline __m256d _mm256_mul_pd(__m256d a, __m256d b)
{
    ERROR("ENABLE_AVX_SUPPORT is OFF");
}
