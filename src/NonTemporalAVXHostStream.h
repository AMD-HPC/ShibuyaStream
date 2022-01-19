//------------------------------------------------------------------------------
/// \file
/// \brief      HostStreamAVX class declaration and inline routines
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include "HostStream.h"

#include <immintrin.h>

//------------------------------------------------------------------------------
/// \brief
///     Represents a streaming workload executed by a CPU.
///     Inherits from the Stream class.
///
template <typename T>
class NonTemporalAVXHostStream: public HostStream<T> {
public:
    /// \brief
    ///     Creates a AVXHostStream object.
    ///
    /// \param[in] label
    ///     the string defining the stream, e.g., C0-C-N0-N0
    ///
    /// \param[in] core_id
    ///     the number of the CPU core executing the workload
    ///
    /// \param[in] workload
    ///     the workload: Copy, Mul, Add, Triad, Dot, HIP (copy)
    ///
    /// \param[in] length
    ///     the number of elements in the stream
    ///
    /// \param[in] duration
    ///     the duration of the iteration in seconds
    ///
    /// \param[in] alpha
    ///     the scaling factor for Mul and Triad
    ///
    /// \param[in] a, b, c
    ///     the arrays to operate on
    ///
    NonTemporalAVXHostStream(std::string label,
                  int core_id,
                  Workload workload,
                  std::size_t length,
                  double duration,
                  T alpha,
                  Array<T>* a,
                  Array<T>* b,
                  Array<T>* c = nullptr)
        : HostStream<T>(label, core_id, workload, length, duration, alpha,
                        a, b, c) {}

    ~NonTemporalAVXHostStream() {}

private:
    /// Implements CPU Copy stream.
    void copy() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        for (std::size_t i = 0; i < this->length_; i += 4) {
            __m256d a_256 = _mm256_load_pd(&a[i]);
            _mm256_stream_pd(&b[i], a_256);
        }
    }

    /// Implements CPU Mul stream.
    void mul() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        T alpha = this->alpha_;
        __m256d alpha_256 = _mm256_set_pd(alpha, alpha, alpha, alpha);
        for (std::size_t i = 0; i < this->length_; i += 4) {
            __m256d a_256 = _mm256_load_pd(&a[i]);
            a_256 = _mm256_mul_pd(alpha_256, a_256);
            _mm256_stream_pd(&b[i], a_256);
        }
    }

    /// Implements CPU Add stream.
    void add() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        T* c = this->c_->host_ptr();
        for (std::size_t i = 0; i < this->length_; i += 4) {
            __m256d a_256 = _mm256_load_pd(&a[i]);
            __m256d b_256 = _mm256_load_pd(&b[i]);
            a_256 = _mm256_add_pd(a_256, b_256);
            _mm256_stream_pd(&c[i], a_256);
        }
    }

    /// Implements CPU Triad stream.
    void triad() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        T* c = this->c_->host_ptr();
        T alpha = this->alpha_;
        __m256d alpha_256 = _mm256_set_pd(alpha, alpha, alpha, alpha);
        for (std::size_t i = 0; i < this->length_; i += 4) {
            __m256d a_256 = _mm256_load_pd(&a[i]);
            __m256d b_256 = _mm256_load_pd(&b[i]);
            a_256 = _mm256_mul_pd(alpha_256, a_256);
            a_256 = _mm256_add_pd(a_256, b_256);
            _mm256_stream_pd(&c[i], a_256);
        }
    }

    /// Implements CPU Dot stream.
    void dot() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        __m256d sum_256 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        for (std::size_t i = 0; i < this->length_; i += 4) {
            __m256d a_256 = _mm256_load_pd(&a[i]);
            __m256d b_256 = _mm256_load_pd(&b[i]);
            a_256 = _mm256_mul_pd(a_256, b_256);
            sum_256 = _mm256_add_pd(sum_256, a_256);
        }

        double sum[4];
        _mm256_stream_pd(sum, sum_256);
        this->dot_sum_ = sum[0]+sum[1]+sum[2]+sum[3];
    }
};
