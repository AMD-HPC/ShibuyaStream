//------------------------------------------------------------------------------
/// \file
/// \brief      HostStream class declaration and inline routines
/// \date       2020-2024
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include "Stream.h"

//------------------------------------------------------------------------------
/// \brief
///     Represents a streaming workload executed by a CPU.
///     Inherits from the Stream class.
///
template <typename T>
class HostStream: public Stream<T> {
public:
    /// \brief
    ///     Creates a HostStream object.
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
    HostStream(std::string label,
               int core_id,
               Workload workload,
               std::size_t length,
               double duration,
               T alpha,
               Array<T>* a,
               Array<T>* b,
               Array<T>* c = nullptr)
        : Stream<T>(label, workload, length, duration, alpha, a, b, c),
          core_id_(core_id) {}

    ~HostStream() {}

    /// Prints core number, workload type, and locations of arrays.
    void printInfo() override
    {
        fprintf(stderr, "\tcore%6d", core_id_);
        this->printWorkload();
        this->printArrays();
    }

private:
    /// Pins the thread to the specified core.
    void setAffinity() override { this->setCoreAffinity(core_id_); }

    /// Implements CPU Copy stream.
    void copy() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        for (std::size_t i = 0; i < this->length_; ++i)
            b[i] = a[i];
    }

    /// Implements CPU Mul stream.
    void mul() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        T alpha = this->alpha_;
        for (std::size_t i = 0; i < this->length_; ++i)
            b[i] = alpha*a[i];
    }

    /// Implements CPU Add stream.
    void add() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        T* c = this->c_->host_ptr();
        for (std::size_t i = 0; i < this->length_; ++i)
            c[i] = a[i]+b[i];
    }

    /// Implements CPU Triad stream.
    void triad() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        T* c = this->c_->host_ptr();
        T alpha = this->alpha_;
        for (std::size_t i = 0; i < this->length_; ++i)
            c[i] = alpha*a[i] + b[i];
    }

    /// Implements CPU Dot stream.
    void dot() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        T sum = T(0.0);
        for (std::size_t i = 0; i < this->length_; ++i)
            sum += a[i]*b[i];

        this->dot_sum_ = sum;
    }

    int core_id_; ///< core number
};
