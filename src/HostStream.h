
#pragma once

#include "Stream.h"

//------------------------------------------------------------------------------
/// \class HostStream
/// \brief stream for a CPU core
template <typename T>
class HostStream: public Stream<T> {
public:
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

    void printInfo() override
    {
        fprintf(stderr, "\tcore%6d", core_id_);
        this->printWorkload();
        this->printArrays();
    }

private:
    void setAffinity() override { this->setCoreAffinity(core_id_); }

    void copy() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        for (std::size_t i = 0; i < this->length_; ++i)
            b[i] = a[i];
    }

    void mul() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        T alpha = this->alpha_;
        for (std::size_t i = 0; i < this->length_; ++i)
            b[i] = alpha*a[i];
    }

    void add() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        T* c = this->c_->host_ptr();
        for (std::size_t i = 0; i < this->length_; ++i)
            c[i] = a[i]+b[i];
    }

    void triad() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        T* c = this->c_->host_ptr();
        T alpha = this->alpha_;
        for (std::size_t i = 0; i < this->length_; ++i)
            c[i] = alpha*a[i] + b[i];
    }

    void dot() override
    {
        T* a = this->a_->host_ptr();
        T* b = this->b_->host_ptr();
        T sum = 0.0;
        for (std::size_t i = 0; i < this->length_; ++i)
            sum += a[i]*b[i];

        this->dot_sum_ = sum;
    }

    int core_id_;
};

