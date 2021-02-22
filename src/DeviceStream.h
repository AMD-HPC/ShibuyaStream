
#pragma once

#include "Stream.h"

// Workaround for NVCC, which does not support __global__ static members.
#if defined(__NVCC__)
template <typename T>
__global__ void copy_kernel(T const* a, T* b)
{
    const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
    b[i] = a[i];
}

template <typename T>
__global__ void mul_kernel(T alpha, T const* a, T* b)
{
    const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
    b[i] = alpha*a[i];
}

template <typename T>
__global__ void add_kernel(T const* a, T const* b, T* c)
{
    const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
    c[i] = a[i]+b[i];
}

template <typename T>
__global__ void triad_kernel(T alpha, T const* a, T const* b, T* c)
{
    const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
    c[i] = alpha*a[i] + b[i];
}

template <typename T>
__global__ void dot_kernel(T const* a, T* b,
                                  std::size_t length, T* dot_sums)
{
    const int group_size_ = 1024;
    __shared__ T sums[group_size_];

    std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
    const int thx = threadIdx.x;

    sums[thx] = 0.0;
    for (; i < length; i += blockDim.x*gridDim.x)
        sums[thx] += a[i]*b[i];

    for (int offset = blockDim.x/2; offset > 0; offset /= 2) {
        __syncthreads();
        if (thx < offset) {
            sums[thx] += sums[thx+offset];
        }
    }

    if (thx == 0)
        dot_sums[blockIdx.x] = sums[0];
}
#endif

//------------------------------------------------------------------------------
/// \class DeviceStream
/// \brief stream for a GPU device
template <typename T>
class DeviceStream: public Stream<T> {
public:
    DeviceStream(std::string label,
                 int device_id,
                 int host_core_id,
                 Workload workload,
                 std::size_t length,
                 double duration,
                 T alpha,
                 Array<T>* a,
                 Array<T>* b,
                 Array<T>* c = nullptr)
        : Stream<T>(label, workload, length, duration, alpha, a, b, c),
          device_id_(device_id), host_core_id_(host_core_id)
    {
        this->a_->registerMem();
        this->b_->registerMem();
        if (workload.type() == Workload::Type::Add ||
            workload.type() == Workload::Type::Triad)
            this->c_->registerMem();

        HIP_CALL(hipHostMalloc(&dot_sums_, sizeof(T)*dot_num_groups_),
                 "Allocation of page-locked memory failed.");
    }

    ~DeviceStream()
    {
        this->a_->unregisterMem();
        this->b_->unregisterMem();
        if (this->workload_.type() == Workload::Type::Add ||
            this->workload_.type() == Workload::Type::Triad)
            this->c_->unregisterMem();
    }

    void printInfo() override
    {
        fprintf(stderr, "\tdevice%4d", device_id_);
        this->printWorkload();
        this->printArrays();
        fprintf(stderr, "\thost core%4d", host_core_id_);
    }

private:
    static const int group_size_ = 1024;
    static const int dot_num_groups_ = 256;

    void setAffinity() override
    {
        // Set the hosting thread.
        this->setCoreAffinity(host_core_id_);
        // Set the streaming device.
        HIP_CALL(hipSetDevice(device_id_),
                 "Setting the device failed.");
    }

    void copy() override
    {
        copy_kernel<<<dim3(this->length_/group_size_),
                      dim3(group_size_),
                      0, 0>>>(
            this->a_->device_ptr(),
            this->b_->device_ptr());
        HIP_CALL(hipDeviceSynchronize(),
                 "Device synchronization failed.");
    }

    void mul() override
    {
        mul_kernel<<<dim3(this->length_/group_size_),
                     dim3(group_size_),
                     0, 0>>>(
            this->alpha_,
            this->a_->device_ptr(),
            this->b_->device_ptr());
        HIP_CALL(hipDeviceSynchronize(),
                 "Device synchronization failed.");
    }

    void add() override
    {
        add_kernel<<<dim3(this->length_/group_size_),
                     dim3(group_size_),
                     0, 0>>>(
            this->a_->device_ptr(),
            this->b_->device_ptr(),
            this->c_->device_ptr());
        HIP_CALL(hipDeviceSynchronize(),
                 "Device synchronization failed.");
    }

    void triad() override
    {
        triad_kernel<<<dim3(this->length_/group_size_),
                       dim3(group_size_),
                       0, 0>>>(
            this->alpha_,
            this->a_->device_ptr(),
            this->b_->device_ptr(),
            this->c_->device_ptr());
        HIP_CALL(hipDeviceSynchronize(),
                 "Device synchronization failed.");
    }

    void dot() override
    {
        dot_kernel<<<dim3(dot_num_groups_),
                     dim3(group_size_),
                     0, 0>>>(
            this->a_->device_ptr(),
            this->b_->device_ptr(),
            this->length_,
            this->dot_sums_);
        HIP_CALL(hipDeviceSynchronize(),
                 "Device synchronization failed.");

        this->dot_sum_ = 0.0;
        for (int i = 0; i < dot_num_groups_; ++i)
            this->dot_sum_ += this->dot_sums_[i];
    }

// __global__ static members are okay for HIPCC.
#if defined(__HIPCC__)
    static __global__ void copy_kernel(T const* a, T* b)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        b[i] = a[i];
    }

    static __global__ void mul_kernel(T alpha, T const* a, T* b)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        b[i] = alpha*a[i];
    }

    static __global__ void add_kernel(T const* a, T const* b, T* c)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        c[i] = a[i]+b[i];
    }

    static __global__ void triad_kernel(T alpha, T const* a, T const* b, T* c)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        c[i] = alpha*a[i] + b[i];
    }

    static __global__ void dot_kernel(T const* a, T* b,
                                      std::size_t length, T* dot_sums)
    {
        __shared__ T sums[group_size_];

        std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        const int thx = threadIdx.x;

        sums[thx] = 0.0;
        for (; i < length; i += blockDim.x*gridDim.x)
            sums[thx] += a[i]*b[i];

        for (int offset = blockDim.x/2; offset > 0; offset /= 2) {
            __syncthreads();
            if (thx < offset) {
                sums[thx] += sums[thx+offset];
            }
        }

        if (thx == 0)
            dot_sums[blockIdx.x] = sums[0];
    }
#endif

    int device_id_;
    int host_core_id_;
    T* dot_sums_;
};
