//------------------------------------------------------------------------------
/// \file
/// \brief      DeviceStream class declaration and inline routines
/// \date       2020-2024
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include "Stream.h"

// Workaround for NVCC, which does not support __global__ static members.
#if defined(USE_CUDA)
template <typename T>
__global__ void copy_kernel(T const* __restrict a, T* __restrict b)
{
    const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
    b[i] = a[i];
}

template <typename T>
__global__ void mul_kernel(T alpha, T const* __restrict a, T* __restrict b)
{
    const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
    b[i] = alpha*a[i];
}

template <typename T>
__global__ void add_kernel(T const* __restrict a, T const* __restrict b,
                           T* __restrict c)
{
    const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
    c[i] = a[i]+b[i];
}

template <typename T>
__global__ void triad_kernel(T alpha, T const* __restrict a,
                             T const* __restrict b, T* __restrict c)
{
    const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
    c[i] = alpha*a[i] + b[i];
}

template <typename T>
__global__ void dot_kernel(T const* __restrict a, T* __restrict b,
                           std::size_t length, T* __restrict dot_sums)
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
/// \brief
///     Represents a streaming workload executed by a GPU.
///     Inherits from the Stream class.
///
template <typename T, int elements_per_item, int chunks_per_group>
class DeviceStream: public Stream<T> {
public:
    /// \brief
    ///     Creates a DeviceStream object.
    ///     Registers the memory and retrieves the GPU pointer for host arrays.
    ///
    /// \param[in] label
    ///     the string defining the stream, e.g., D0-C-D0-D0
    ///
    /// \param[in] device_id
    ///     the number of the GPU device executing the workload
    ///
    /// \param[in] host_core_id
    ///     the number of the CPU core launching the GPU operations
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
          num_groups_(length/group_size_/elements_per_item/chunks_per_group),
          device_id_(device_id), host_core_id_(host_core_id)
    {
        this->a_->registerMem();
        this->b_->registerMem();
        if (workload.type() == Workload::Type::Add ||
            workload.type() == Workload::Type::Triad)
            this->c_->registerMem();

        HIP_CALL(hipHostMalloc(&dot_sums_, sizeof(T)*num_groups_),
                 "Allocation of page-locked memory failed.");
    }

    /// Destroys a DeviceStream object.
    /// Unregisters the `a` and `b` arrays.
    /// Unregisters the `c` array if the workload is Add or Triad.
    ~DeviceStream()
    {
        this->a_->unregisterMem();
        this->b_->unregisterMem();
        if (this->workload_.type() == Workload::Type::Add ||
            this->workload_.type() == Workload::Type::Triad)
            this->c_->unregisterMem();
    }

    /// Prints device number, workload type, locations of arrays,
    /// and core number (the core launching the GPU operations).
    void printInfo() override
    {
        fprintf(stderr, "\tdevice%4d", device_id_);
        this->printWorkload();
        this->printArrays();
        fprintf(stderr, "\thost core%4d", host_core_id_);
    }

protected:
    /// work-group size for streaming kernels
    static constexpr int group_size_ = 1024;

    int num_groups_; /// number of groups for streaming kernels
    T* dot_sums_;    /// partial sum from each work-group (Dot workload)

private:
    /// Pins the controlling thread to the given core.
    /// Sets the device for the streaming workload.
    void setAffinity() override
    {
        // Set the hosting thread.
        this->setCoreAffinity(host_core_id_);
        // Set the streaming device.
        HIP_CALL(hipSetDevice(device_id_),
                 "Setting the device failed.");
    }

    /// Launches the GPU Copy kernel.
    void copy() override
    {
        copy_kernel<<<dim3(num_groups_),
                      dim3(group_size_),
                      0, 0>>>(
            this->a_->device_ptr(),
            this->b_->device_ptr());
        HIP_CALL(hipDeviceSynchronize(),
                 "Device synchronization failed.");
    }

    /// Launches the GPU Mul kernel.
    void mul() override
    {
        mul_kernel<<<dim3(num_groups_),
                     dim3(group_size_),
                     0, 0>>>(
            this->alpha_,
            this->a_->device_ptr(),
            this->b_->device_ptr());
        HIP_CALL(hipDeviceSynchronize(),
                 "Device synchronization failed.");
    }

    /// Launches the GPU Add kernel.
    void add() override
    {
        add_kernel<<<dim3(num_groups_),
                     dim3(group_size_),
                     0, 0>>>(
            this->a_->device_ptr(),
            this->b_->device_ptr(),
            this->c_->device_ptr());
        HIP_CALL(hipDeviceSynchronize(),
                 "Device synchronization failed.");
    }

    /// Launches the GPU Triad kernel.
    void triad() override
    {
        triad_kernel<<<dim3(num_groups_),
                       dim3(group_size_),
                       0, 0>>>(
            this->alpha_,
            this->a_->device_ptr(),
            this->b_->device_ptr(),
            this->c_->device_ptr());
        HIP_CALL(hipDeviceSynchronize(),
                 "Device synchronization failed.");
    }

    /// Launches the GPU Dot kernel.
    /// Reduces the partial sums on the host.
    void dot() override
    {
        dot_kernel<<<dim3(num_groups_),
                     dim3(group_size_),
                     0, 0>>>(
            this->a_->device_ptr(),
            this->b_->device_ptr(),
            this->dot_sums_);
        HIP_CALL(hipDeviceSynchronize(),
                 "Device synchronization failed.");

        this->dot_sum_ = T(0.0);
        for (int i = 0; i < num_groups_; ++i)
            this->dot_sum_ += this->dot_sums_[i];
    }

// __global__ static members are okay for HIPCC.
#if defined(USE_HIP)
    // Return the offset for each group.
    static __device__ uint32_t offset()
    {
        return (blockDim.x*blockIdx.x + threadIdx.x)*elements_per_item;
    }

    // Return the stride for each group.
    static __device__ uint32_t stride()
    {
        return gridDim.x*blockDim.x*elements_per_item;
    }

    /// Implements the Copy kernel.
    static __global__ void copy_kernel(T const* __restrict a, T* __restrict b)
    {
        const auto offs = offset();
        const auto strd = stride();
        for (auto j = 0u; j < chunks_per_group; ++j)
            for (auto i = 0u; i < elements_per_item; ++i)
                b[offs + j*strd + i] = a[offs + j*strd + i];
    }

    /// Implements the Mul kernel.
    static __global__ void mul_kernel(T alpha,
                                      T const* __restrict a, T* __restrict b)
    {
        const auto offs = offset();
        const auto strd = stride();
        for (auto j = 0u; j < chunks_per_group; ++j)
            for (auto i = 0u; i < elements_per_item; ++i)
                b[offs + j*strd + i] = alpha*a[offs + j*strd + i];
    }

    /// Implements the Add kernel.
    static __global__ void add_kernel(T const* __restrict a,
                                      T const* __restrict b, T* __restrict c)
    {
        const auto offs = offset();
        const auto strd = stride();
        for (auto j = 0u; j < chunks_per_group; ++j)
            for (auto i = 0u; i < elements_per_item; ++i)
                c[offs + j*strd + i] = a[offs + j*strd + i]
                                     + b[offs + j*strd + i];
    }

    /// Implements the Triad kernel.
    static __global__ void triad_kernel(T alpha, T const* __restrict a,
                                        T const* __restrict b, T* __restrict c)
    {
        const auto offs = offset();
        const auto strd = stride();
        for (auto j = 0u; j < chunks_per_group; ++j)
            for (auto i = 0u; i < elements_per_item; ++i)
                c[offs + j*strd + i] = a[offs + j*strd + i]*alpha
                                     + b[offs + j*strd + i];
    }

    /// Implements the Dot kernel.
    /// First, each work-item computes its partial sum.
    /// Then, each work-group reduces the sums from its threads.
    static __global__ void dot_kernel(T const* __restrict a, T* __restrict b,
                                      T* __restrict dot_sums)
    {
        __shared__ T sums[group_size_];

        const auto offs = offset();
        const auto strd = stride();
        sums[threadIdx.x] = T(0.0);
        for (auto j = 0u; j < chunks_per_group; ++j)
            for (auto i = 0u; i < elements_per_item; ++i)
                sums[threadIdx.x] += a[offs + j*strd + i]*b[offs + j*strd + i];

        for (auto i = blockDim.x/2; i > 0; i /= 2) {
            __syncthreads();
            if (threadIdx.x < i) {
                sums[threadIdx.x] += sums[threadIdx.x+i];
            }
        }

        if (threadIdx.x == 0)
            dot_sums[blockIdx.x] = sums[0];
    }
#endif

    int device_id_;    ///< device number
    int host_core_id_; ///< CPU core launching GPU operations
};
