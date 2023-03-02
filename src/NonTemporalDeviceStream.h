//------------------------------------------------------------------------------
/// \file
/// \brief      DeviceStream class declaration and inline routines
/// \date       2020-2022
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include "DeviceStream.h"

//------------------------------------------------------------------------------
/// \brief
///     Represents a streaming workload executed by a GPU.
///     Inherits from the DeviceStream class.
///     Uses LLVM intrinsics for non-temporal loads and stores.
///
template <typename T>
class NonTemporalDeviceStream: public DeviceStream<T> {
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
    NonTemporalDeviceStream(std::string label,
                 int device_id,
                 int host_core_id,
                 Workload workload,
                 std::size_t length,
                 double duration,
                 T alpha,
                 Array<T>* a,
                 Array<T>* b,
                 Array<T>* c = nullptr)
        : DeviceStream<T>(label, device_id, host_core_id, workload, length,
                          duration, alpha, a, b, c) {}

    ~NonTemporalDeviceStream() {}

private:
    static const int group_size_ = DeviceStream<T>::group_size_;
    static const int dot_num_groups_ = DeviceStream<T>::dot_num_groups_;

    /// Launches the GPU Copy kernel.
    void copy() override
    {
        non_temporal_copy_kernel<<<dim3(this->length_/group_size_),
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
        non_temporal_mul_kernel<<<dim3(this->length_/group_size_),
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
        non_temporal_add_kernel<<<dim3(this->length_/group_size_),
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
        non_temporal_triad_kernel<<<dim3(this->length_/group_size_),
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
        non_temporal_dot_kernel<<<dim3(dot_num_groups_),
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

    /// Implements the Copy kernel.
    static __global__ void non_temporal_copy_kernel(T const* a, T* b)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        __builtin_nontemporal_store(__builtin_nontemporal_load(&a[i]), &b[i]);
    }

    /// Implements the Mul kernel.
    static __global__ void non_temporal_mul_kernel(T alpha, T const* a, T* b)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        __builtin_nontemporal_store(__builtin_nontemporal_load(&a[i])*alpha,
                                    &b[i]);
    }

    /// Implements the Add kernel.
    static __global__ void non_temporal_add_kernel(T const* a, T const* b, T* c)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        __builtin_nontemporal_store(__builtin_nontemporal_load(&a[i])+
                                    __builtin_nontemporal_load(&b[i]), &c[i]);
    }

    /// Implements the Triad kernel.
    static __global__
    void non_temporal_triad_kernel(T alpha, T const* a, T const* b, T* c)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        __builtin_nontemporal_store(__builtin_nontemporal_load(&a[i])*alpha +
                                    __builtin_nontemporal_load(&b[i]), &c[i]);
    }

    /// Implements the Dot kernel.
    /// First, each work-item computes its partial sum.
    /// Then, each work-group reduces the sums from its threads.
    static __global__
    void non_temporal_dot_kernel(T const* a, T* b,
                                 std::size_t length, T* dot_sums)
    {
        __shared__ T sums[group_size_];

        std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        const int thx = threadIdx.x;

        sums[thx] = 0.0;
        for (; i < length; i += blockDim.x*gridDim.x)
            sums[thx] += __builtin_nontemporal_load(&a[i])*
                         __builtin_nontemporal_load(&b[i]);

        for (int offset = blockDim.x/2; offset > 0; offset /= 2) {
            __syncthreads();
            if (thx < offset) {
                sums[thx] += sums[thx+offset];
            }
        }

        if (thx == 0)
            __builtin_nontemporal_store(sums[0], &dot_sums[blockIdx.x]);
    }
};
