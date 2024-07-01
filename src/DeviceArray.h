//------------------------------------------------------------------------------
/// \file
/// \brief      DeviceArray class declaration and inline routines
/// \date       2020-2024
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include "Array.h"

#if defined(USE_HIP)
    #include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
    #include <cuda_runtime.h>
    #include "hip2cuda.h"
#endif

// Kernels defined as global functions for NVCC,
// which does not support static device members.
#if defined(USE_CUDA)
    template <typename T>
    __global__ void init_kernel(T* a, T start, T step)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        a[i] = start + step*i;
    }

    template <typename T>
    __global__ void check_kernel(T* a, T start, T step)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        /// \todo Improve error reporting. Remove device-side assert.
        assert(a[i] == start + step*i);
    }
#endif

//------------------------------------------------------------------------------
/// \brief
///     Represents an array in device memory.
///     Inherits from the Array class.
///
template <typename T>
class DeviceArray: public Array<T> {
public:
    /// Allocates memory on the specified device.
    DeviceArray(int device_id, std::size_t size)
        : Array<T>(size), device_id_(device_id)
    {
        HIP_CALL(hipSetDevice(device_id_),
                 "Setting the device failed.");
        HIP_CALL(hipMalloc(&this->device_ptr_, this->size_),
                 "Allocation of device memory failed.");
    }

    ~DeviceArray()
    {
        (void)hipSetDevice(device_id_);
        (void)hipFree(this->device_ptr_);
    }

    T* host_ptr() override { return(this->device_ptr_); }
    void registerMem() override {}
    void unregisterMem() override {}
    void printInfo() override { fprintf(stderr, "\tdevice%4d", device_id_); }

    /// Initializes an array in device memory
    /// with values starting at `start`
    /// and growing by `step`.
    void init(T start, T step) override
    {
        HIP_CALL(hipSetDevice(device_id_),
                 "Setting the device failed.");
        init_kernel<<<dim3(this->length_/group_size_),
                      dim3(group_size_),
                      0, 0>>>(this->device_ptr_, start, step);
        HIP_CALL(hipDeviceSynchronize(),
                 "Device synchronization failed.");
    }

    /// Checks that an array in device memory
    /// contains values starting at `start`
    /// and growing by `step`.
    void check(T start, T step) override
    {
        HIP_CALL(hipSetDevice(device_id_),
                 "Setting the device failed.");
        check_kernel<<<dim3(this->length_/group_size_),
                       dim3(group_size_),
                       0, 0>>>(this->device_ptr_, start, step);
        HIP_CALL(hipDeviceSynchronize(),
                 "Device synchronization failed.");
    }

private:
    /// work-group size for initialization and validation
    static const int group_size_ = 256;

// As opposed to NVCC, HIPCC supports static device members.
#if defined(USE_HIP)
    /// Initializes an array in device memory.
    static __global__ void init_kernel(T* a, T start, T step)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        a[i] = start + step*i;
    }

    /// Checks the contents of an array in device memory.
    static __global__ void check_kernel(T* a, T start, T step)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        /// \todo Improve error reporting. Remove device-side assert.
        assert(a[i] == start + step*i);
    }
#endif

    int device_id_; ///< device number
};
