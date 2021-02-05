
#pragma once

#include "Array.h"

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__NVCC__)
#include <cuda_runtime.h>
#include "hip2cuda.h"
#endif

//------------------------------------------------------------------------------
/// \class DeviceArray
/// \brief array in device memory
#if defined(__NVCC__)
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

template <typename T>
class DeviceArray: public Array<T> {
public:
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
        hipSetDevice(device_id_);
        hipFree(this->device_ptr_);
    }

    T* host_ptr() override { return(this->device_ptr_); }
    void registerMem() override {}
    void unregisterMem() override {}
    void printInfo() override { fprintf(stderr, "\tdevice%4d", device_id_); }

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
    static const int group_size_ = 256;

#if defined(__HIPCC__)
    static __global__ void init_kernel(T* a, T start, T step)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        a[i] = start + step*i;
    }

    static __global__ void check_kernel(T* a, T start, T step)
    {
        const std::size_t i = (std::size_t)blockIdx.x*blockDim.x + threadIdx.x;
        /// \todo Improve error reporting. Remove device-side assert.
        assert(a[i] == start + step*i);
    }
#endif

    int device_id_;
};
