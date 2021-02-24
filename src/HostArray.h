//------------------------------------------------------------------------------
/// \file
/// \brief      HostArray class declaration and inline routines
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include "Array.h"

#include <cstdio>

#include <numa.h>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__NVCC__)
#include <cuda_runtime.h>
#include "hip2cuda.h"
#endif

//------------------------------------------------------------------------------
/// \brief
///     Represents an array in host memory.
///     Inherits from the Array class.
///
template <typename T>
class HostArray: public Array<T> {
public:
    /// Allocates memory from the specified NUMA node.
    HostArray(int numa_id, std::size_t size)
        : Array<T>(size), numa_id_(numa_id)
    {
        this->host_ptr_ = (T*)numa_alloc_onnode(this->size_, numa_id_);
        ASSERT(this->host_ptr_ != nullptr, "NUMA allocation failed.");
    }

    ~HostArray()
    {
        numa_free(this->host_ptr_, this->size_);
    }

    T* host_ptr() override { return host_ptr_; }

    /// Page-locks the memory.
    /// Retrieves the device pointer.
    void registerMem() override
    {
        HIP_CALL(hipHostRegister(this->host_ptr_,
                                 this->size_,
                                 hipHostRegisterMapped),
                 "Registration of host memory failed.");
        HIP_CALL(hipHostGetDevicePointer((void**)(&this->device_ptr_),
                                         (void*)this->host_ptr_, 0),
                 "Retrieving of device pointer to host memory failed.");

    }

    /// Page-unlocks the memory.
    void unregisterMem() override
    {
        HIP_CALL(hipHostUnregister(this->host_ptr_),
                 "Unregistering of device pointer to host memory failed.");
    }

    /// Prints the NUMA node number.
    void printInfo() override { fprintf(stderr, "\tnode%6d", numa_id_); }

    /// Initializes an array in host memory
    /// with values starting at `start`
    /// and growing by `step`.
    void init(T start, T step) override
    {
        T val = start;
        for (std::size_t i = 0; i < this->length_; ++i) {
            host_ptr_[i] = val;
            val += step;
        }
    }

    /// Checks that an array in host memory
    /// contains values starting at `start`
    /// and growing by `step`.
    void check(T start, T step) override
    {
        T val = start;
        for (std::size_t i = 0; i < this->length_; ++i) {
            ASSERT(host_ptr_[i] == val, "Host correctness check failed.");
            val += step;
        }
    }

private:
    int numa_id_; ///< NUMA node number
    T* host_ptr_; ///< host pointer to host memory
};
