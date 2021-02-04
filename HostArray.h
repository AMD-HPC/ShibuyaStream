
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
// \class HostArray
// \brief array in host memory
template <typename T>
class HostArray: public Array<T> {
public:
    HostArray(int numa_id, std::size_t size)
        : Array<T>(size), numa_id_(numa_id)
    {
        this->host_ptr_ = (T*)numa_alloc_onnode(this->size_, numa_id_);
        assert(this->host_ptr_ != nullptr);
    }

    ~HostArray()
    {
        numa_free(this->host_ptr_, this->size_);
    }

    T* host_ptr() override { return host_ptr_; }

    void registerMem() override
    {
        CALL_HIP(hipHostRegister(this->host_ptr_,
                                 this->size_,
                                 hipHostRegisterMapped));
        CALL_HIP(hipHostGetDevicePointer((void**)(&this->device_ptr_),
                                         (void*)this->host_ptr_, 0));
    }

    void unregisterMem() override
    {
        CALL_HIP(hipHostUnregister(this->host_ptr_));
    }

    void printInfo() override { fprintf(stderr, "\tnode%6d", numa_id_); }

    void init(T start, T step) override
    {
        T val = start;
        for (std::size_t i = 0; i < this->length_; ++i) {
            host_ptr_[i] = val;
            val += step;
        }
    }

    void check(T start, T step) override
    {
        T val = start;
        for (std::size_t i = 0; i < this->length_; ++i) {
            assert(host_ptr_[i] == val);
            val += step;
        }
    }

private:
    int numa_id_; ///< NUMA node number
    T* host_ptr_; ///< host pointer to host memory
};
