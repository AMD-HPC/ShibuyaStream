
#pragma once

#include "Exception.h"

#include <cstdlib>

//------------------------------------------------------------------------------
/// \class Array
/// \brief parent class for HostArray and DeviceArray
template <typename T>
class Array {
public:
    static Array<T>* make(char type, int id, std::size_t length);

    Array(std::size_t length)
        : length_(length),
          size_(length*sizeof(T)) {}
    virtual ~Array() {}

    T* device_ptr() { return device_ptr_; }
    virtual T* host_ptr() = 0;
    virtual void registerMem() = 0;
    virtual void unregisterMem() = 0;
    virtual void printInfo() = 0;
    virtual void init(T start, T step) = 0;
    virtual void check(T start, T step) = 0;

protected:
    std::size_t size_;   ///< size in bytes
    std::size_t length_; ///< length in elements
    T* device_ptr_;      ///< device ptr. (device mem. or registered host mem.)
};
