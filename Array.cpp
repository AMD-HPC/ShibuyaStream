
#include "Array.h"
#include "HostArray.h"
#include "DeviceArray.h"

//------------------------------------------------------------------------------
// \brief Creates either a HostArray or a DeviceArray.
template <typename T>
Array<T>*
Array<T>::make(char type, int id, size_t length)
{
    switch (type) {
        case 'N': return new HostArray<T>(id, length);
        case 'D': return new DeviceArray<T>(id, length);
        default: assert(false);
    }
    return nullptr; // suppressing NVCC warning
}

template
Array<double>*
Array<double>::make(char type, int id, size_t length);
