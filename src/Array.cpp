//------------------------------------------------------------------------------
/// \file
/// \brief      implementations of Array methods
/// \date       2020-2024
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#include "Array.h"
#include "HostArray.h"
#include "DeviceArray.h"

//------------------------------------------------------------------------------
/// \brief
///     Creates either a HostArray or a DeviceArray.
///
/// \param[in] type
///     type of the array, N for host (NUMA), D for device
///
/// \param[in] id
///     number of the NUMA node or number of the device
///
/// \param[in] length
///     number of elements in the array
///
template <typename T>
Array<T>*
Array<T>::make(char type, int id, size_t length)
{
    switch (type) {
        case 'N': return new HostArray<T>(id, length);
        case 'D': return new DeviceArray<T>(id, length);
        default: ERROR("Invalid array type.");
    }
    return nullptr; // suppressing NVCC warning
}

template
Array<double>*
Array<double>::make(char type, int id, size_t length);
