//------------------------------------------------------------------------------
/// \file
/// \brief      HIP to CUDA name replacements
/// \date       2020-2022
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#define hipDeviceSynchronize        cudaDeviceSynchronize
#define hipDeviceEnablePeerAccess   cudaDeviceEnablePeerAccess
#define hipFree                     cudaFree
#define hipGetDeviceCount           cudaGetDeviceCount
#define hipHostGetDevicePointer     cudaHostGetDevicePointer
#define hipHostMalloc               cudaMallocHost
#define hipHostRegister             cudaHostRegister
#define hipHostRegisterMapped       cudaHostRegisterMapped
#define hipHostUnregister           cudaHostUnregister
#define hipMalloc                   cudaMalloc
#define hipMemcpy                   cudaMemcpy
#define hipMemcpyDefault            cudaMemcpyDefault
#define hipSetDevice                cudaSetDevice
#define hipSuccess                  cudaSuccess
