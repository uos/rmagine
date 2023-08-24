#include "rmagine/util/cuda/CudaDebug.hpp"

#include <cuda_runtime.h>

void cudaAssert(
   cudaError_t code, 
   const char* file, 
   const char* func,
   int line)
{
    if(code != cudaSuccess)
    {
        std::stringstream ss;
        ss << "CUDA Error! Name: " << cudaGetErrorName(code) << ", Message: " << cudaGetErrorString(code) << "\n";
        throw rmagine::CudaException(ss.str(), file, func, line);
    }
}