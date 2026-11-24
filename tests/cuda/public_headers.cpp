// Every installed rmagine_cuda header, included together at the top of a plain
// C++ translation unit with nothing above them. Downstream code compiles these
// headers with a host C++ compiler, so a header that needs a GPU compiler or a
// standard header above it is a regression no other test here can see: the rest
// of the suite always reaches these headers through some other include.
//
// This does not check per-header independence. Every include below the first
// one sees whatever the ones above it pulled in, and two of these headers do
// fail as the first include of a translation unit, the same way with the CUDA
// and with the ROCm headers, both predating this change: math/linalg.cuh
// declares __device__ functions with no runtime header above it, and
// util/cuda/CudaHelper.hpp throws std::runtime_error without <stdexcept>.

#include <rmagine/util/cuda/cuda_to_hip.h>
#include <rmagine/util/cuda/curand_to_hiprand.h>
#include <rmagine/util/cuda/CudaContext.hpp>
#include <rmagine/util/cuda/CudaDebug.hpp>
#include <rmagine/util/cuda/CudaHelper.hpp>
#include <rmagine/util/cuda/CudaStream.hpp>
#include <rmagine/util/cuda/random.cuh>
#include <rmagine/noise/NoiseCuda.hpp>
#include <rmagine/noise/GaussianNoiseCuda.hpp>
#include <rmagine/noise/RelGaussianNoiseCuda.hpp>
#include <rmagine/noise/UniformDustNoiseCuda.hpp>
#include <rmagine/map/mesh_preprocessing.cuh>
#include <rmagine/math/linalg.cuh>
#include <rmagine/math/math_batched.cuh>
#include <rmagine/math/memory_math.cuh>
#include <rmagine/math/statistics.cuh>

#include <iostream>

namespace rm = rmagine;

int main()
{
    // Also check that the headers declare a usable runtime, not just that they
    // parse: a device count query goes through the same symbols they map.
    int devices = 0;
    if(cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1)
    {
        std::cout << "FAIL: no GPU device visible" << std::endl;
        return 1;
    }

    rm::CudaStream stream;

    std::cout << "PASS: public headers include cleanly, " << devices
        << " device(s) visible" << std::endl;
    return 0;
}
