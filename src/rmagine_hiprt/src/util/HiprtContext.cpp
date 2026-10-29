/**
 * @file HiprtContext.cpp
 * @brief HIPRT context implementation
 */

#include "rmagine/util/hiprt/HiprtContext.hpp"

#include <hiprt/hiprt.h>
#include <hip/hip_runtime.h>

#include <iostream>
#include <stdexcept>

namespace rmagine
{

HiprtContext::HiprtContext()
{
    // Get current HIP device
    int device = 0;
    hipError_t hip_err = hipGetDevice(&device);
    if (hip_err != hipSuccess) {
        throw std::runtime_error("HiprtContext: hipGetDevice failed");
    }
    m_device_index = device;

    // Create HIPRT context
    hiprtContextCreationInput input;
    input.ctxt = nullptr;  // Use current HIP context
    input.device = device;
    input.deviceType = hiprtDeviceAMD;

    hiprtError err = hiprtCreateContext(HIPRT_API_VERSION, input, m_ctx);
    if (err != hiprtSuccess) {
        throw std::runtime_error("HiprtContext: hiprtCreateContext failed with error " + std::to_string(err));
    }

    std::cout << "[HiprtContext] Created on device " << device << std::endl;
}

HiprtContext::~HiprtContext()
{
    if (m_ctx) {
        hiprtDestroyContext(m_ctx);
        m_ctx = nullptr;
    }
}

} // namespace rmagine
