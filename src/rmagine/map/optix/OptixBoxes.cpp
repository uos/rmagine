#include "rmagine/map/optix/OptixBoxes.hpp"

#include "rmagine/map/optix/OptixAccelerationStructure.hpp"

#include "rmagine/math/assimp_conversions.h"
#include "rmagine/util/optix/OptixDebug.hpp"
#include "rmagine/types/MemoryCuda.hpp"
#include "rmagine/util/GenericAlign.hpp"
#include "rmagine/map/mesh_preprocessing.cuh"
#include "rmagine/math/math.cuh"

#include <optix.h>
#include <optix_stubs.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <rmagine/util/prints.h>

namespace rmagine
{

OptixBoxes::OptixBoxes(OptixContextPtr context)
:Base(context)
{
    std::cout << "[OptixBoxes::OptixBoxes()] constructed." << std::endl;
}

OptixBoxes::~OptixBoxes()
{
    std::cout << "[OptixBoxes::~OptixBoxes()] destroyed." << std::endl;
}

void OptixBoxes::apply()
{
    // no
}

void OptixBoxes::commit()
{
    // build/update acceleration structure
    if(!m_as)
    {
        // No acceleration structure exists yet!
        m_as = std::make_shared<OptixAccelerationStructure>();
        std::cout << "Build acceleration structure" << std::endl;
    } else {
        // update existing structure
        std::cout << "Update acceleration structure" << std::endl;
    }

    CUdeviceptr tmp = reinterpret_cast<CUdeviceptr>(boxes.raw());

    const uint32_t aabb_input_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };

    OptixBuildInput aabb_input = {};

    aabb_input.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = &tmp;
    aabb_input.customPrimitiveArray.numPrimitives = boxes.size();
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords = 1;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
                m_ctx->ref(),
                &accel_options,
                &aabb_input,
                1, // Number of build inputs
                &gas_buffer_sizes
                ) );

    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &d_temp_buffer_gas ),
        gas_buffer_sizes.tempSizeInBytes) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &m_as->buffer ),
                gas_buffer_sizes.outputSizeInBytes
                ) );

    OPTIX_CHECK( optixAccelBuild(
                m_ctx->ref(),
                0,                  // CUDA stream
                &accel_options,
                &aabb_input,
                1,                  // num build inputs
                d_temp_buffer_gas,
                gas_buffer_sizes.tempSizeInBytes,
                m_as->buffer,
                gas_buffer_sizes.outputSizeInBytes,
                &m_as->handle,
                nullptr,            // emitted property list
                0                   // num emitted properties
    ));

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
}



} // namespace rmagine