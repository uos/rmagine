#include "rmagine/map/optix/OptixSpheres.hpp"

#include "rmagine/map/optix/OptixAccelerationStructure.hpp"

#include "rmagine/math/assimp_conversions.h"
#include "rmagine/util/optix/OptixDebug.hpp"
#include "rmagine/types/MemoryCuda.hpp"
#include "rmagine/util/GenericAlign.hpp"

#include <optix.h>
#include <optix_stubs.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <rmagine/util/prints.h>

namespace rmagine
{

OptixSpheres::OptixSpheres(OptixContextPtr context)
:Base(context)
{
    std::cout << "[OptixSpheres::OptixSpheres()] constructed." << std::endl;
}

OptixSpheres::~OptixSpheres()
{
    std::cout << "[OptixSpheres::~OptixSpheres()] destroyed." << std::endl;
}

void OptixSpheres::apply()
{

}

void OptixSpheres::commit()
{
    if(!m_as)
    {
        // No acceleration structure exists yet!
        m_as = std::make_shared<OptixAccelerationStructure>();
        std::cout << "Build acceleration structure" << std::endl;
    } else {
        // update existing structure
        std::cout << "Update acceleration structure" << std::endl;
    }

    const uint32_t spheres_input_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    OptixBuildInput spheres_input = {};

    // ONLY IN VERSION >= 7.5
    #if OPTIX_VERSION >= 75000
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;

    CUdeviceptr tmp_centers = reinterpret_cast<CUdeviceptr>(centers_.raw());

    // VERTICES
    spheres_input.sphereArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
    spheres_input.sphereArray.vertexStrideInBytes = sizeof(Point);
    spheres_input.sphereArray.numVertices   = centers_.size();
    spheres_input.sphereArray.vertexBuffers = &tmp_centers;

    // TODO

    #else
    std::cout << "Need to make custom primitives!" << std::endl;

    throw std::runtime_error("Compile rmagine with optix 7.5 again to use OptixSpheres!");

    #endif

}

} // namespace rmagine