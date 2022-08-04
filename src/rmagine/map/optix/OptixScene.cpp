#include "rmagine/map/optix/OptixScene.hpp"

#include "rmagine/map/optix/OptixGeometry.hpp"

#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/util/optix/OptixDebug.hpp>

#include <optix_stubs.h>


namespace rmagine
{

OptixScene::OptixScene(OptixContextPtr context)
:OptixEntity(context)
{
    
}

OptixScene::OptixScene(OptixGeometryPtr geom, OptixContextPtr context)
:OptixEntity(context)
,m_geom(geom)
{
    
}

OptixScene::~OptixScene()
{

}

void OptixScene::setRoot(OptixGeometryPtr geom)
{
    m_geom = geom;
}

OptixGeometryPtr OptixScene::getRoot() const
{
    return m_geom;
}

unsigned int OptixScene::add(OptixGeometryPtr geom)
{
    unsigned int id = gen.get();
    m_geometries[id] = geom;
    m_ids[geom] = id;
    return id;
}

unsigned int OptixScene::get(OptixGeometryPtr geom) const
{
    return m_ids.at(geom);
}

std::map<unsigned int, OptixGeometryPtr> OptixScene::geometries() const
{
    return m_geometries;
}

std::unordered_map<OptixGeometryPtr, unsigned int> OptixScene::ids() const
{
    return m_ids;
}


// void OptixScene::commit()
// {
//     // make flat buffer
//     Memory<OptixInstance, RAM> instances(m_instances.size());
//     std::unordered_map<unsigned int, unsigned int> id_map;

//     unsigned int id = 0;
//     for(auto elem : m_instances)
//     {
//         instances[id] = elem.second->data();
//         id_map[elem.first] = id;
//         id++;
//     }

//     Memory<OptixInstance, VRAM_CUDA> instances_gpu;
//     instances_gpu = instances;

//     OptixBuildInput instance_input = {};
//     // TODO check: OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS
//     instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
//     instance_input.instanceArray.numInstances = instances.size();
//     instance_input.instanceArray.instances = reinterpret_cast<CUdeviceptr>(instances.raw());

//     OptixAccelBuildOptions ias_accel_options = {};
//     ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
//     ias_accel_options.motionOptions.numKeys = 1;
    
//     if(m_as)
//     {
//         std::cout << "Update existing scene!" << std::endl;
//         ias_accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
//     } else {
//         m_as = std::make_shared<OptixAccelerationStructure>();
//         ias_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
//     }

//     OptixAccelBufferSizes ias_buffer_sizes;
//     OPTIX_CHECK( optixAccelComputeMemoryUsage( 
//         m_ctx->ref(), 
//         &ias_accel_options, 
//         &instance_input, 
//         1, 
//         &ias_buffer_sizes ) );

//     CUdeviceptr d_temp_buffer_ias;
//     CUDA_CHECK( cudaMalloc(
//         reinterpret_cast<void**>( &d_temp_buffer_ias ),
//         ias_buffer_sizes.tempSizeInBytes) );

//     CUDA_CHECK( cudaMalloc(
//                 reinterpret_cast<void**>( &m_as->buffer ),
//                 ias_buffer_sizes.outputSizeInBytes
//                 ) );
// }

} // namespace rmagine