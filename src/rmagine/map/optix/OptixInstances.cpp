#include "rmagine/map/optix/OptixInstances.hpp"

#include "rmagine/map/optix/OptixInst.hpp"
#include "rmagine/map/optix/OptixAccelerationStructure.hpp"

#include <rmagine/util/optix/OptixDebug.hpp>
#include <rmagine/util/GenericAlign.hpp>

#include <optix_stubs.h>
#include <rmagine/types/MemoryCuda.hpp>

namespace rmagine
{

OptixInstances::OptixInstances(OptixContextPtr context)
:Base(context)
{

}

OptixInstances::~OptixInstances()
{

}

void OptixInstances::apply()
{
    // TODO: pass transform to every child instance
}

void OptixInstances::commit()
{
    build_acc_old();
}

unsigned int OptixInstances::add(OptixInstPtr inst)
{
    unsigned int inst_id = gen.get();
    inst->setId(inst_id);
    m_instances[inst_id] = inst;
    m_ids[inst] = inst_id;
    return inst_id;
}

unsigned int OptixInstances::get(OptixInstPtr inst) const
{
    return m_ids.at(inst);
}

OptixInstPtr OptixInstances::get(unsigned int id) const
{
    return m_instances.at(id);
}

std::map<unsigned int, OptixInstPtr> OptixInstances::instances() const
{
    return m_instances;
}

std::unordered_map<OptixInstPtr, unsigned int> OptixInstances::ids() const
{
    return m_ids;
}


// PRIVATE
// void OptixInstances::build_acc()
// {
//     std::cout << "[OptixInstances::commit()] !!!" << std::endl;
//     std::cout << "- Instances: " << m_instances.size() << std::endl;

//     Memory<CUdeviceptr, RAM> inst_h(m_instances.size());
//     Memory<CUdeviceptr, VRAM_CUDA> inst_d;

//     size_t i=0;
//     for(auto elem : m_instances)
//     {
//         std::cout << elem.first << " -> " << i << std::endl;
//         inst_h[i] = elem.second->data_gpu();
//         i++;
//     }

//     // upload pointers
//     inst_d = inst_h;

//     OptixBuildInput instance_input = {};
//     instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS;
//     instance_input.instanceArray.numInstances = m_instances.size();
//     instance_input.instanceArray.instances = reinterpret_cast<CUdeviceptr>(inst_d.raw());

//     OptixAccelBuildOptions ias_accel_options = {};
    
//     unsigned int build_flags = OPTIX_BUILD_FLAG_NONE;
//     { // BUILD FLAGS
//         build_flags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
//         build_flags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
//         build_flags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
//     }
//     ias_accel_options.buildFlags = build_flags;

//     ias_accel_options.motionOptions.numKeys = 1;

//     if(m_as)
//     {
//         ias_accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
//     } else {
//         // first commit
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

//     cudaDeviceSynchronize();

//     CUdeviceptr d_temp_buffer_ias;
//     CUDA_CHECK( cudaMalloc(
//         reinterpret_cast<void**>( &d_temp_buffer_ias ),
//         ias_buffer_sizes.tempSizeInBytes) );

//     CUDA_CHECK( cudaMalloc(
//                 reinterpret_cast<void**>( &m_as->buffer ),
//                 ias_buffer_sizes.outputSizeInBytes
//                 ) );

//     cudaDeviceSynchronize();

//     OPTIX_CHECK(optixAccelBuild( 
//         m_ctx->ref(), 
//         0, 
//         &ias_accel_options, 
//         &instance_input, 
//         1, 
//         d_temp_buffer_ias,
//         ias_buffer_sizes.tempSizeInBytes, 
//         m_as->buffer,
//         ias_buffer_sizes.outputSizeInBytes,
//         &(m_as->handle),
//         nullptr, 
//         0 
//     ));

//     CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_ias ) ) );
// }

void OptixInstances::build_acc_old()
{
    // std::cout << "[OptixInstances::build_acc_old()] !!!" << std::endl;
    // std::cout << "- Instances: " << m_instances.size() << std::endl;

    // Memory<OptixInstance, RAM> inst_h(m_instances.size());
    // 

    std::vector<OptixInstance> inst_vec;
    // TODO get only instances that requires update
    
    for(auto elem : m_instances)
    {
        unsigned int inst_id = elem.first;
        OptixInstPtr inst = elem.second;
        // if(inst->m_changed)
        // {
            std::cout << "Instance " << inst_id << " changed!" << std::endl;
            inst_vec.push_back(inst->data());
        // }
    }

    if(m_as)
    {
        // std::cout << OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID << std::endl;
        // reverse
        // std::reverse(inst_vec.begin(), inst_vec.end());
        inst_vec[5].visibilityMask = 0;
    }

    Memory<OptixInstance, RAM> inst_h(inst_vec.size());
    std::copy(inst_vec.begin(), inst_vec.end(), inst_h.raw());
    Memory<OptixInstance, VRAM_CUDA> inst_d = inst_h;

    if(m_as)
    {
        std::cout << "UPDATE " << inst_d.size() << " instances" << std::endl;
    } else {
        std::cout << "COMMIT " << inst_d.size() << " instances" << std::endl;
    }

    OptixBuildInput instance_input = {};
    instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.numInstances = inst_d.size();
    instance_input.instanceArray.instances = reinterpret_cast<CUdeviceptr>(inst_d.raw());

    OptixAccelBuildOptions ias_accel_options = {};

    unsigned int build_flags = OPTIX_BUILD_FLAG_NONE;
    { // BUILD FLAGS
        build_flags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        build_flags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        build_flags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
    }
    ias_accel_options.buildFlags = build_flags;
    ias_accel_options.motionOptions.numKeys = 1;
    
    if(m_as)
    {
        // update
        ias_accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
    } else {
        // first commit
        m_as = std::make_shared<OptixAccelerationStructure>();
        ias_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    }

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( 
        m_ctx->ref(), 
        &ias_accel_options,
        &instance_input, 
        1, 
        &ias_buffer_sizes ) );

    CUdeviceptr d_temp_buffer_ias;
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &d_temp_buffer_ias ),
        ias_buffer_sizes.tempSizeInBytes) );

    if(m_as->buffer && (m_as->buffer_size != ias_buffer_sizes.outputSizeInBytes ) )
    {
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_as->buffer ) ) );
        m_as->buffer = 0;
        m_as->buffer_size = 0;
    }

    if(!m_as->buffer)
    {
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &m_as->buffer ),
            ias_buffer_sizes.outputSizeInBytes
        ));
        m_as->buffer_size = ias_buffer_sizes.outputSizeInBytes;
    }

    OPTIX_CHECK(optixAccelBuild( 
        m_ctx->ref(), 
        0, 
        &ias_accel_options, 
        &instance_input, 
        1, 
        d_temp_buffer_ias,
        ias_buffer_sizes.tempSizeInBytes, 
        m_as->buffer,
        ias_buffer_sizes.outputSizeInBytes,
        &(m_as->handle),
        nullptr, 
        0 
    ));

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_ias ) ) );

    // set every child to commited
    for(auto elem : m_instances)
    {
        elem.second->m_changed = false;
    }
}

} // namespace rmagine

