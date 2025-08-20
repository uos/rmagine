#include "SimulatorVulkan.hpp"



namespace rmagine
{

template<typename SensorModelRamT, typename SensorModelDeviceT>
void SimulatorVulkan<SensorModelRamT, SensorModelDeviceT>::setTsb(Memory<Transform, RAM>& tsbMem)
{
    this->tsbMem = tsbMem;
}


template <typename SensorModelRamT, typename SensorModelDeviceT>
template <typename BundleT>
inline void SimulatorVulkan<SensorModelRamT, SensorModelDeviceT>::simulate(Memory<Transform, VULKAN_DEVICE_LOCAL> &tbmMem, BundleT &ret)
{
    VulkanResultsData resultsData{};
    set_vulkan_results_data(ret, resultsData);

    Memory<VulkanResultsData, RAM> resultsMem_ram(1);
    resultsMem_ram[0] = resultsData;
    simulate(tbmMem, resultsMem_ram);
}


template <typename SensorModelRamT, typename SensorModelDeviceT>
template <typename BundleT>
inline BundleT SimulatorVulkan<SensorModelRamT, SensorModelDeviceT>::simulate(Memory<Transform, VULKAN_DEVICE_LOCAL> &tbmMem)
{
    BundleT res;
    resize_memory_bundle<VULKAN_DEVICE_LOCAL>(res, newDimensions.width, newDimensions.height, tbmMem.size());
    simulate(tbmMem, res);
    return res;
}


template<typename SensorModelRamT, typename SensorModelDeviceT>
void SimulatorVulkan<SensorModelRamT, SensorModelDeviceT>::simulate(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem, Memory<VulkanResultsData, RAM>& resultsMem_ram)
{
    newDimensions.depth = tbmMem.size();
    if(newDimensions.width == uint64_t(~0) || newDimensions.height == uint64_t(~0) || newDimensions.depth == uint64_t(~0))
    {
        throw std::runtime_error("incalid new sensor dimensions!");
    }
    if(!check_results_data_size(resultsMem_ram, newDimensions.width*newDimensions.height*newDimensions.depth))
    {
        throw std::runtime_error("resultsMem must have at least enough space for every ray!");
    }


    ShaderDefineFlags newShaderDefines = sensorType | get_result_flags(resultsMem_ram);
    resultsMem = resultsMem_ram;


    bool rerecordCommandBuffer = false;
    //check whether other shaders are needed
    //if they are, a new pipeline need to get fetched
    if(previousShaderDefines != newShaderDefines)
    {
        //update current pipeline/shader configuration
        previousShaderDefines = newShaderDefines;

        pipeline = vulkan_context->getPipeline(newShaderDefines);
        std::cout << "recieved pipeline & shader binding table" << std::endl;

        rerecordCommandBuffer = true;
    }
    //check whether buffers have changed
    //if the previous buffers are not the same, the folloing functions need to be called
    //sensorMem, resultsMem & tsbMem dont have to get checked as they are always the same anyways
    if(previousBuffers.vertexID  != map->scene()->vertexptr->getID() ||
       previousBuffers.indexID   != map->scene()->indexptr->getID()  ||/*
       previousBuffers.sensorID  != sensorMem.getID()  ||
       previousBuffers.resultsID != resultsMem.getID() ||
       previousBuffers.tsbID     != tsbMem.getID()     ||*/
       previousBuffers.tbmID     != tbmMem.getID()     ||
       previousBuffers.tlasID    != map->scene()->as()->this_shared<TopLevelAccelerationStructure>()->getID())
    {
        //update used buffers
        previousBuffers.vertexID  = map->scene()->vertexptr->getID();
        previousBuffers.indexID   = map->scene()->indexptr->getID();
        previousBuffers.sensorID  = sensorMem.getID();
        previousBuffers.resultsID = resultsMem.getID();
        previousBuffers.tsbID     = tsbMem.getID();
        previousBuffers.tbmID     = tbmMem.getID();
        previousBuffers.tlasID    = map->scene()->as()->this_shared<TopLevelAccelerationStructure>()->getID();

        descriptorSet->updateDescriptorSet(map->scene()->vertexptr->getBuffer(), map->scene()->indexptr->getBuffer(), sensorMem.getBuffer(), resultsMem.getBuffer(), tsbMem.getBuffer(), tbmMem.getBuffer(), map->scene()->as()->this_shared<TopLevelAccelerationStructure>());
        std::cout << "updated descriptor set" << std::endl;

        rerecordCommandBuffer = true;
    }
    //check whether dimensions have changed
    //if they have changed the raytracing command needs to be rerecorded
    if(rerecordCommandBuffer ||
       previousDimensions.width  != newDimensions.width  || 
       previousDimensions.height != newDimensions.height || 
       previousDimensions.depth  != newDimensions.depth)
    {
        //update dimensions
        previousDimensions.width = newDimensions.width;
        previousDimensions.height = newDimensions.height;
        previousDimensions.depth = newDimensions.depth;

        commandBuffer->recordRayTracingToCommandBuffer(descriptorSet, pipeline, newDimensions.width, newDimensions.height, newDimensions.depth);
        std::cout << "rerecorded instructions to command buffer" << std::endl;
    }

    commandBuffer->submitRecordedCommandAndWait();
}


template<typename SensorModelRamT, typename SensorModelDeviceT>
void SimulatorVulkan<SensorModelRamT, SensorModelDeviceT>::resetBufferHistory()
{
    previousBuffers.indexID = 0;
    previousBuffers.vertexID = 0;
    previousBuffers.sensorID = 0;
    previousBuffers.resultsID = 0;
    previousBuffers.tsbID = 0;
    previousBuffers.tbmID = 0;
    previousBuffers.tlasID = 0;
}


template<typename SensorModelRamT, typename SensorModelDeviceT>
void SimulatorVulkan<SensorModelRamT, SensorModelDeviceT>::resetPipeline()
{
    pipeline.reset();
}


template<typename SensorModelRamT, typename SensorModelDeviceT>
void SimulatorVulkan<SensorModelRamT, SensorModelDeviceT>::cleanup()
{
    std::cout << "cleaning up..." << std::endl;;

    descriptorSet->cleanup();
    std::cout << "cleaned up descriptor set." << std::endl;

    commandBuffer->cleanup();
    std::cout << "cleaned up command buffer." << std::endl;

    std::cout << "done." << std::endl;
}


template <typename SensorModelRamT, typename SensorModelDeviceT>
inline void SimulatorVulkan<SensorModelRamT, SensorModelDeviceT>::checkTemplateArgs()
{
    if constexpr(std::is_same<SensorModelRamT, SphericalModel>::value)
    {
        sensorType = ShaderDefines::Def_Sphere;
        if constexpr(!std::is_same<SensorModelDeviceT, SphericalModel>::value)
            throw std::runtime_error("constructed illegal simulator (Sphere)");
    }
    else if constexpr(std::is_same<SensorModelRamT, PinholeModel>::value)
    {
        sensorType = ShaderDefines::Def_Pinhole;
        if constexpr(!std::is_same<SensorModelDeviceT, PinholeModel>::value)
            throw std::runtime_error("constructed illegal simulator (Pinhole)");
    }
    else if constexpr(std::is_same<SensorModelRamT, O1DnModel>::value)
    {
        sensorType = ShaderDefines::Def_O1Dn;
        if constexpr(!std::is_same<SensorModelDeviceT, O1DnModel_<VULKAN_DEVICE_LOCAL>>::value)
            throw std::runtime_error("constructed illegal simulator (O1Dn)");
    }
    else if constexpr(std::is_same<SensorModelRamT, OnDnModel>::value)
    {
        sensorType = ShaderDefines::Def_OnDn;
        if constexpr(!std::is_same<SensorModelDeviceT, OnDnModel_<VULKAN_DEVICE_LOCAL>>::value)
            throw std::runtime_error("constructed illegal simulator (OnDn)");
    }
    else
    {
        throw std::runtime_error("constructed illegal simulator");
    }
}

} // namespace rmagine