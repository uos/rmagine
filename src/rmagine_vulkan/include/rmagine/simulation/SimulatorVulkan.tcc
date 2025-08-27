#include "SimulatorVulkan.hpp"



namespace rmagine
{

template<typename SensorModelRamT>
void SimulatorVulkan<SensorModelRamT>::setTsb(const Memory<Transform, RAM>& tsbMem)
{
    this->tsbMem = tsbMem;
}


template <typename SensorModelRamT>
template <typename BundleT>
inline void SimulatorVulkan<SensorModelRamT>::simulate(Memory<Transform, VULKAN_DEVICE_LOCAL> &tbmMem, BundleT &ret)
{
    VulkanResultsData results{};
    set_vulkan_results_data(ret, results);

    Memory<VulkanResultsData, RAM> resultsMem_ram(1);
    resultsMem_ram[0] = results;
    simulate(tbmMem, resultsMem_ram);
}


template <typename SensorModelRamT>
template <typename BundleT>
inline BundleT SimulatorVulkan<SensorModelRamT>::simulate(Memory<Transform, VULKAN_DEVICE_LOCAL> &tbmMem)
{
    BundleT res;
    resize_memory_bundle<VULKAN_DEVICE_LOCAL>(res, newDimensions.width, newDimensions.height, tbmMem.size());
    simulate(tbmMem, res);
    return res;
}


template<typename SensorModelRamT>
void SimulatorVulkan<SensorModelRamT>::simulate(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem, Memory<VulkanResultsData, RAM>& resultsMem_ram)
{
    newDimensions.depth = tbmMem.size();
    if(newDimensions.width == 0 || newDimensions.height == 0 || newDimensions.depth == 0)
    {
        throw std::runtime_error("invalid new sensor dimensions!");
    }


    ShaderDefineFlags newShaderDefines = sensorType | get_result_flags(resultsMem_ram);


    //upload addresses if neccessary
    updateAddresses(tbmMem, resultsMem_ram);
    

    bool rerecordCommandBuffer = false;
    //check whether other shaders are needed
    //if they are, a new pipeline need to get fetched
    if(previousShaderDefines != newShaderDefines)
    {
        //update current pipeline/shader configuration
        previousShaderDefines = newShaderDefines;

        pipeline = vulkan_context->getPipeline(newShaderDefines);
        std::cout << "recieved pipeline, shaders & shader binding table" << std::endl;

        rerecordCommandBuffer = true;
    }
    //check whether buffers have changed
    //if the previous buffers are not the same, the folloing functions need to be called
    //sensorMem, resultsMem & tsbMem dont have to get checked as they are always the same anyways
    if(previousBuffers.asAddress      != map->scene()->as()->getDeviceAddress() ||
       previousBuffers.mapDataAddress != map->scene()->as()->this_shared<TopLevelAccelerationStructure>()->m_asInstancesDescriptions.getBuffer()->getBufferDeviceAddress())
    {
        //update used buffers
        previousBuffers.asAddress      = map->scene()->as()->getDeviceAddress();
        previousBuffers.mapDataAddress = map->scene()->as()->this_shared<TopLevelAccelerationStructure>()->m_asInstancesDescriptions.getBuffer()->getBufferDeviceAddress();

        descriptorSet->updateDescriptorSet(map->scene()->as(), 
                                           map->scene()->as()->this_shared<TopLevelAccelerationStructure>()->m_asInstancesDescriptions.getBuffer(), 
                                           sensorMem.getBuffer(), resultsMem.getBuffer(), tsbMem.getBuffer(), origsDirsAndTransformsMem.getBuffer());
        std::cout << "updated descriptor set" << std::endl;

        rerecordCommandBuffer = true;
    }
    //check whether dimensions, descriptorset or pipeline have changed
    //if they have changed the raytracing command needs to be rerecorded
    if(rerecordCommandBuffer ||
       previousDimensions.width  != newDimensions.width  || 
       previousDimensions.height != newDimensions.height || 
       previousDimensions.depth  != newDimensions.depth)
    {
        //update dimensions
        previousDimensions.width  = newDimensions.width;
        previousDimensions.height = newDimensions.height;
        previousDimensions.depth  = newDimensions.depth;

        commandBuffer->recordRayTracingToCommandBuffer(descriptorSet, pipeline, newDimensions.width, newDimensions.height, newDimensions.depth);
        std::cout << "(re)recorded instructions to command buffer" << std::endl;
    }

    commandBuffer->submitRecordedCommandAndWait();
}


template<typename SensorModelRamT>
void SimulatorVulkan<SensorModelRamT>::updateAddresses(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem, Memory<VulkanResultsData, RAM>& resultsMem_ram)
{
    if(previousBuffers.resultsAddresses.hitsAddress        != resultsMem_ram[0].hitsAddress        ||
       previousBuffers.resultsAddresses.rangesAddress      != resultsMem_ram[0].rangesAddress      ||
       previousBuffers.resultsAddresses.pointsAddress      != resultsMem_ram[0].pointsAddress      ||
       previousBuffers.resultsAddresses.normalsAddress     != resultsMem_ram[0].normalsAddress     ||
       previousBuffers.resultsAddresses.primitiveIdAddress != resultsMem_ram[0].primitiveIdAddress ||
       previousBuffers.resultsAddresses.instanceIdAddress  != resultsMem_ram[0].instanceIdAddress  ||
       previousBuffers.resultsAddresses.geometryIdAddress  != resultsMem_ram[0].geometryIdAddress)
    {
        previousBuffers.resultsAddresses = resultsMem_ram[0];

        resultsMem = resultsMem_ram;
    }

    if(previousBuffers.origsDirsAndTransformsAddresses.tbmAddress   != tbmMem.getBuffer()->getBufferDeviceAddress() ||
       previousBuffers.origsDirsAndTransformsAddresses.origsAddress != 0 ||
       previousBuffers.origsDirsAndTransformsAddresses.dirsAddress  != 0)
    {
        Memory<VulkanOrigsDirsAndTransformsData, RAM> origsDirsAndTransformsMem_ram(1);
        origsDirsAndTransformsMem_ram[0].tbmAddress   = tbmMem.getBuffer()->getBufferDeviceAddress();
        origsDirsAndTransformsMem_ram[0].origsAddress = 0;
        origsDirsAndTransformsMem_ram[0].dirsAddress  = 0;

        previousBuffers.origsDirsAndTransformsAddresses = origsDirsAndTransformsMem_ram[0];

        origsDirsAndTransformsMem = origsDirsAndTransformsMem_ram;
    }
}


template<typename SensorModelRamT>
void SimulatorVulkan<SensorModelRamT>::resetBufferHistory()
{
    previousBuffers.asAddress = 0;
    previousBuffers.mapDataAddress = 0;

    previousBuffers.resultsAddresses = {};
    previousBuffers.origsDirsAndTransformsAddresses = {};
}


template<typename SensorModelRamT>
void SimulatorVulkan<SensorModelRamT>::resetPipeline()
{
    pipeline.reset();
}


template<typename SensorModelRamT>
void SimulatorVulkan<SensorModelRamT>::cleanup()
{
    std::cout << "cleaning up..." << std::endl;;

    descriptorSet->cleanup();
    std::cout << "cleaned up descriptor set." << std::endl;

    commandBuffer->cleanup();
    std::cout << "cleaned up command buffer." << std::endl;

    std::cout << "done." << std::endl;
}


template <typename SensorModelRamT>
inline void SimulatorVulkan<SensorModelRamT>::checkTemplateArgs()
{
    if constexpr(std::is_same<SensorModelRamT, SphericalModel>::value)
    {
        sensorType = ShaderDefines::Def_Sphere;
    }
    else if constexpr(std::is_same<SensorModelRamT, PinholeModel>::value)
    {
        sensorType = ShaderDefines::Def_Pinhole;
    }
    else if constexpr(std::is_same<SensorModelRamT, O1DnModel>::value)
    {
        sensorType = ShaderDefines::Def_O1Dn;
    }
    else if constexpr(std::is_same<SensorModelRamT, OnDnModel>::value)
    {
        sensorType = ShaderDefines::Def_OnDn;
    }
    else
    {
        throw std::runtime_error("constructed invalid simulator");
    }
}

} // namespace rmagine