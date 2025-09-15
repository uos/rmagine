#include "SimulatorVulkan.hpp"



namespace rmagine
{

template<typename SensorModelRamT>
SimulatorVulkan<SensorModelRamT>::SimulatorVulkan(VulkanMapPtr map) : 
    vulkan_context(get_vulkan_context()), map(map), sensorMem(1), 
    tsbMem(1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT), 
    resultsMem(1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT), 
    tbmAndSensorSpecificMem(1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)
    // uniform buffers might slightly increase performance for small readonly buffers
{
    checkTemplateArgs();

    commandBuffer = std::make_shared<CommandBuffer>(vulkan_context);
    descriptorSet = std::make_shared<DescriptorSet>(vulkan_context);
}

template<typename SensorModelRamT>
SimulatorVulkan<SensorModelRamT>::~SimulatorVulkan()
{
    std::cout << "Destroying SimulatorVulkan" << std::endl;
    resetShaderBindingTable();
    descriptorSet.reset();
    commandBuffer.reset();
    std::cout << "SimulatorVulkan destroyed" << std::endl;
}



template<typename SensorModelRamT>
void SimulatorVulkan<SensorModelRamT>::setMap(VulkanMapPtr map)
{
    this->map = map;
}


template<typename SensorModelRamT>
void SimulatorVulkan<SensorModelRamT>::setTsb(const Memory<Transform, RAM>& tsbMem)
{
    this->tsbMem = tsbMem;
}


template<typename SensorModelRamT>
void SimulatorVulkan<SensorModelRamT>::setTsb(const Transform& Tsb)
{
    Memory<Transform, RAM> tmp(1);
    tmp[0] = Tsb;
    setTsb(tmp);
}


template <typename SensorModelRamT>
template<typename BundleT>
void SimulatorVulkan<SensorModelRamT>::simulate(const Transform& tbm, BundleT& ret)
{
    Memory<Transform, RAM> tbmMem_ram(1);
    tbmMem_ram[0] = tbm;
    Memory<Transform, VULKAN_DEVICE_LOCAL> tbmMem(1);
    tbmMem = tbmMem_ram;
    simulate(tbmMem, ret);
}


template <typename SensorModelRamT>
template <typename BundleT>
BundleT SimulatorVulkan<SensorModelRamT>::simulate(Memory<Transform, VULKAN_DEVICE_LOCAL> &tbmMem)
{
    BundleT res;
    resize_memory_bundle<VULKAN_DEVICE_LOCAL>(res, newDimensions.width, newDimensions.height, tbmMem.size());
    simulate(tbmMem, res);
    return res;
}


template <typename SensorModelRamT>
template <typename BundleT>
void SimulatorVulkan<SensorModelRamT>::simulate(Memory<Transform, VULKAN_DEVICE_LOCAL> &tbmMem, BundleT &ret)
{
    VulkanResultsAddresses results{};
    set_vulkan_results_data(ret, results);

    Memory<VulkanResultsAddresses, RAM> resultsMem_ram(1);
    resultsMem_ram[0] = results;
    simulate(tbmMem, resultsMem_ram);
}


template<typename SensorModelRamT>
void SimulatorVulkan<SensorModelRamT>::simulate(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem, Memory<VulkanResultsAddresses, RAM>& resultsMem_ram)
{
    newDimensions.depth = tbmMem.size();
    if(newDimensions.width == 0 || newDimensions.height == 0 || newDimensions.depth == 0)
    {
        throw std::runtime_error("invalid new sensor dimensions!");
    }


    //upload addresses if neccessary
    updateResultsAddresses(resultsMem_ram);
    updateTbmAndSensorSpecificAddresses(tbmMem);
    

    bool rerecordCommandBuffer = false;
    //check whether other shaders are needed
    //if they are, a new shaderBindingTable (+ pipeline & shader) need to get fetched
    ShaderDefineFlags newShaderDefines = sensorType | get_result_flags(resultsMem_ram);
    if(previousShaderDefines != newShaderDefines)
    {
        //update current shaderBindingTable configuration
        previousShaderDefines = newShaderDefines;

        shaderBindingTable = vulkan_context->getShaderBindingTable(newShaderDefines);
        std::cout << "recieved shader binding table with pipeline & shaders" << std::endl;

        rerecordCommandBuffer = true;
    }
    //check whether buffers have changed
    //if the previous buffers are not the same, the folloing functions need to be called
    //sensorMem, resultsMem & tsbMem dont have to get checked as they are always the same anyways
    if(previousAddresses.asAddress      != map->scene()->as()->getDeviceAddress() ||
       previousAddresses.mapDataAddress != map->scene()->as()->this_shared<TopLevelAccelerationStructure>()->m_asInstancesDescriptions.getBuffer()->getBufferDeviceAddress())
    {
        //update used buffers
        previousAddresses.asAddress      = map->scene()->as()->getDeviceAddress();
        previousAddresses.mapDataAddress = map->scene()->as()->this_shared<TopLevelAccelerationStructure>()->m_asInstancesDescriptions.getBuffer()->getBufferDeviceAddress();

        descriptorSet->updateDescriptorSet(map->scene()->as(), 
                                           map->scene()->as()->this_shared<TopLevelAccelerationStructure>()->m_asInstancesDescriptions.getBuffer(), 
                                           sensorMem.getBuffer(), resultsMem.getBuffer(), tsbMem.getBuffer(), tbmAndSensorSpecificMem.getBuffer());
        std::cout << "updated descriptor set" << std::endl;

        rerecordCommandBuffer = true;
    }
    //check whether dimensions, descriptorset or shaderBindingTable have changed
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

        commandBuffer->recordRayTracingToCommandBuffer(descriptorSet, shaderBindingTable, newDimensions.width, newDimensions.height, newDimensions.depth);
        std::cout << "(re)recorded instructions to command buffer" << std::endl;
    }

    commandBuffer->submitRecordedCommandAndWait();
}


template<typename SensorModelRamT>
void SimulatorVulkan<SensorModelRamT>::updateResultsAddresses(Memory<VulkanResultsAddresses, RAM>& resultsMem_ram)
{
    if(previousAddresses.resultsAddresses.hitsAddress        != resultsMem_ram[0].hitsAddress        ||
       previousAddresses.resultsAddresses.rangesAddress      != resultsMem_ram[0].rangesAddress      ||
       previousAddresses.resultsAddresses.pointsAddress      != resultsMem_ram[0].pointsAddress      ||
       previousAddresses.resultsAddresses.normalsAddress     != resultsMem_ram[0].normalsAddress     ||
       previousAddresses.resultsAddresses.primitiveIdAddress != resultsMem_ram[0].primitiveIdAddress ||
       previousAddresses.resultsAddresses.instanceIdAddress  != resultsMem_ram[0].instanceIdAddress  ||
       previousAddresses.resultsAddresses.geometryIdAddress  != resultsMem_ram[0].geometryIdAddress)
    {
        previousAddresses.resultsAddresses = resultsMem_ram[0];

        resultsMem = resultsMem_ram;
    }
}


template<typename SensorModelRamT>
void SimulatorVulkan<SensorModelRamT>::updateTbmAndSensorSpecificAddresses(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem)
{
    if(previousAddresses.tbmAndSensorSpecificAddresses.tbmAddress   != tbmMem.getBuffer()->getBufferDeviceAddress() ||
       previousAddresses.tbmAndSensorSpecificAddresses.origsAddress != 0 ||
       previousAddresses.tbmAndSensorSpecificAddresses.dirsAddress  != 0)
    {
        Memory<VulkanTbmAndSensorSpecificAddresses, RAM> origsDirsAndTransformsMem_ram(1);
        origsDirsAndTransformsMem_ram[0].tbmAddress   = tbmMem.getBuffer()->getBufferDeviceAddress();
        origsDirsAndTransformsMem_ram[0].origsAddress = 0;
        origsDirsAndTransformsMem_ram[0].dirsAddress  = 0;

        previousAddresses.tbmAndSensorSpecificAddresses = origsDirsAndTransformsMem_ram[0];

        tbmAndSensorSpecificMem = origsDirsAndTransformsMem_ram;
    }
}


template<typename SensorModelRamT>
void SimulatorVulkan<SensorModelRamT>::resetAddressHistory()
{
    previousAddresses.asAddress = 0;
    previousAddresses.mapDataAddress = 0;

    previousAddresses.resultsAddresses = {};
    previousAddresses.tbmAndSensorSpecificAddresses = {};
}


template<typename SensorModelRamT>
void SimulatorVulkan<SensorModelRamT>::resetShaderBindingTable()
{
    shaderBindingTable.reset();
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