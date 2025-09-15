#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include <rmagine/math/types.h>
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/VulkanContext.hpp>
#include <rmagine/util/vulkan/ShaderBindingTable.hpp>
#include <rmagine/util/VulkanContext.hpp>
#include <rmagine/map/VulkanMap.hpp>
#include "vulkan/DescriptorSet.hpp"
#include "SimulatorVulkanUtil.hpp"



namespace rmagine
{

template<typename SensorModelRamT>
class SimulatorVulkan
{
private:
    ShaderDefines sensorType;

protected:
    VulkanContextPtr vulkan_context = nullptr;
    VulkanMapPtr map = nullptr;
    
    CommandBufferPtr commandBuffer = nullptr;
    DescriptorSetPtr descriptorSet = nullptr;
    ShaderBindingTablePtr shaderBindingTable = nullptr;

    //Simulator internal Memory
    Memory<SensorModelRamT, VULKAN_DEVICE_LOCAL> sensorMem;
    Memory<Transform, VULKAN_DEVICE_LOCAL> tsbMem;

    //Simulator internal Memory: Memory for the deviceAddresses of external Buffers
    //moving these buffers to the gpu via their deviceAddresses means that the Descriptorset does not have to get updated so often
    //(that would take more time and would mean and the command would have to get rerecorded as well)
    Memory<VulkanResultsAddresses, VULKAN_DEVICE_LOCAL> resultsMem;
    Memory<VulkanTbmAndSensorSpecificAddresses, VULKAN_DEVICE_LOCAL> tbmAndSensorSpecificMem;

    //for checking whether buffers have changed
    struct PreviousAddresses{
        VkDeviceAddress asAddress = 0;
        VkDeviceAddress mapDataAddress = 0;

        VulkanResultsAddresses resultsAddresses{};
        VulkanTbmAndSensorSpecificAddresses tbmAndSensorSpecificAddresses{};
    }previousAddresses;

    ShaderDefineFlags previousShaderDefines = 0;

    VulkanDimensions previousDimensions;

    VulkanDimensions newDimensions;

public:
    SimulatorVulkan(VulkanMapPtr map);

    ~SimulatorVulkan();

    SimulatorVulkan(const SimulatorVulkan&) = delete;//TODO: maybe create custom copy constructor (needs to ceate its own descriporSet, CommandBuffer, sensorMem, tsbMem, resultsMem & tbmAndSensorSpecificMem)


    void setMap(VulkanMapPtr map);

    void setTsb(const Memory<Transform, RAM>& tsbMem);
    void setTsb(const Transform& Tsb);

    void setModel(const Memory<SensorModelRamT, RAM>& sensorMem_ram);
    void setModel(const SensorModelRamT& sensor);

    template<typename BundleT>
    void simulate(const Transform& tbm, BundleT& ret);

    template<typename BundleT>
    BundleT simulate(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem);

    template<typename BundleT>
    void simulate(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem, BundleT& ret);
    
    void simulate(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem, Memory<VulkanResultsAddresses, RAM>& resultsMem_ram);

protected:
    void updateResultsAddresses(Memory<VulkanResultsAddresses, RAM>& resultsMem_ram);
    virtual void updateTbmAndSensorSpecificAddresses(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem);

    void resetAddressHistory();

    void resetShaderBindingTable();

private:
    void checkTemplateArgs();
};

} // namespace rmagine

#include "SimulatorVulkan.tcc"
