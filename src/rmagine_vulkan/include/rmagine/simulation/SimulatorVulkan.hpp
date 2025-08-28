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
#include <rmagine/util/vulkan/Pipeline.hpp>
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
    PipelinePtr pipeline = nullptr;

    //Simulator internal Memory
    Memory<SensorModelRamT, VULKAN_DEVICE_LOCAL> sensorMem;
    Memory<Transform, VULKAN_DEVICE_LOCAL> tsbMem;

    //Memory for the deviceAddresses of external Buffers
    //moving these buffers to the gpu via their deviceAddresses means that the Descriptorset does not have to get updated so often
    //(that would take more time and would mean and the command would have to get rerecorded as well)
    Memory<VulkanResultsAddresses, VULKAN_DEVICE_LOCAL> resultsMem;
    Memory<VulkanTbmAndSensorSpecificAddresses, VULKAN_DEVICE_LOCAL> origsDirsAndTransformsMem;

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
    SimulatorVulkan(VulkanMapPtr map) : vulkan_context(get_vulkan_context()), map(map), sensorMem(1), tsbMem(1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT), resultsMem(1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT), origsDirsAndTransformsMem(1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)
    {
        checkTemplateArgs();

        commandBuffer = std::make_shared<CommandBuffer>();
        descriptorSet = std::make_shared<DescriptorSet>();
    }

    ~SimulatorVulkan()
    {
        std::cout << "destroying SimulatorVulkan" << std::endl;
        resetPipeline();
        descriptorSet->cleanup();
        commandBuffer->cleanup();
    }

    SimulatorVulkan(const SimulatorVulkan&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues

    void setTsb(const Memory<Transform, RAM>& tsbMem);
    void setTsb(const Transform& Tsb);//TODO

    void setModel(const Memory<SensorModelRamT, RAM>& sensorMem_ram);
    void setModel(const SensorModelRamT& sensorMem_ram);//TODO

    template<typename BundleT>
    BundleT simulate(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem);

    template<typename BundleT>
    void simulate(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem, BundleT& ret);
    
    void simulate(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem, Memory<VulkanResultsAddresses, RAM>& resultsMem_ram);

protected:
    void updateResultsAddresses(Memory<VulkanResultsAddresses, RAM>& resultsMem_ram);
    virtual void updateTbmAndSensorSpecificAddresses(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem);

    void resetAddressHistory();

    void resetPipeline();

private:
    void checkTemplateArgs();
};

} // namespace rmagine

#include "SimulatorVulkan.tcc"
