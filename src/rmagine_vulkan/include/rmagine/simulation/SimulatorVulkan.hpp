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

    Memory<SensorModelRamT, VULKAN_DEVICE_LOCAL> sensorMem;
    Memory<Transform, VULKAN_DEVICE_LOCAL> tsbMem;
    Memory<VulkanResultsData, VULKAN_DEVICE_LOCAL> resultsMem;
    Memory<VulkanOrigsDirsAndTransformsData, VULKAN_DEVICE_LOCAL> origsDirsAndTransformsMem;

    //for checking whether buffers have changed
    struct PreviousBuffers{
        VkDeviceAddress asAddress = 0;
        VkDeviceAddress mapDataAddress = 0;

        VulkanResultsData resultsAddresses{};
        VulkanOrigsDirsAndTransformsData origsDirsAndTransformsAddresses{};

    }previousBuffers;//TODO: rename previous addresses

    ShaderDefineFlags previousShaderDefines = 0;

    struct PreviousDimensions{
        uint64_t width = 0;
        uint64_t height = 0;
        uint64_t depth = 0;
    }previousDimensions;

    struct NewDimensions{
        uint64_t width = 0;
        uint64_t height = 0;
        uint64_t depth = 0;
    }newDimensions;


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
        resetBufferHistory();
        resetPipeline();
        cleanup();
    }

    SimulatorVulkan(const SimulatorVulkan&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues

    void setTsb(const Memory<Transform, RAM>& tsbMem);
    void setTsb(const Transform& Tsb);//TODO

    void setModel(const Memory<SensorModelRamT, RAM>& sensorMem_ram);
    void setModel(const SensorModelRamT& sensorMem_ram);//TODO

    template<typename BundleT>
    void simulate(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem, BundleT& ret);

    template<typename BundleT>
    BundleT simulate(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem);
    
    void simulate(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem, Memory<VulkanResultsData, RAM>& resultsMem_ram);

protected:
    virtual void updateAddresses(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem, Memory<VulkanResultsData, RAM>& resultsMem_ram);

    void resetBufferHistory();

    void resetPipeline();

    void cleanup();

private:
    void checkTemplateArgs();
};

} // namespace rmagine

#include "SimulatorVulkan.tcc"
