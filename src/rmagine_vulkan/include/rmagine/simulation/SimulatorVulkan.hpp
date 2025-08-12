#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include "../util/VulkanContext.hpp"
#include "../../rmagine_core/math/Types.hpp"
#include "../map/VulkanMap.hpp"
#include "simulatorComponents/DescriptorSet.hpp"
#include "../util/contextComponents/Pipeline.hpp"
#include "SimulatorVulkanUtil.hpp"



namespace rmagine
{

template<typename SensorModelRamT, typename SensorModelDeviceT>
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

    Memory<SensorModelDeviceT, VULKAN_DEVICE_LOCAL> sensorMem;
    Memory<Transform, VULKAN_DEVICE_LOCAL> tsbMem;
    Memory<VulkanResultsData, VULKAN_DEVICE_LOCAL> resultsMem;

    //for checking whether buffers have changed
    struct PreviousBuffers{
        size_t vertexID = 0;
        size_t indexID = 0;
        size_t sensorID = 0;
        size_t resultsID = 0;
        size_t tsbID = 0;
        size_t tbmID = 0;
        size_t tlasID = 0;
    }previousBuffers;

    ShaderDefineFlags previousShaderDefines = 0;

    struct PreviousDimensions{
        uint64_t width = uint64_t(~0);
        uint64_t height = uint64_t(~0);
        uint64_t depth = uint64_t(~0);
    }previousDimensions;

    struct NewDimensions{
        uint64_t width = uint64_t(~0);
        uint64_t height = uint64_t(~0);
        uint64_t depth = uint64_t(~0);
    }newDimensions;


public:
    SimulatorVulkan(VulkanMapPtr map) : vulkan_context(get_vulkan_context()), map(map), sensorMem(1), tsbMem(1), resultsMem(1)
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

    void setTsb(Memory<Transform, RAM>& tsbMem);

    void setModel(Memory<SensorModelRamT, RAM>& sensorMem_ram);

    template<typename BundleT>
    void simulate(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem, BundleT& ret);

    template<typename BundleT>
    BundleT simulate(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem);
    
    void simulate(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem, Memory<VulkanResultsData, RAM>& resultsMem_ram);

protected:
    void resetBufferHistory();

    void resetPipeline();

    void cleanup();

private:
    void checkTemplateArgs();
};

} // namespace rmagine

#include "SimulatorVulkan.tcc"
