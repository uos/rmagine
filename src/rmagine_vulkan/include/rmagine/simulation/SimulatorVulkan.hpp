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
    Memory<SensorModelRamT, DEVICE_LOCAL_VULKAN> sensorMem;
    Memory<Transform, DEVICE_LOCAL_VULKAN> tsbMem;

    //Simulator internal Memory: Memory for the deviceAddresses of external Buffers
    //moving these buffers to the gpu via their deviceAddresses means that the Descriptorset does not have to get updated so often
    //(that would take more time and would mean and the command would have to get rerecorded as well)
    Memory<VulkanResultsAddresses, DEVICE_LOCAL_VULKAN> resultsMem;
    Memory<VulkanTbmAndSensorSpecificAddresses, DEVICE_LOCAL_VULKAN> tbmAndSensorSpecificMem;

    //for checking whether buffers have changed
    struct PreviousAddresses{
        //cant use deviceaddress for this
        //as the new buffer/as could thoretically coincidentally have the same deviceaddress as the old buffer/as
        //thus another unique identifyer is needed 
        //maybe there is a better option, as there are only finitely many (SIZE_MAX) ids?
        size_t asID = 0;
        size_t mapDataID = 0;

        VulkanResultsAddresses resultsAddresses{};
        VulkanTbmAndSensorSpecificAddresses tbmAndSensorSpecificAddresses{};
    }previousAddresses;

    ShaderDefineFlags previousShaderDefines = 0;

    VulkanDimensions previousDimensions;

    VulkanDimensions newDimensions;

public:
    SimulatorVulkan(VulkanMapPtr map);

    ~SimulatorVulkan();

    SimulatorVulkan(const SimulatorVulkan& other);


    void setMap(VulkanMapPtr map);

    void setTsb(const Memory<Transform, RAM>& tsbMem);
    void setTsb(const Transform& Tsb);

    void setModel(const Memory<SensorModelRamT, RAM>& sensorMem_ram);
    void setModel(const SensorModelRamT& sensor);

    template<typename BundleT>
    void simulate(const Transform& tbm, BundleT& ret);

    template<typename BundleT>
    BundleT simulate(Memory<Transform, DEVICE_LOCAL_VULKAN>& tbmMem);

    template<typename BundleT>
    void simulate(Memory<Transform, DEVICE_LOCAL_VULKAN>& tbmMem, BundleT& ret);
    
    void simulate(Memory<Transform, DEVICE_LOCAL_VULKAN>& tbmMem, Memory<VulkanResultsAddresses, RAM>& resultsMem_ram);

protected:
    void updateResultsAddresses(Memory<VulkanResultsAddresses, RAM>& resultsMem_ram);
    virtual void updateTbmAndSensorSpecificAddresses(Memory<Transform, DEVICE_LOCAL_VULKAN>& tbmMem);

    void resetAddressHistory();

    void resetShaderBindingTable();

private:
    void checkTemplateArgs();
};

} // namespace rmagine

#include "SimulatorVulkan.tcc"
