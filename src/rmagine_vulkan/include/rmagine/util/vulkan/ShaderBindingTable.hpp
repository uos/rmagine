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

#include "../VulkanUtil.hpp"
#include "../MemoryVulkan.hpp"
#include "Device.hpp"
#include "Shader.hpp"
#include "Pipeline.hpp"



namespace rmagine
{

class ShaderBindingTable
{
private:
    DevicePtr device = nullptr;
    ExtensionFunctionsPtr extensionFunctionsPtr = nullptr;

    Memory<char, VULKAN_DEVICE_LOCAL> shaderBindingTableMemory;

    //shader data
    VkStridedDeviceAddressRegionKHR rchitShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR rgenShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR rmissShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR callableShaderBindingTable{};

public:
    ShaderBindingTable(PipelinePtr pipeline);
    
    ShaderBindingTable(DevicePtr device, PipelinePtr pipeline, ExtensionFunctionsPtr extensionFunctionsPtr);

    ~ShaderBindingTable() {}

    ShaderBindingTable(const ShaderBindingTable&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues


    void cleanup();

    VkStridedDeviceAddressRegionKHR* getRayGenerationShaderBindingTablePtr();

    VkStridedDeviceAddressRegionKHR* getClosestHitShaderBindingTablePtr();

    VkStridedDeviceAddressRegionKHR* getMissShaderBindingTablePtr();

    VkStridedDeviceAddressRegionKHR* getCallableShaderBindingTablePtr();

private:
    void createShaderBindingTable(PipelinePtr pipeline);
};

using ShaderBindingTablePtr = std::shared_ptr<ShaderBindingTable>;

} // namespace rmagine
