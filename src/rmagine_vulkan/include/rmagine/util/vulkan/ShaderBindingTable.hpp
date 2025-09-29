#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include <rmagine/util/VulkanContextUtil.hpp>
#include <rmagine/types/MemoryVulkan.hpp>
#include "ShaderUtil.hpp"
#include "Shader.hpp"
#include "RayTracingPipeline.hpp"



namespace rmagine
{

class ShaderBindingTable
{
private:
    DeviceWPtr device;

    RayTracingPipelinePtr pipeline = nullptr;

    Memory<char, DEVICE_LOCAL_VULKAN> shaderBindingTableMemory;

    //shader data
    VkStridedDeviceAddressRegionKHR rchitShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR rgenShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR rmissShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR callableShaderBindingTable{};

public:
    ShaderBindingTable(DeviceWPtr device, RayTracingPipelineLayoutWPtr pipelineLayout, ShaderDefineFlags shaderDefines);

    ~ShaderBindingTable();

    ShaderBindingTable(const ShaderBindingTable&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues


    RayTracingPipelinePtr getPipeline();

    VkStridedDeviceAddressRegionKHR* getRayGenerationShaderBindingTablePtr();

    VkStridedDeviceAddressRegionKHR* getClosestHitShaderBindingTablePtr();

    VkStridedDeviceAddressRegionKHR* getMissShaderBindingTablePtr();

    VkStridedDeviceAddressRegionKHR* getCallableShaderBindingTablePtr();

private:
    void createShaderBindingTable();
};

using ShaderBindingTablePtr = std::shared_ptr<ShaderBindingTable>;

} // namespace rmagine
