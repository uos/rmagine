#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include <rmagine/util/VulkanUtil.hpp>
#include <rmagine/types/MemoryVulkan.hpp>
#include "ShaderUtil.hpp"
#include "Shader.hpp"
#include "Pipeline.hpp"



namespace rmagine
{

class ShaderBindingTable
{
private:
    VulkanContextWPtr vulkan_context;

    PipelinePtr pipeline = nullptr;

    Memory<char, DEVICE_LOCAL_VULKAN> shaderBindingTableMemory;

    //shader data
    VkStridedDeviceAddressRegionKHR rchitShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR rgenShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR rmissShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR callableShaderBindingTable{};

public:
    ShaderBindingTable(VulkanContextWPtr vulkan_context, ShaderDefineFlags shaderDefines);

    ~ShaderBindingTable();

    ShaderBindingTable(const ShaderBindingTable&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues


    PipelinePtr getPipeline();

    VkStridedDeviceAddressRegionKHR* getRayGenerationShaderBindingTablePtr();

    VkStridedDeviceAddressRegionKHR* getClosestHitShaderBindingTablePtr();

    VkStridedDeviceAddressRegionKHR* getMissShaderBindingTablePtr();

    VkStridedDeviceAddressRegionKHR* getCallableShaderBindingTablePtr();

private:
    void createShaderBindingTable();
};

using ShaderBindingTablePtr = std::shared_ptr<ShaderBindingTable>;

} // namespace rmagine
