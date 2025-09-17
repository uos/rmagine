#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include <rmagine/util/VulkanUtil.hpp>
#include <rmagine/util/vulkan/memory/Buffer.hpp>
#include "Fence.hpp"



namespace rmagine
{

//forward declaration
class DescriptorSet;
using DescriptorSetPtr = std::shared_ptr<DescriptorSet>;
class ShaderBindingTable;
using ShaderBindingTablePtr = std::shared_ptr<ShaderBindingTable>;
class Pipeline;
using PipelinePtr = std::shared_ptr<Pipeline>;



class CommandBuffer
{
private:
    VulkanContextPtr vulkan_context;

    FencePtr fence = nullptr;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

public:
    CommandBuffer(VulkanContextPtr vulkan_context);

    ~CommandBuffer();

    CommandBuffer(const CommandBuffer&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues


    void recordRayTracingToCommandBuffer(DescriptorSetPtr descriptorSet, ShaderBindingTablePtr shaderBindingTable, uint32_t width = 1, uint32_t height = 1, uint32_t depth = 1);

    void recordBuildingASToCommandBuffer(VkAccelerationStructureBuildGeometryInfoKHR& accelerationStructureBuildGeometryInfo, const VkAccelerationStructureBuildRangeInfoKHR* accelerationStructureBuildRangeInfos);

    void recordCopyBufferToCommandBuffer(BufferPtr scrBuffer, BufferPtr dstBuffer, VkDeviceSize size, VkDeviceSize srcOffset = 0, VkDeviceSize dstOffset = 0);

    void submitRecordedCommandAndWait();

    VkCommandBuffer getCommandbuffer();

private:
    void createCommandBuffer();
};

using CommandBufferPtr = std::shared_ptr<CommandBuffer>;

} // namespace rmagine
