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

#include <rmagine/util/VulkanUtil.hpp>
#include <rmagine/util/vulkan/Device.hpp>
#include <rmagine/util/vulkan/CommandPool.hpp>
#include <rmagine/util/vulkan/DescriptorSetLayout.hpp>
#include <rmagine/util/vulkan/PipelineLayout.hpp>
#include <rmagine/util/vulkan/CommandPool.hpp>
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
    DevicePtr device = nullptr;
    ExtensionFunctionsPtr extensionFunctionsPtr = nullptr;
    CommandPoolPtr commandPool = nullptr;
    PipelineLayoutPtr pipelineLayout = nullptr;

    FencePtr fence = nullptr;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

public:
    CommandBuffer();

    CommandBuffer(DevicePtr device, ExtensionFunctionsPtr extensionFunctionsPtr, CommandPoolPtr commandPool, PipelineLayoutPtr pipelineLayout);

    ~CommandBuffer() {}

    CommandBuffer(const CommandBuffer&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues


    void recordRayTracingToCommandBuffer(DescriptorSetPtr descriptorSet, PipelinePtr pipeline, uint32_t width = 1, uint32_t height = 1, uint32_t depth = 1);

    void recordBuildingASToCommandBuffer(VkAccelerationStructureBuildGeometryInfoKHR& accelerationStructureBuildGeometryInfo, const VkAccelerationStructureBuildRangeInfoKHR* accelerationStructureBuildRangeInfos);

    void recordCopyBufferToCommandBuffer(BufferPtr scrBuffer, BufferPtr dstBuffer);

    void submitRecordedCommandAndWait();

    void cleanup();

    VkCommandBuffer getCommandbuffer();

    VkCommandBuffer* getCommandbufferPtr();

private:
    void createCommandBuffer();
};

using CommandBufferPtr = std::shared_ptr<CommandBuffer>;

} // namespace rmagine
