#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>
#include <mutex>

#include <vulkan/vulkan.h>

#include <rmagine/util/VulkanContext.hpp>
#include <rmagine/util/VulkanUtil.hpp>
#include <rmagine/util/vulkan/memory/Buffer.hpp>
#include <rmagine/util/vulkan/memory/DeviceMemory.hpp>
#include <rmagine/util/vulkan/command/CommandBuffer.hpp>



namespace rmagine
{

enum VulkanMemoryUsage
{
    Usage_Default,
    Usage_Uniform,
    Usage_AccelerationStructureMeshData,
    Usage_AccelerationStructureInstanceData,
    Usage_AccelerationStructureScratch,
    Usage_AccelerationStructure,
    Usage_ShaderBindingTable,
    //Last Element: only use it to get the size of this enum excluding this element
    VULKAN_MEMORY_USEAGE_SIZE
};

VkBufferUsageFlags get_buffer_usage_flags(VulkanMemoryUsage memUsage);

size_t get_new_mem_id();



namespace vulkan
{

void memcpyDeviceToDevice(BufferPtr srcBuffer, BufferPtr dstBuffer, VkDeviceSize size, VkDeviceSize srcOffset, VkDeviceSize dstOffset);

void memcpyHostToDevice(const void* src, DeviceMemoryPtr dstDeviceMemory, VkDeviceSize size, VkDeviceSize dstOffset);

void memcpyDeviceToHost(DeviceMemoryPtr srcDeviceMemory, void* dst, VkDeviceSize size, VkDeviceSize srcOffset);

} // namespace vulkan

} // namespace rmagine
