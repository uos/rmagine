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

/**
 * translates an element of the VulkanMemoryUsage enum into vulkan buffer usage flags
 * 
 * @param memUsage memory usage
 * 
 * @return corressponding vulkan buffer usage flags
 */
VkBufferUsageFlags get_buffer_usage_flags(VulkanMemoryUsage memUsage);

/**
 * generates and returns a new mem id
 * 
 * @return new mem id
 */
size_t get_new_mem_id();



namespace vulkan
{

/**
 * copy data from (host visible & device local) vulkan device memory to (host visible & device local) vulkan device memory
 * 
 * @param srcBuffer buffer of the source memory
 * 
 * @param dstBuffer buffer of the destination memory
 * 
 * @param size number of bytes that get copied
 * 
 * @param srcOffset offset into the source memory
 * 
 * @param dstOffset offset into the destination memory
 */
void memcpyDeviceToDevice(BufferPtr srcBuffer, BufferPtr dstBuffer, VkDeviceSize size, VkDeviceSize srcOffset, VkDeviceSize dstOffset);

/**
 * copy data from host memory to host visible vulkan device memory
 * 
 * @param src pointer to source host memory
 * 
 * @param dstDeviceMemory device memory of the destination memory
 * 
 * @param size number of bytes that get copied
 * 
 * @param dstOffset offset into the destination memory
 */
void memcpyHostToDevice(const void* src, DeviceMemoryPtr dstDeviceMemory, VkDeviceSize size, VkDeviceSize dstOffset);

/**
 * copy data from host visible vulkan device memory to host memory
 * 
 * @param srcDeviceMemory device memory of the source memory
 * 
 * @param dst pointer to destination host memory
 * 
 * @param size number of bytes that get copied
 * 
 * @param srcOffset offset into the source memory
 */
void memcpyDeviceToHost(DeviceMemoryPtr srcDeviceMemory, void* dst, VkDeviceSize size, VkDeviceSize srcOffset);

} // namespace vulkan

} // namespace rmagine
