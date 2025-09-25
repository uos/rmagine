#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include "Buffer.hpp"



namespace rmagine
{

class DeviceMemory
{
private:
    VulkanContextWPtr vulkan_context;
    DevicePtr device = nullptr;

    BufferPtr buffer = nullptr;

    VkDeviceMemory deviceMemory = VK_NULL_HANDLE;

    std::mutex deviceMemoryMtx;

public:
    DeviceMemory(VkMemoryPropertyFlags memoryPropertyFlags, BufferPtr buffer);

    ~DeviceMemory();

    DeviceMemory(const DeviceMemory&) = delete;


    void copyToDeviceMemory(const void* src);

    void copyToDeviceMemory(const void* src, size_t offset, size_t stride);

    void copyFromDeviceMemory(void* dst);

    void copyFromDeviceMemory(void* dst, size_t offset, size_t stride);

    int getMemoryHandle();

    BufferPtr getBuffer();
    
private:
    /**
     * create the DeviceMemory
     */
    void allocateDeviceMemory(VkMemoryPropertyFlags memoryPropertyFlags, bool withAllocateFlags);
};

using DeviceMemoryPtr = std::shared_ptr<DeviceMemory>;

} // namespace rmagine
