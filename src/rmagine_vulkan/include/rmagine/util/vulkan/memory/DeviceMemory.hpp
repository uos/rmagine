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
    DevicePtr device = nullptr;
    BufferPtr buffer = nullptr;

    VkDeviceMemory deviceMemory = VK_NULL_HANDLE;

public:
    /**
     * THIS CONSTRUCTOR MUST NOT BE CALLED FROM THE CONSTRUCTOR OF THE VULKAN-CONTEXT
     */
    DeviceMemory(VkMemoryPropertyFlags memoryPropertyFlags, BufferPtr buffer);

    DeviceMemory(VkMemoryPropertyFlags memoryPropertyFlags, DevicePtr device, BufferPtr buffer);

    ~DeviceMemory();

    DeviceMemory(const DeviceMemory&) = delete;


    void copyToDeviceMemory(const void* src);

    void copyToDeviceMemory(const void* src, size_t offset, size_t stride);

    void copyFromDeviceMemory(void* dst);

    void copyFromDeviceMemory(void* dst, size_t offset, size_t stride);

    /**
     * free the DeviceMemory
     */
    void cleanup();

    BufferPtr getBuffer();
    
private:
    /**
     * create the DeviceMemory
     */
    void allocateDeviceMemory(VkMemoryPropertyFlags memoryPropertyFlags, bool withAllocateFlags);
};

using DeviceMemoryPtr = std::shared_ptr<DeviceMemory>;

} // namespace rmagine
