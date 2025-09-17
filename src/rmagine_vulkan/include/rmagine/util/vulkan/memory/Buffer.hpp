#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>
#include <mutex>

#include <vulkan/vulkan.h>

#include <rmagine/util/VulkanUtil.hpp>
#include <rmagine/util/vulkan/Device.hpp>



namespace rmagine
{

class Buffer
{
private:
    VulkanContextWPtr vulkan_context;
    DevicePtr device = nullptr;

    VkDeviceSize bufferSize = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceAddress deviceAddress = 0;

    std::mutex bufferMtx;

public:
    Buffer(VkDeviceSize bufferSize, VkBufferUsageFlags bufferUsageFlags);

    ~Buffer();

    Buffer(const Buffer&) = delete;
    
    
    VkDeviceAddress getBufferDeviceAddress();

    VkDeviceSize getBufferSize();

    VkBuffer getBuffer();

private:
    /**
     * create the Buffer
     */
    void createBuffer(VkBufferUsageFlags bufferUsageFlags);
};

using BufferPtr = std::shared_ptr<Buffer>;

} // namespace rmagine
