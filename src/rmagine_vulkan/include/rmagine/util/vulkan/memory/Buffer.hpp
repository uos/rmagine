#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include <rmagine/util/VulkanUtil.hpp>
#include <rmagine/util/vulkan/Device.hpp>



namespace rmagine
{

class Buffer
{
private:
    DevicePtr device = nullptr;
    ExtensionFunctionsPtr extensionFunctionsPtr = nullptr;

    VkDeviceSize bufferSize = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceAddress deviceAddress = 0;
public:
    /**
     * THIS CONSTRUCTOR MUST NOT BE CALLED FROM THE CONSTRUCTOR OF THE VULKAN-CONTEXT
     */
    Buffer(VkDeviceSize bufferSize, VkBufferUsageFlags bufferUsageFlags);

    Buffer(VkDeviceSize bufferSize, VkBufferUsageFlags bufferUsageFlags, DevicePtr device, ExtensionFunctionsPtr extensionFunctionsPtr);

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
