#include "rmagine/util/vulkan/memory/Buffer.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

Buffer::Buffer(VkDeviceSize bufferSize, VkBufferUsageFlags bufferUsageFlags) : device(get_vulkan_context()->getDevice()), extensionFunctionsPtr(get_vulkan_context()->getExtensionFunctionsPtr()), bufferSize(bufferSize)
{
    createBuffer(bufferUsageFlags);
}

Buffer::Buffer(VkDeviceSize bufferSize, VkBufferUsageFlags bufferUsageFlags, DevicePtr device, ExtensionFunctionsPtr extensionFunctionsPtr) : device(device), extensionFunctionsPtr(extensionFunctionsPtr), bufferSize(bufferSize)
{
    createBuffer(bufferUsageFlags);
}



void Buffer::createBuffer(VkBufferUsageFlags bufferUsageFlags)
{
    if(buffer != VK_NULL_HANDLE)
    {
        throw std::runtime_error("tried to create a buffer that already exists!");
    }

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = bufferSize;
    bufferCreateInfo.usage = bufferUsageFlags;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferCreateInfo.queueFamilyIndexCount = 1;
    bufferCreateInfo.pQueueFamilyIndices = device->getQueueFamilyIndexPtr();

    if(vkCreateBuffer(device->getLogicalDevice(), &bufferCreateInfo, nullptr, &buffer) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create buffer!");
    }
}



void Buffer::cleanup()
{
    if(buffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(device->getLogicalDevice(), buffer, nullptr);
        buffer = VK_NULL_HANDLE;
    }
}



VkDeviceAddress Buffer::getBufferDeviceAddress()
{
    if(buffer == VK_NULL_HANDLE)
    {
        throw std::runtime_error("tried to get the buffer device address without having created a buffer first!");
    }

    if(deviceAddress != 0)
        return deviceAddress;

    VkBufferDeviceAddressInfo bufferDeviceAddressInfo{};
    bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bufferDeviceAddressInfo.buffer = buffer;
    
    deviceAddress = extensionFunctionsPtr->pvkGetBufferDeviceAddressKHR(device->getLogicalDevice(), &bufferDeviceAddressInfo);
    return deviceAddress;
}

VkDeviceSize Buffer::getBufferSize()
{
    return bufferSize;
}

VkBuffer Buffer::getBuffer()
{
    return buffer;
}

} // namespace rmagine
