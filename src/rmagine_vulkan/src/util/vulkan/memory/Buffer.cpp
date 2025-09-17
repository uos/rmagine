#include "rmagine/util/vulkan/memory/Buffer.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

Buffer::Buffer(VkDeviceSize bufferSize, VkBufferUsageFlags bufferUsageFlags) :
    vulkan_context(get_vulkan_context_weak()),
    device(vulkan_context.lock()->getDevice()),
    bufferSize(bufferSize)
{
    createBuffer(bufferUsageFlags);
}

Buffer::~Buffer()
{
    if(buffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(device->getLogicalDevice(), buffer, nullptr);
    }
    device.reset();
}



void Buffer::createBuffer(VkBufferUsageFlags bufferUsageFlags)
{
    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = bufferSize;
    bufferCreateInfo.usage = bufferUsageFlags;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferCreateInfo.queueFamilyIndexCount = 1;
    bufferCreateInfo.pQueueFamilyIndices = vulkan_context.lock()->getDevice()->getQueueFamilyIndexPtr();

    if(vkCreateBuffer(vulkan_context.lock()->getDevice()->getLogicalDevice(), &bufferCreateInfo, nullptr, &buffer) != VK_SUCCESS)
    {
        throw std::runtime_error("[Buffer::createBuffer()] ERROR - failed to create buffer!");
    }
}


VkDeviceAddress Buffer::getBufferDeviceAddress()
{
    std::lock_guard<std::mutex> guard(bufferMtx);
    if(deviceAddress != 0)
        return deviceAddress;

    VkBufferDeviceAddressInfo bufferDeviceAddressInfo{};
    bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bufferDeviceAddressInfo.buffer = buffer;

    deviceAddress = vkGetBufferDeviceAddress(vulkan_context.lock()->getDevice()->getLogicalDevice(), &bufferDeviceAddressInfo);
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
