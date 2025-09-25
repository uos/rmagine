#include "rmagine/util/vulkan/memory/DeviceMemory.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

DeviceMemory::DeviceMemory(VkMemoryPropertyFlags memoryPropertyFlags, BufferPtr buffer) :
    vulkan_context(get_vulkan_context_weak()),
    device(vulkan_context.lock()->getDevice()),
    buffer(buffer)
{
    allocateDeviceMemory(memoryPropertyFlags, true);
}

DeviceMemory::~DeviceMemory()
{
    if(deviceMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(device->getLogicalDevice(), deviceMemory, nullptr);
    }
    device.reset();
}



void DeviceMemory::allocateDeviceMemory(VkMemoryPropertyFlags memoryPropertyFlags, bool withAllocateFlags)
{
    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(vulkan_context.lock()->getDevice()->getLogicalDevice(), buffer->getBuffer(), &memoryRequirements);

    VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
    vkGetPhysicalDeviceMemoryProperties(vulkan_context.lock()->getDevice()->getPhysicalDevice(), &physicalDeviceMemoryProperties);

    //find compatible memory type
    uint32_t memoryTypeIndex = uint32_t(~0);
    for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        if ((memoryRequirements.memoryTypeBits & (1 << i)) && 
            (physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & memoryPropertyFlags) == memoryPropertyFlags)
        {
            memoryTypeIndex = i;
            break;
        }
    }

    VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
    memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo memoryAllocateInfo{};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.pNext = (withAllocateFlags ? &memoryAllocateFlagsInfo : nullptr);
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = memoryTypeIndex;

    if(vkAllocateMemory(vulkan_context.lock()->getDevice()->getLogicalDevice(), &memoryAllocateInfo, nullptr, &deviceMemory) != VK_SUCCESS)
    {
        throw std::runtime_error("[DeviceMemory::allocateDeviceMemory()] ERROR - failed to allocate device memory!");
    }

    //bind memory to buffer
    if(vkBindBufferMemory(vulkan_context.lock()->getDevice()->getLogicalDevice(), buffer->getBuffer(), deviceMemory, 0) != VK_SUCCESS)
    {
        throw std::runtime_error("[DeviceMemory::allocateDeviceMemory()] ERROR - failed to bind buffer!");
    }
}


void DeviceMemory::copyToDeviceMemory(const void* src)
{
    copyToDeviceMemory(src, 0, buffer->getBufferSize());
}


void DeviceMemory::copyToDeviceMemory(const void *src, size_t offset, size_t stride)
{
    std::lock_guard<std::mutex> guard(deviceMemoryMtx);
    if(offset >= buffer->getBufferSize() || offset + stride > buffer->getBufferSize())
    {
        throw std::runtime_error("[DeviceMemory::copyToDeviceMemory()] ERROR - offset and/or stride too large for this device memory!");
    }

    void *hostMemoryBuffer;
    if(vkMapMemory(vulkan_context.lock()->getDevice()->getLogicalDevice(), deviceMemory, offset, stride, 0, &hostMemoryBuffer) != VK_SUCCESS)//only map the data that gets written to
    {
        throw std::runtime_error("[DeviceMemory::copyToDeviceMemory()] ERROR - failed to map memory!");
    }
    memcpy(hostMemoryBuffer, src, stride);
    vkUnmapMemory(vulkan_context.lock()->getDevice()->getLogicalDevice(), deviceMemory);
}


void DeviceMemory::copyFromDeviceMemory(void *dst)
{
    copyFromDeviceMemory(dst, 0, buffer->getBufferSize());
}


void DeviceMemory::copyFromDeviceMemory(void* dst, size_t offset, size_t stride)
{
    std::lock_guard<std::mutex> guard(deviceMemoryMtx);
    if(offset >= buffer->getBufferSize() || offset + stride > buffer->getBufferSize())
    {
        throw std::runtime_error("[DeviceMemory::copyFromDeviceMemory()] ERROR - offset and/or stride too large for this device memory!");
    }

    void *hostMemoryBuffer;
    if(vkMapMemory(vulkan_context.lock()->getDevice()->getLogicalDevice(), deviceMemory, offset, stride, 0, &hostMemoryBuffer) != VK_SUCCESS)//only map the data that gets read from
    {
        throw std::runtime_error("[DeviceMemory::copyFromDeviceMemory()] ERROR - failed to map memory!");
    }
    memcpy(dst, hostMemoryBuffer, stride);
    vkUnmapMemory(vulkan_context.lock()->getDevice()->getLogicalDevice(), deviceMemory);
}


int DeviceMemory::getMemoryHandle()
{
    int fd = -1;

    VkMemoryGetFdInfoKHR vkMemoryGetFdInfoKHR{};
    vkMemoryGetFdInfoKHR.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    vkMemoryGetFdInfoKHR.pNext      = nullptr;
    vkMemoryGetFdInfoKHR.memory     = deviceMemory;
    vkMemoryGetFdInfoKHR.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    // vkMemoryGetFdInfoKHR.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT; // windows 8.10 or greater
    // vkMemoryGetFdInfoKHR.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT; // not windows 8.10 or greater

    if(vulkan_context.lock()->extensionFuncs.vkGetMemoryFdKHR(vulkan_context.lock()->getDevice()->getLogicalDevice(), &vkMemoryGetFdInfoKHR, &fd) != VK_SUCCESS)
    {
        throw std::runtime_error("[DeviceMemory::copyFromDeviceMemory()] ERROR - failed to retrieve handle for device memory!");
    }

    return fd;
}


BufferPtr DeviceMemory::getBuffer()
{
    return buffer;
}

} // namespace rmagine
