#include "rmagine/util/vulkan/memory/DeviceMemory.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

DeviceMemory::DeviceMemory(VkMemoryPropertyFlags memoryPropertyFlags, BufferPtr buffer) : device(get_vulkan_context()->getDevice()), buffer(buffer)
{
    allocateDeviceMemory(memoryPropertyFlags, true);
}

DeviceMemory::DeviceMemory(VkMemoryPropertyFlags memoryPropertyFlags, DevicePtr device, BufferPtr buffer) : device(device), buffer(buffer)
{
    allocateDeviceMemory(memoryPropertyFlags, true);
}



void DeviceMemory::allocateDeviceMemory(VkMemoryPropertyFlags memoryPropertyFlags, bool withAllocateFlags)
{
    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device->getLogicalDevice(), buffer->getBuffer(), &memoryRequirements);

    VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
    vkGetPhysicalDeviceMemoryProperties(device->getPhysicalDevice(), &physicalDeviceMemoryProperties);

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

    if(vkAllocateMemory(device->getLogicalDevice(), &memoryAllocateInfo, nullptr, &deviceMemory) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate device memory!");
    }

    //bind memory to buffer
    if(vkBindBufferMemory(device->getLogicalDevice(), buffer->getBuffer(), deviceMemory, 0) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to bind buffer!");
    }
}


void DeviceMemory::map(void** ptr)
{
    map(ptr, 0, buffer->getBufferSize());
}


void DeviceMemory::map(void** ptr, size_t offset, size_t stride)
{
    if(persistentlyMapped)
    {
        throw std::runtime_error("cant map a buffer that has already been mapped!");
    }
    if(offset >= buffer->getBufferSize() || offset + stride > buffer->getBufferSize())
    {
        throw std::runtime_error("offset and/or stride too large for this buffer!");
    }

    persistentlyMapped = true;
    if(vkMapMemory(device->getLogicalDevice(), deviceMemory, offset, stride, 0, ptr) != VK_SUCCESS)//only map the data that gets written to
    {
        throw std::runtime_error("failed to map memory!");
    }
}


void DeviceMemory::unMap()
{
    if(!persistentlyMapped)
    {
        vkUnmapMemory(device->getLogicalDevice(), deviceMemory);
        persistentlyMapped = false;
    }
}


bool DeviceMemory::isPersistentlyMapped()
{
    return persistentlyMapped;
}


void DeviceMemory::copyToDeviceMemory(const void* src)
{
    copyToDeviceMemory(src, 0, buffer->getBufferSize());
}


void DeviceMemory::copyToDeviceMemory(const void *src, size_t offset, size_t stride)
{
    if(persistentlyMapped)
    {
        throw std::runtime_error("cant map a buffer that has already been mapped!");
    }
    if(offset >= buffer->getBufferSize() || offset + stride > buffer->getBufferSize())
    {
        throw std::runtime_error("offset and/or stride too large for this buffer!");
    }

    void *hostMemoryBuffer;
    if(vkMapMemory(device->getLogicalDevice(), deviceMemory, offset, stride, 0, &hostMemoryBuffer) != VK_SUCCESS)//only map the data that gets written to
    {
        throw std::runtime_error("failed to map memory!");
    }
    memcpy(hostMemoryBuffer, src, stride);
    vkUnmapMemory(device->getLogicalDevice(), deviceMemory);
}


void DeviceMemory::copyFromDeviceMemory(void *dst)
{
    copyFromDeviceMemory(dst, 0, buffer->getBufferSize());
}


void DeviceMemory::copyFromDeviceMemory(void* dst, size_t offset, size_t stride)
{
    if(persistentlyMapped)
    {
        throw std::runtime_error("cant map a buffer that has already been mapped!");
    }
    if(offset >= buffer->getBufferSize() || offset + stride > buffer->getBufferSize())
    {
        throw std::runtime_error("offset and/or stride too large for this buffer!");
    }

    void *hostMemoryBuffer;
    if(vkMapMemory(device->getLogicalDevice(), deviceMemory, offset, stride, 0, &hostMemoryBuffer) != VK_SUCCESS)//only map the data that gets read from
    {
        throw std::runtime_error("failed to map memory!");
    }
    memcpy(dst, hostMemoryBuffer, stride);
    vkUnmapMemory(device->getLogicalDevice(), deviceMemory);
}


void DeviceMemory::cleanup()
{
    if(persistentlyMapped)
    {
        unMap();
    }
    if(deviceMemory  != VK_NULL_HANDLE)
    {
        vkFreeMemory(device->getLogicalDevice(), deviceMemory, nullptr);
        deviceMemory = VK_NULL_HANDLE;
    }
}

} // namespace rmagine
