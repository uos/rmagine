#include "MemoryVulkan.hpp"

namespace rmagine
{

//// MemoryView - VULKAN_HOST_VISIBLE

template<typename DataT>
size_t MemoryView<DataT, VULKAN_HOST_VISIBLE>::size() const
{
    if(memoryData == nullptr)
    {
        return 0;
    }
    return memoryData->size;
}


template <typename DataT>
size_t MemoryView<DataT, VULKAN_HOST_VISIBLE>::getID() const
{
    if(memoryData == nullptr)
    {
        return 0;
    }
    return memoryData->memID;
}


template<typename DataT>
BufferPtr MemoryView<DataT, VULKAN_HOST_VISIBLE>::getBuffer() const
{
    if(memoryData == nullptr)
    {
        return nullptr;
    }
    return memoryData->buffer;
}


template<typename DataT>
DeviceMemoryPtr MemoryView<DataT, VULKAN_HOST_VISIBLE>::getDeviceMemory() const
{
    if(memoryData == nullptr)
    {
        return nullptr;
    }
    return memoryData->deviceMemory;
}






//// Memory - VULKAN_HOST_VISIBLE

template<typename DataT>
Memory<DataT, VULKAN_HOST_VISIBLE>::Memory()
{
    bufferDeviceAddress = 0;
}

template<typename DataT>
Memory<DataT, VULKAN_HOST_VISIBLE>::Memory(size_t N)
{
    resize(N);
}

template<typename DataT>
Memory<DataT, VULKAN_HOST_VISIBLE>::Memory(size_t N, VkBufferUsageFlags bufferUsageFlags)
{
    resize(N, bufferUsageFlags);
}


template<typename DataT>
void Memory<DataT, VULKAN_HOST_VISIBLE>::resize(size_t N)
{
    resize(N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}


template<typename DataT>
void Memory<DataT, VULKAN_HOST_VISIBLE>::resize(size_t N, VkBufferUsageFlags bufferUsageFlags)
{
    MemoryDataPtr newMemoryData = std::make_shared<MemoryData>();
    newMemoryData->size = N;

    newMemoryData->buffer = std::make_shared<Buffer>(N*sizeof(DataT), bufferUsageFlags | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    newMemoryData->deviceMemory =std::make_shared<DeviceMemory>(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, newMemoryData->buffer);

    if(Base::size() != 0)//CHECK: test if the copying works as intended
    {
        get_vulkan_context()->getDefaultCommandBuffer()->recordCopyBufferToCommandBuffer(memoryData->buffer, newMemoryData->buffer);
        get_vulkan_context()->getDefaultCommandBuffer()->submitRecordedCommandAndWait();
    }

    memoryData.reset();
    memoryData = newMemoryData;
    bufferDeviceAddress = newMemoryData->buffer->getBufferDeviceAddress();
}


template<typename DataT>
template<typename MemT2>
Memory<DataT, VULKAN_HOST_VISIBLE>& Memory<DataT, VULKAN_HOST_VISIBLE>::operator=(const Memory<DataT, MemT2>& o)
{
    if(this->size() != o.size())
    {
        throw std::runtime_error("Memory (VULKAN_HOST_VISIBLE) MemT2 assignment of different sizes");
        // this->resize(o.size());
    }
    copy(o, *this);
    return *this;
}






//// VULKAN_HOST_VISIBLE

template<typename DataT>
DataT* VULKAN_HOST_VISIBLE::alloc(size_t N)
{
    (void) N;
    throw std::runtime_error("VULKAN_HOST_VISIBLE (alloc): The Program should never call this function and always use the template specialization instead!");
    return nullptr;
}

template<typename DataT>
DataT* VULKAN_HOST_VISIBLE::realloc(DataT* mem, size_t Nold, size_t Nnew)
{
    (void) mem;
    (void) Nold;
    (void) Nnew;
    throw std::runtime_error("VULKAN_HOST_VISIBLE (realloc): The Program should never call this function and always use the template specialization instead!");
    return nullptr;
}

template<typename DataT>
void VULKAN_HOST_VISIBLE::free(DataT* mem, size_t N)
{
    (void) mem;
    (void) N;
    throw std::runtime_error("VULKAN_HOST_VISIBLE (free): The Program should never call this function and always use the template specialization instead!");
}

} // namespace rmagine