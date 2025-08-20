#include "MemoryVulkan.hpp"

namespace rmagine
{

//MemoryView - VULKAN_DEVICE_LOCAL

template<typename DataT>
size_t MemoryView<DataT, VULKAN_DEVICE_LOCAL>::size() const
{
    if(memoryData == nullptr)
    {
        return 0;
    }
    return memoryData->size;
}


template <typename DataT>
size_t MemoryView<DataT, VULKAN_DEVICE_LOCAL>::getID() const
{
    if(memoryData == nullptr)
    {
        return 0;
    }
    return memoryData->memID;
}


template<typename DataT>
BufferPtr MemoryView<DataT, VULKAN_DEVICE_LOCAL>::getBuffer() const
{
    if(memoryData == nullptr)
    {
        return nullptr;
    }
    return memoryData->buffer;
}


template<typename DataT>
BufferPtr MemoryView<DataT, VULKAN_DEVICE_LOCAL>::getStagingBuffer() const
{
    if(memoryData == nullptr)
    {
        return nullptr;
    }
    return memoryData->stagingBuffer;
}


template<typename DataT>
DeviceMemoryPtr MemoryView<DataT, VULKAN_DEVICE_LOCAL>::getDeviceMemory() const
{
    if(memoryData == nullptr)
    {
        return nullptr;
    }
    return memoryData->deviceMemory;
}


template<typename DataT>
DeviceMemoryPtr MemoryView<DataT, VULKAN_DEVICE_LOCAL>::getStagingDeviceMemory() const
{
    if(memoryData == nullptr)
    {
        return nullptr;
    }
    return memoryData->stagingDeviceMemory;
}






//Memory - VULKAN_DEVICE_LOCAL

template<typename DataT>
Memory<DataT, VULKAN_DEVICE_LOCAL>::Memory()
{
    bufferDeviceAddress = 0;
}

template<typename DataT>
Memory<DataT, VULKAN_DEVICE_LOCAL>::Memory(size_t N)
{
    resize(N);
}

template<typename DataT>
Memory<DataT, VULKAN_DEVICE_LOCAL>::Memory(size_t N, VkBufferUsageFlags bufferUsageFlags)
{
    resize(N, bufferUsageFlags);
}


template<typename DataT>
void Memory<DataT, VULKAN_DEVICE_LOCAL>::resize(size_t N)
{
    resize(N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}


template<typename DataT>
void Memory<DataT, VULKAN_DEVICE_LOCAL>::resize(size_t N, VkBufferUsageFlags bufferUsageFlags)
{
    MemoryDataPtr newMemoryData = std::make_shared<MemoryData>();
    newMemoryData->size = N;

    newMemoryData->stagingBuffer = std::make_shared<Buffer>(N*sizeof(DataT), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    newMemoryData->stagingDeviceMemory =std::make_shared<DeviceMemory>(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, newMemoryData->stagingBuffer);
    
    newMemoryData->buffer = std::make_shared<Buffer>(N*sizeof(DataT), bufferUsageFlags | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    newMemoryData->deviceMemory =std::make_shared<DeviceMemory>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, newMemoryData->buffer);

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
Memory<DataT, VULKAN_DEVICE_LOCAL>& Memory<DataT, VULKAN_DEVICE_LOCAL>::operator=(const Memory<DataT, MemT2>& o)
{
    if(this->size() != o.size())
    {
        throw std::runtime_error("Memory (VULKAN_DEVICE_LOCAL) MemT2 assignment of different sizes");
        // this->resize(o.size());
    }
    copy(o, *this);
    return *this;
}






//// VULKAN_DEVICE_LOCAL
template<typename DataT>
DataT* VULKAN_DEVICE_LOCAL::alloc(size_t N)
{
    (void) N;
    throw std::runtime_error("VULKAN_DEVICE_LOCAL (alloc): The program should never call this function and always use the template specialization instead!");
    return nullptr;
}

template<typename DataT>
DataT* VULKAN_DEVICE_LOCAL::realloc(DataT* mem, size_t Nold, size_t Nnew)
{
    (void) mem;
    (void) Nold;
    (void) Nnew;
    throw std::runtime_error("VULKAN_DEVICE_LOCAL (realloc): The program should never call this function and always use the template specialization instead!");
    return nullptr;
}

template<typename DataT>
void VULKAN_DEVICE_LOCAL::free(DataT* mem, size_t N)
{
    (void) mem;
    (void) N;
    throw std::runtime_error("VULKAN_DEVICE_LOCAL (free): The program should never call this function and always use the template specialization instead!");
}

} // namespace rmagine
