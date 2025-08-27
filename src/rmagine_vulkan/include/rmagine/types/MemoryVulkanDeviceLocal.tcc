#include "MemoryVulkan.hpp"

namespace rmagine
{

//// MemoryView - VULKAN_DEVICE_LOCAL

template<typename DataT>
size_t MemoryView<DataT, VULKAN_DEVICE_LOCAL>::size() const
{
    return m_size;
}


template <typename DataT>
size_t MemoryView<DataT, VULKAN_DEVICE_LOCAL>::getID() const
{
    return m_memID;
}


template<typename DataT>
BufferPtr MemoryView<DataT, VULKAN_DEVICE_LOCAL>::getBuffer() const
{
    return m_buffer;
}


template<typename DataT>
BufferPtr MemoryView<DataT, VULKAN_DEVICE_LOCAL>::getStagingBuffer() const
{
    return m_stagingBuffer;
}


template<typename DataT>
DeviceMemoryPtr MemoryView<DataT, VULKAN_DEVICE_LOCAL>::getDeviceMemory() const
{
    return m_deviceMemory;
}


template<typename DataT>
DeviceMemoryPtr MemoryView<DataT, VULKAN_DEVICE_LOCAL>::getStagingDeviceMemory() const
{
    return m_stagingDeviceMemory;
}






//// Memory - VULKAN_DEVICE_LOCAL

template<typename DataT>
Memory<DataT, VULKAN_DEVICE_LOCAL>::Memory()
{
    
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
    if(N != 0)
    {
        size_t newSize = N;

        BufferPtr newStagingBuffer = std::make_shared<Buffer>(N*sizeof(DataT), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        DeviceMemoryPtr newStagingDeviceMemory =std::make_shared<DeviceMemory>(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, newStagingBuffer);
        
        BufferPtr newBuffer = std::make_shared<Buffer>(N*sizeof(DataT), bufferUsageFlags | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        DeviceMemoryPtr newDeviceMemory =std::make_shared<DeviceMemory>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, newBuffer);

        //CHECK: test if the copying works as intended
        get_vulkan_context()->getDefaultCommandBuffer()->recordCopyBufferToCommandBuffer(m_buffer, newBuffer);
        get_vulkan_context()->getDefaultCommandBuffer()->submitRecordedCommandAndWait();

        m_buffer.reset();
        m_deviceMemory.reset();
        stagingBuffer.reset();
        stagingDeviceMemory.reset();

        m_size = newSize;
        m_memID = MemoryHelper::GetNewMemID();
        m_buffer = newBuffer;
        m_deviceMemory = newDeviceMemory;
        stagingBuffer = newStagingBuffer;
        stagingDeviceMemory = newStagingDeviceMemory;
    }
    else
    {
        m_size = 0;
        m_buffer = nullptr;
        m_deviceMemory = nullptr;
        stagingBuffer = nullptr;
        stagingDeviceMemory = nullptr;
    }
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
