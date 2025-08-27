#include "MemoryVulkan.hpp"

namespace rmagine
{

//// MemoryView - VULKAN_HOST_VISIBLE

template<typename DataT>
size_t MemoryView<DataT, VULKAN_HOST_VISIBLE>::size() const
{
    return m_size;
}


template <typename DataT>
size_t MemoryView<DataT, VULKAN_HOST_VISIBLE>::getID() const
{
    return m_memID;
}


template<typename DataT>
BufferPtr MemoryView<DataT, VULKAN_HOST_VISIBLE>::getBuffer() const
{
    return m_buffer;
}


template<typename DataT>
DeviceMemoryPtr MemoryView<DataT, VULKAN_HOST_VISIBLE>::getDeviceMemory() const
{
    return m_deviceMemory;
}






//// Memory - VULKAN_HOST_VISIBLE

template<typename DataT>
Memory<DataT, VULKAN_HOST_VISIBLE>::Memory()
{
    
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
    if(N != 0)
    {
        size_t newSize = N;

        BufferPtr newBuffer = std::make_shared<Buffer>(N*sizeof(DataT), bufferUsageFlags | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        DeviceMemoryPtr newDeviceMemory =std::make_shared<DeviceMemory>(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, newBuffer);

        //CHECK: test if the copying works as intended
        get_vulkan_context()->getDefaultCommandBuffer()->recordCopyBufferToCommandBuffer(m_buffer, newBuffer);
        get_vulkan_context()->getDefaultCommandBuffer()->submitRecordedCommandAndWait();

        m_buffer.reset();
        m_deviceMemory.reset();

        m_size = newSize;
        m_memID = MemoryHelper::GetNewMemID();
        m_buffer = newBuffer;
        m_deviceMemory = newDeviceMemory;
    }
    else
    {
        m_size = 0;
        m_buffer = nullptr;
        m_deviceMemory = nullptr;
    }
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