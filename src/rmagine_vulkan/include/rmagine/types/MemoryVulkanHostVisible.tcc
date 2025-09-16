#include "MemoryVulkan.hpp"

namespace rmagine
{

//// MemoryView - VULKAN_HOST_VISIBLE

template<typename DataT>
MemoryView<DataT, VULKAN_HOST_VISIBLE>::MemoryView(
    size_t p_size, size_t p_offset, VulkanMemoryUsage p_memoryUsage, 
    BufferPtr p_buffer, DeviceMemoryPtr p_deviceMemory) :
    m_size(p_size), m_offset(p_offset), m_memoryUsage(p_memoryUsage),
    m_buffer(p_buffer), m_deviceMemory(p_deviceMemory)
{

}



template<typename DataT>
size_t MemoryView<DataT, VULKAN_HOST_VISIBLE>::size() const
{
    return m_size;
}


template<typename DataT>
size_t MemoryView<DataT, VULKAN_HOST_VISIBLE>::offset() const
{
    return m_offset;
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
Memory<DataT, VULKAN_HOST_VISIBLE>::Memory() : Memory(0)
{
    
}

template<typename DataT>
Memory<DataT, VULKAN_HOST_VISIBLE>::Memory(size_t N) : Memory(N, VulkanMemoryUsage::Usage_Default)
{
    
}

template<typename DataT>
Memory<DataT, VULKAN_HOST_VISIBLE>::Memory(size_t N, VulkanMemoryUsage memoryUsage)
    : Base(N, 0, memoryUsage, nullptr, nullptr)
{
    if(N > 0)
    {
        m_buffer = std::make_shared<Buffer>(N*sizeof(DataT), get_buffer_usage_flags(m_memoryUsage));
        m_deviceMemory = std::make_shared<DeviceMemory>(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_buffer);

        m_memID = get_new_mem_id();
        #ifdef VDEBUG
            std::cout << "VULKAN_HOST_VISIBLE: new m_memID = " << m_memID << std::endl;
        #endif
    }
}

template<typename DataT>
Memory<DataT, VULKAN_HOST_VISIBLE>::~Memory()
{
    #ifdef VDEBUG
        if(m_memID != 0)
            std::cout << "VULKAN_HOST_VISIBLE: retired m_memID = " << m_memID << std::endl;
    #endif
}


template<typename DataT>
void Memory<DataT, VULKAN_HOST_VISIBLE>::resize(size_t N)
{
    if(N == m_size)
    {
        return;
    }
    else if(N == 0)
    {
        m_size = 0;
        m_buffer = nullptr;
        m_deviceMemory = nullptr;
        return;
    }

    size_t newSize = N;

    BufferPtr newBuffer = std::make_shared<Buffer>(N*sizeof(DataT), get_buffer_usage_flags(m_memoryUsage));
    DeviceMemoryPtr newDeviceMemory = std::make_shared<DeviceMemory>(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, newBuffer);

    //copy from old buffer to new buffer, if old buffer wasnt empty
    if(m_size != 0)
    {
        //TODO: use copy here when its done for non equal sizes
        get_mem_command_buffer()->recordCopyBufferToCommandBuffer(m_buffer, newBuffer);
        get_mem_command_buffer()->submitRecordedCommandAndWait();
    }

    m_size = newSize;
    m_buffer = newBuffer;
    m_deviceMemory = newDeviceMemory;

    #ifdef VDEBUG
        if(m_memID != 0)
            std::cout << "VULKAN_HOST_VISIBLE: retired m_memID = " << m_memID << " (resize)" << std::endl;
    #endif
    m_memID = get_new_mem_id();
    #ifdef VDEBUG
        std::cout << "VULKAN_HOST_VISIBLE: new m_memID = " << m_memID << " (resize)" << std::endl;
    #endif
}


template<typename DataT>
template<typename MemT2>
Memory<DataT, VULKAN_HOST_VISIBLE>& Memory<DataT, VULKAN_HOST_VISIBLE>::operator=(const Memory<DataT, MemT2>& o)
{
    if(this->size() != o.size())
    {
        // throw std::runtime_error("Memory (VULKAN_HOST_VISIBLE) MemT2 assignment of different sizes");
        this->resize(o.size());
    }
    copy(o, *this);
    return *this;
}


template <typename DataT>
size_t Memory<DataT, VULKAN_HOST_VISIBLE>::getID() const
{
    return m_memID;
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