#include "MemoryVulkan.hpp"

namespace rmagine
{

//// MemoryView - VULKAN_HOST_VISIBLE

template<typename DataT>
MemoryView<DataT, VULKAN_HOST_VISIBLE>::MemoryView(
    size_t p_size, size_t p_offset, VulkanMemoryUsage p_memoryUsage, DeviceMemoryPtr p_deviceMemory) :
    m_size(p_size), m_offset(p_offset), m_memoryUsage(p_memoryUsage), m_deviceMemory(p_deviceMemory)
{

}



template<typename DataT>
MemoryView<DataT, VULKAN_HOST_VISIBLE>& MemoryView<DataT, VULKAN_HOST_VISIBLE>::operator=(const MemoryView<DataT, VULKAN_HOST_VISIBLE>& o)
{
    if(this->size() != o.size())
    {
        throw std::invalid_argument("[MemoryView<DataT, VULKAN_HOST_VISIBLE>::operator=()] ERROR - VULKAN_HOST_VISIBLE assignment of different sizes");
    }

    copy(o, *this);
    return *this;
}


template<typename DataT>
template<typename MemT2>
MemoryView<DataT, VULKAN_HOST_VISIBLE>& MemoryView<DataT, VULKAN_HOST_VISIBLE>::operator=(const MemoryView<DataT, MemT2>& o)
{
    if(this->size() != o.size())
    {
        throw std::invalid_argument("[MemoryView<DataT, VULKAN_HOST_VISIBLE>::operator=()] ERROR - MemT2 assignment of different sizes");
    }

    copy(o, *this);
    return *this;
}


template<typename DataT>
MemoryView<DataT, VULKAN_HOST_VISIBLE> MemoryView<DataT, VULKAN_HOST_VISIBLE>::slice(size_t idx_start, size_t idx_end)
{
    return MemoryView<DataT, VULKAN_HOST_VISIBLE>(idx_end - idx_start, idx_start, m_memoryUsage, m_deviceMemory);
}


template<typename DataT>
const MemoryView<DataT, VULKAN_HOST_VISIBLE> MemoryView<DataT, VULKAN_HOST_VISIBLE>::slice(size_t idx_start, size_t idx_end) const
{
    return MemoryView<DataT, VULKAN_HOST_VISIBLE>(idx_end - idx_start, idx_start, m_memoryUsage, m_deviceMemory);
}


template<typename DataT>
MemoryView<DataT, VULKAN_HOST_VISIBLE> MemoryView<DataT, VULKAN_HOST_VISIBLE>::operator()(size_t idx_start, size_t idx_end)
{
    return slice(idx_start, idx_end);
}


template<typename DataT>
const MemoryView<DataT, VULKAN_HOST_VISIBLE> MemoryView<DataT, VULKAN_HOST_VISIBLE>::operator()(size_t idx_start, size_t idx_end) const
{
    return slice(idx_start, idx_end);
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
    return m_deviceMemory->getBuffer();
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
        BufferPtr buffer = std::make_shared<Buffer>(N*sizeof(DataT), get_buffer_usage_flags(m_memoryUsage));
        m_deviceMemory = std::make_shared<DeviceMemory>(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer);

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
        m_deviceMemory = nullptr;
        return;
    }

    size_t newSize = N;

    BufferPtr newBuffer = std::make_shared<Buffer>(N*sizeof(DataT), get_buffer_usage_flags(m_memoryUsage));
    DeviceMemoryPtr newDeviceMemory = std::make_shared<DeviceMemory>(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, newBuffer);

    //copy from old buffer to new buffer, if old buffer wasnt empty
    if(m_size != 0)
    {
        vulkan_memcpy_device_to_device(m_deviceMemory->getBuffer(), newBuffer, sizeof(DataT) * std::min(m_size, newSize), 0, 0);
    }

    m_size = newSize;
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
Memory<DataT, VULKAN_HOST_VISIBLE>& Memory<DataT, VULKAN_HOST_VISIBLE>::operator=(const MemoryView<DataT, VULKAN_HOST_VISIBLE>& o)
{
    if(this->size() != o.size())
    {
        this->resize(o.size());
    }

    Base::operator=(o);
    return *this;
}


template<typename DataT>
template<typename MemT2>
Memory<DataT, VULKAN_HOST_VISIBLE>& Memory<DataT, VULKAN_HOST_VISIBLE>::operator=(const MemoryView<DataT, MemT2>& o)
{
    if(this->size() != o.size())
    {
        this->resize(o.size());
    }

    Base::operator=(o);
    return *this;
}


template<typename DataT>
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