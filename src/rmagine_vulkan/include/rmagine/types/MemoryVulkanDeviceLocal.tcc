#include "MemoryVulkan.hpp"

namespace rmagine
{

//// MemoryView - VULKAN_DEVICE_LOCAL

template<typename DataT>
MemoryView<DataT, VULKAN_DEVICE_LOCAL>::MemoryView(
    size_t p_size, size_t p_offset, VulkanMemoryUsage p_memoryUsage, DeviceMemoryPtr p_deviceMemory, DeviceMemoryPtr p_stagingDeviceMemory) :
    m_size(p_size), m_offset(p_offset), m_memoryUsage(p_memoryUsage), m_deviceMemory(p_deviceMemory), m_stagingDeviceMemory(p_stagingDeviceMemory)
{

}



template<typename DataT>
MemoryView<DataT, VULKAN_DEVICE_LOCAL>& MemoryView<DataT, VULKAN_DEVICE_LOCAL>::operator=(const MemoryView<DataT, VULKAN_DEVICE_LOCAL>& o)
{
    if(this->size() != o.size())
    {
        throw std::invalid_argument("[MemoryView<DataT, VULKAN_DEVICE_LOCAL>::operator=()] ERROR - VULKAN_DEVICE_LOCAL assignment of different sizes");
    }

    copy(o, *this);
    return *this;
}


template<typename DataT>
template<typename MemT2>
MemoryView<DataT, VULKAN_DEVICE_LOCAL>& MemoryView<DataT, VULKAN_DEVICE_LOCAL>::operator=(const MemoryView<DataT, MemT2>& o)
{
    if(this->size() != o.size())
    {
        throw std::invalid_argument("[MemoryView<DataT, VULKAN_DEVICE_LOCAL>::operator=()] ERROR - MemT2 assignment of different sizes");
    }

    copy(o, *this);
    return *this;
}


template<typename DataT>
MemoryView<DataT, VULKAN_DEVICE_LOCAL> MemoryView<DataT, VULKAN_DEVICE_LOCAL>::slice(size_t idx_start, size_t idx_end)
{
    return MemoryView<DataT, VULKAN_DEVICE_LOCAL>(idx_end - idx_start, idx_start, m_memoryUsage, m_deviceMemory, m_stagingDeviceMemory);
}


template<typename DataT>
const MemoryView<DataT, VULKAN_DEVICE_LOCAL> MemoryView<DataT, VULKAN_DEVICE_LOCAL>::slice(size_t idx_start, size_t idx_end) const
{
    return MemoryView<DataT, VULKAN_DEVICE_LOCAL>(idx_end - idx_start, idx_start, m_memoryUsage, m_deviceMemory, m_stagingDeviceMemory);
}


template<typename DataT>
MemoryView<DataT, VULKAN_DEVICE_LOCAL> MemoryView<DataT, VULKAN_DEVICE_LOCAL>::operator()(size_t idx_start, size_t idx_end)
{
    return slice(idx_start, idx_end);
}


template<typename DataT>
const MemoryView<DataT, VULKAN_DEVICE_LOCAL> MemoryView<DataT, VULKAN_DEVICE_LOCAL>::operator()(size_t idx_start, size_t idx_end) const
{
    return slice(idx_start, idx_end);
}


template<typename DataT>
size_t MemoryView<DataT, VULKAN_DEVICE_LOCAL>::size() const
{
    return m_size;
}


template<typename DataT>
size_t MemoryView<DataT, VULKAN_DEVICE_LOCAL>::offset() const
{
    return m_offset;
}


template<typename DataT>
BufferPtr MemoryView<DataT, VULKAN_DEVICE_LOCAL>::getBuffer() const
{
    return m_deviceMemory->getBuffer();
}


template<typename DataT>
BufferPtr MemoryView<DataT, VULKAN_DEVICE_LOCAL>::getStagingBuffer() const
{
    if(m_memoryUsage == VulkanMemoryUsage::Usage_AccelerationStructure || m_memoryUsage == VulkanMemoryUsage::Usage_AccelerationStructureScratch)
        return nullptr;
    return m_stagingDeviceMemory->getBuffer();
}


template<typename DataT>
DeviceMemoryPtr MemoryView<DataT, VULKAN_DEVICE_LOCAL>::getDeviceMemory() const
{
    return m_deviceMemory;
}


template<typename DataT>
DeviceMemoryPtr MemoryView<DataT, VULKAN_DEVICE_LOCAL>::getStagingDeviceMemory() const
{
    if(m_memoryUsage == VulkanMemoryUsage::Usage_AccelerationStructure || m_memoryUsage == VulkanMemoryUsage::Usage_AccelerationStructureScratch)
        return nullptr;
    return m_stagingDeviceMemory;
}






//// Memory - VULKAN_DEVICE_LOCAL

template<typename DataT>
Memory<DataT, VULKAN_DEVICE_LOCAL>::Memory() : Memory(0)
{
    
}

template<typename DataT>
Memory<DataT, VULKAN_DEVICE_LOCAL>::Memory(size_t N) : Memory(N, VulkanMemoryUsage::Usage_Default)
{
    
}

template<typename DataT>
Memory<DataT, VULKAN_DEVICE_LOCAL>::Memory(size_t N, VulkanMemoryUsage memoryUsage) :
    Base(N, 0, memoryUsage, nullptr, nullptr)
{
    if(N > 0)
    {
        //acceleration structure and its scratch buffer dont need staging - they get written to by a command
        if(m_memoryUsage != VulkanMemoryUsage::Usage_AccelerationStructure && m_memoryUsage != VulkanMemoryUsage::Usage_AccelerationStructureScratch)
        {
            BufferPtr stagingBuffer = std::make_shared<Buffer>(N*sizeof(DataT), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
            m_stagingDeviceMemory = std::make_shared<DeviceMemory>(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer);
        }

        BufferPtr buffer = std::make_shared<Buffer>(N*sizeof(DataT), get_buffer_usage_flags(m_memoryUsage));
        m_deviceMemory = std::make_shared<DeviceMemory>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer);

        m_memID = get_new_mem_id();
        #ifdef VDEBUG
            std::cout << "VULKAN_DEVICE_LOCAL: new m_memID = " << m_memID << std::endl;
        #endif
    }
}

template<typename DataT>
Memory<DataT, VULKAN_DEVICE_LOCAL>::~Memory()
{
    #ifdef VDEBUG
        if(m_memID != 0)
            std::cout << "VULKAN_DEVICE_LOCAL: retired m_memID = " << m_memID << std::endl;
    #endif
}


template<typename DataT>
void Memory<DataT, VULKAN_DEVICE_LOCAL>::resize(size_t N)
{
    if(N == m_size)
    {
        return;
    }
    else if(N == 0)
    {
        m_size = 0;
        m_deviceMemory = nullptr;
        m_stagingDeviceMemory = nullptr;
        return;
    }

    size_t newSize = N;

    //acceleration structure and its scratch buffer dont need staging - they get written to by a command
    BufferPtr newStagingBuffer = nullptr;
    DeviceMemoryPtr newStagingDeviceMemory = nullptr;
    if(m_memoryUsage != VulkanMemoryUsage::Usage_AccelerationStructure && m_memoryUsage != VulkanMemoryUsage::Usage_AccelerationStructureScratch)
    {
        newStagingBuffer = std::make_shared<Buffer>(N*sizeof(DataT), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        newStagingDeviceMemory = std::make_shared<DeviceMemory>(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, newStagingBuffer);
    }

    BufferPtr newBuffer = std::make_shared<Buffer>(N*sizeof(DataT), get_buffer_usage_flags(m_memoryUsage));
    DeviceMemoryPtr newDeviceMemory = std::make_shared<DeviceMemory>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, newBuffer);

    //copy from old buffer to new buffer, if old buffer wasnt empty
    if(m_size != 0)
    {
        vulkan::memcpyDeviceToDevice(m_deviceMemory->getBuffer(), newBuffer, sizeof(DataT) * std::min(m_size, newSize), 0, 0);
    }

    m_size = newSize;
    m_deviceMemory = newDeviceMemory;
    m_stagingDeviceMemory = newStagingDeviceMemory;

    #ifdef VDEBUG
        if(m_memID != 0)
            std::cout << "VULKAN_DEVICE_LOCAL: retired m_memID = " << m_memID << " (resize)" << std::endl;
    #endif
    m_memID = get_new_mem_id();
    #ifdef VDEBUG
        std::cout << "VULKAN_DEVICE_LOCAL: new m_memID = " << m_memID << " (resize)" << std::endl;
    #endif
}


template<typename DataT>
Memory<DataT, VULKAN_DEVICE_LOCAL>& Memory<DataT, VULKAN_DEVICE_LOCAL>::operator=(const MemoryView<DataT, VULKAN_DEVICE_LOCAL>& o)
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
Memory<DataT, VULKAN_DEVICE_LOCAL>& Memory<DataT, VULKAN_DEVICE_LOCAL>::operator=(const MemoryView<DataT, MemT2>& o)
{
    if(this->size() != o.size())
    {
        this->resize(o.size());
    }

    Base::operator=(o);
    return *this;
}


template<typename DataT>
size_t Memory<DataT, VULKAN_DEVICE_LOCAL>::getID() const
{
    return m_memID;
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
