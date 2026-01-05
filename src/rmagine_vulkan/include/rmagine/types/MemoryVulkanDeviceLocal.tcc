#include "MemoryVulkan.hpp"

namespace rmagine
{

//// MemoryView - DEVICE_LOCAL_VULKAN

template<typename DataT>
MemoryView<DataT, DEVICE_LOCAL_VULKAN>::MemoryView(
    size_t p_size, size_t p_offset, VulkanMemoryUsage p_memoryUsage, DeviceMemoryPtr p_deviceMemory, DeviceMemoryPtr p_stagingDeviceMemory, CommandBufferPtr p_commandBuffer) :
    m_size(p_size), m_offset(p_offset), m_memoryUsage(p_memoryUsage), m_deviceMemory(p_deviceMemory), m_stagingDeviceMemory(p_stagingDeviceMemory), m_commandBuffer(p_commandBuffer)
{

}



template<typename DataT>
MemoryView<DataT, DEVICE_LOCAL_VULKAN>& MemoryView<DataT, DEVICE_LOCAL_VULKAN>::operator=(const MemoryView<DataT, DEVICE_LOCAL_VULKAN>& o)
{
    if(this->size() != o.size())
    {
        throw std::invalid_argument("[MemoryView<DataT, DEVICE_LOCAL_VULKAN>::operator=()] ERROR - DEVICE_LOCAL_VULKAN assignment of different sizes");
    }

    copy(o, *this);
    return *this;
}


template<typename DataT>
template<typename MemT2>
MemoryView<DataT, DEVICE_LOCAL_VULKAN>& MemoryView<DataT, DEVICE_LOCAL_VULKAN>::operator=(const MemoryView<DataT, MemT2>& o)
{
    if(this->size() != o.size())
    {
        throw std::invalid_argument("[MemoryView<DataT, DEVICE_LOCAL_VULKAN>::operator=()] ERROR - MemT2 assignment of different sizes");
    }

    copy(o, *this);
    return *this;
}


template<typename DataT>
MemoryView<DataT, DEVICE_LOCAL_VULKAN> MemoryView<DataT, DEVICE_LOCAL_VULKAN>::slice(size_t idx_start, size_t idx_end)
{
    if(idx_start >= m_size || idx_end > m_size || idx_start >= idx_end)
    {
        throw std::invalid_argument("[MemoryView<DataT, DEVICE_LOCAL_VULKAN>::slice()] ERROR - invlaid indicies");
    }

    return MemoryView<DataT, DEVICE_LOCAL_VULKAN>(idx_end - idx_start, idx_start, m_memoryUsage, m_deviceMemory, m_stagingDeviceMemory, m_commandBuffer);
}


template<typename DataT>
const MemoryView<DataT, DEVICE_LOCAL_VULKAN> MemoryView<DataT, DEVICE_LOCAL_VULKAN>::slice(size_t idx_start, size_t idx_end) const
{
    if(idx_start >= m_size || idx_end > m_size || idx_start >= idx_end)
    {
        throw std::invalid_argument("[MemoryView<DataT, DEVICE_LOCAL_VULKAN>::slice()] ERROR - invlaid indicies");
    }

    return MemoryView<DataT, DEVICE_LOCAL_VULKAN>(idx_end - idx_start, idx_start, m_memoryUsage, m_deviceMemory, m_stagingDeviceMemory, m_commandBuffer);
}


template<typename DataT>
MemoryView<DataT, DEVICE_LOCAL_VULKAN> MemoryView<DataT, DEVICE_LOCAL_VULKAN>::operator()(size_t idx_start, size_t idx_end)
{
    return slice(idx_start, idx_end);
}


template<typename DataT>
const MemoryView<DataT, DEVICE_LOCAL_VULKAN> MemoryView<DataT, DEVICE_LOCAL_VULKAN>::operator()(size_t idx_start, size_t idx_end) const
{
    return slice(idx_start, idx_end);
}


template<typename DataT>
size_t MemoryView<DataT, DEVICE_LOCAL_VULKAN>::size() const
{
    return m_size;
}


template<typename DataT>
size_t MemoryView<DataT, DEVICE_LOCAL_VULKAN>::offset() const
{
    return m_offset;
}


template<typename DataT>
VulkanMemoryUsage MemoryView<DataT, DEVICE_LOCAL_VULKAN>::getMemoryUsage() const
{
    return m_memoryUsage;
}


template<typename DataT>
BufferPtr MemoryView<DataT, DEVICE_LOCAL_VULKAN>::getBuffer() const
{
    return m_deviceMemory->getBuffer();
}


template<typename DataT>
BufferPtr MemoryView<DataT, DEVICE_LOCAL_VULKAN>::getStagingBuffer() const
{
    if(m_memoryUsage == VulkanMemoryUsage::Usage_AccelerationStructure || m_memoryUsage == VulkanMemoryUsage::Usage_AccelerationStructureScratch)
        return nullptr;
    return m_stagingDeviceMemory->getBuffer();
}


template<typename DataT>
DeviceMemoryPtr MemoryView<DataT, DEVICE_LOCAL_VULKAN>::getDeviceMemory() const
{
    return m_deviceMemory;
}


template<typename DataT>
DeviceMemoryPtr MemoryView<DataT, DEVICE_LOCAL_VULKAN>::getStagingDeviceMemory() const
{
    if(m_memoryUsage == VulkanMemoryUsage::Usage_AccelerationStructure || m_memoryUsage == VulkanMemoryUsage::Usage_AccelerationStructureScratch)
        return nullptr;
    return m_stagingDeviceMemory;
}


template<typename DataT>
CommandBufferPtr MemoryView<DataT, DEVICE_LOCAL_VULKAN>::getCommandBuffer() const
{
    return m_commandBuffer;
}






//// Memory - DEVICE_LOCAL_VULKAN

template<typename DataT>
Memory<DataT, DEVICE_LOCAL_VULKAN>::Memory() : Memory(0)
{
    
}

template<typename DataT>
Memory<DataT, DEVICE_LOCAL_VULKAN>::Memory(size_t N) : Memory(N, VulkanMemoryUsage::Usage_Default)
{
    
}

template<typename DataT>
Memory<DataT, DEVICE_LOCAL_VULKAN>::Memory(size_t N, VulkanMemoryUsage memoryUsage) :
    Base(N, 0, memoryUsage, nullptr, nullptr, std::make_shared<CommandBuffer>(get_vulkan_context()))
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
        #if defined(VDEBUG)
            std::cout << "[DEVICE_LOCAL_VULKAN] new m_memID = " << m_memID << std::endl;
        #endif
    }
}

template<typename DataT>
Memory<DataT, DEVICE_LOCAL_VULKAN>::Memory(const MemoryView<DataT, DEVICE_LOCAL_VULKAN>& o) : Memory(o.size(), o.getMemoryUsage())
{
    copy(o, *this);
}

template<typename DataT>
Memory<DataT, DEVICE_LOCAL_VULKAN>::Memory(const Memory<DataT, DEVICE_LOCAL_VULKAN>& o) : Memory(o.size(), o.getMemoryUsage())
{
    copy(o, *this);
}

template<typename DataT>
template<typename MemT2>
Memory<DataT, DEVICE_LOCAL_VULKAN>::Memory(const MemoryView<DataT, MemT2>& o) : 
    Memory(o.size(), VulkanMemoryUsage::Usage_Default)//has to be default i guess...
{
    copy(o, *this);
}

template<typename DataT>
Memory<DataT, DEVICE_LOCAL_VULKAN>::Memory(Memory<DataT, DEVICE_LOCAL_VULKAN>&& o) noexcept :
    Base(o.size(), o.offset()/*should always be 0*/, o.getMemoryUsage(), o.m_deviceMemory, o.m_stagingDeviceMemory, o.m_commandBuffer)
{
    o.m_deviceMemory = nullptr;
    o.m_stagingDeviceMemory = nullptr;
    o.m_size = 0;
    // o.m_offset = 0;/*should already be 0*/
}

template<typename DataT>
Memory<DataT, DEVICE_LOCAL_VULKAN>::~Memory()
{
    #if defined(VDEBUG)
        if(m_memID != 0)
            std::cout << "[DEVICE_LOCAL_VULKAN] retired m_memID = " << m_memID << std::endl;
    #endif
}


template<typename DataT>
void Memory<DataT, DEVICE_LOCAL_VULKAN>::resize(size_t N)
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

    #if defined(VDEBUG)
        if(m_memID != 0)
            std::cout << "[DEVICE_LOCAL_VULKAN] retired m_memID = " << m_memID << " (resize)" << std::endl;
    #endif
    m_memID = get_new_mem_id();
    #if defined(VDEBUG)
        std::cout << "[DEVICE_LOCAL_VULKAN] new m_memID = " << m_memID << " (resize)" << std::endl;
    #endif
}


template<typename DataT>
Memory<DataT, DEVICE_LOCAL_VULKAN>& Memory<DataT, DEVICE_LOCAL_VULKAN>::operator=(const MemoryView<DataT, DEVICE_LOCAL_VULKAN>& o)
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
Memory<DataT, DEVICE_LOCAL_VULKAN>& Memory<DataT, DEVICE_LOCAL_VULKAN>::operator=(const MemoryView<DataT, MemT2>& o)
{
    if(this->size() != o.size())
    {
        this->resize(o.size());
    }

    Base::operator=(o);
    return *this;
}


template<typename DataT>
size_t Memory<DataT, DEVICE_LOCAL_VULKAN>::getID() const
{
    return m_memID;
}






//// DEVICE_LOCAL_VULKAN

template<typename DataT>
DataT* DEVICE_LOCAL_VULKAN::alloc(size_t N)
{
    (void) N;
    throw std::runtime_error("DEVICE_LOCAL_VULKAN (alloc): The program should never call this function and always use the template specialization instead!");
    return nullptr;
}

template<typename DataT>
DataT* DEVICE_LOCAL_VULKAN::realloc(DataT* mem, size_t Nold, size_t Nnew)
{
    (void) mem;
    (void) Nold;
    (void) Nnew;
    throw std::runtime_error("DEVICE_LOCAL_VULKAN (realloc): The program should never call this function and always use the template specialization instead!");
    return nullptr;
}

template<typename DataT>
void DEVICE_LOCAL_VULKAN::free(DataT* mem, size_t N)
{
    (void) mem;
    (void) N;
    throw std::runtime_error("DEVICE_LOCAL_VULKAN (free): The program should never call this function and always use the template specialization instead!");
}

} // namespace rmagine
