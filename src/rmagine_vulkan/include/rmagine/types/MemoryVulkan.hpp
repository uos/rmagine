#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include <rmagine/util/VulkanContext.hpp>
#include <rmagine/util/VulkanUtil.hpp>
#include <rmagine/util/vulkan/memory/Buffer.hpp>
#include <rmagine/util/vulkan/memory/DeviceMemory.hpp>
#include <rmagine/util/vulkan/command/CommandBuffer.hpp>
#include <rmagine/types/Memory.hpp>
#include "MemoryVulkanUtil.hpp"



namespace rmagine
{

struct VULKAN_HOST_VISIBLE
{
    template<typename DataT>
    static DataT* alloc(size_t N);

    template<typename DataT>
    static DataT* realloc(DataT* mem, size_t Nold, size_t Nnew);

    template<typename DataT>
    static void free(DataT* mem, size_t N);
};



struct VULKAN_DEVICE_LOCAL
{
    template<typename DataT>
    static DataT* alloc(size_t N);

    template<typename DataT>
    static DataT* realloc(DataT* mem, size_t Nold, size_t Nnew);

    template<typename DataT>
    static void free(DataT* mem, size_t N);
};



//// MemoryView - VULKAN_HOST_VISIBLE

template<typename DataT>
class MemoryView<DataT, VULKAN_HOST_VISIBLE>
{
protected:
    size_t m_size = 0;
    size_t m_offset = 0;
    VulkanMemoryUsage m_memoryUsage = VulkanMemoryUsage::Usage_Default;
    DeviceMemoryPtr m_deviceMemory = nullptr;

public:
    MemoryView() = delete;

    MemoryView(size_t p_size, size_t p_offset, VulkanMemoryUsage p_memoryUsage, DeviceMemoryPtr p_deviceMemory);

    
    static MemoryView<DataT, VULKAN_HOST_VISIBLE> Empty()
    {
        MemoryView(0, 0, VulkanMemoryUsage::Usage_Default, nullptr, nullptr);
    }

    bool empty() const
    {
        return (m_size == 0);
    }

    MemoryView<DataT, VULKAN_HOST_VISIBLE>& operator=(const MemoryView<DataT, VULKAN_HOST_VISIBLE>& o);

    template<typename MemT2>
    MemoryView<DataT, VULKAN_HOST_VISIBLE>& operator=(const MemoryView<DataT, MemT2>& o);

    MemoryView<DataT, VULKAN_HOST_VISIBLE> slice(size_t idx_start, size_t idx_end);

    const MemoryView<DataT, VULKAN_HOST_VISIBLE> slice(size_t idx_start, size_t idx_end) const;

    MemoryView<DataT, VULKAN_HOST_VISIBLE> operator()(size_t idx_start, size_t idx_end);

    const MemoryView<DataT, VULKAN_HOST_VISIBLE> operator()(size_t idx_start, size_t idx_end) const;

    size_t size() const;

    size_t offset() const;

    BufferPtr getBuffer() const;

    DeviceMemoryPtr getDeviceMemory() const;
};



//// Memory - VULKAN_HOST_VISIBLE

template<typename DataT>
class Memory<DataT, VULKAN_HOST_VISIBLE> : public MemoryView<DataT, VULKAN_HOST_VISIBLE>
{
private:
    size_t m_memID = 0;

public:
    using Base = MemoryView<DataT, VULKAN_HOST_VISIBLE>;

    Memory();
    
    Memory(size_t size);

    /**
     * some memory objects are used for special puposes, such as:
     * building or holding an accleration structure,
     * holding the shaderBindingTable or
     * being a uniform buffer (for better performance)
     * in these cases special bufferUsageFlags besides the default storage buffer are needed
     */
    Memory(size_t size, VulkanMemoryUsage memoryUsage);

    ~Memory();


    void resize(size_t N);

    Memory<DataT, VULKAN_HOST_VISIBLE>& operator=(const MemoryView<DataT, VULKAN_HOST_VISIBLE>& o);

    Memory<DataT, VULKAN_HOST_VISIBLE>& operator=(const Memory<DataT, VULKAN_HOST_VISIBLE>& o)
    {
        const MemoryView<DataT, VULKAN_HOST_VISIBLE>& c = o;
        return operator=(c);
    }

    template<typename MemT2>
    Memory<DataT, VULKAN_HOST_VISIBLE>& operator=(const MemoryView<DataT, MemT2>& o);

    template<typename MemT2>
    Memory<DataT, VULKAN_HOST_VISIBLE>& operator=(const Memory<DataT, MemT2>& o)
    {
        const MemoryView<DataT, MemT2>& c = o;
        return operator=(c);
    }

    size_t getID() const;

protected:
    using Base::m_size;
    using Base::m_offset;
    using Base::m_memoryUsage;
    using Base::m_deviceMemory;
};



//// MemoryView - VULKAN_DEVICE_LOCAL

template<typename DataT>
class MemoryView<DataT, VULKAN_DEVICE_LOCAL>
{
protected:
    size_t m_size = 0;
    size_t m_offset = 0;
    VulkanMemoryUsage m_memoryUsage = VulkanMemoryUsage::Usage_Default;
    DeviceMemoryPtr m_deviceMemory = nullptr;
    DeviceMemoryPtr m_stagingDeviceMemory = nullptr;

public:
    MemoryView() = delete;

    MemoryView(size_t p_size, size_t p_offset, VulkanMemoryUsage p_memoryUsage, DeviceMemoryPtr p_deviceMemory, DeviceMemoryPtr p_stagingDeviceMemory);

    
    static MemoryView<DataT, VULKAN_DEVICE_LOCAL> Empty()
    {
        MemoryView(0, 0, VulkanMemoryUsage::Usage_Default, nullptr, nullptr, nullptr, nullptr);
    }

    bool empty() const
    {
        return (m_size == 0);
    }

    MemoryView<DataT, VULKAN_DEVICE_LOCAL>& operator=(const MemoryView<DataT, VULKAN_DEVICE_LOCAL>& o);

    template<typename MemT2>
    MemoryView<DataT, VULKAN_DEVICE_LOCAL>& operator=(const MemoryView<DataT, MemT2>& o);

    MemoryView<DataT, VULKAN_DEVICE_LOCAL> slice(size_t idx_start, size_t idx_end);

    const MemoryView<DataT, VULKAN_DEVICE_LOCAL> slice(size_t idx_start, size_t idx_end) const;

    MemoryView<DataT, VULKAN_DEVICE_LOCAL> operator()(size_t idx_start, size_t idx_end);

    const MemoryView<DataT, VULKAN_DEVICE_LOCAL> operator()(size_t idx_start, size_t idx_end) const;

    size_t size() const;

    size_t offset() const;

    BufferPtr getBuffer() const;

    BufferPtr getStagingBuffer() const;

    DeviceMemoryPtr getDeviceMemory() const;

    DeviceMemoryPtr getStagingDeviceMemory() const;
};



//// Memory - VULKAN_DEVICE_LOCAL

template<typename DataT>
class Memory<DataT, VULKAN_DEVICE_LOCAL> : public MemoryView<DataT, VULKAN_DEVICE_LOCAL>
{
private:
    size_t m_memID = 0;

public:
    using Base = MemoryView<DataT, VULKAN_DEVICE_LOCAL>;

    Memory();
    
    Memory(size_t size);

    /**
     * some memory objects are used for special puposes, such as:
     * building or holding an accleration structure,
     * holding the shaderBindingTable or
     * being a uniform buffer (for better performance)
     * in these cases special bufferUsageFlags besides the default storage buffer are needed
     */
    Memory(size_t size, VulkanMemoryUsage memoryUsage);

    ~Memory();


    void resize(size_t N);

    Memory<DataT, VULKAN_DEVICE_LOCAL>& operator=(const MemoryView<DataT, VULKAN_DEVICE_LOCAL>& o);

    inline Memory<DataT, VULKAN_DEVICE_LOCAL>& operator=(const Memory<DataT, VULKAN_DEVICE_LOCAL>& o)
    {
        const MemoryView<DataT, VULKAN_DEVICE_LOCAL>& c = o;
        return operator=(c);
    }

    template<typename MemT2>
    Memory<DataT, VULKAN_DEVICE_LOCAL>& operator=(const MemoryView<DataT, MemT2>& o);

    template<typename MemT2>
    inline Memory<DataT, VULKAN_DEVICE_LOCAL>& operator=(const Memory<DataT, MemT2>& o)
    {
        const MemoryView<DataT, MemT2>& c = o;
        return operator=(c);
    }

    size_t getID() const;

protected:
    using Base::m_size;
    using Base::m_offset;
    using Base::m_memoryUsage;
    using Base::m_deviceMemory;
    using Base::m_stagingDeviceMemory;
};






//////////////////////////
///   HOST TO DEVICE   ///
//////////////////////////

template<typename DataT>
void copy(const MemoryView<DataT, RAM>& from, MemoryView<DataT, VULKAN_HOST_VISIBLE>& to)
{
    if(to.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkan_memcpy_host_to_device(from.raw(), to.getDeviceMemory(), sizeof(DataT) * to.size(), sizeof(DataT) * to.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, RAM>& from, MemoryView<DataT, VULKAN_DEVICE_LOCAL>& to)
{
    if(to.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkan_memcpy_host_to_device(from.raw(), to.getStagingDeviceMemory(), sizeof(DataT) * to.size(), sizeof(DataT) * to.offset());
    vulkan_memcpy_device_to_device(to.getStagingBuffer(), to.getBuffer(), sizeof(DataT) * to.size(), sizeof(DataT) * to.offset(), sizeof(DataT) * to.offset());
}


//////////////////////////
///   DEVICE TO HOST   ///
//////////////////////////

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_HOST_VISIBLE>& from, MemoryView<DataT, RAM>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }
    
    vulkan_memcpy_device_to_host(from.getDeviceMemory(), to.raw(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_DEVICE_LOCAL>& from, MemoryView<DataT, RAM>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkan_memcpy_device_to_device(from.getBuffer(), from.getStagingBuffer(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset(), sizeof(DataT) * from.offset());
    vulkan_memcpy_device_to_host(from.getStagingDeviceMemory(), to.raw(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset());
}


////////////////////////////
///   DEVICE TO DEVICE   ///
////////////////////////////

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_HOST_VISIBLE>& from, MemoryView<DataT, VULKAN_HOST_VISIBLE>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkan_memcpy_device_to_device(from.getBuffer(), to.getBuffer(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset(), sizeof(DataT) * to.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_DEVICE_LOCAL>& from, MemoryView<DataT, VULKAN_DEVICE_LOCAL>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkan_memcpy_device_to_device(from.getBuffer(), to.getBuffer(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset(), sizeof(DataT) * to.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_HOST_VISIBLE>& from, MemoryView<DataT, VULKAN_DEVICE_LOCAL>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkan_memcpy_device_to_device(from.getBuffer(), to.getBuffer(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset(), sizeof(DataT) * to.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_DEVICE_LOCAL>& from, MemoryView<DataT, VULKAN_HOST_VISIBLE>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkan_memcpy_device_to_device(from.getBuffer(), to.getBuffer(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset(), sizeof(DataT) * to.offset());
}

} // namespace rmagine

#include "MemoryVulkanHostVisible.tcc"
#include "MemoryVulkanDeviceLocal.tcc"
