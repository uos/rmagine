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

struct HOST_VISIBLE_VULKAN
{
    template<typename DataT>
    static DataT* alloc(size_t N);

    template<typename DataT>
    static DataT* realloc(DataT* mem, size_t Nold, size_t Nnew);

    template<typename DataT>
    static void free(DataT* mem, size_t N);
};



struct DEVICE_LOCAL_VULKAN
{
    template<typename DataT>
    static DataT* alloc(size_t N);

    template<typename DataT>
    static DataT* realloc(DataT* mem, size_t Nold, size_t Nnew);

    template<typename DataT>
    static void free(DataT* mem, size_t N);
};



//// MemoryView - HOST_VISIBLE_VULKAN

template<typename DataT>
class MemoryView<DataT, HOST_VISIBLE_VULKAN>
{
protected:
    size_t m_size = 0;
    size_t m_offset = 0;
    VulkanMemoryUsage m_memoryUsage = VulkanMemoryUsage::Usage_Default;
    DeviceMemoryPtr m_deviceMemory = nullptr;

public:
    MemoryView() = delete;

    MemoryView(size_t p_size, size_t p_offset, VulkanMemoryUsage p_memoryUsage, DeviceMemoryPtr p_deviceMemory);

    
    static MemoryView<DataT, HOST_VISIBLE_VULKAN> Empty()
    {
        MemoryView(0, 0, VulkanMemoryUsage::Usage_Default, nullptr, nullptr);
    }

    bool empty() const
    {
        return (m_size == 0);
    }

    MemoryView<DataT, HOST_VISIBLE_VULKAN>& operator=(const MemoryView<DataT, HOST_VISIBLE_VULKAN>& o);

    template<typename MemT2>
    MemoryView<DataT, HOST_VISIBLE_VULKAN>& operator=(const MemoryView<DataT, MemT2>& o);

    MemoryView<DataT, HOST_VISIBLE_VULKAN> slice(size_t idx_start, size_t idx_end);

    const MemoryView<DataT, HOST_VISIBLE_VULKAN> slice(size_t idx_start, size_t idx_end) const;

    MemoryView<DataT, HOST_VISIBLE_VULKAN> operator()(size_t idx_start, size_t idx_end);

    const MemoryView<DataT, HOST_VISIBLE_VULKAN> operator()(size_t idx_start, size_t idx_end) const;

    size_t size() const;

    size_t offset() const;

    VulkanMemoryUsage getMemoryUsage() const;

    BufferPtr getBuffer() const;

    DeviceMemoryPtr getDeviceMemory() const;
};



//// Memory - HOST_VISIBLE_VULKAN

template<typename DataT>
class Memory<DataT, HOST_VISIBLE_VULKAN> : public MemoryView<DataT, HOST_VISIBLE_VULKAN>
{
private:
    size_t m_memID = 0;

public:
    using Base = MemoryView<DataT, HOST_VISIBLE_VULKAN>;

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

    // Copy Constructors:

    Memory(const MemoryView<DataT, HOST_VISIBLE_VULKAN>& o);

    Memory(const Memory<DataT, HOST_VISIBLE_VULKAN>& o);

    template<typename MemT2>
    Memory(const MemoryView<DataT, MemT2>& o);

    Memory(Memory<DataT, HOST_VISIBLE_VULKAN>&& o) noexcept;

    ~Memory();


    void resize(size_t N);

    Memory<DataT, HOST_VISIBLE_VULKAN>& operator=(const MemoryView<DataT, HOST_VISIBLE_VULKAN>& o);

    Memory<DataT, HOST_VISIBLE_VULKAN>& operator=(const Memory<DataT, HOST_VISIBLE_VULKAN>& o)
    {
        const MemoryView<DataT, HOST_VISIBLE_VULKAN>& c = o;
        return operator=(c);
    }

    template<typename MemT2>
    Memory<DataT, HOST_VISIBLE_VULKAN>& operator=(const MemoryView<DataT, MemT2>& o);

    template<typename MemT2>
    Memory<DataT, HOST_VISIBLE_VULKAN>& operator=(const Memory<DataT, MemT2>& o)
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



//// MemoryView - DEVICE_LOCAL_VULKAN

template<typename DataT>
class MemoryView<DataT, DEVICE_LOCAL_VULKAN>
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

    
    static MemoryView<DataT, DEVICE_LOCAL_VULKAN> Empty()
    {
        MemoryView(0, 0, VulkanMemoryUsage::Usage_Default, nullptr, nullptr, nullptr, nullptr);
    }

    bool empty() const
    {
        return (m_size == 0);
    }

    MemoryView<DataT, DEVICE_LOCAL_VULKAN>& operator=(const MemoryView<DataT, DEVICE_LOCAL_VULKAN>& o);

    template<typename MemT2>
    MemoryView<DataT, DEVICE_LOCAL_VULKAN>& operator=(const MemoryView<DataT, MemT2>& o);

    MemoryView<DataT, DEVICE_LOCAL_VULKAN> slice(size_t idx_start, size_t idx_end);

    const MemoryView<DataT, DEVICE_LOCAL_VULKAN> slice(size_t idx_start, size_t idx_end) const;

    MemoryView<DataT, DEVICE_LOCAL_VULKAN> operator()(size_t idx_start, size_t idx_end);

    const MemoryView<DataT, DEVICE_LOCAL_VULKAN> operator()(size_t idx_start, size_t idx_end) const;

    size_t size() const;

    size_t offset() const;

    VulkanMemoryUsage getMemoryUsage() const;

    BufferPtr getBuffer() const;

    BufferPtr getStagingBuffer() const;

    DeviceMemoryPtr getDeviceMemory() const;

    DeviceMemoryPtr getStagingDeviceMemory() const;
};



//// Memory - DEVICE_LOCAL_VULKAN

template<typename DataT>
class Memory<DataT, DEVICE_LOCAL_VULKAN> : public MemoryView<DataT, DEVICE_LOCAL_VULKAN>
{
private:
    size_t m_memID = 0;

public:
    using Base = MemoryView<DataT, DEVICE_LOCAL_VULKAN>;

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

    // Copy Constructors:

    Memory(const MemoryView<DataT, DEVICE_LOCAL_VULKAN>& o);

    Memory(const Memory<DataT, DEVICE_LOCAL_VULKAN>& o);

    template<typename MemT2>
    Memory(const MemoryView<DataT, MemT2>& o);

    Memory(Memory<DataT, DEVICE_LOCAL_VULKAN>&& o) noexcept;

    ~Memory();


    void resize(size_t N);

    Memory<DataT, DEVICE_LOCAL_VULKAN>& operator=(const MemoryView<DataT, DEVICE_LOCAL_VULKAN>& o);

    inline Memory<DataT, DEVICE_LOCAL_VULKAN>& operator=(const Memory<DataT, DEVICE_LOCAL_VULKAN>& o)
    {
        const MemoryView<DataT, DEVICE_LOCAL_VULKAN>& c = o;
        return operator=(c);
    }

    template<typename MemT2>
    Memory<DataT, DEVICE_LOCAL_VULKAN>& operator=(const MemoryView<DataT, MemT2>& o);

    template<typename MemT2>
    inline Memory<DataT, DEVICE_LOCAL_VULKAN>& operator=(const Memory<DataT, MemT2>& o)
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
void copy(const MemoryView<DataT, RAM>& from, MemoryView<DataT, HOST_VISIBLE_VULKAN>& to)
{
    if(to.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkan::memcpyHostToDevice(from.raw(), to.getDeviceMemory(), sizeof(DataT) * to.size(), sizeof(DataT) * to.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, RAM>& from, MemoryView<DataT, DEVICE_LOCAL_VULKAN>& to)
{
    if(to.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkan::memcpyHostToDevice(from.raw(), to.getStagingDeviceMemory(), sizeof(DataT) * to.size(), sizeof(DataT) * to.offset());
    vulkan::memcpyDeviceToDevice(to.getStagingBuffer(), to.getBuffer(), sizeof(DataT) * to.size(), sizeof(DataT) * to.offset(), sizeof(DataT) * to.offset());
}


//////////////////////////
///   DEVICE TO HOST   ///
//////////////////////////

template<typename DataT>
void copy(const MemoryView<DataT, HOST_VISIBLE_VULKAN>& from, MemoryView<DataT, RAM>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }
    
    vulkan::memcpyDeviceToHost(from.getDeviceMemory(), to.raw(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, DEVICE_LOCAL_VULKAN>& from, MemoryView<DataT, RAM>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkan::memcpyDeviceToDevice(from.getBuffer(), from.getStagingBuffer(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset(), sizeof(DataT) * from.offset());
    vulkan::memcpyDeviceToHost(from.getStagingDeviceMemory(), to.raw(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset());
}


////////////////////////////
///   DEVICE TO DEVICE   ///
////////////////////////////

template<typename DataT>
void copy(const MemoryView<DataT, HOST_VISIBLE_VULKAN>& from, MemoryView<DataT, HOST_VISIBLE_VULKAN>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkan::memcpyDeviceToDevice(from.getBuffer(), to.getBuffer(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset(), sizeof(DataT) * to.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, DEVICE_LOCAL_VULKAN>& from, MemoryView<DataT, DEVICE_LOCAL_VULKAN>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkan::memcpyDeviceToDevice(from.getBuffer(), to.getBuffer(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset(), sizeof(DataT) * to.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, HOST_VISIBLE_VULKAN>& from, MemoryView<DataT, DEVICE_LOCAL_VULKAN>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkan::memcpyDeviceToDevice(from.getBuffer(), to.getBuffer(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset(), sizeof(DataT) * to.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, DEVICE_LOCAL_VULKAN>& from, MemoryView<DataT, HOST_VISIBLE_VULKAN>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkan::memcpyDeviceToDevice(from.getBuffer(), to.getBuffer(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset(), sizeof(DataT) * to.offset());
}

} // namespace rmagine

#include "MemoryVulkanHostVisible.tcc"
#include "MemoryVulkanDeviceLocal.tcc"
