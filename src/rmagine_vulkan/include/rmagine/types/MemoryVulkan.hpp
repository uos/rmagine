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
#include <rmagine/types/Memory.hpp>



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



// used for debugging to see when memory objects get created and destroyed
class MemoryHelper
{
private:
    // TODO: maybe hold these as global staging buffers for device local memory
    // static BufferPtr m_stagingBuffer;
    // static DeviceMemoryPtr m_stagingDeviceMemory;

    static size_t MemIDcounter;

public:
    static size_t GetNewMemID();
};



//// MemoryView - VULKAN_HOST_VISIBLE

template<typename DataT>
class MemoryView<DataT, VULKAN_HOST_VISIBLE>
{
protected:
    size_t m_size = 0;
    size_t m_offset = 0;
    VkBufferUsageFlags m_bufferUsageFlags = 0;
    BufferPtr m_buffer = nullptr;
    DeviceMemoryPtr m_deviceMemory = nullptr;

public:
    MemoryView() = delete;

    MemoryView(size_t p_size, size_t p_offset, VkBufferUsageFlags p_bufferUsageFlags, 
               BufferPtr p_buffer, DeviceMemoryPtr p_deviceMemory);

    
    static MemoryView<DataT, VULKAN_HOST_VISIBLE> Empty()
    {
        MemoryView(0, 0, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, nullptr, nullptr);
    }

    bool empty() const
    {
        return (m_size == 0);
    }

    // MemoryView<DataT, VULKAN_HOST_VISIBLE>& operator=(const MemoryView<DataT, VULKAN_HOST_VISIBLE>& o);

    // template<typename MemT2>
    // MemoryView<DataT, VULKAN_HOST_VISIBLE>& operator=(const MemoryView<DataT, MemT2>& o);

    // DataT& at(size_t idx);

    // const DataT& at(size_t idx) const;

    // DataT& operator[](size_t idx);

    // const DataT& operator[](size_t idx) const;

    // DataT& operator*();

    // const DataT& operator*() const;

    // MemoryView<DataT, MemT> slice(size_t idx_start, size_t idx_end);

    // const MemoryView<DataT, MemT> slice(size_t idx_start, size_t idx_end) const;

    // MemoryView<DataT, MemT> operator()(size_t idx_start, size_t idx_end);

    // const MemoryView<DataT, MemT> operator()(size_t idx_start, size_t idx_end) const;

    // TODO:
    // raw() & operator->() cannot be implementet.
    // i am not sure about at(size_t idx), operator[](size_t idx) & operator*().
    // but slice(size_t idx_start, size_t idx_end) & operator()(size_t idx_start, size_t idx_end) should probably work (i will just need to save an offset and a stride)

    size_t size() const;

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
    Memory(size_t size, VkBufferUsageFlags bufferUsageFlags);

    ~Memory();


    void resize(size_t N);

    void resize(size_t N, VkBufferUsageFlags bufferUsageFlags);

    Memory<DataT, VULKAN_HOST_VISIBLE>& operator=(const Memory<DataT, VULKAN_HOST_VISIBLE>& o) = default;//TODO: make it work like the other operator= function

    // Memory<DataT, VULKAN_HOST_VISIBLE>& operator=(const MemoryView<DataT, VULKAN_HOST_VISIBLE>& o);

    template<typename MemT2>
    Memory<DataT, VULKAN_HOST_VISIBLE>& operator=(const Memory<DataT, MemT2>& o);

    // template<typename MemT2>
    // Memory<DataT, VULKAN_HOST_VISIBLE>& operator=(const MemoryView<DataT, MemT2>& o);

    size_t getID() const;

protected:
    using Base::m_size;
    using Base::m_offset;
    using Base::m_bufferUsageFlags;
    using Base::m_buffer;
    using Base::m_deviceMemory;
};



//// MemoryView - VULKAN_DEVICE_LOCAL

template<typename DataT>
class MemoryView<DataT, VULKAN_DEVICE_LOCAL>
{
protected:
    size_t m_size = 0;
    size_t m_offset = 0;
    VkBufferUsageFlags m_bufferUsageFlags = 0;
    BufferPtr m_buffer = nullptr;
    DeviceMemoryPtr m_deviceMemory = nullptr;
    BufferPtr m_stagingBuffer = nullptr;
    DeviceMemoryPtr m_stagingDeviceMemory = nullptr;

public:
    MemoryView() = delete;

    MemoryView(size_t p_size, size_t p_offset, VkBufferUsageFlags p_bufferUsageFlags, 
               BufferPtr p_buffer, DeviceMemoryPtr p_deviceMemory,
               BufferPtr p_stagingBuffer, DeviceMemoryPtr p_stagingDeviceMemory);

    
    static MemoryView<DataT, VULKAN_DEVICE_LOCAL> Empty()
    {
        MemoryView(0, 0, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, nullptr, nullptr, nullptr, nullptr);
    }

    bool empty() const
    {
        return (m_size == 0);
    }

    // MemoryView<DataT, VULKAN_DEVICE_LOCAL>& operator=(const MemoryView<DataT, VULKAN_DEVICE_LOCAL>& o);

    // template<typename MemT2>
    // MemoryView<DataT, VULKAN_DEVICE_LOCAL>& operator=(const MemoryView<DataT, MemT2>& o);


    // DataT& at(size_t idx);

    // const DataT& at(size_t idx) const;

    // DataT& operator[](size_t idx);

    // const DataT& operator[](size_t idx) const;

    // DataT& operator*();

    // const DataT& operator*() const;

    // MemoryView<DataT, MemT> slice(size_t idx_start, size_t idx_end);

    // const MemoryView<DataT, MemT> slice(size_t idx_start, size_t idx_end) const;

    // MemoryView<DataT, MemT> operator()(size_t idx_start, size_t idx_end);

    // const MemoryView<DataT, MemT> operator()(size_t idx_start, size_t idx_end) const;

    // TODO:
    // raw() & operator->() cannot be implementet.
    // i am not sure about at(size_t idx), operator[](size_t idx) & operator*().
    // but slice(size_t idx_start, size_t idx_end) & operator()(size_t idx_start, size_t idx_end) should probably work (i will just need to save an offset and a stride)

    size_t size() const;

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
    Memory(size_t size, VkBufferUsageFlags bufferUsageFlags);

    ~Memory();


    void resize(size_t N);

    Memory<DataT, VULKAN_DEVICE_LOCAL>& operator=(const Memory<DataT, VULKAN_DEVICE_LOCAL>& o) = default;//TODO: make it work like the other operator= function

    // Memory<DataT, VULKAN_DEVICE_LOCAL>& operator=(const MemoryView<DataT, VULKAN_DEVICE_LOCAL>& o);

    template<typename MemT2>
    Memory<DataT, VULKAN_DEVICE_LOCAL>& operator=(const Memory<DataT, MemT2>& o);

    // template<typename MemT2>
    // Memory<DataT, VULKAN_DEVICE_LOCAL>& operator=(const MemoryView<DataT, MemT2>& o);

    size_t getID() const;

protected:
    using Base::m_size;
    using Base::m_offset;
    using Base::m_bufferUsageFlags;
    using Base::m_buffer;
    using Base::m_deviceMemory;
    using Base::m_stagingBuffer;
    using Base::m_stagingDeviceMemory;
};






//////////////////////////
///   HOST TO DEVICE   ///
//////////////////////////

template<typename DataT>
void copy(const MemoryView<DataT, RAM>& from, MemoryView<DataT, VULKAN_HOST_VISIBLE>& to)
{
    if(to.size() == 0)
    {
        throw std::runtime_error("cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::runtime_error("given memory objects have to have the same size!");
    }

    to.getDeviceMemory()->copyToDeviceMemory(from.raw());
}

template<typename DataT>
void copy(const MemoryView<DataT, RAM>& from, MemoryView<DataT, VULKAN_DEVICE_LOCAL>& to)
{
    if(to.size() == 0)
    {
        throw std::runtime_error("cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::runtime_error("given memory objects have to have the same size!");
    }

    to.getStagingDeviceMemory()->copyToDeviceMemory(from.raw());
    get_vulkan_context()->getDefaultCommandBuffer()->recordCopyBufferToCommandBuffer(to.getStagingBuffer(), to.getBuffer());
    get_vulkan_context()->getDefaultCommandBuffer()->submitRecordedCommandAndWait();
}


//////////////////////////
///   DEVICE TO HOST   ///
//////////////////////////

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_HOST_VISIBLE>& from, MemoryView<DataT, RAM>& to)
{
    if(from.size() == 0)
    {
        throw std::runtime_error("cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::runtime_error("given memory objects have to have the same size!");
    }
    
    from.getDeviceMemory()->copyFromDeviceMemory(to.raw());
}

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_DEVICE_LOCAL>& from, MemoryView<DataT, RAM>& to)
{
    if(from.size() == 0)
    {
        throw std::runtime_error("cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::runtime_error("given memory objects have to have the same size!");
    }

    get_vulkan_context()->getDefaultCommandBuffer()->recordCopyBufferToCommandBuffer(from.getBuffer(), from.getStagingBuffer());
    get_vulkan_context()->getDefaultCommandBuffer()->submitRecordedCommandAndWait();
    from.getStagingDeviceMemory()->copyFromDeviceMemory(to.raw());
}


////////////////////////////
///   DEVICE TO DEVICE   ///
////////////////////////////

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_HOST_VISIBLE>& from, MemoryView<DataT, VULKAN_HOST_VISIBLE>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::runtime_error("cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::runtime_error("given memory objects have to have the same size!");
    }

    get_vulkan_context()->getDefaultCommandBuffer()->recordCopyBufferToCommandBuffer(from.getBuffer(), to.getBuffer());
    get_vulkan_context()->getDefaultCommandBuffer()->submitRecordedCommandAndWait();
}

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_DEVICE_LOCAL>& from, MemoryView<DataT, VULKAN_DEVICE_LOCAL>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::runtime_error("cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::runtime_error("given memory objects have to have the same size!");
    }

    get_vulkan_context()->getDefaultCommandBuffer()->recordCopyBufferToCommandBuffer(from.getBuffer(), to.getBuffer());
    get_vulkan_context()->getDefaultCommandBuffer()->submitRecordedCommandAndWait();
}

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_HOST_VISIBLE>& from, MemoryView<DataT, VULKAN_DEVICE_LOCAL>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::runtime_error("cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::runtime_error("given memory objects have to have the same size!");
    }

    get_vulkan_context()->getDefaultCommandBuffer()->recordCopyBufferToCommandBuffer(from.getBuffer(), to.getBuffer());
    get_vulkan_context()->getDefaultCommandBuffer()->submitRecordedCommandAndWait();
}

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_DEVICE_LOCAL>& from, MemoryView<DataT, VULKAN_HOST_VISIBLE>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::runtime_error("cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::runtime_error("given memory objects have to have the same size!");
    }

    get_vulkan_context()->getDefaultCommandBuffer()->recordCopyBufferToCommandBuffer(from.getBuffer(), to.getBuffer());
    get_vulkan_context()->getDefaultCommandBuffer()->submitRecordedCommandAndWait();
}

} // namespace rmagine

#include "MemoryVulkanHostVisible.tcc"
#include "MemoryVulkanDeviceLocal.tcc"
