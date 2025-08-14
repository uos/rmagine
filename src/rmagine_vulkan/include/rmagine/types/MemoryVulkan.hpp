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

struct VULKAN_DEVICE_LOCAL
{
    template<typename DataT>
    static DataT* alloc(size_t N);

    template<typename DataT>
    static DataT* realloc(DataT* mem, size_t Nold, size_t Nnew);

    template<typename DataT>
    static void free(DataT* mem, size_t N);
};



struct MemoryData
{
    size_t size = 0;
    size_t memID = 0;
    BufferPtr buffer = nullptr;
    DeviceMemoryPtr deviceMemory = nullptr;
    BufferPtr stagingBuffer = nullptr;
    DeviceMemoryPtr stagingDeviceMemory = nullptr;

    MemoryData()
    {
        memID = getNewMemID();
        std::cout << "New Memory with memID: " << memID << std::endl;
    };

    ~MemoryData()
    {
        std::cout << "Destroying Memory with memID: " << memID << std::endl;
        if(deviceMemory != nullptr)
            deviceMemory->cleanup();
        if(buffer != nullptr)
            buffer->cleanup();
        if(stagingDeviceMemory != nullptr)
            stagingDeviceMemory->cleanup();
        if(stagingBuffer != nullptr)
            stagingBuffer->cleanup();
    }

    MemoryData(const MemoryData&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues

private:
    static size_t memIDcounter;

    static size_t getNewMemID();
};
using MemoryDataPtr = std::shared_ptr<MemoryData>;



template<typename DataT>
class MemoryView<DataT, VULKAN_DEVICE_LOCAL>
{
protected:
    VkDeviceAddress bufferDeviceAddress; // needs to be read on gpu
    MemoryDataPtr memoryData = nullptr;  // reduces footprint on gpu, bundels data & allows memoryViews to access the same data as the correspondeing memory objects easily

public:
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

    size_t getID() const;

    BufferPtr getBuffer() const;

    BufferPtr getStagingBuffer() const;

    DeviceMemoryPtr getDeviceMemory() const;

    DeviceMemoryPtr getStagingDeviceMemory() const;
};



template<typename DataT>
class Memory<DataT, VULKAN_DEVICE_LOCAL> : public MemoryView<DataT, VULKAN_DEVICE_LOCAL>
{
public:
    using Base = MemoryView<DataT, VULKAN_DEVICE_LOCAL>;

    Memory();
    
    Memory(size_t size);

    Memory(size_t size, VkBufferUsageFlags bufferUsageFlags);

    ~Memory() {};


    void resize(size_t N);

    void resize(size_t N, VkBufferUsageFlags bufferUsageFlags);

    Memory<DataT, VULKAN_DEVICE_LOCAL>& operator=(const Memory<DataT, VULKAN_DEVICE_LOCAL>& o) = default;//TODO: make it work like the other operator= function

    // Memory<DataT, VULKAN_DEVICE_LOCAL>& operator=(const MemoryView<DataT, VULKAN_DEVICE_LOCAL>& o);

    template<typename MemT2>
    Memory<DataT, VULKAN_DEVICE_LOCAL>& operator=(const Memory<DataT, MemT2>& o);

    // template<typename MemT2>
    // Memory<DataT, VULKAN_DEVICE_LOCAL>& operator=(const MemoryView<DataT, MemT2>& o);

protected:
    using Base::bufferDeviceAddress;
    using Base::memoryData;
};






//////////////////////////
///   HOST TO DEVICE   ///
//////////////////////////
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

} // namespace rmagine

#include "MemoryVulkan.tcc"
