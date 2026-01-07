#pragma once

#include "VulkanCudaInterop.hpp"



namespace rmagine
{

struct VULKAN_AS_CUDA
{
    template<typename DataT>
    static DataT* alloc(size_t N);

    template<typename DataT>
    static DataT* realloc(DataT* mem, size_t Nold, size_t Nnew);

    template<typename DataT>
    static void free(DataT* mem, size_t N);
};



//// MemoryView - VULKAN_AS_CUDA

template<typename DataT>
class MemoryView<DataT, VULKAN_AS_CUDA>
{
public:
    MemoryView() = delete;

    static MemoryView<DataT, VULKAN_AS_CUDA> Empty()
    {
        return MemoryView<DataT, VULKAN_AS_CUDA>(nullptr, 0, nullptr);
    }

    MemoryView(DataT* mem, size_t N, CudaExternalMemoryPtr externalMemory);

    template<typename MemT2>
    MemoryView(const MemoryView<DataT, MemT2>& vulkanMem);

    // no virtual: we dont want to destruct memory of a view
    ~MemoryView();

    RMAGINE_INLINE_FUNCTION
    bool empty() const 
    {
        return m_mem == nullptr;
    }

    // Copy for assignment of same VULKAN_AS_CUDA
    MemoryView<DataT, VULKAN_AS_CUDA>& operator=(const MemoryView<DataT, VULKAN_AS_CUDA>& o);
    
    // Copy for assignment of different MemT
    template<typename MemT2>
    MemoryView<DataT, VULKAN_AS_CUDA>& operator=(const MemoryView<DataT, MemT2>& o);

    RMAGINE_FUNCTION
    DataT* raw();
    
    RMAGINE_FUNCTION
    const DataT* raw() const;

    RMAGINE_FUNCTION
    DataT* operator->()
    {
        return raw();
    }

    RMAGINE_FUNCTION
    const DataT* operator->() const
    {
        return raw();
    }

    RMAGINE_FUNCTION
    DataT& at(size_t idx)
    {
        return m_mem[idx];
    }

    RMAGINE_FUNCTION
    const DataT& at(size_t idx) const
    {
        return m_mem[idx];
    }

    RMAGINE_FUNCTION
    DataT& operator[](size_t idx)
    {
        return m_mem[idx];
    }

    RMAGINE_FUNCTION
    const DataT& operator[](size_t idx) const
    {
        return m_mem[idx];
    }

    RMAGINE_FUNCTION
    const DataT& operator*() const {
        return *m_mem;
    }

    RMAGINE_FUNCTION
    DataT& operator*() {
        return *m_mem;
    }

    RMAGINE_FUNCTION
    size_t size() const {
        return m_size;
    }

    MemoryView<DataT, VULKAN_AS_CUDA> slice(size_t idx_start, size_t idx_end)
    {
        return MemoryView<DataT, VULKAN_AS_CUDA>(m_mem + idx_start, idx_end - idx_start, m_externalMemory);
    }

    // problem solved: just dont write to it
    const MemoryView<DataT, VULKAN_AS_CUDA> slice(size_t idx_start, size_t idx_end) const 
    {
        return MemoryView<DataT, VULKAN_AS_CUDA>(m_mem + idx_start, idx_end - idx_start, m_externalMemory);
    }

    MemoryView<DataT, VULKAN_AS_CUDA> operator()(size_t idx_start, size_t idx_end)
    {
        return slice(idx_start, idx_end);
    }

    const MemoryView<DataT, VULKAN_AS_CUDA> operator()(size_t idx_start, size_t idx_end) const
    {
        return slice(idx_start, idx_end);
    }

    CudaExternalMemoryPtr getExternalMemory() const
    {
        return m_externalMemory;
    }

protected:
    DataT* m_mem = nullptr;
    size_t m_size = 0;
    CudaExternalMemoryPtr m_externalMemory;
};



//// Memory - VULKAN_AS_CUDA

template<typename DataT>
class Memory<DataT, VULKAN_AS_CUDA>
{
    /**
     * You should never construct a Memory<DataT, VULKAN_AS_CUDA> Object. Only the MemoryView<DataT, VULKAN_AS_CUDA> exists.
     */
    Memory() = delete;
};



//////////////////////////////////////////////////////////
///   VULKAN-AS-CUDA-DEVICE TO VULKAN-AS-CUDA-DEVICE   ///
//////////////////////////////////////////////////////////

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_AS_CUDA>& from, MemoryView<DataT, VULKAN_AS_CUDA>& to)
{
    cuda::memcpyDeviceToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size());
}



////////////////////////////////////////////////
///   VULKAN-AS-CUDA-DEVICE TO CUDA-DEVICE   ///
////////////////////////////////////////////////

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_AS_CUDA>& from, MemoryView<DataT, UNIFIED_CUDA>& to)
{
    cuda::memcpyDeviceToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_AS_CUDA>& from, MemoryView<DataT, VRAM_CUDA>& to)
{
    cuda::memcpyDeviceToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size());
}



////////////////////////////////////////////////
///   CUDA-DEVICE TO VULKAN-AS-CUDA-DEVICE   ///
////////////////////////////////////////////////

template<typename DataT>
void copy(const MemoryView<DataT, UNIFIED_CUDA>& from, MemoryView<DataT, VULKAN_AS_CUDA>& to)
{
    cuda::memcpyDeviceToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

template<typename DataT>
void copy(const MemoryView<DataT, VRAM_CUDA>& from, MemoryView<DataT, VULKAN_AS_CUDA>& to)
{
    cuda::memcpyDeviceToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size());
}



/////////////////////////////////////////
///   VULKAN-AS-CUDA-DEVICE TO HOST   ///
/////////////////////////////////////////

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_AS_CUDA>& from, MemoryView<DataT, RAM_CUDA>& to)
{
    cuda::memcpyDeviceToHost(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_AS_CUDA>& from, MemoryView<DataT, RAM>& to)
{
    cuda::memcpyDeviceToHost(to.raw(), from.raw(), sizeof(DataT) * from.size());
}



/////////////////////////////////////////
///   HOST TO VULKAN-AS-CUDA-DEVICE   ///
/////////////////////////////////////////

template<typename DataT>
void copy(const MemoryView<DataT, RAM_CUDA>& from, MemoryView<DataT, VULKAN_AS_CUDA>& to)
{
    cuda::memcpyHostToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

template<typename DataT>
void copy(const MemoryView<DataT, RAM>& from, MemoryView<DataT, VULKAN_AS_CUDA>& to)
{
    cuda::memcpyHostToDevice(to.raw(), from.raw(), sizeof(DataT) * from.size());
}

} // namespace rmagine

#include "MemoryVulkanCudaInterop.tcc"
