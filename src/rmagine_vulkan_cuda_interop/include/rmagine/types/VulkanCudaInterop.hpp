#pragma once

#include <driver_types.h>

#include <rmagine/types/Memory.hpp>
#include "rmagine/types/MemoryVulkan.hpp"
#include "rmagine/types/MemoryCuda.hpp"
#include "rmagine/util/cuda/CudaDebug.hpp"



namespace rmagine
{

/**
 * Wrapper struct for cudaExternalMemory_t to ensure its destruction when shared_ptr gets freed
 */
struct CudaExternalMemory
{
    cudaExternalMemory_t externalMemory;

    ~CudaExternalMemory()
    {
        RM_CUDA_CHECK(cudaDestroyExternalMemory(externalMemory));
    }
};
using CudaExternalMemoryPtr = std::shared_ptr<CudaExternalMemory>;



namespace vulkanCudaInterop
{

/**
 * import a piece of vulkan memory to cuda, 
 * allows you to access a piece of vulkan memory through a cuda pointer
 * 
 * @param deviceMemory vulkan device memory that get imported to cuda
 * 
 * @param externalMemory wrapper for a cuda struct that manages the external memory
 * 
 * @param count number of bytes that get imported
 * 
 * @param byteOffset offset into the vulkan device memory
 * 
 * @return cuda ptr that accesses the vulkan memory
 */
void* importVulkanMemToCuda(DeviceMemoryPtr deviceMemory, CudaExternalMemoryPtr externalMemory, VkDeviceSize count, VkDeviceSize byteOffset);

/**
 * copy data from vulkan memory to a cuda memory
 * 
 * @param srcDeviceMemory (soucre) vulkan device memory
 * 
 * @param dst (destination) cuda pointer
 * 
 * @param count number of bytes that get copied
 * 
 * @param offset offset into the vulkan device memory
 */
void memcpyVulkanDeviceToCudaDevice(DeviceMemoryPtr srcDeviceMemory, void* dst, VkDeviceSize count, VkDeviceSize srcByteOffset);

/**
 * copy data from a cuda memory to a vulkan memory
 * 
 * @param src (soucre) cuda pointer
 * 
 * @param dstDeviceMemory (destination) vulkan device memory
 * 
 * @param count number of bytes that get copied
 * 
 * @param offset offset into the vulkan device memory
 */
void memcpyCudaDeviceToVulkanDevice(const void* src, DeviceMemoryPtr dstDeviceMemory, VkDeviceSize count, VkDeviceSize dstByteOffset);

} // namespace vulkanCudaInterop



////////////////////////////////////////
///   VULKAN-DEVICE TO CUDA-DEVICE   ///
////////////////////////////////////////

template<typename DataT>
void copy(const MemoryView<DataT, HOST_VISIBLE_VULKAN>& from, MemoryView<DataT, UNIFIED_CUDA>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkanCudaInterop::memcpyVulkanDeviceToCudaDevice(from.getDeviceMemory(), to.raw(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, HOST_VISIBLE_VULKAN>& from, MemoryView<DataT, VRAM_CUDA>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkanCudaInterop::memcpyVulkanDeviceToCudaDevice(from.getDeviceMemory(), to.raw(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, DEVICE_LOCAL_VULKAN>& from, MemoryView<DataT, UNIFIED_CUDA>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkanCudaInterop::memcpyVulkanDeviceToCudaDevice(from.getDeviceMemory(), to.raw(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, DEVICE_LOCAL_VULKAN>& from, MemoryView<DataT, VRAM_CUDA>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkanCudaInterop::memcpyVulkanDeviceToCudaDevice(from.getDeviceMemory(), to.raw(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset());
}



////////////////////////////////////////
///   CUDA-DEVICE TO VULKAN-DEVICE   ///
////////////////////////////////////////

template<typename DataT>
void copy(const MemoryView<DataT, UNIFIED_CUDA>& from, MemoryView<DataT, HOST_VISIBLE_VULKAN>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkanCudaInterop::memcpyCudaDeviceToVulkanDevice(from.raw(), to.getDeviceMemory(), sizeof(DataT) * to.size(), sizeof(DataT) * to.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, VRAM_CUDA>& from, MemoryView<DataT, HOST_VISIBLE_VULKAN>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkanCudaInterop::memcpyCudaDeviceToVulkanDevice(from.raw(), to.getDeviceMemory(), sizeof(DataT) * to.size(), sizeof(DataT) * to.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, UNIFIED_CUDA>& from, MemoryView<DataT, DEVICE_LOCAL_VULKAN>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkanCudaInterop::memcpyCudaDeviceToVulkanDevice(from.raw(), to.getDeviceMemory(), sizeof(DataT) * to.size(), sizeof(DataT) * to.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, VRAM_CUDA>& from, MemoryView<DataT, DEVICE_LOCAL_VULKAN>& to)
{
    if(from.size() == 0 || to.size() == 0)
    {
        throw std::invalid_argument("[copy()] ERROR - cant be called when size() is 0!");
    }
    if(to.size() != from.size())
    {
        throw std::invalid_argument("[copy()] ERROR - memoryViews need to have the same size!");
    }

    vulkanCudaInterop::memcpyCudaDeviceToVulkanDevice(from.raw(), to.getDeviceMemory(), sizeof(DataT) * to.size(), sizeof(DataT) * to.offset());
}

} // namespace rmagine
