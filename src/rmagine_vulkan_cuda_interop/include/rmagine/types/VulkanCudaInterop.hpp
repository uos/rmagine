#include <driver_types.h>

#include "rmagine/types/MemoryVulkan.hpp"
#include "rmagine/types/MemoryCuda.hpp"
#include "rmagine/util/cuda/CudaDebug.hpp"



namespace rmagine
{

namespace vulkanCudaInterop
{

    void importVulkanMemToCuda(DeviceMemoryPtr deviceMemory, VkDeviceSize size, VkDeviceSize srcOffset,
                               void** cudaPtr, cudaExternalMemory_t& cuExternalMemory);

    void memcpyVulkanDeviceToCudaDevice(DeviceMemoryPtr srcDeviceMemory, void* dst, VkDeviceSize size, VkDeviceSize srcOffset);
    void memcpyCudaDeviceToVulkanDevice(const void* src, DeviceMemoryPtr dstDeviceMemory, VkDeviceSize size, VkDeviceSize dstOffset);

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
