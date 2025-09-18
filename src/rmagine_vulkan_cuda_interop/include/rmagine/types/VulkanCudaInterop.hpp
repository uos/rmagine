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
void copy(const MemoryView<DataT, VULKAN_HOST_VISIBLE>& from, MemoryView<DataT, UNIFIED_CUDA>& to)
{
    memcpyVulkanDeviceToCudaDevice(from.getDeviceMemory(), to.raw(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_HOST_VISIBLE>& from, MemoryView<DataT, VRAM_CUDA>& to)
{
    memcpyVulkanDeviceToCudaDevice(from.getDeviceMemory(), to.raw(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_DEVICE_LOCAL>& from, MemoryView<DataT, UNIFIED_CUDA>& to)
{
    memcpyVulkanDeviceToCudaDevice(from.getDeviceMemory(), to.raw(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, VULKAN_DEVICE_LOCAL>& from, MemoryView<DataT, VRAM_CUDA>& to)
{
    memcpyVulkanDeviceToCudaDevice(from.getDeviceMemory(), to.raw(), sizeof(DataT) * from.size(), sizeof(DataT) * from.offset());
}



////////////////////////////////////////
///   CUDA-DEVICE TO VULKAN-DEVICE   ///
////////////////////////////////////////

template<typename DataT>
void copy(const MemoryView<DataT, UNIFIED_CUDA>& from, MemoryView<DataT, VULKAN_HOST_VISIBLE>& to)
{
    memcpyCudaDeviceToVulkanDevice(from.raw(), to.getDeviceMemory(), sizeof(DataT) * to.size(), sizeof(DataT) * to.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, VRAM_CUDA>& from, MemoryView<DataT, VULKAN_HOST_VISIBLE>& to)
{
    memcpyCudaDeviceToVulkanDevice(from.raw(), to.getDeviceMemory(), sizeof(DataT) * to.size(), sizeof(DataT) * to.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, UNIFIED_CUDA>& from, MemoryView<DataT, VULKAN_DEVICE_LOCAL>& to)
{
    memcpyCudaDeviceToVulkanDevice(from.raw(), to.getDeviceMemory(), sizeof(DataT) * to.size(), sizeof(DataT) * to.offset());
}

template<typename DataT>
void copy(const MemoryView<DataT, VRAM_CUDA>& from, MemoryView<DataT, VULKAN_DEVICE_LOCAL>& to)
{
    memcpyCudaDeviceToVulkanDevice(from.raw(), to.getDeviceMemory(), sizeof(DataT) * to.size(), sizeof(DataT) * to.offset());
}

} // namespace rmagine
