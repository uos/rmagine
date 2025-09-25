#include "rmagine/types/VulkanCudaInterop.hpp"



namespace rmagine
{

namespace vulkanCudaInterop
{

void importVulkanMemToCuda(DeviceMemoryPtr deviceMemory, VkDeviceSize size, VkDeviceSize srcOffset,
                           void** cudaPtr, cudaExternalMemory_t& cuExternalMemory)
{
    // source: https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleVulkan
    //         https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html

    if(size + srcOffset > deviceMemory->getBuffer()->getBufferSize())
    {
        throw std::invalid_argument("[importVulkanMemToCuda()] ERROR - size and/or srcOffset too large!");
    }


    cudaExternalMemoryHandleDesc cuExternalMemoryHandleDesc{};
    cuExternalMemoryHandleDesc.type      = cudaExternalMemoryHandleTypeOpaqueFd;       // linux
    // cuExternalMemoryHandleDesc.type   = cudaExternalMemoryHandleTypeOpaqueWin32;    // windows 8.10 or greater
    // cuExternalMemoryHandleDesc.type   = cudaExternalMemoryHandleTypeOpaqueWin32Kmt; // less than windows 8.10 
    cuExternalMemoryHandleDesc.size      = size;
    cuExternalMemoryHandleDesc.handle.fd = deviceMemory->getMemoryHandle();
    cuExternalMemoryHandleDesc.flags     = 0;

    RM_CUDA_CHECK(cudaImportExternalMemory(&cuExternalMemory, &cuExternalMemoryHandleDesc));


    cudaExternalMemoryBufferDesc cuExternalMemBufferDesc{};
    cuExternalMemBufferDesc.offset = srcOffset;
    cuExternalMemBufferDesc.size   = size;
    cuExternalMemBufferDesc.flags  = 0;

    RM_CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(cudaPtr, cuExternalMemory, &cuExternalMemBufferDesc));
}



void memcpyVulkanDeviceToCudaDevice(DeviceMemoryPtr srcDeviceMemory, void* dst, VkDeviceSize size, VkDeviceSize srcOffset)
{
    void* intermediateCudaPtr = nullptr;
    cudaExternalMemory_t cuExternalMemory;
    importVulkanMemToCuda(srcDeviceMemory, size, srcOffset, &intermediateCudaPtr, cuExternalMemory);

    cuda::memcpyDeviceToDevice(dst, intermediateCudaPtr, size);

    RM_CUDA_CHECK(cudaDestroyExternalMemory(cuExternalMemory));
}

void memcpyCudaDeviceToVulkanDevice(const void* src, DeviceMemoryPtr dstDeviceMemory, VkDeviceSize size, VkDeviceSize dstOffset)
{
    void* intermediateCudaPtr = nullptr;
    cudaExternalMemory_t cuExternalMemory;
    importVulkanMemToCuda(dstDeviceMemory, size, dstOffset, &intermediateCudaPtr, cuExternalMemory);

    cuda::memcpyDeviceToDevice(intermediateCudaPtr, src, size);

    RM_CUDA_CHECK(cudaDestroyExternalMemory(cuExternalMemory));
}

} // namespace vulkanCudaInterop

} // namespace rmagine
