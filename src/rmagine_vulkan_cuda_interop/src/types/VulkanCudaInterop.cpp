#include "rmagine/types/VulkanCudaInterop.hpp"



namespace rmagine
{

namespace vulkanCudaInterop
{

void importVulkanMemToCuda(DeviceMemoryPtr deviceMemory, VkDeviceSize count, VkDeviceSize byteOffset,
                           void** cudaPtrPtr, cudaExternalMemory_t& cuExternalMemory)
{
    // source: https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleVulkan
    //         https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html

    if(count + byteOffset > deviceMemory->getBuffer()->getBufferSize())
    {
        throw std::invalid_argument("[importVulkanMemToCuda()] ERROR - count and/or byteOffset too large!");
    }


    cudaExternalMemoryHandleDesc cuExternalMemoryHandleDesc{};
    cuExternalMemoryHandleDesc.type      = cudaExternalMemoryHandleTypeOpaqueFd;       // linux
    // cuExternalMemoryHandleDesc.type   = cudaExternalMemoryHandleTypeOpaqueWin32;    // windows 8.10 or greater
    // cuExternalMemoryHandleDesc.type   = cudaExternalMemoryHandleTypeOpaqueWin32Kmt; // less than windows 8.10 
    cuExternalMemoryHandleDesc.size      = count;
    cuExternalMemoryHandleDesc.handle.fd = deviceMemory->getMemoryHandle();
    cuExternalMemoryHandleDesc.flags     = 0;

    RM_CUDA_CHECK(cudaImportExternalMemory(&cuExternalMemory, &cuExternalMemoryHandleDesc));


    cudaExternalMemoryBufferDesc cuExternalMemBufferDesc{};
    cuExternalMemBufferDesc.offset = byteOffset;
    cuExternalMemBufferDesc.size   = count;
    cuExternalMemBufferDesc.flags  = 0;

    RM_CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(cudaPtrPtr, cuExternalMemory, &cuExternalMemBufferDesc));
}



void memcpyVulkanDeviceToCudaDevice(DeviceMemoryPtr srcDeviceMemory, void* dst, VkDeviceSize count, VkDeviceSize srcByteOffset)
{
    void* intermediateCudaPtr = nullptr;
    cudaExternalMemory_t cuExternalMemory;
    importVulkanMemToCuda(srcDeviceMemory, count, srcByteOffset, &intermediateCudaPtr, cuExternalMemory);

    cuda::memcpyDeviceToDevice(dst, intermediateCudaPtr, count);

    RM_CUDA_CHECK(cudaDestroyExternalMemory(cuExternalMemory));
}

void memcpyCudaDeviceToVulkanDevice(const void* src, DeviceMemoryPtr dstDeviceMemory, VkDeviceSize count, VkDeviceSize dstByteOffset)
{
    void* intermediateCudaPtr = nullptr;
    cudaExternalMemory_t cuExternalMemory;
    importVulkanMemToCuda(dstDeviceMemory, count, dstByteOffset, &intermediateCudaPtr, cuExternalMemory);

    cuda::memcpyDeviceToDevice(intermediateCudaPtr, src, count);

    RM_CUDA_CHECK(cudaDestroyExternalMemory(cuExternalMemory));
}

} // namespace vulkanCudaInterop

} // namespace rmagine
