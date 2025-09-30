#include "rmagine/types/VulkanCudaInterop.hpp"



namespace rmagine
{

namespace vulkanCudaInterop
{

void* importVulkanMemToCuda(DeviceMemoryPtr deviceMemory, CudaExternalMemoryPtr externalMemory, VkDeviceSize count, VkDeviceSize byteOffset)
{
    // source: https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleVulkan
    //         https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html

    if(count + byteOffset > deviceMemory->getBuffer()->getBufferSize())
    {
        throw std::invalid_argument("[importVulkanMemToCuda()] ERROR - count and/or byteOffset too large!");
    }

    void* ptr = nullptr;

    cudaExternalMemoryHandleDesc cuExternalMemoryHandleDesc{};
    cuExternalMemoryHandleDesc.type      = cudaExternalMemoryHandleTypeOpaqueFd;       // linux
    // cuExternalMemoryHandleDesc.type   = cudaExternalMemoryHandleTypeOpaqueWin32;    // windows 8.10 or greater
    // cuExternalMemoryHandleDesc.type   = cudaExternalMemoryHandleTypeOpaqueWin32Kmt; // less than windows 8.10 
    cuExternalMemoryHandleDesc.size      = count;
    cuExternalMemoryHandleDesc.handle.fd = deviceMemory->getMemoryHandle();
    cuExternalMemoryHandleDesc.flags     = 0;

    RM_CUDA_CHECK(cudaImportExternalMemory(&(externalMemory->externalMemory), &cuExternalMemoryHandleDesc));

    cudaExternalMemoryBufferDesc cuExternalMemBufferDesc{};
    cuExternalMemBufferDesc.offset = byteOffset;
    cuExternalMemBufferDesc.size   = count;
    cuExternalMemBufferDesc.flags  = 0;

    RM_CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&ptr, externalMemory->externalMemory, &cuExternalMemBufferDesc));

    return ptr;
}



void memcpyVulkanDeviceToCudaDevice(DeviceMemoryPtr srcDeviceMemory, void* dst, VkDeviceSize count, VkDeviceSize srcByteOffset)
{
    CudaExternalMemoryPtr cuExternalMemory = std::make_shared<CudaExternalMemory>();
    void* intermediateCudaPtr = importVulkanMemToCuda(srcDeviceMemory, cuExternalMemory, count, srcByteOffset);

    cuda::memcpyDeviceToDevice(dst, intermediateCudaPtr, count);
}

void memcpyCudaDeviceToVulkanDevice(const void* src, DeviceMemoryPtr dstDeviceMemory, VkDeviceSize count, VkDeviceSize dstByteOffset)
{
    CudaExternalMemoryPtr cuExternalMemory = std::make_shared<CudaExternalMemory>();
    void* intermediateCudaPtr = importVulkanMemToCuda(dstDeviceMemory, cuExternalMemory, count, dstByteOffset);

    cuda::memcpyDeviceToDevice(intermediateCudaPtr, src, count);
}

} // namespace vulkanCudaInterop

} // namespace rmagine
