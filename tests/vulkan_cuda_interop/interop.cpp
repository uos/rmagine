#include <iostream>

#include <rmagine/types/VulkanCudaInterop.hpp>
#include <rmagine/util/exceptions.h>

#include <stdexcept>
#include <cassert>

using namespace rmagine;



template<typename MemT>
void cmp_mem(MemoryView<uint64_t, RAM>& originalMem, MemoryView<uint64_t, MemT>& newMem, std::string note = "")
{
    std::cout << note << std::endl;

    Memory<uint64_t, RAM> newMem_ram(newMem.size());
    newMem_ram = newMem;

    if(memcmp(originalMem.raw(), newMem_ram.raw(), originalMem.size() * sizeof(uint64_t)) != 0)
    {
        std::cout << "original Mem:" << std::endl;
        for(size_t i = 0; i < originalMem.size(); i++)
        {
            std::cout << originalMem[i] << ", ";
        }
        std::cout << std::endl;
        
        std::cout << "new Mem:" << std::endl;
        for(size_t i = 0; i < newMem_ram.size(); i++)
        {
            std::cout << newMem_ram[i] << ", ";
        }
        std::cout << std::endl;

        throw VulkanCudaInteropException(note + ": content not equal!");
    }
}



int main(int argc, char** argv)
{
    std::cout << "Start testing." << std::endl;

    // fill some memory
    Memory<uint64_t, RAM> nums_ram(10);
    for(size_t i = 0; i < nums_ram.size(); i++)
    {
        nums_ram[i] = i;
    }

    // starting vulkan mem
    Memory<uint64_t, DEVICE_LOCAL_VULKAN> nums_vulkan_devLocal(nums_ram.size());
    nums_vulkan_devLocal = nums_ram;
    Memory<uint64_t, HOST_VISIBLE_VULKAN> nums_vulkan_hostVis(nums_ram.size());
    nums_vulkan_hostVis = nums_ram;
    
    // starting cuda mem
    Memory<uint64_t, UNIFIED_CUDA> nums_cuda_uni(nums_ram.size());
    nums_cuda_uni = nums_ram;
    Memory<uint64_t, VRAM_CUDA> nums_cuda_vram(nums_ram.size());
    nums_cuda_vram = nums_ram;


    // vulkan to cuda
    {
        Memory<uint64_t, UNIFIED_CUDA> nums_cuda_uni_2(nums_ram.size());
        nums_cuda_uni_2 = nums_vulkan_hostVis;
        cmp_mem(nums_ram, nums_cuda_uni_2, "HOST_VISIBLE_VULKAN to UNIFIED_CUDA");

        Memory<uint64_t, UNIFIED_CUDA> nums_cuda_uni_3(nums_ram.size());
        nums_cuda_uni_3 = nums_vulkan_devLocal;
        cmp_mem(nums_ram, nums_cuda_uni_3, "DEVICE_LOCAL_VULKAN to UNIFIED_CUDA");

        Memory<uint64_t, VRAM_CUDA> nums_cuda_vram_2(nums_ram.size());
        nums_cuda_vram_2 = nums_vulkan_hostVis;
        cmp_mem(nums_ram, nums_cuda_vram_2, "HOST_VISIBLE_VULKAN to VRAM_CUDA");

        Memory<uint64_t, VRAM_CUDA> nums_cuda_vram_3(nums_ram.size());
        nums_cuda_vram_3 = nums_vulkan_devLocal;
        cmp_mem(nums_ram, nums_cuda_vram_3, "DEVICE_LOCAL_VULKAN to VRAM_CUDA");
    }


    // cuda to vulkan
    {
        Memory<uint64_t, HOST_VISIBLE_VULKAN> nums_vulkan_hostVis_2(nums_ram.size());
        nums_vulkan_hostVis_2 = nums_cuda_uni;
        cmp_mem(nums_ram, nums_vulkan_hostVis_2, "UNIFIED_CUDA to HOST_VISIBLE_VULKAN");

        Memory<uint64_t, HOST_VISIBLE_VULKAN> nums_vulkan_hostVis_3(nums_ram.size());
        nums_vulkan_hostVis_3 = nums_cuda_vram;
        cmp_mem(nums_ram, nums_vulkan_hostVis_3, "VRAM_CUDA to HOST_VISIBLE_VULKAN");

        Memory<uint64_t, DEVICE_LOCAL_VULKAN> nums_vulkan_devLocal_2(nums_ram.size());
        nums_vulkan_devLocal_2 = nums_cuda_uni;
        cmp_mem(nums_ram, nums_vulkan_devLocal_2, "UNIFIED_CUDA to DEVICE_LOCAL_VULKAN");

        Memory<uint64_t, DEVICE_LOCAL_VULKAN> nums_vulkan_devLocal_3(nums_ram.size());
        nums_vulkan_devLocal_3 = nums_cuda_vram;
        cmp_mem(nums_ram, nums_vulkan_devLocal_3, "VRAM_CUDA to DEVICE_LOCAL_VULKAN");
    }


    std::cout << "Done testing." << std::endl;

    return 0;
}
