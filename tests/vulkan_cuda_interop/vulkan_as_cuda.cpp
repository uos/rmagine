#include <iostream>

#include <rmagine/types/MemoryVulkanCudaInterop.hpp>
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
    
    // vulkan as cuda mem
    MemoryView<uint64_t, VULKAN_AS_CUDA> nums_cuda_devLocal(nums_vulkan_devLocal);
    cmp_mem(nums_ram, nums_cuda_devLocal, "nums_cuda_devLocal");

    MemoryView<uint64_t, VULKAN_AS_CUDA> nums_cuda_hostVis(nums_vulkan_hostVis);
    cmp_mem(nums_ram, nums_cuda_hostVis, "nums_cuda_hostVis");
    
    MemoryView<uint64_t, RAM> nums_ram_2 = nums_ram.slice(4, 8);

    MemoryView<uint64_t, VULKAN_AS_CUDA> nums_cuda_devLocal_2 = nums_cuda_devLocal.slice(4, 8);
    cmp_mem(nums_ram_2, nums_cuda_devLocal_2, "nums_cuda_devLocal_2");

    MemoryView<uint64_t, VULKAN_AS_CUDA> nums_vulkan_hostVis_2 = nums_cuda_hostVis.slice(4, 8);
    cmp_mem(nums_ram_2, nums_vulkan_hostVis_2, "nums_vulkan_hostVis_2");

    std::cout << "Done testing." << std::endl;

    return 0;
}
