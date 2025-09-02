#include "rmagine/types/MemoryVulkan.hpp"



namespace rmagine
{

// BufferPtr MemoryHelper::MemStagingBuffer = nullptr;
// DeviceMemoryPtr MemoryHelper::MemStagingDeviceMemory = nullptr;

CommandBufferPtr MemoryHelper::MemCommandBuffer = nullptr;

size_t MemoryHelper::MemIDcounter = 0;

size_t MemoryHelper::GetNewMemID()
{
    if(MemIDcounter == SIZE_MAX)
    {
        #ifdef VDEBUG
            std::cout << "[MemoryHelper::GetNewMemID()] DEBUG WARNING - created too many MemIDs, restarting at 1!" << std::endl;
        #endif
        ++MemIDcounter;//skip 0 - it is supposed to be an invalid value
    }
    return ++MemIDcounter;
}

CommandBufferPtr MemoryHelper::GetMemCommandBuffer()
{
    if(MemCommandBuffer == nullptr)
    {
        MemCommandBuffer = std::make_shared<CommandBuffer>(get_vulkan_context());
    }
    return MemCommandBuffer;
}

} // namespace rmagine