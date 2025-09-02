#include "rmagine/types/MemoryVulkan.hpp"



namespace rmagine
{

const VkBufferUsageFlags get_buffer_usage_flags[VulkanMemoryUsage::VULKAN_MEMORY_USEAGE_SIZE] = {
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT                                                                        | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT                                                                        | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR                                      | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT                                                                                                                                              | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR                                                                                                                          | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR                                                              | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
};



BufferPtr MemoryHelper::MemStagingBuffer = nullptr;
DeviceMemoryPtr MemoryHelper::MemStagingDeviceMemory = nullptr;

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