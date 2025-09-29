#include "rmagine/types/MemoryVulkanUtil.hpp"



namespace rmagine
{

const VkBufferUsageFlags get_buffer_usage_flags_arr[VulkanMemoryUsage::VULKAN_MEMORY_USEAGE_SIZE] = {
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT                                                                        | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT                                                                        | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR                                      | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT                                                                                                                                              | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR                                                                                                                          | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR                                                              | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
};

VkBufferUsageFlags get_buffer_usage_flags(VulkanMemoryUsage memUsage)
{
    return get_buffer_usage_flags_arr[memUsage];
}



size_t mem_id_counter = 0;

size_t get_new_mem_id()
{
    if(mem_id_counter == SIZE_MAX)
    {
        #if defined(VDEBUG)
            std::cout << "[get_new_mem_id()] WARNING - created too many MemIDs, restarting at 1!" << std::endl;
        #endif
        ++mem_id_counter;//skip 0 - it is supposed to be an invalid value
    }
    return ++mem_id_counter;
}



namespace vulkan
{

std::mutex vulkan_memcpy_mutex;

CommandBufferPtr mem_command_buffer = nullptr;

void memcpyDeviceToDevice(BufferPtr srcBuffer, BufferPtr dstBuffer, VkDeviceSize count, VkDeviceSize srcByteOffset, VkDeviceSize dstByteOffset)
{
    std::lock_guard<std::mutex> guard(vulkan_memcpy_mutex);
    if(mem_command_buffer == nullptr)
    {
        mem_command_buffer = std::make_shared<CommandBuffer>(get_vulkan_context());
    }

    mem_command_buffer->recordCopyBufferToCommandBuffer(srcBuffer, dstBuffer, count, srcByteOffset, dstByteOffset);
    mem_command_buffer->submitRecordedCommandAndWait();
}

void memcpyHostToDevice(const void* src, DeviceMemoryPtr dstDeviceMemory, VkDeviceSize count, VkDeviceSize dstByteOffset)
{
    dstDeviceMemory->copyToDeviceMemory(src, dstByteOffset, count);
}

void memcpyDeviceToHost(DeviceMemoryPtr srcDeviceMemory, void* dst, VkDeviceSize count, VkDeviceSize srcByteOffset)
{
    srcDeviceMemory->copyFromDeviceMemory(dst, srcByteOffset, count);
}

} // namespace vulkan

} // namespace rmagine
