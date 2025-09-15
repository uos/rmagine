#include "rmagine/util/vulkan/CommandPool.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{
CommandPool::CommandPool(DevicePtr device) : device(device)
{
    createCommandPool();
}

CommandPool::~CommandPool()
{
    std::cout << "Destroying CommandPool" << std::endl;
    if(commandPool != VK_NULL_HANDLE)
    {
        vkResetCommandPool(device->getLogicalDevice(), commandPool, VkCommandPoolResetFlagBits::VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
        vkDestroyCommandPool(device->getLogicalDevice(), commandPool, nullptr);
    }
    device.reset();
    std::cout << "CommandPool destroyed" << std::endl;
}



void CommandPool::createCommandPool()
{
    VkCommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolCreateInfo.queueFamilyIndex = device->getQueueFamilyIndex();

    if(vkCreateCommandPool(device->getLogicalDevice(), &commandPoolCreateInfo, nullptr,  &commandPool) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create command pool!");
    }
}


VkCommandPool CommandPool::getCommandPool()
{
    return commandPool;
}

} // namespace rmagine
