#include "rmagine/util/vulkan/CommandPool.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{
CommandPool::CommandPool(VulkanContextWPtr vulkan_context) : vulkan_context(vulkan_context), device(vulkan_context.lock()->getDevice())
{
    createCommandPool();
}

CommandPool::~CommandPool()
{
    std::cout << "Destroying CommandPool" << std::endl;
    if(commandPool != VK_NULL_HANDLE)
    {
        vkResetCommandPool(device->getLogicalDevice(), commandPool, VkCommandPoolResetFlagBits::VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
        vkDestroyCommandPool(device->getLogicalDevice(), commandPool, NULL);
    }
    device.reset();
    std::cout << "CommandPool destroyed" << std::endl;
}



void CommandPool::createCommandPool()
{
    VkCommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolCreateInfo.queueFamilyIndex = vulkan_context.lock()->getDevice()->getQueueFamilyIndex();

    if(vkCreateCommandPool(vulkan_context.lock()->getDevice()->getLogicalDevice(), &commandPoolCreateInfo, NULL,  &commandPool) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create command pool!");
    }
}


VkCommandPool CommandPool::getCommandPool()
{
    return commandPool;
}

} // namespace rmagine
