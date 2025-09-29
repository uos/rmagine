#include "rmagine/util/vulkan/command/Fence.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

Fence::Fence(VulkanContextPtr vulkan_context) : vulkan_context(vulkan_context)
{
    createFence();
}

Fence::~Fence()
{
    if(fence != VK_NULL_HANDLE)
    {
        vkDestroyFence(vulkan_context->getDevice()->getLogicalDevice(), fence, nullptr);
    }
}



void Fence::createFence()
{
    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    if(vkCreateFence(vulkan_context->getDevice()->getLogicalDevice(), &fenceCreateInfo, nullptr, &fence) != VK_SUCCESS)
    {
        throw std::runtime_error("[Fence::createFence()] ERROR - failed to create fence!");
    }
}


void Fence::submitWithFenceAndWait(VkSubmitInfo& submitInfo)
{
    submitWithFence(submitInfo);
    waitForFence();
}


void Fence::submitWithFence(VkSubmitInfo& submitInfo)
{
    if(vkQueueSubmit(vulkan_context->getDevice()->getQueue(), 1,  &submitInfo, fence) != VK_SUCCESS)
    {
        throw std::runtime_error("[Fence::submitWithFence()] ERROR - failed to submit build!");
    }
}


void Fence::waitForFence()
{
    if(vkWaitForFences(vulkan_context->getDevice()->getLogicalDevice(), 1, &fence, true, UINT64_MAX) != VK_SUCCESS)
    {
        throw std::runtime_error("[Fence::waitForFence()] ERROR - failed to wait for fence!");
    }

    vkResetFences(vulkan_context->getDevice()->getLogicalDevice(), 1, &fence);
}

} // namespace rmagine
