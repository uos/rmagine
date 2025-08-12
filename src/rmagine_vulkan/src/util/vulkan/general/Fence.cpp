#include "Fence.hpp"
#include "../VulkanContext.hpp"



namespace rmagine
{

Fence::Fence() : device(get_vulkan_context()->getDevice())
{
    createFence();
}

Fence::Fence(DevicePtr device) : device(device)
{
    createFence();
}



void Fence::createFence()
{
    if(fence != VK_NULL_HANDLE)
    {
        throw std::runtime_error("tried to create a fence that already exists!");
    }

    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    if(vkCreateFence(device->getLogicalDevice(), &fenceCreateInfo, nullptr, &fence) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create fence!");
    }
}


void Fence::submitWithFenceAndWait(VkSubmitInfo& submitInfo)
{
    submitWithFence(submitInfo);
    waitForFence();
}


void Fence::submitWithFence(VkSubmitInfo& submitInfo)
{
    if(vkQueueSubmit(device->getQueue(), 1,  &submitInfo, fence) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to submit build!");
    }
}


void Fence::waitForFence()
{
    if(vkWaitForFences(device->getLogicalDevice(), 1, &fence, true, UINT64_MAX) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to wait for fence!");
    }

    vkResetFences(device->getLogicalDevice(), 1, &fence);
}


void Fence::cleanup()
{
    if(fence != VK_NULL_HANDLE)
    {
        vkDestroyFence(device->getLogicalDevice(), fence, nullptr);
        fence = VK_NULL_HANDLE;
    }
}

} // namespace rmagine
