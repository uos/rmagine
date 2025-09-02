#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include <rmagine/util/VulkanUtil.hpp>
#include <rmagine/util/vulkan/Device.hpp>



namespace rmagine
{

class Fence
{
private:
    VulkanContextWPtr vulkan_context;
    DevicePtr device = nullptr;

    VkFence fence = VK_NULL_HANDLE;

public:
    Fence(VulkanContextWPtr vulkan_context);

    ~Fence();

    Fence(const Fence&) = delete;


    /**
     * combines submitWithFence(VkSubmitInfo&) and waitForFence()
     * 
     * @param submitInfo task you want the gpu to complete
     */
    void submitWithFenceAndWait(VkSubmitInfo& submitInfo);

    /**
     * submit a task to the gpu, 
     * you can then call waitForFence() to wait until this task has been completed
     * 
     * @param submitInfo task you want the gpu to complete
     */
    void submitWithFence(VkSubmitInfo& submitInfo);

    /**
     * wait until the task that has been submit to the gpu with this Fence has been completed
     */
    void waitForFence();
    
private:
    /**
     * create the Fence
     */
    void createFence();
};

using FencePtr = std::shared_ptr<Fence>;

} // namespace rmagine
