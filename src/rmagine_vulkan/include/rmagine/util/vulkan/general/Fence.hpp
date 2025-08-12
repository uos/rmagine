#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>
#include "../contextComponents/Device.hpp"



namespace rmagine
{

class Fence
{
private:
    DevicePtr device = nullptr;

    VkFence fence = VK_NULL_HANDLE;

public:
    /**
     * THIS CONSTRUCTOR MUST NOT BE CALLED FROM THE CONSTRUCTOR OF THE VULKAN-CONTEXT
     */
    Fence();

    Fence(DevicePtr device);

    ~Fence() {}

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

    /**
     * free the Fence
     */
    void cleanup();
    
private:
    /**
     * create the Fence
     */
    void createFence();
};

using FencePtr = std::shared_ptr<Fence>;

} // namespace rmagine
