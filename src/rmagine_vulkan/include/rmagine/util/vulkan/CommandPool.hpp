#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include <rmagine/util/VulkanUtil.hpp>
#include "Device.hpp"



namespace rmagine
{

class CommandPool
{
private:
    DevicePtr device = nullptr;

    VkCommandPool commandPool = VK_NULL_HANDLE;

public:
    CommandPool(DevicePtr device) : device(device)
    {
        createCommandPool();
    }

    ~CommandPool() {}

    CommandPool(const CommandPool&) = delete;


    void cleanup();

    VkCommandPool getCommandPool();

private:
    void createCommandPool();
};

using CommandPoolPtr = std::shared_ptr<CommandPool>;

} // namespace rmagine
