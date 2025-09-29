#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include <rmagine/util/VulkanContextUtil.hpp>
#include "Device.hpp"



namespace rmagine
{

class CommandPool
{
private:
    DeviceWPtr device;

    VkCommandPool commandPool = VK_NULL_HANDLE;

public:
    CommandPool(DeviceWPtr device);

    ~CommandPool();

    CommandPool(const CommandPool&) = delete;


    VkCommandPool getCommandPool();

private:
    void createCommandPool();
};

using CommandPoolPtr = std::shared_ptr<CommandPool>;

} // namespace rmagine
