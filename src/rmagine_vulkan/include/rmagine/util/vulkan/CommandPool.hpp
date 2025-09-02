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
    VulkanContextWPtr vulkan_context;
    DevicePtr device = nullptr;

    VkCommandPool commandPool = VK_NULL_HANDLE;

public:
    CommandPool(VulkanContextWPtr vulkan_context);

    ~CommandPool();

    CommandPool(const CommandPool&) = delete;


    VkCommandPool getCommandPool();

private:
    void createCommandPool();
};

using CommandPoolPtr = std::shared_ptr<CommandPool>;

} // namespace rmagine
