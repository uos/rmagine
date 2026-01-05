#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include <rmagine/util/VulkanContext.hpp>



namespace rmagine
{

class VulkanEntity : public std::enable_shared_from_this<VulkanEntity>
{
protected:
    VulkanContextPtr vulkan_context = nullptr;
    
public:
    std::string name;

    VulkanEntity() : vulkan_context(get_vulkan_context()) {}

    virtual ~VulkanEntity() {}

    
    template<typename T>
    inline std::shared_ptr<T> this_shared()
    {
        return std::dynamic_pointer_cast<T>(shared_from_this());
    }
};

using VulkanEntityPtr = std::shared_ptr<VulkanEntity>;

} // namespace rmagine
