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

#include "../../util/MemoryVulkan.hpp"



namespace rmagine
{

class VulkanInst : VulkanGeometry
{
private:
    /* data */
public:
    VulkanInst(/* args */);
    ~VulkanInst();
};

using VulkanInstPtr = std::shared_ptr<VulkanInst>;

} // namespace rmagine
