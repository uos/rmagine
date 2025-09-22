#include "rmagine/util/vulkan/RayTracingPipelineLayout.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

RayTracingPipelineLayout::RayTracingPipelineLayout(DevicePtr device, DescriptorSetLayoutPtr descriptorSetLayout) : device(device), descriptorSetLayout(descriptorSetLayout)
{
    createPipelineLayout();
}

RayTracingPipelineLayout::~RayTracingPipelineLayout()
{
    if(pipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device->getLogicalDevice(), pipelineLayout, nullptr);
    }
    device.reset();
}



void RayTracingPipelineLayout::createPipelineLayout()
{
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayout->getDescriptorSetLayoutPtr();

    if(vkCreatePipelineLayout(device->getLogicalDevice(), &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("[RayTracingPipelineLayout::createPipelineLayout()] ERROR - Failed to create pipeline layout!");
    }
}


VkPipelineLayout RayTracingPipelineLayout::getPipelineLayout()
{
    return pipelineLayout;
}

} // namespace rmagine
