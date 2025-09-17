#include "rmagine/util/vulkan/PipelineLayout.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

PipelineLayout::PipelineLayout(DevicePtr device, DescriptorSetLayoutPtr descriptorSetLayout) : device(device), descriptorSetLayout(descriptorSetLayout)
{
    createPipelineLayout();
}

PipelineLayout::~PipelineLayout()
{
    if(pipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device->getLogicalDevice(), pipelineLayout, nullptr);
    }
    device.reset();
}



void PipelineLayout::createPipelineLayout()
{
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayout->getDescriptorSetLayoutPtr();

    if(vkCreatePipelineLayout(device->getLogicalDevice(), &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("[PipelineLayout::createPipelineLayout()] ERROR - Failed to create pipeline layout!");
    }
}


VkPipelineLayout PipelineLayout::getPipelineLayout()
{
    return pipelineLayout;
}

} // namespace rmagine
