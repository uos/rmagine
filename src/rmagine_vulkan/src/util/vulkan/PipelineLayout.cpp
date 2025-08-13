#include "rmagine/util/vulkan/PipelineLayout.hpp"



namespace rmagine
{

void PipelineLayout::createPipelineLayout()
{
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayout->getDescriptorSetLayoutPtr();

    if(vkCreatePipelineLayout(device->getLogicalDevice(), &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create pipeline layout!");
    }
}


void PipelineLayout::cleanup()
{
    if(pipelineLayout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device->getLogicalDevice(), pipelineLayout, nullptr);
}



VkPipelineLayout PipelineLayout::getPipelineLayout()
{
    return pipelineLayout;
}

} // namespace rmagine
