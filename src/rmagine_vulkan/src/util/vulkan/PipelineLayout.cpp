#include "rmagine/util/vulkan/PipelineLayout.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

PipelineLayout::PipelineLayout(VulkanContextWPtr vulkan_context) : vulkan_context(vulkan_context), device(vulkan_context.lock()->getDevice())
{
    createPipelineLayout();
}

PipelineLayout::~PipelineLayout()
{
    std::cout << "Destroying PipelineLayout" << std::endl;
    if(pipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device->getLogicalDevice(), pipelineLayout, nullptr);
    }
    device.reset();
    std::cout << "PipelineLayout destroyed" << std::endl;
}



void PipelineLayout::createPipelineLayout()
{
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = vulkan_context.lock()->getDescriptorSetLayout()->getDescriptorSetLayoutPtr();

    if(vkCreatePipelineLayout(vulkan_context.lock()->getDevice()->getLogicalDevice(), &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create pipeline layout!");
    }
}


VkPipelineLayout PipelineLayout::getPipelineLayout()
{
    return pipelineLayout;
}

} // namespace rmagine
