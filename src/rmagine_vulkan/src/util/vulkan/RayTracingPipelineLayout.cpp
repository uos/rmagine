#include "rmagine/util/vulkan/RayTracingPipelineLayout.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

RayTracingPipelineLayout::RayTracingPipelineLayout(DeviceWPtr device, DescriptorSetLayoutWPtr descriptorSetLayout) : device(device), descriptorSetLayout(descriptorSetLayout)
{
    createPipelineLayout();
    createPipelineCache();
}

RayTracingPipelineLayout::~RayTracingPipelineLayout()
{
    if(pipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device.lock()->getLogicalDevice(), pipelineLayout, nullptr);
    }
    if(pipelineCache != VK_NULL_HANDLE)
    {
        vkDestroyPipelineCache(device.lock()->getLogicalDevice(), pipelineCache, nullptr);
    }
    device.reset();
}



void RayTracingPipelineLayout::createPipelineLayout()
{
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayout.lock()->getDescriptorSetLayoutPtr();

    if(vkCreatePipelineLayout(device.lock()->getLogicalDevice(), &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("[RayTracingPipelineLayout::createPipelineLayout()] ERROR - Failed to create pipeline layout!");
    }
}



void RayTracingPipelineLayout::createPipelineCache()
{
    VkPipelineCacheCreateInfo pipelineCacheCreateInfo{};
    pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    // pipelineCacheCreateInfo.flags = VK_PIPELINE_CACHE_CREATE_INTERNALLY_SYNCHRONIZED_MERGE_BIT_KHR; //TODO: check if this is neccessary/makes a difference

    if(vkCreatePipelineCache(device.lock()->getLogicalDevice(), &pipelineCacheCreateInfo, nullptr, &pipelineCache) != VK_SUCCESS)
    {
        throw std::runtime_error("[RayTracingPipeline::createPipelineCache()] ERROR - Failed to create pipeline cache!");
    }
}



VkPipelineLayout RayTracingPipelineLayout::getPipelineLayout()
{
    return pipelineLayout;
}


VkPipelineCache RayTracingPipelineLayout::getPipelineCache()
{
    return pipelineCache;
}

} // namespace rmagine
