#include "rmagine/util/vulkan/Pipeline.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

Pipeline::Pipeline(VulkanContextWPtr vulkan_context, ShaderDefineFlags shaderDefines) : vulkan_context(vulkan_context), device(vulkan_context.lock()->getDevice())
{
    createPipelineCache();
    createPipeline(shaderDefines);
}


Pipeline::~Pipeline() 
{
    if(pipelineCache != VK_NULL_HANDLE)
    {
        vkDestroyPipelineCache(device->getLogicalDevice(), pipelineCache, nullptr);
    }
    if(pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device->getLogicalDevice(), pipeline, nullptr);
    }
    device.reset();
}



void Pipeline::createPipelineCache()
{
    VkPipelineCacheCreateInfo pipelineCacheCreateInfo{};
    pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;

    if(vkCreatePipelineCache(vulkan_context.lock()->getDevice()->getLogicalDevice(), &pipelineCacheCreateInfo, nullptr, &pipelineCache) != VK_SUCCESS)
    {
        throw std::runtime_error("[Pipeline::createPipelineCache()] ERROR - Failed to create pipeline cache!");
    }
}


void Pipeline::createPipeline(ShaderDefineFlags shaderDefines)
{
    std::vector<VkPipelineShaderStageCreateInfo> pipelineShaderStageCreateInfoList = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .pNext = nullptr,
         .flags = 0,
         .stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
         .module = get_vulkan_context()->getShader(ShaderType::CHit, shaderDefines)->getShaderModule(),
         .pName = "main",
         .pSpecializationInfo = nullptr},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .pNext = nullptr,
         .flags = 0,
         .stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
         .module = get_vulkan_context()->getShader(ShaderType::RGen, shaderDefines)->getShaderModule(),
         .pName = "main",
         .pSpecializationInfo = nullptr},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .pNext = nullptr,
         .flags = 0,
         .stage = VK_SHADER_STAGE_MISS_BIT_KHR,
         .module = get_vulkan_context()->getShader(ShaderType::Miss, shaderDefines)->getShaderModule(),
         .pName = "main",
         .pSpecializationInfo = nullptr}};
    
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> rayTracingShaderGroupCreateInfoList = {
        {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
         .pNext = nullptr,
         .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
         .generalShader = VK_SHADER_UNUSED_KHR,
         .closestHitShader = 0,
         .anyHitShader = VK_SHADER_UNUSED_KHR,
         .intersectionShader = VK_SHADER_UNUSED_KHR,
         .pShaderGroupCaptureReplayHandle = nullptr},
        {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
         .pNext = nullptr,
         .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
         .generalShader = 1,
         .closestHitShader = VK_SHADER_UNUSED_KHR,
         .anyHitShader = VK_SHADER_UNUSED_KHR,
         .intersectionShader = VK_SHADER_UNUSED_KHR,
         .pShaderGroupCaptureReplayHandle = nullptr},
        {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
         .pNext = nullptr,
         .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
         .generalShader = 2,
         .closestHitShader = VK_SHADER_UNUSED_KHR,
         .anyHitShader = VK_SHADER_UNUSED_KHR,
         .intersectionShader = VK_SHADER_UNUSED_KHR,
         .pShaderGroupCaptureReplayHandle = nullptr}};

    VkRayTracingPipelineCreateInfoKHR pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pipelineCreateInfo.stageCount = (uint32_t)pipelineShaderStageCreateInfoList.size();
    pipelineCreateInfo.pStages = pipelineShaderStageCreateInfoList.data();
    pipelineCreateInfo.groupCount = (uint32_t)rayTracingShaderGroupCreateInfoList.size();
    pipelineCreateInfo.pGroups = rayTracingShaderGroupCreateInfoList.data();
    pipelineCreateInfo.maxPipelineRayRecursionDepth = 1;
    pipelineCreateInfo.layout = vulkan_context.lock()->getPipelineLayout()->getPipelineLayout(),
    pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineCreateInfo.basePipelineIndex = 0;

    if(vulkan_context.lock()->extensionFuncs.vkCreateRayTracingPipelinesKHR(vulkan_context.lock()->getDevice()->getLogicalDevice(), VK_NULL_HANDLE, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("[Pipeline::createPipeline()] ERROR - Failed to create pipeline!");
    }
}


VkPipeline Pipeline::getPipeline()
{
    return pipeline;
}

} // namespace rmagine
