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
    std::vector<VkPipelineShaderStageCreateInfo> pipelineShaderStageCreateInfoList(3);
    pipelineShaderStageCreateInfoList[0] = {};//closest hit
    pipelineShaderStageCreateInfoList[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfoList[0].pNext = nullptr;
    pipelineShaderStageCreateInfoList[0].flags = 0;
    pipelineShaderStageCreateInfoList[0].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    pipelineShaderStageCreateInfoList[0].module = get_vulkan_context()->getShader(ShaderType::CHit, shaderDefines)->getShaderModule();
    pipelineShaderStageCreateInfoList[0].pName = "main";
    pipelineShaderStageCreateInfoList[0].pSpecializationInfo = nullptr;
    pipelineShaderStageCreateInfoList[1] = {};//ray generation
    pipelineShaderStageCreateInfoList[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfoList[1].pNext = nullptr;
    pipelineShaderStageCreateInfoList[1].flags = 0;
    pipelineShaderStageCreateInfoList[1].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    pipelineShaderStageCreateInfoList[1].module = get_vulkan_context()->getShader(ShaderType::RGen, shaderDefines)->getShaderModule();
    pipelineShaderStageCreateInfoList[1].pName = "main";
    pipelineShaderStageCreateInfoList[1].pSpecializationInfo = nullptr;
    pipelineShaderStageCreateInfoList[2] = {};//miss
    pipelineShaderStageCreateInfoList[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfoList[2].pNext = nullptr;
    pipelineShaderStageCreateInfoList[2].flags = 0;
    pipelineShaderStageCreateInfoList[2].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    pipelineShaderStageCreateInfoList[2].module = get_vulkan_context()->getShader(ShaderType::Miss, shaderDefines)->getShaderModule();
    pipelineShaderStageCreateInfoList[2].pName = "main";
    pipelineShaderStageCreateInfoList[2].pSpecializationInfo = nullptr;
    
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> rayTracingShaderGroupCreateInfoList(3);
    rayTracingShaderGroupCreateInfoList[0] = {};//closest hit
    rayTracingShaderGroupCreateInfoList[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    rayTracingShaderGroupCreateInfoList[0].pNext = nullptr;
    rayTracingShaderGroupCreateInfoList[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    rayTracingShaderGroupCreateInfoList[0].generalShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[0].closestHitShader = 0;
    rayTracingShaderGroupCreateInfoList[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[0].intersectionShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[0].pShaderGroupCaptureReplayHandle = nullptr;
    rayTracingShaderGroupCreateInfoList[1] = {};//ray generation
    rayTracingShaderGroupCreateInfoList[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    rayTracingShaderGroupCreateInfoList[1].pNext = nullptr,
    rayTracingShaderGroupCreateInfoList[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    rayTracingShaderGroupCreateInfoList[1].generalShader = 1;
    rayTracingShaderGroupCreateInfoList[1].closestHitShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[1].anyHitShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[1].intersectionShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[1].pShaderGroupCaptureReplayHandle = nullptr;
    rayTracingShaderGroupCreateInfoList[2] = {};//miss
    rayTracingShaderGroupCreateInfoList[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    rayTracingShaderGroupCreateInfoList[2].pNext = nullptr;
    rayTracingShaderGroupCreateInfoList[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    rayTracingShaderGroupCreateInfoList[2].generalShader = 2;
    rayTracingShaderGroupCreateInfoList[2].closestHitShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[2].anyHitShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[2].intersectionShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[2].pShaderGroupCaptureReplayHandle = nullptr;

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
