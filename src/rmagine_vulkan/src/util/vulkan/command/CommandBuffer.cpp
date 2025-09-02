#include "rmagine/util/vulkan/command/CommandBuffer.hpp"
#include "rmagine/util/VulkanContext.hpp"
#include "rmagine/util/vulkan/ShaderBindingTable.hpp"
#include "rmagine/simulation/vulkan/DescriptorSet.hpp"



namespace rmagine
{

CommandBuffer::CommandBuffer(VulkanContextPtr vulkan_context) : vulkan_context(vulkan_context)
{
    createCommandBuffer();
    fence = std::make_shared<Fence>(vulkan_context);
}

CommandBuffer::~CommandBuffer()
{
    std::cout << "Destroying CommandBuffer" << std::endl;
    if(fence != nullptr)
        fence.reset();
    std::cout << "CommandBuffer destroyed" << std::endl;
}



void CommandBuffer::createCommandBuffer()
{
    VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = vulkan_context->getCommandPool()->getCommandPool();
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    
    std::vector<VkCommandBuffer> commandBuffers = std::vector<VkCommandBuffer>(1, VK_NULL_HANDLE);
    if(vkAllocateCommandBuffers(vulkan_context->getDevice()->getLogicalDevice(), &commandBufferAllocateInfo, commandBuffers.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create command buffer(s)!");
    }
    commandBuffer = commandBuffers.front();
}


void CommandBuffer::recordRayTracingToCommandBuffer(DescriptorSetPtr descriptorSet, ShaderBindingTablePtr shaderBindingTable, uint32_t width, uint32_t height, uint32_t depth)
{
    VkCommandBufferBeginInfo rtxCommandBufferBeginInfo{};
    rtxCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    rtxCommandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;//so it can be used more than once (before: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)

    if(vkBeginCommandBuffer(commandBuffer, &rtxCommandBufferBeginInfo) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to begin recording commmands to the command buffer!");
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, shaderBindingTable->getPipeline()->getPipeline());

    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        vulkan_context->getPipelineLayout()->getPipelineLayout(),
        0,
        1,
        descriptorSet->getDescriptorSetPtr(),
        0,
        nullptr);
        
    vulkan_context->extensionFuncs.vkCmdTraceRaysKHR(
        commandBuffer,
        shaderBindingTable->getRayGenerationShaderBindingTablePtr(),
        shaderBindingTable->getMissShaderBindingTablePtr(),
        shaderBindingTable->getClosestHitShaderBindingTablePtr(),
        shaderBindingTable->getCallableShaderBindingTablePtr(),
        width,
        height,
        depth);

    if(vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to stop recording commmands to the command buffer!");
    }
}


void CommandBuffer::recordBuildingASToCommandBuffer(VkAccelerationStructureBuildGeometryInfoKHR& accelerationStructureBuildGeometryInfo, const VkAccelerationStructureBuildRangeInfoKHR* accelerationStructureBuildRangeInfos)
{
    VkCommandBufferBeginInfo commandBufferBeginInfo{};
    commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to begin recording commmands to the command buffer!");
    }
    
    vulkan_context->extensionFuncs.vkCmdBuildAccelerationStructuresKHR(
        commandBuffer,
        1,
        &accelerationStructureBuildGeometryInfo,
        &accelerationStructureBuildRangeInfos);

    if(vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to stop recording commmands to the command buffer!");
    }
}


void CommandBuffer::recordCopyBufferToCommandBuffer(BufferPtr scrBuffer, BufferPtr dstBuffer)
{
    VkCommandBufferBeginInfo commandBufferBeginInfo{};
    commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to begin recording commmands to the command buffer!");
    }
    
    VkBufferCopy copyRegion{};
    copyRegion.size = std::min(scrBuffer->getBufferSize(), dstBuffer->getBufferSize());
    vkCmdCopyBuffer(commandBuffer, scrBuffer->getBuffer(), dstBuffer->getBuffer(), 1, &copyRegion);

    if(vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to stop recording commmands to the command buffer!");
    }
}


void CommandBuffer::submitRecordedCommandAndWait()
{
    VkSubmitInfo accelerationStructureBuildSubmitInfo{};
    accelerationStructureBuildSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    accelerationStructureBuildSubmitInfo.commandBufferCount = 1;
    accelerationStructureBuildSubmitInfo.pCommandBuffers = &commandBuffer;

    fence->submitWithFence(accelerationStructureBuildSubmitInfo);
    fence->waitForFence();
}


VkCommandBuffer CommandBuffer::getCommandbuffer()
{
    return commandBuffer;
}

} // namespace rmagine
