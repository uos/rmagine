#include "CommandBuffer.hpp"
#include "../VulkanContext.hpp"
#include "../../simulators/simulatorComponents/DescriptorSet.hpp"
#include "../contextComponents/ShaderBindingTable.hpp"
#include "../contextComponents/Pipeline.hpp"



namespace rmagine
{

CommandBuffer::CommandBuffer() : device(get_vulkan_context()->getDevice()), extensionFunctionsPtr(get_vulkan_context()->getExtensionFunctionsPtr()), commandPool(get_vulkan_context()->getCommandPool()), pipelineLayout(get_vulkan_context()->getPipelineLayout())
{
    createCommandBuffer();
    fence = std::make_shared<Fence>();
}

CommandBuffer::CommandBuffer(DevicePtr device, ExtensionFunctionsPtr extensionFunctionsPtr, CommandPoolPtr commandPool, PipelineLayoutPtr pipelineLayout) : device(device), extensionFunctionsPtr(extensionFunctionsPtr), commandPool(commandPool), pipelineLayout(pipelineLayout)
{
    createCommandBuffer();
    fence = std::make_shared<Fence>(device);
}



void CommandBuffer::createCommandBuffer()
{
    VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool->getCommandPool();
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    
    std::vector<VkCommandBuffer> commandBuffers = std::vector<VkCommandBuffer>(1, VK_NULL_HANDLE);
    if(vkAllocateCommandBuffers(device->getLogicalDevice(), &commandBufferAllocateInfo, commandBuffers.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create command buffer(s)!");
    }
    commandBuffer = commandBuffers.front();
}


void CommandBuffer::recordRayTracingToCommandBuffer(DescriptorSetPtr descriptorSet, PipelinePtr pipeline, uint32_t width, uint32_t height, uint32_t depth)
{
    VkCommandBufferBeginInfo rtxCommandBufferBeginInfo{};
    rtxCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    rtxCommandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;//so it can be used more than once (before: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)

    if(vkBeginCommandBuffer(commandBuffer, &rtxCommandBufferBeginInfo) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to begin recording commmands to the command buffer!");
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline->getPipeline());

    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        pipelineLayout->getPipelineLayout(),
        0,
        1,
        descriptorSet->getDescriptorSetPtr(),
        0,
        nullptr);
        
    extensionFunctionsPtr->pvkCmdTraceRaysKHR(
        commandBuffer,
        pipeline->getShaderBindingTable()->getRayGenerationShaderBindingTablePtr(),
        pipeline->getShaderBindingTable()->getMissShaderBindingTablePtr(),
        pipeline->getShaderBindingTable()->getClosestHitShaderBindingTablePtr(),
        pipeline->getShaderBindingTable()->getCallableShaderBindingTablePtr(),
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
    
    extensionFunctionsPtr->pvkCmdBuildAccelerationStructuresKHR(
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



void CommandBuffer::cleanup()
{
    if(fence != nullptr)
        fence->cleanup();
}



VkCommandBuffer CommandBuffer::getCommandbuffer()
{
    return commandBuffer;
}

VkCommandBuffer* CommandBuffer::getCommandbufferPtr()
{
    return &commandBuffer;
}

} // namespace rmagine
