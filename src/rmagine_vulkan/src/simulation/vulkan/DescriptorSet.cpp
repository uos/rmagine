#include "rmagine/simulation/vulkan/DescriptorSet.hpp"
#include "rmagine/map/vulkan/accelerationStructure/AccelerationStructure.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

DescriptorSet::DescriptorSet(VulkanContextPtr vulkan_context) : vulkan_context(vulkan_context)
{
    allocateDescriptorSet();
}

DescriptorSet::~DescriptorSet()
{
    
}


void DescriptorSet::allocateDescriptorSet()
{
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = vulkan_context->getDescriptorSetLayout()->getDescriptorPool();
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = vulkan_context->getDescriptorSetLayout()->getDescriptorSetLayoutPtr();

    std::vector<VkDescriptorSet> descriptorSets = std::vector<VkDescriptorSet>(1, VK_NULL_HANDLE);
    if(vkAllocateDescriptorSets(vulkan_context->getDevice()->getLogicalDevice(), &descriptorSetAllocateInfo, descriptorSets.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("[DescriptorSet::allocateDescriptorSet()] ERROR - failed to allocate descriptor sets (you may have created more DescriptorSets/Simulators than currently possible)!");
    }
    descriptorSet = descriptorSets.front();
}


void DescriptorSet::updateDescriptorSet(AccelerationStructurePtr accelerationStructure, BufferPtr mapDataBuffer, 
                                        BufferPtr sensorBuffer, BufferPtr resultsBuffer, 
                                        BufferPtr tsbBuffer, BufferPtr origsDirsAndTransformsBuffer)
{
    VkWriteDescriptorSetAccelerationStructureKHR accelerationStructureDescriptorInfo{};
    accelerationStructureDescriptorInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    accelerationStructureDescriptorInfo.accelerationStructureCount = 1;
    accelerationStructureDescriptorInfo.pAccelerationStructures = accelerationStructure->getAcceleratiionStructurePtr();
  
    VkDescriptorBufferInfo mapDataDescriptorInfo{};
    mapDataDescriptorInfo.buffer = mapDataBuffer->getBuffer();
    mapDataDescriptorInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo sensorDescriptorInfo{};
    sensorDescriptorInfo.buffer = sensorBuffer->getBuffer();
    sensorDescriptorInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo resultsDescriptorInfo{};
    resultsDescriptorInfo.buffer = resultsBuffer->getBuffer();
    resultsDescriptorInfo.range = VK_WHOLE_SIZE;
    
    VkDescriptorBufferInfo tsbDescriptorInfo{};
    tsbDescriptorInfo.buffer = tsbBuffer->getBuffer();
    tsbDescriptorInfo.range = VK_WHOLE_SIZE;
    
    VkDescriptorBufferInfo origsDirsAndTransformsDescriptorInfo{};
    origsDirsAndTransformsDescriptorInfo.buffer = origsDirsAndTransformsBuffer->getBuffer();
    origsDirsAndTransformsDescriptorInfo.range = VK_WHOLE_SIZE;

    std::vector<VkWriteDescriptorSet> writeDescriptorSetList = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,//acceleration structure
         .pNext = &accelerationStructureDescriptorInfo,
         .dstSet = descriptorSet,
         .dstBinding = 0,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
         .pImageInfo = nullptr,
         .pBufferInfo = nullptr,
         .pTexelBufferView = nullptr},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,//mapData buffer
         .pNext = nullptr,
         .dstSet = descriptorSet,
         .dstBinding = 1,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pImageInfo = nullptr,
         .pBufferInfo = &mapDataDescriptorInfo,
         .pTexelBufferView = nullptr},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,//sensor buffer
         .pNext = nullptr,
         .dstSet = descriptorSet,
         .dstBinding = 2,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,//TODO: for some reason cant be made into a uniform buffer, i dont know why... (maybe becaue it does not always hold the same type of data?)
         .pImageInfo = nullptr,
         .pBufferInfo = &sensorDescriptorInfo,
         .pTexelBufferView = nullptr},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,//results buffer
         .pNext = nullptr,
         .dstSet = descriptorSet,
         .dstBinding = 3,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .pImageInfo = nullptr,
         .pBufferInfo = &resultsDescriptorInfo,
         .pTexelBufferView = nullptr},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,//tbs buffer
         .pNext = nullptr,
         .dstSet = descriptorSet,
         .dstBinding = 4,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .pImageInfo = nullptr,
         .pBufferInfo = &tsbDescriptorInfo,
         .pTexelBufferView = nullptr},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,//origsDirsAndTransforms buffer
         .pNext = nullptr,
         .dstSet = descriptorSet,
         .dstBinding = 5,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .pImageInfo = nullptr,
         .pBufferInfo = &origsDirsAndTransformsDescriptorInfo,
         .pTexelBufferView = nullptr}};

    vkUpdateDescriptorSets(vulkan_context->getDevice()->getLogicalDevice(), writeDescriptorSetList.size(), writeDescriptorSetList.data(), 0, nullptr);
}


VkDescriptorSet *DescriptorSet::getDescriptorSetPtr()
{
    return &descriptorSet;
}

} // namespace rmagine
