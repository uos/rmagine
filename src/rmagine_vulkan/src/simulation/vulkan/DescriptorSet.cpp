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

    std::vector<VkWriteDescriptorSet> writeDescriptorSetList(6);
    writeDescriptorSetList[0] = {};//acceleration structure
    writeDescriptorSetList[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetList[0].pNext = &accelerationStructureDescriptorInfo;
    writeDescriptorSetList[0].dstSet = descriptorSet;
    writeDescriptorSetList[0].dstBinding = 0;
    writeDescriptorSetList[0].dstArrayElement = 0;
    writeDescriptorSetList[0].descriptorCount = 1;
    writeDescriptorSetList[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    writeDescriptorSetList[0].pImageInfo = nullptr;
    writeDescriptorSetList[0].pBufferInfo = nullptr;
    writeDescriptorSetList[0].pTexelBufferView = nullptr;
    writeDescriptorSetList[1] = {};//mapData
    writeDescriptorSetList[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetList[1].pNext = nullptr;
    writeDescriptorSetList[1].dstSet = descriptorSet;
    writeDescriptorSetList[1].dstBinding = 1;
    writeDescriptorSetList[1].dstArrayElement = 0;
    writeDescriptorSetList[1].descriptorCount = 1;
    writeDescriptorSetList[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSetList[1].pImageInfo = nullptr;
    writeDescriptorSetList[1].pBufferInfo = &mapDataDescriptorInfo;
    writeDescriptorSetList[1].pTexelBufferView = nullptr;
    writeDescriptorSetList[2] = {};//sensor
    writeDescriptorSetList[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetList[2].pNext = nullptr;
    writeDescriptorSetList[2].dstSet = descriptorSet;
    writeDescriptorSetList[2].dstBinding = 2;
    writeDescriptorSetList[2].dstArrayElement = 0;
    writeDescriptorSetList[2].descriptorCount = 1;
    writeDescriptorSetList[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;//TODO: for some reason cant be made into a uniform buffer, i dont know why... (maybe becaue it does not always hold the same type of data?)
    writeDescriptorSetList[2].pImageInfo = nullptr;
    writeDescriptorSetList[2].pBufferInfo = &sensorDescriptorInfo;
    writeDescriptorSetList[2].pTexelBufferView = nullptr;
    writeDescriptorSetList[3] = {};//results
    writeDescriptorSetList[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetList[3].pNext = nullptr;
    writeDescriptorSetList[3].dstSet = descriptorSet;
    writeDescriptorSetList[3].dstBinding = 3;
    writeDescriptorSetList[3].dstArrayElement = 0;
    writeDescriptorSetList[3].descriptorCount = 1;
    writeDescriptorSetList[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writeDescriptorSetList[3].pImageInfo = nullptr;
    writeDescriptorSetList[3].pBufferInfo = &resultsDescriptorInfo;
    writeDescriptorSetList[3].pTexelBufferView = nullptr;
    writeDescriptorSetList[4] = {};//tbs
    writeDescriptorSetList[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetList[4].pNext = nullptr;
    writeDescriptorSetList[4].dstSet = descriptorSet;
    writeDescriptorSetList[4].dstBinding = 4;
    writeDescriptorSetList[4].dstArrayElement = 0;
    writeDescriptorSetList[4].descriptorCount = 1;
    writeDescriptorSetList[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writeDescriptorSetList[4].pImageInfo = nullptr;
    writeDescriptorSetList[4].pBufferInfo = &tsbDescriptorInfo;
    writeDescriptorSetList[4].pTexelBufferView = nullptr;
    writeDescriptorSetList[5] = {};//tbmAndSensorSpecific
    writeDescriptorSetList[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetList[5].pNext = nullptr;
    writeDescriptorSetList[5].dstSet = descriptorSet;
    writeDescriptorSetList[5].dstBinding = 5;
    writeDescriptorSetList[5].dstArrayElement = 0;
    writeDescriptorSetList[5].descriptorCount = 1;
    writeDescriptorSetList[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writeDescriptorSetList[5].pImageInfo = nullptr;
    writeDescriptorSetList[5].pBufferInfo = &origsDirsAndTransformsDescriptorInfo;
    writeDescriptorSetList[5].pTexelBufferView = nullptr;

    vkUpdateDescriptorSets(vulkan_context->getDevice()->getLogicalDevice(), writeDescriptorSetList.size(), writeDescriptorSetList.data(), 0, nullptr);
}


VkDescriptorSet *DescriptorSet::getDescriptorSetPtr()
{
    return &descriptorSet;
}

} // namespace rmagine
