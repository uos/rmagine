#include "rmagine/simulation/vulkan/DescriptorSet.hpp"
#include "rmagine/map/vulkan/accelerationStructure/TopLevelAccelerationStructure.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

DescriptorSet::DescriptorSet() : device(get_vulkan_context()->getDevice()), descriptorSetLayout(get_vulkan_context()->getDescriptorSetLayout())
{
    allocateDescriptorSet();
}

DescriptorSet::DescriptorSet(DevicePtr device, DescriptorSetLayoutPtr descriptorSetLayout) : device(device), descriptorSetLayout(descriptorSetLayout)
{
    allocateDescriptorSet();
}

void DescriptorSet::allocateDescriptorSet()
{
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = descriptorSetLayout->getDescriptorPool();
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = descriptorSetLayout->getDescriptorSetLayoutPtr();

    std::vector<VkDescriptorSet> descriptorSets = std::vector<VkDescriptorSet>(1, VK_NULL_HANDLE);
    if(vkAllocateDescriptorSets(device->getLogicalDevice(), &descriptorSetAllocateInfo, descriptorSets.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate descriptor sets (you may have created more DescriptorSets/Simulators than currently possible)!");
    }
    descriptorSet = descriptorSets.front();
}


void DescriptorSet::updateDescriptorSet(BufferPtr vertexBuffer, BufferPtr indexBuffer, 
                                        BufferPtr sensorBuffer, BufferPtr resultsBuffer, 
                                        BufferPtr tsbBuffer, BufferPtr tbmBuffer, 
                                        TopLevelAccelerationStructurePtr topLevelAccelerationStructure)
{
    VkWriteDescriptorSetAccelerationStructureKHR accelerationStructureDescriptorInfo{};
    accelerationStructureDescriptorInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    accelerationStructureDescriptorInfo.accelerationStructureCount = 1;
    accelerationStructureDescriptorInfo.pAccelerationStructures = topLevelAccelerationStructure->getAcceleratiionStructurePtr();

    VkDescriptorBufferInfo sensorDescriptorInfo{};
    sensorDescriptorInfo.buffer = sensorBuffer->getBuffer();
    sensorDescriptorInfo.range = VK_WHOLE_SIZE;
  
    VkDescriptorBufferInfo indexDescriptorInfo{};
    indexDescriptorInfo.buffer = indexBuffer->getBuffer();
    indexDescriptorInfo.range = VK_WHOLE_SIZE;
  
    VkDescriptorBufferInfo vertexDescriptorInfo{};
    vertexDescriptorInfo.buffer = vertexBuffer->getBuffer();
    vertexDescriptorInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo resultsDescriptorInfo{};
    resultsDescriptorInfo.buffer = resultsBuffer->getBuffer();
    resultsDescriptorInfo.range = VK_WHOLE_SIZE;
    
    VkDescriptorBufferInfo tsbDescriptorInfo{};
    tsbDescriptorInfo.buffer = tsbBuffer->getBuffer();
    tsbDescriptorInfo.range = VK_WHOLE_SIZE;
    
    VkDescriptorBufferInfo tbmDescriptorInfo{};
    tbmDescriptorInfo.buffer = tbmBuffer->getBuffer();
    tbmDescriptorInfo.range = VK_WHOLE_SIZE;

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
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,//sensor uniform buffer
         .pNext = nullptr,
         .dstSet = descriptorSet,
         .dstBinding = 1,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pImageInfo = nullptr,
         .pBufferInfo = &sensorDescriptorInfo,
         .pTexelBufferView = nullptr},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,//index buffer
         .pNext = nullptr,
         .dstSet = descriptorSet,
         .dstBinding = 2,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pImageInfo = nullptr,
         .pBufferInfo = &indexDescriptorInfo,
         .pTexelBufferView = nullptr},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,//vertex buffer
         .pNext = nullptr,
         .dstSet = descriptorSet,
         .dstBinding = 3,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pImageInfo = nullptr,
         .pBufferInfo = &vertexDescriptorInfo,
         .pTexelBufferView = nullptr},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,//results buffer
         .pNext = nullptr,
         .dstSet = descriptorSet,
         .dstBinding = 4,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pImageInfo = nullptr,
         .pBufferInfo = &resultsDescriptorInfo,
         .pTexelBufferView = nullptr},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,//tbs buffer
         .pNext = nullptr,
         .dstSet = descriptorSet,
         .dstBinding = 5,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pImageInfo = nullptr,
         .pBufferInfo = &tsbDescriptorInfo,
         .pTexelBufferView = nullptr},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,//tbm buffer
         .pNext = nullptr,
         .dstSet = descriptorSet,
         .dstBinding = 6,
         .dstArrayElement = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pImageInfo = nullptr,
         .pBufferInfo = &tbmDescriptorInfo,
         .pTexelBufferView = nullptr}};

    vkUpdateDescriptorSets(device->getLogicalDevice(), writeDescriptorSetList.size(), writeDescriptorSetList.data(), 0, NULL);
}


void DescriptorSet::cleanup()
{
    //nothing?
}



VkDescriptorSet *DescriptorSet::getDescriptorSetPtr()
{
    return &descriptorSet;
}

} // namespace rmagine
