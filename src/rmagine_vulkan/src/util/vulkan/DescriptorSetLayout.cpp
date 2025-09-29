#include "rmagine/util/vulkan/DescriptorSetLayout.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

DescriptorSetLayout::DescriptorSetLayout(DeviceWPtr device) : device(device)
{
    createDescriptorPool();
    createDescriptorSetLayout();
}

DescriptorSetLayout::~DescriptorSetLayout()
{
    if(descriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device.lock()->getLogicalDevice(), descriptorSetLayout, nullptr);
    }
    if(descriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device.lock()->getLogicalDevice(), descriptorPool, nullptr);
    }
    device.reset();
}



void DescriptorSetLayout::createDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> descriptorPoolSizeList(3);
    descriptorPoolSizeList[0] = {};//accelaration structure
    descriptorPoolSizeList[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    descriptorPoolSizeList[0].descriptorCount = 1;
    descriptorPoolSizeList[1] = {};//mapData, sensor
    descriptorPoolSizeList[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorPoolSizeList[1].descriptorCount = 2;
    descriptorPoolSizeList[2] = {};//result, tsb & tbmAndSensorSpecific
    descriptorPoolSizeList[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    descriptorPoolSizeList[2].descriptorCount = 3;

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descriptorPoolCreateInfo.maxSets = MAX_NUM_OF_DESCRIPTOR_SETS;
    descriptorPoolCreateInfo.poolSizeCount = (uint32_t)descriptorPoolSizeList.size();
    descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSizeList.data();

    if(vkCreateDescriptorPool(device.lock()->getLogicalDevice(), &descriptorPoolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS)
    {
        throw std::runtime_error("[DescriptorSetLayout::createDescriptorPool()] ERROR - failed to create descriptor pool!");
    }
}


void DescriptorSetLayout::createDescriptorSetLayout()
{
    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindingList(6);
    descriptorSetLayoutBindingList[0] = {};//accelaration structure
    descriptorSetLayoutBindingList[0].binding = 0;
    descriptorSetLayoutBindingList[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    descriptorSetLayoutBindingList[0].descriptorCount = 1;
    descriptorSetLayoutBindingList[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    descriptorSetLayoutBindingList[0].pImmutableSamplers = nullptr;
    descriptorSetLayoutBindingList[1] = {};//mapData
    descriptorSetLayoutBindingList[1].binding = 1;
    descriptorSetLayoutBindingList[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBindingList[1].descriptorCount = 1;
    descriptorSetLayoutBindingList[1].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    descriptorSetLayoutBindingList[1].pImmutableSamplers = nullptr;
    descriptorSetLayoutBindingList[2] = {};//sensor
    descriptorSetLayoutBindingList[2].binding = 2;
    descriptorSetLayoutBindingList[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBindingList[2].descriptorCount = 1;
    descriptorSetLayoutBindingList[2].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    descriptorSetLayoutBindingList[2].pImmutableSamplers = nullptr;
    descriptorSetLayoutBindingList[3] = {};//results
    descriptorSetLayoutBindingList[3].binding = 3;
    descriptorSetLayoutBindingList[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorSetLayoutBindingList[3].descriptorCount = 1;
    descriptorSetLayoutBindingList[3].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
    descriptorSetLayoutBindingList[3].pImmutableSamplers = nullptr;
    descriptorSetLayoutBindingList[4] = {};//tsb
    descriptorSetLayoutBindingList[4].binding = 4;
    descriptorSetLayoutBindingList[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorSetLayoutBindingList[4].descriptorCount = 1;
    descriptorSetLayoutBindingList[4].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    descriptorSetLayoutBindingList[4].pImmutableSamplers = nullptr;
    descriptorSetLayoutBindingList[5] = {};//tbmAndSensorSpecific
    descriptorSetLayoutBindingList[5].binding = 5;
    descriptorSetLayoutBindingList[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorSetLayoutBindingList[5].descriptorCount = 1;
    descriptorSetLayoutBindingList[5].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    descriptorSetLayoutBindingList[5].pImmutableSamplers = nullptr;


    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = (uint32_t)descriptorSetLayoutBindingList.size();
    descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindingList.data();
    
    if(vkCreateDescriptorSetLayout(device.lock()->getLogicalDevice(), &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("[DescriptorSetLayout::createDescriptorSetLayout()] ERROR - failed to create descriptor set layout!");
    }
}


VkDescriptorPool DescriptorSetLayout::getDescriptorPool()
{
    return descriptorPool;
}


VkDescriptorSetLayout* DescriptorSetLayout::getDescriptorSetLayoutPtr()
{
    return &descriptorSetLayout;
}

} // namespace rmagine
