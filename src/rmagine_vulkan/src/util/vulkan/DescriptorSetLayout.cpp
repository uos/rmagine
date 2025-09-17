#include "rmagine/util/vulkan/DescriptorSetLayout.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

DescriptorSetLayout::DescriptorSetLayout(DevicePtr device) : device(device)
{
    createDescriptorPool();
    createDescriptorSetLayout();
}

DescriptorSetLayout::~DescriptorSetLayout()
{
    std::cout << "Destroying DescriptorSetLayout" << std::endl;
    if(descriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device->getLogicalDevice(), descriptorSetLayout, nullptr);
    }
    if(descriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device->getLogicalDevice(), descriptorPool, nullptr);
    }
    device.reset();
    std::cout << "DescriptorSetLayout destroyed" << std::endl;
}



void DescriptorSetLayout::createDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> descriptorPoolSizeList = {
        {.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, .descriptorCount = 1},//accelaration structure
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2},//MapData, Sensor
        {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 3}};//Result, tsb & origsDirsAndTransforms

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descriptorPoolCreateInfo.maxSets = MAX_NUM_OF_DESCRIPTOR_SETS;
    descriptorPoolCreateInfo.poolSizeCount = (uint32_t)descriptorPoolSizeList.size();
    descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSizeList.data();

    if(vkCreateDescriptorPool(device->getLogicalDevice(), &descriptorPoolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS)
    {
        throw std::runtime_error("[DescriptorSetLayout::createDescriptorPool()] ERROR - failed to create descriptor pool!");
    }
}


void DescriptorSetLayout::createDescriptorSetLayout()
{
    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindingList = {
        {.binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
         .pImmutableSamplers = nullptr},//accelaration structure
        {.binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
         .pImmutableSamplers = nullptr},//mapData
        {.binding = 2,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
         .pImmutableSamplers = nullptr},//sensor
        {.binding = 3,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
         .pImmutableSamplers = nullptr},//results
        {.binding = 4,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
         .pImmutableSamplers = nullptr},//tsb
        {.binding = 5,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
         .pImmutableSamplers = nullptr}};//tbm

    
    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = (uint32_t)descriptorSetLayoutBindingList.size();
    descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindingList.data();
    
    if(vkCreateDescriptorSetLayout(device->getLogicalDevice(), &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
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
