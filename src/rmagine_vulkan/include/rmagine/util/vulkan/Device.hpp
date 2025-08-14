#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>



namespace rmagine
{

class Device
{
private:
    //validation layers for debugging
    const std::vector<const char*> validationLayers{"VK_LAYER_KHRONOS_validation"/*, "VK_LAYER_PRINTF_ENABLE", "VK_LAYER_PRINTF_TO_STDOUT", "VK_LAYER_PRINTF_VERBOSE"*/};

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR physicalDeviceRayTracingPipelineProperties{};
    uint32_t queueFamilyIndex = uint32_t(~0);
    VkDevice logicalDevice = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;

public:
    Device()
    {
        createInstance();
        choosePhysicalDevice();
        createLogicalDevice();
    }

    ~Device() {}

    Device(const Device&) = delete;


    void cleanup();
    
    VkDevice getLogicalDevice();

    VkPhysicalDevice getPhysicalDevice();

    VkQueue getQueue();

    uint32_t getQueueFamilyIndex();

    uint32_t* getQueueFamilyIndexPtr();

    VkDeviceSize getShaderGroupBaseAlignment();

    VkDeviceSize getShaderGroupHandleSize();

private:
    
    void createInstance(std::string appName = "VulkanApp");

    void choosePhysicalDevice();

    void createLogicalDevice();
};

using DevicePtr = std::shared_ptr<Device>;

} // namespace rmagine
