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
    VkDevice logicalDevice = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;

    uint32_t queueFamilyIndex = uint32_t(~0);

    uint32_t shaderGroupBaseAlignment = 0;
    uint32_t shaderGroupHandleSize = 0;
    
    uint32_t maxPrimitiveCount = 0;
    uint32_t maxGeometryCount = 0;
    uint32_t maxInstanceCount = 0;

public:
    Device();

    ~Device();

    Device(const Device&) = delete;


    VkDevice getLogicalDevice();

    VkPhysicalDevice getPhysicalDevice();

    VkQueue getQueue();

    uint32_t getQueueFamilyIndex();

    uint32_t* getQueueFamilyIndexPtr();

    VkDeviceSize getShaderGroupBaseAlignment();

    VkDeviceSize getShaderGroupHandleSize();

    uint32_t getMaxPrimitiveCount();

    uint32_t getMaxGeometryCount();

    uint32_t getMaxInstanceCount();

private:
    void createInstance();

    void choosePhysicalDevice();
    void printMemoryInfo(VkPhysicalDeviceMemoryProperties2 physicalDeviceMemoryProperties2);
    bool evaluatePhysicalDeviceType(VkPhysicalDeviceType currentPysicalDeviceType, VkPhysicalDeviceType newPysicalDeviceType);
    bool evaluatePhysicalDeviceFeatures(const VkPhysicalDevice &physicalDevice);

    void chooseQueueFamily();
    void createLogicalDevice();
};

using DevicePtr = std::shared_ptr<Device>;

} // namespace rmagine
