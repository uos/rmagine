#include "rmagine/util/vulkan/Device.hpp"



namespace rmagine
{

Device::Device()
{
    createInstance();
    choosePhysicalDevice();
    chooseQueueFamily();
    createLogicalDevice();
}

Device::~Device()
{
    std::cout << "Destroying Device" << std::endl;
    if(logicalDevice != VK_NULL_HANDLE)
    {
        vkDestroyDevice(logicalDevice, nullptr);
    }
    if(instance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(instance, nullptr);
    }
    std::cout << "Device destroyed" << std::endl;
}



void Device::createInstance()
{
    VkApplicationInfo ApplicationInfo{};
    ApplicationInfo.pApplicationName = "rmagine-vulkan";
    ApplicationInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instanceCreateInfo{};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pApplicationInfo = &ApplicationInfo;

    // enalbe validation layers
    #ifdef VDEBUG
        std::cout << "Mode: Debug" << std::endl;
        instanceCreateInfo.enabledLayerCount = (uint32_t)validationLayers.size();
        instanceCreateInfo.ppEnabledLayerNames = validationLayers.data();
    #else
        std::cout << "Mode: Release" << std::endl;
        instanceCreateInfo.enabledLayerCount = 0;
    #endif

    if (vkCreateInstance(&instanceCreateInfo, nullptr, &instance) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create instance!");
    }
}


void Device::choosePhysicalDevice()
{
    // get list physical devices
    uint32_t physicalDeviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);
    std::cout << "Number of compatible pysical devices found: "<< physicalDeviceCount << std::endl;
    if(physicalDeviceCount == 0)
    {
        throw std::runtime_error("Failed to find any compatible pysical devices!");
    }
    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data());


    // iterate over all available physical devices and choose one
    // potential issue: if there are multiple physical device on the same level, the one that appears later in the list will get chosen
    // you could maybe introduce a tiebraker like amount of vram for example for those cases...
    std::cout << "Pysical devices:" << std::endl;
    VkPhysicalDeviceType pysicalDeviceType = VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_OTHER;
    for(const auto& dev : physicalDevices)
    {
        VkPhysicalDeviceProperties2 physicalDeviceProperties2{};
        physicalDeviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        vkGetPhysicalDeviceProperties2(dev, &physicalDeviceProperties2);

        // VkPhysicalDeviceMemoryProperties2 physicalDeviceMemoryProperties2{};
        // physicalDeviceMemoryProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        // vkGetPhysicalDeviceMemoryProperties2(dev, &physicalDeviceMemoryProperties2);

        // list all the physical devices found 
        switch(physicalDeviceProperties2.properties.deviceType)
        {
        case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            std::cout << " - Discrete GPU:    ";
            break;
        case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            std::cout << " - Integranted GPU: ";
            break;
        case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            std::cout << " - Virtual GPU:     ";
            break;
        case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_CPU:
            std::cout << " - CPU:             ";
            break;
        case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_OTHER:
            std::cout << " - Unknown:         ";
            break;
        default:
            break;
        }
        std::cout << physicalDeviceProperties2.properties.deviceName << std::endl;

        if(evaluatePhysicalDeviceType(pysicalDeviceType, physicalDeviceProperties2.properties.deviceType) && evaluatePhysicalDeviceFeatures(dev))
        {
            physicalDevice = dev;
            pysicalDeviceType = physicalDeviceProperties2.properties.deviceType;
        }
    }
    if(physicalDevice == VK_NULL_HANDLE)
    {
        throw std::runtime_error("Unable to find compatible physical device!");
    }


    // get physical device properties
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR physicalDeviceRayTracingPipelineProperties{};
    physicalDeviceRayTracingPipelineProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    physicalDeviceRayTracingPipelineProperties.pNext = nullptr;

    VkPhysicalDeviceAccelerationStructurePropertiesKHR physicalDeviceAccelerationStructureProperties{};
    physicalDeviceAccelerationStructureProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
    physicalDeviceAccelerationStructureProperties.pNext = &physicalDeviceRayTracingPipelineProperties;

    VkPhysicalDeviceProperties2 physicalDeviceProperties2{};
    physicalDeviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    physicalDeviceProperties2.pNext = &physicalDeviceAccelerationStructureProperties;
    vkGetPhysicalDeviceProperties2(physicalDevice, &physicalDeviceProperties2);

    shaderGroupBaseAlignment = physicalDeviceRayTracingPipelineProperties.shaderGroupBaseAlignment;
    shaderGroupHandleSize = physicalDeviceRayTracingPipelineProperties.shaderGroupHandleSize;
    
    maxPrimitiveCount = physicalDeviceAccelerationStructureProperties.maxPrimitiveCount;
    maxGeometryCount = physicalDeviceAccelerationStructureProperties.maxGeometryCount;
    maxInstanceCount = physicalDeviceAccelerationStructureProperties.maxInstanceCount;

    //print physical device properties on chosen physical device
    std::cout << "Chosen physical device: " << physicalDeviceProperties2.properties.deviceName << std::endl;
    const uint32_t apiVesion = physicalDeviceProperties2.properties.apiVersion;
    std::cout << "Vulkan version: " << VK_VERSION_MAJOR(apiVesion) << "." << VK_VERSION_MINOR(apiVesion) << "." << VK_VERSION_PATCH(apiVesion) << std::endl;
}


bool Device::evaluatePhysicalDeviceType(VkPhysicalDeviceType currentPysicalDeviceType, VkPhysicalDeviceType newPysicalDeviceType)
{
    if(newPysicalDeviceType == VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        return true;

    if(newPysicalDeviceType == VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU &&
       currentPysicalDeviceType != VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        return true;

    if(newPysicalDeviceType == VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU &&
       currentPysicalDeviceType != VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
       currentPysicalDeviceType != VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
        return true;

    if(newPysicalDeviceType == VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_CPU &&
       currentPysicalDeviceType != VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
       currentPysicalDeviceType != VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU &&
       currentPysicalDeviceType != VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)
        return true;

    if(newPysicalDeviceType == VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_OTHER &&
       currentPysicalDeviceType != VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
       currentPysicalDeviceType != VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU &&
       currentPysicalDeviceType != VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU &&
       currentPysicalDeviceType != VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_CPU)
        return true;

    return false;
}


bool Device::evaluatePhysicalDeviceFeatures(const VkPhysicalDevice &physicalDevice)
{
    //check physical device features
    VkPhysicalDeviceBufferDeviceAddressFeatures physicalDeviceBufferDeviceAddressFeatures{};
    physicalDeviceBufferDeviceAddressFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    physicalDeviceBufferDeviceAddressFeatures.pNext = nullptr;

    VkPhysicalDeviceShaderFloat16Int8Features physicalDeviceShaderFloat16Int8Features{};
    physicalDeviceShaderFloat16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    physicalDeviceShaderFloat16Int8Features.pNext = &physicalDeviceBufferDeviceAddressFeatures;

    VkPhysicalDevice8BitStorageFeatures physicalDevice8BitStorageFeatures{};
    physicalDevice8BitStorageFeatures.sType =  VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    physicalDevice8BitStorageFeatures.pNext = &physicalDeviceShaderFloat16Int8Features;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR physicalDeviceAccelerationStructureFeatures{};
    physicalDeviceAccelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
    physicalDeviceAccelerationStructureFeatures.pNext = &physicalDevice8BitStorageFeatures;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR physicalDeviceRayTracingPipelineFeatures{};
    physicalDeviceRayTracingPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    physicalDeviceRayTracingPipelineFeatures.pNext = &physicalDeviceAccelerationStructureFeatures;

    VkPhysicalDeviceFeatures2 physicalDeviceFeatures2{};
    physicalDeviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    physicalDeviceFeatures2.pNext = &physicalDeviceRayTracingPipelineFeatures;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &physicalDeviceFeatures2);

    //all of these physical device features are needed for this program
    if(physicalDeviceBufferDeviceAddressFeatures.bufferDeviceAddress     == VK_TRUE &&
       physicalDeviceShaderFloat16Int8Features.shaderInt8                == VK_TRUE &&
       physicalDevice8BitStorageFeatures.storageBuffer8BitAccess         == VK_TRUE &&
       physicalDeviceAccelerationStructureFeatures.accelerationStructure == VK_TRUE &&
       physicalDeviceRayTracingPipelineFeatures.rayTracingPipeline       == VK_TRUE &&
       physicalDeviceFeatures2.features.shaderInt64                      == VK_TRUE)
    {
        return true;
    }

    return false;
}


void Device::chooseQueueFamily()
{
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());

    for(size_t i = 0; i < queueFamilyProperties.size(); i++)
    {
        if((queueFamilyProperties[i].queueFlags & VkQueueFlagBits::VK_QUEUE_COMPUTE_BIT) && (queueFamilyProperties[i].queueFlags & VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT))
        {
            queueFamilyIndex = i;
            break;
        }
    }
    if(queueFamilyIndex == uint32_t(~0))
    {
        throw std::runtime_error("Could not find compatible queue family!");
    }
    std::cout << "Queue family index: " << queueFamilyIndex << "    (Out of " << queueFamilyProperties.size() << " queue families)"<< std::endl;
}


void Device::createLogicalDevice()
{
    //physical device features
    VkPhysicalDeviceBufferDeviceAddressFeatures physicalDeviceBufferDeviceAddressFeatures{};
    physicalDeviceBufferDeviceAddressFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    physicalDeviceBufferDeviceAddressFeatures.pNext = nullptr;
    physicalDeviceBufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceShaderFloat16Int8Features physicalDeviceShaderFloat16Int8Features{};
    physicalDeviceShaderFloat16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    physicalDeviceShaderFloat16Int8Features.pNext = &physicalDeviceBufferDeviceAddressFeatures;
    physicalDeviceShaderFloat16Int8Features.shaderInt8 = VK_TRUE;

    VkPhysicalDevice8BitStorageFeatures physicalDevice8BitStorageFeatures{};
    physicalDevice8BitStorageFeatures.sType =  VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    physicalDevice8BitStorageFeatures.pNext = &physicalDeviceShaderFloat16Int8Features;
    physicalDevice8BitStorageFeatures.storageBuffer8BitAccess = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR physicalDeviceAccelerationStructureFeatures{};
    physicalDeviceAccelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
    physicalDeviceAccelerationStructureFeatures.pNext = &physicalDevice8BitStorageFeatures;
    physicalDeviceAccelerationStructureFeatures.accelerationStructure = VK_TRUE;
    // physicalDeviceAccelerationStructureFeatures.descriptorBindingAccelerationStructureUpdateAfterBind = VK_TRUE; // TODO: might need this

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR physicalDeviceRayTracingPipelineFeatures{};
    physicalDeviceRayTracingPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    physicalDeviceRayTracingPipelineFeatures.pNext = &physicalDeviceAccelerationStructureFeatures;
    physicalDeviceRayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;

    VkPhysicalDeviceFeatures physicalDeviceFeatures{};
    physicalDeviceFeatures.shaderInt64 = VK_TRUE;


    float queuePriority = 1.0f;

    VkDeviceQueueCreateInfo deviceQueueCreateInfo{};
    deviceQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    deviceQueueCreateInfo.queueFamilyIndex = queueFamilyIndex;
    deviceQueueCreateInfo.queueCount = 1;
    deviceQueueCreateInfo.pQueuePriorities = &queuePriority;


    std::vector<const char *> deviceExtensionList = {
        "VK_KHR_ray_tracing_pipeline",
        "VK_KHR_acceleration_structure",
        "VK_EXT_descriptor_indexing",
        "VK_KHR_maintenance3",
        "VK_KHR_buffer_device_address",
        "VK_KHR_deferred_host_operations"};


    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pNext = &physicalDeviceRayTracingPipelineFeatures;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
    deviceCreateInfo.pEnabledFeatures = &physicalDeviceFeatures;
    deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensionList.size();
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensionList.data();

    // enalbe validation layers
    #ifdef VDEBUG
        deviceCreateInfo.enabledLayerCount = (uint32_t)validationLayers.size();
        deviceCreateInfo.ppEnabledLayerNames = validationLayers.data();
    #else
        deviceCreateInfo.enabledLayerCount = 0;
    #endif


    if(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &logicalDevice) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(logicalDevice, queueFamilyIndex, 0, &queue);
}


VkDevice Device::getLogicalDevice()
{
    return logicalDevice;
}

VkPhysicalDevice Device::getPhysicalDevice()
{
    return physicalDevice;
}

VkQueue Device::getQueue()
{
    return queue;
}

uint32_t Device::getQueueFamilyIndex()
{
    return queueFamilyIndex;
}

uint32_t* Device::getQueueFamilyIndexPtr()
{
    return &queueFamilyIndex;
}

VkDeviceSize Device::getShaderGroupBaseAlignment()
{
    return shaderGroupBaseAlignment;
}

VkDeviceSize Device::getShaderGroupHandleSize()
{
    return shaderGroupHandleSize;
}

uint32_t Device::getMaxPrimitiveCount()
{
    return maxPrimitiveCount;
}

uint32_t Device::getMaxGeometryCount()
{
    return maxGeometryCount;
}

uint32_t Device::getMaxInstanceCount()
{
    return maxInstanceCount;
}

} // namespace rmagine
