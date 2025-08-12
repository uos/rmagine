#include "Device.hpp"



namespace rmagine
{

void Device::createInstance(std::string appName)
{
    VkApplicationInfo AppInfo{};
    AppInfo.pApplicationName = appName.c_str();
    AppInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instanceCreateInfo{};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pApplicationInfo = &AppInfo;

    //enalbe validation layers & and create debug messenger create info
    #ifdef VDEBUG
        std::cout << "Mode: Debug" << std::endl;
        instanceCreateInfo.enabledLayerCount = (uint32_t)validationLayers.size();
        instanceCreateInfo.ppEnabledLayerNames = validationLayers.data();
    #else
        std::cout << "Mode: Release" << std::endl;
        instanceCreateInfo.enabledLayerCount = 0;
    #endif

    //create instance
    if (vkCreateInstance(&instanceCreateInfo, nullptr, &instance) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create instance!");
    }
}


void Device::choosePhysicalDevice()
{
    //get list physical devices
    uint32_t physicalDeviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);

    std::cout << "Number of compatible pysical devices found: "<< physicalDeviceCount << std::endl;
    if(physicalDeviceCount == 0)
    {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data());


    //choose a physical device
    enum VkPhysicalDeviceType pysicalDeviceType = VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_OTHER;
    for(const auto& d : physicalDevices)
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(d, &props);

        std::cout << props.deviceName << std::endl;

        switch (props.deviceType)
        {
        case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            std::cout << " - Discrete GPU" << std::endl;
            physicalDevice = d;
            pysicalDeviceType = props.deviceType;
            break;
        
        case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            std::cout << " - Integranted GPU" << std::endl;
            if(pysicalDeviceType != VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            {
                physicalDevice = d;
                pysicalDeviceType = props.deviceType;
            }
            break;
        
        case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            std::cout << " - Virtual GPU" << std::endl;
            if(pysicalDeviceType != VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && pysicalDeviceType != VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
            {
                physicalDevice = d;
                pysicalDeviceType = props.deviceType;
            }
            break;
        
        case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_CPU:
            std::cout << " - CPU" << std::endl;
            if(pysicalDeviceType != VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && pysicalDeviceType != VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && pysicalDeviceType != VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)
            {
                physicalDevice = d;
                pysicalDeviceType = props.deviceType;
            }
            break;
        
        case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_OTHER:
            std::cout << " - Unknown" << std::endl;
            if(pysicalDeviceType == VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_OTHER)
            {
                physicalDevice = d;
                pysicalDeviceType = props.deviceType;
            }
            break;
        
        default:
            break;
        }
    }


    //get pys device props
    VkPhysicalDeviceProperties physicalDeviceProperties{};
    vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

    
    //for raytraycing
    physicalDeviceRayTracingPipelineProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    physicalDeviceRayTracingPipelineProperties.pNext = nullptr;

    VkPhysicalDeviceProperties2 physicalDeviceProperties2{};
    physicalDeviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    physicalDeviceProperties2.pNext = &physicalDeviceRayTracingPipelineProperties;
    physicalDeviceProperties2.properties = physicalDeviceProperties;
    vkGetPhysicalDeviceProperties2(physicalDevice, &physicalDeviceProperties2);

    
    //print info on chosen physicalDevice
    std::cout << "Chosen physical device: " << physicalDeviceProperties.deviceName << std::endl;
    const uint32_t apiVesion = physicalDeviceProperties.apiVersion;
    std::cout << "Vulkan version: " << VK_VERSION_MAJOR(apiVesion) << "." << VK_VERSION_MINOR(apiVesion) << "." << VK_VERSION_PATCH(apiVesion) << std::endl;
}


void Device::createLogicalDevice()
{
    //physical device features
    // VkPhysicalDeviceBufferDeviceAddressFeatures physicalDeviceBufferDeviceAddressFeatures{};
    // physicalDeviceBufferDeviceAddressFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    // physicalDeviceBufferDeviceAddressFeatures.pNext = nullptr;
    // physicalDeviceBufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR physicalDeviceAccelerationStructureFeatures {};
    physicalDeviceAccelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
    physicalDeviceAccelerationStructureFeatures.pNext = nullptr; //&physicalDeviceBufferDeviceAddressFeatures;
    physicalDeviceAccelerationStructureFeatures.accelerationStructure = VK_TRUE;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR physicalDeviceRayTracingPipelineFeatures{};
    physicalDeviceRayTracingPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    physicalDeviceRayTracingPipelineFeatures.pNext = &physicalDeviceAccelerationStructureFeatures;
    physicalDeviceRayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;

    VkPhysicalDeviceVulkan12Features physicalDeviceFeatures12{};
    physicalDeviceFeatures12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    physicalDeviceFeatures12.shaderInt8 = VK_TRUE;
    physicalDeviceFeatures12.storageBuffer8BitAccess = VK_TRUE;
    physicalDeviceFeatures12.bufferDeviceAddress = VK_TRUE;
    physicalDeviceFeatures12.descriptorIndexing = VK_TRUE;
    physicalDeviceFeatures12.pNext = &physicalDeviceRayTracingPipelineFeatures;

    VkPhysicalDeviceFeatures physicalDeviceFeatures{};
    physicalDeviceFeatures.shaderInt64 = VK_TRUE;


    //choose queue family
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
        throw std::runtime_error("could not find compatible queue family!");
    }
    std::cout << "Queue family index: " << queueFamilyIndex << "    (Out of " << queueFamilyProperties.size() << " queue families)"<< std::endl;


    //create logicalDevice
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
    deviceCreateInfo.pNext = &physicalDeviceFeatures12;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
    deviceCreateInfo.pEnabledFeatures = &physicalDeviceFeatures;
    deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensionList.size();
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensionList.data();


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


void Device::cleanup()
{
    if(logicalDevice != VK_NULL_HANDLE)
        vkDestroyDevice(logicalDevice, nullptr);
    if(instance != VK_NULL_HANDLE)
        vkDestroyInstance(instance, nullptr);
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
    return physicalDeviceRayTracingPipelineProperties.shaderGroupBaseAlignment;
}

VkDeviceSize Device::getShaderGroupHandleSize()
{
    return physicalDeviceRayTracingPipelineProperties.shaderGroupHandleSize;
}

} // namespace rmagine
