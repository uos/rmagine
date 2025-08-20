#include "rmagine/map/vulkan/accelerationStructure/AccelerationStructure.hpp"
#include "rmagine/util/VulkanContext.hpp"
#include "VulkanMesh.hpp"
#include "VulkanInst.hpp"



namespace rmagine
{

AccelerationStructure::AccelerationStructure(VkAccelerationStructureTypeKHR accelerationStructureType) : 
    accelerationStructureType(accelerationStructureType),
    device(get_vulkan_context()->getDevice()), 
    extensionFunctionsPtr(get_vulkan_context()->getExtensionFunctionsPtr())
{

}

AccelerationStructure::AccelerationStructure(VkAccelerationStructureTypeKHR accelerationStructureType, DevicePtr device, ExtensionFunctionsPtr extensionFunctionsPtr) :
    accelerationStructureType(accelerationStructureType), 
    device(device), 
    extensionFunctionsPtr(extensionFunctionsPtr)
{
    
}

AccelerationStructure::~AccelerationStructure()
{
    cleanup();
}

void AccelerationStructure::createAccelerationStructure(
    std::vector<VkAccelerationStructureGeometryKHR>& accelerationStructureGeometrys, 
    std::vector<VkAccelerationStructureBuildRangeInfoKHR>& accelerationStructureBuildRangeInfos)
{
    VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo{};
    accelerationStructureBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    accelerationStructureBuildGeometryInfo.type = accelerationStructureType;
    accelerationStructureBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    accelerationStructureBuildGeometryInfo.geometryCount = accelerationStructureGeometrys.size();
    accelerationStructureBuildGeometryInfo.pGeometries = accelerationStructureGeometrys.data();
    accelerationStructureBuildGeometryInfo.scratchData = {};
    accelerationStructureBuildGeometryInfo.scratchData.deviceAddress = 0;

    VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo{};
    accelerationStructureBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    accelerationStructureBuildSizesInfo.accelerationStructureSize = 0;
    accelerationStructureBuildSizesInfo.updateScratchSize = 0;
    accelerationStructureBuildSizesInfo.buildScratchSize = 0;

    std::vector<uint32_t> maxPrimitiveCountList;
    for(size_t i = 0; i < accelerationStructureBuildRangeInfos.size(); i++)
    {
        maxPrimitiveCountList.push_back(accelerationStructureBuildRangeInfos[i].primitiveCount);
    }

    createAccelerationStructureBufferAndDeviceMemory(maxPrimitiveCountList, accelerationStructureBuildGeometryInfo, accelerationStructureBuildSizesInfo);

    buildAccelerationStructure(accelerationStructureBuildRangeInfos, accelerationStructureBuildGeometryInfo, accelerationStructureBuildSizesInfo);
}


void AccelerationStructure::createAccelerationStructureBufferAndDeviceMemory(
    std::vector<uint32_t>& maxPrimitiveCountList, 
    VkAccelerationStructureBuildGeometryInfoKHR& accelerationStructureBuildGeometryInfo, 
    VkAccelerationStructureBuildSizesInfoKHR& accelerationStructureBuildSizesInfo)
{
    if(accelerationStructureBuffer != nullptr)
    {
        throw std::runtime_error("acceleration structure buffer has already been created!");
    }

    extensionFunctionsPtr->pvkGetAccelerationStructureBuildSizesKHR(
        device->getLogicalDevice(), 
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, 
        &accelerationStructureBuildGeometryInfo, 
        maxPrimitiveCountList.data(), 
        &accelerationStructureBuildSizesInfo);

    accelerationStructureBuffer = std::make_shared<Buffer>(
        accelerationStructureBuildSizesInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    accelerationStructureDeviceMemory = std::make_shared<DeviceMemory>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, accelerationStructureBuffer);

    VkAccelerationStructureCreateInfoKHR accelerationStructureCreateInfo{};
    accelerationStructureCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    accelerationStructureCreateInfo.buffer = accelerationStructureBuffer->getBuffer();
    accelerationStructureCreateInfo.size = accelerationStructureBuildSizesInfo.accelerationStructureSize;
    accelerationStructureCreateInfo.type = accelerationStructureType;
    accelerationStructureCreateInfo.deviceAddress = 0;

    if(extensionFunctionsPtr->pvkCreateAccelerationStructureKHR(device->getLogicalDevice(), &accelerationStructureCreateInfo, nullptr, &accelerationStructure) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to creates acceleration structure!");
    }
    
    VkAccelerationStructureDeviceAddressInfoKHR accelerationStructureDeviceAddressInfo{};
    accelerationStructureDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    accelerationStructureDeviceAddressInfo.accelerationStructure = accelerationStructure;

    accelerationStructureDeviceAddress = 
        extensionFunctionsPtr->pvkGetAccelerationStructureDeviceAddressKHR(device->getLogicalDevice(), &accelerationStructureDeviceAddressInfo);
}


void AccelerationStructure::buildAccelerationStructure(
    std::vector<VkAccelerationStructureBuildRangeInfoKHR>& accelerationStructureBuildRangeInfos, 
    VkAccelerationStructureBuildGeometryInfoKHR& accelerationStructureBuildGeometryInfo, 
    VkAccelerationStructureBuildSizesInfoKHR& accelerationStructureBuildSizesInfo)
{
    if(accelerationStructureBuffer == nullptr)
    {
        throw std::runtime_error("acceleration structure buffer has to get created before acceleration structure can be build!");
    }

    accelerationStructureScratchBuffer = std::make_shared<Buffer>(
        accelerationStructureBuildSizesInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    accelerationStructureScratchDeviceMemory = std::make_shared<DeviceMemory>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, accelerationStructureScratchBuffer);

    accelerationStructureBuildGeometryInfo.dstAccelerationStructure = accelerationStructure;
    accelerationStructureBuildGeometryInfo.scratchData.deviceAddress = accelerationStructureScratchBuffer->getBufferDeviceAddress();

    get_vulkan_context()->getDefaultCommandBuffer()->recordBuildingASToCommandBuffer(accelerationStructureBuildGeometryInfo, accelerationStructureBuildRangeInfos.data());
    get_vulkan_context()->getDefaultCommandBuffer()->submitRecordedCommandAndWait();

    std::cout << "acceleration structure has been build" << std::endl;
}


VkDeviceAddress AccelerationStructure::getDeviceAddress()
{
    return accelerationStructureDeviceAddress;
}


VkAccelerationStructureKHR* AccelerationStructure::getAcceleratiionStructurePtr()
{
    return &accelerationStructure;
}


void AccelerationStructure::cleanup()
{
    if(accelerationStructure != VK_NULL_HANDLE)
    {
        accelerationStructureDeviceMemory->cleanup();
        accelerationStructureBuffer->cleanup();
        accelerationStructureScratchDeviceMemory->cleanup();
        accelerationStructureScratchBuffer->cleanup();

        extensionFunctionsPtr->pvkDestroyAccelerationStructureKHR(device->getLogicalDevice(), accelerationStructure, nullptr);
        accelerationStructure = VK_NULL_HANDLE;
    }
}


size_t AccelerationStructure::getID()
{
    return asID;
}



size_t AccelerationStructure::asIDcounter = 0;

size_t AccelerationStructure::getNewAsID()
{
    if(asIDcounter == SIZE_MAX)
    {
        throw std::runtime_error("You created way too many top level acceleration structures!"); 
    }
    return ++asIDcounter;
}



VkAccelerationStructureBuildRangeInfoKHR AccelerationStructure::GetASBuildRange(VulkanScenePtr scene)
{
    VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
    accelerationStructureBuildRangeInfo.firstVertex = 0;
    accelerationStructureBuildRangeInfo.primitiveOffset = 0;
    accelerationStructureBuildRangeInfo.primitiveCount = scene->numOfChildNodes();
    accelerationStructureBuildRangeInfo.transformOffset = 0;
    return accelerationStructureBuildRangeInfo;
}

VkAccelerationStructureBuildRangeInfoKHR AccelerationStructure::GetASBuildRange(VulkanMeshPtr mesh)
{
    VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
    accelerationStructureBuildRangeInfo.firstVertex = 0;
    accelerationStructureBuildRangeInfo.primitiveOffset = 0;
    accelerationStructureBuildRangeInfo.primitiveCount = mesh->faces.size();
    accelerationStructureBuildRangeInfo.transformOffset = 0;
    return accelerationStructureBuildRangeInfo;
}

VkAccelerationStructureGeometryKHR AccelerationStructure::GetASGeometry(VulkanScenePtr scene)
{
    VkAccelerationStructureGeometryKHR accelerationStructureGeometry{};
    accelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    accelerationStructureGeometry.geometry = {};
    accelerationStructureGeometry.geometry.instances = {};
    accelerationStructureGeometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    accelerationStructureGeometry.geometry.instances.arrayOfPointers = VK_FALSE;
    accelerationStructureGeometry.geometry.instances.data = {};
    accelerationStructureGeometry.geometry.instances.data.deviceAddress = mesh->getASInstances().getBuffer()->getBufferDeviceAddress();
    return accelerationStructureGeometry;
}

VkAccelerationStructureGeometryKHR AccelerationStructure::GetASGeometry(VulkanMeshPtr mesh)
{
    VkAccelerationStructureGeometryKHR accelerationStructureGeometry{};
    accelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    accelerationStructureGeometry.geometry = {};
    accelerationStructureGeometry.geometry.triangles = {};
    accelerationStructureGeometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    accelerationStructureGeometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    accelerationStructureGeometry.geometry.triangles.vertexData = {};
    accelerationStructureGeometry.geometry.triangles.vertexData.deviceAddress = mesh->vertices.getBuffer()->getBufferDeviceAddress();
    accelerationStructureGeometry.geometry.triangles.vertexStride = sizeof(float) * 3;
    accelerationStructureGeometry.geometry.triangles.maxVertex = mesh->vertices.size();
    accelerationStructureGeometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    accelerationStructureGeometry.geometry.triangles.indexData = {};
    accelerationStructureGeometry.geometry.triangles.indexData.deviceAddress = mesh->faces.getBuffer()->getBufferDeviceAddress();
    accelerationStructureGeometry.geometry.triangles.transformData = {};
    accelerationStructureGeometry.geometry.triangles.transformData.deviceAddress = mesh->transformMatrix.getBuffer()->getBufferDeviceAddress();
    return accelerationStructureGeometry;
}

} // namespace rmagine
