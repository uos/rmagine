#include "rmagine/map/vulkan/accelerationStructure/TopLevelAccelerationStructure.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

void TopLevelAccelerationStructure::createAccelerationStructure(std::vector<VkAccelerationStructureGeometryKHR>& accelerationStructureGeometrys, std::vector<VkAccelerationStructureBuildRangeInfoKHR>& accelerationStructureBuildRangeInfos)
{
    // VkAccelerationStructureGeometryDataKHR accelerationStructureGeometryData{};
    // accelerationStructureGeometryData.instances = {};
    // accelerationStructureGeometryData.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    // accelerationStructureGeometryData.instances.arrayOfPointers = VK_FALSE;
    // accelerationStructureGeometryData.instances.data = {};
    // accelerationStructureGeometryData.instances.data.deviceAddress = bottomLevelAccelerationStructureInstance->getInstanceMemory().getBuffer()->getBufferDeviceAddress();

    // VkAccelerationStructureGeometryKHR accelerationStructureGeometry{};
    // accelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    // accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    // accelerationStructureGeometry.geometry = accelerationStructureGeometryData;
    // accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    
    VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo{};
    accelerationStructureBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    accelerationStructureBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    accelerationStructureBuildGeometryInfo.geometryCount = accelerationStructureGeometrys.size();//TODO: only works for 1 right now
    accelerationStructureBuildGeometryInfo.pGeometries = accelerationStructureGeometrys.data();
    accelerationStructureBuildGeometryInfo.scratchData = {};
    accelerationStructureBuildGeometryInfo.scratchData.deviceAddress = 0;

    VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo{};
    accelerationStructureBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    accelerationStructureBuildSizesInfo.accelerationStructureSize = 0;
    accelerationStructureBuildSizesInfo.updateScratchSize = 0;
    accelerationStructureBuildSizesInfo.buildScratchSize = 0;

    // uint32_t primitiveCount = 1;
    std::vector<uint32_t> topLevelMaxPrimitiveCountList;
    for(size_t i = 0; i < accelerationStructureBuildRangeInfos.size(); i++)
    {
        topLevelMaxPrimitiveCountList.push_back(accelerationStructureBuildRangeInfos[i].primitiveCount);
    }

    createAccelerationStructureBufferAndDeviceMemory(topLevelMaxPrimitiveCountList, accelerationStructureBuildGeometryInfo, accelerationStructureBuildSizesInfo, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR);
    
    buildAccelerationStructure(accelerationStructureBuildRangeInfos, accelerationStructureBuildGeometryInfo, accelerationStructureBuildSizesInfo);
}

size_t TopLevelAccelerationStructure::getID()
{
    return tlasID;
}



size_t TopLevelAccelerationStructure::tlasIDcounter = 0;

size_t TopLevelAccelerationStructure::getNewTlasID()
{
    if(tlasIDcounter == SIZE_MAX)
    {
        throw std::runtime_error("You created way too many top level acceleration structures!"); 
    }
    return ++tlasIDcounter;
}

} // namespace rmagine
