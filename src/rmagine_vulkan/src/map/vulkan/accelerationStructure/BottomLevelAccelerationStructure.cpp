#include "rmagine/map/vulkan/accelerationStructure/BottomLevelAccelerationStructure.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

void BottomLevelAccelerationStructure::createAccelerationStructure(std::vector<VkAccelerationStructureGeometryKHR>& accelerationStructureGeometrys, std::vector<VkAccelerationStructureBuildRangeInfoKHR>& accelerationStructureBuildRangeInfos)
{
    // VkAccelerationStructureGeometryDataKHR accelerationStructureGeometryData{};
    // accelerationStructureGeometryData.triangles = {};
    // accelerationStructureGeometryData.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    // accelerationStructureGeometryData.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    // accelerationStructureGeometryData.triangles.vertexData = {};
    // accelerationStructureGeometryData.triangles.vertexData.deviceAddress = vertexMem.getBuffer()->getBufferDeviceAddress();
    // accelerationStructureGeometryData.triangles.vertexStride = sizeof(float) * 3;
    // accelerationStructureGeometryData.triangles.maxVertex = numVerticies;
    // accelerationStructureGeometryData.triangles.indexType = VK_INDEX_TYPE_UINT32;
    // accelerationStructureGeometryData.triangles.indexData = {};
    // accelerationStructureGeometryData.triangles.indexData.deviceAddress = indexMem.getBuffer()->getBufferDeviceAddress();
    // accelerationStructureGeometryData.triangles.transformData = {};
    // accelerationStructureGeometryData.triangles.transformData.deviceAddress = 0;

    // VkAccelerationStructureGeometryKHR accelerationStructureGeometry{};
    // accelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    // accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    // accelerationStructureGeometry.geometry = accelerationStructureGeometryData;
    // accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo{};
    accelerationStructureBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
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

    // uint32_t primitiveCount = numTriangles;
    std::vector<uint32_t> bottomLevelMaxPrimitiveCountList;
    for(size_t i = 0; i < accelerationStructureBuildRangeInfos.size(); i++)
    {
        bottomLevelMaxPrimitiveCountList.push_back(accelerationStructureBuildRangeInfos[i].primitiveCount);
    }

    createAccelerationStructureBufferAndDeviceMemory(bottomLevelMaxPrimitiveCountList, accelerationStructureBuildGeometryInfo, accelerationStructureBuildSizesInfo, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);

    buildAccelerationStructure(accelerationStructureBuildRangeInfos, accelerationStructureBuildGeometryInfo, accelerationStructureBuildSizesInfo);
}

} // namespace rmagine
