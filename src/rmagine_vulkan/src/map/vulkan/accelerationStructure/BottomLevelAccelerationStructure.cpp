#include "rmagine/map/vulkan/accelerationStructure/BottomLevelAccelerationStructure.hpp"
#include "rmagine/map/vulkan/VulkanMesh.hpp"
#include "rmagine/map/vulkan/VulkanInst.hpp"
#include "rmagine/map/vulkan/VulkanScene.hpp"



namespace rmagine
{

BottomLevelAccelerationStructure::BottomLevelAccelerationStructure(std::map<unsigned int, VulkanGeometryPtr>& geometries) : 
    AccelerationStructure(VkAccelerationStructureTypeKHR::VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR),
    m_meshDescriptions_ram(geometries.size()),
    m_meshDescriptions(geometries.size())
{
    std::vector<VkAccelerationStructureGeometryKHR> accelerationStructureGeometrys;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> accelerationStructureBuildRangeInfos;

    size_t idx = 0;
    for(auto const& geometry : geometries)
    {
        VulkanMeshPtr mesh = geometry.second->this_shared<VulkanMesh>();

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
        accelerationStructureGeometrys.push_back(accelerationStructureGeometry);

        VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
        accelerationStructureBuildRangeInfo.firstVertex = 0;
        accelerationStructureBuildRangeInfo.primitiveOffset = 0;
        accelerationStructureBuildRangeInfo.primitiveCount = mesh->faces.size();
        accelerationStructureBuildRangeInfo.transformOffset = 0;
        accelerationStructureBuildRangeInfos.push_back(accelerationStructureBuildRangeInfo);

        m_meshDescriptions_ram[idx].vertexAddress = mesh->vertices.getBuffer()->getBufferDeviceAddress();
        m_meshDescriptions_ram[idx].faceAddress = mesh->faces.getBuffer()->getBufferDeviceAddress();
        m_meshDescriptions_ram[idx].faceNormalAddress = mesh->face_normals.getBuffer()->getBufferDeviceAddress();
        m_meshDescriptions_ram[idx].vertexNormalAddress = mesh->vertex_normals.getBuffer()->getBufferDeviceAddress();

        idx++;
    }
    m_meshDescriptions = m_meshDescriptions_ram;

    createAccelerationStructure(accelerationStructureGeometrys, accelerationStructureBuildRangeInfos);
}

BottomLevelAccelerationStructure::~BottomLevelAccelerationStructure()
{

}

} // namespace rmagine