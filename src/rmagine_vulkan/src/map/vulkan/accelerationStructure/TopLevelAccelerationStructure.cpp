#include "rmagine/map/vulkan/accelerationStructure/TopLevelAccelerationStructure.hpp"
#include "rmagine/map/vulkan/accelerationStructure/BottomLevelAccelerationStructure.hpp"
#include "rmagine/map/vulkan/VulkanMesh.hpp"
#include "rmagine/map/vulkan/VulkanInst.hpp"
#include "rmagine/map/vulkan/VulkanScene.hpp"



namespace rmagine
{

TopLevelAccelerationStructure::TopLevelAccelerationStructure(std::map<unsigned int, VulkanGeometryPtr>& geometries) : 
    AccelerationStructure(VkAccelerationStructureTypeKHR::VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR),
    m_asInstances_ram(geometries.size()),
    m_asInstances(geometries.size(), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR),
    m_blasMeshDescriptions_ram(geometries.size()),
    m_blasMeshDescriptions(geometries.size())
{
    std::vector<VkAccelerationStructureGeometryKHR> accelerationStructureGeometrys;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> accelerationStructureBuildRangeInfos;

    size_t idx = 0;
    for(auto const& geometry : geometries)
    {
        VulkanInstPtr inst = geometry.second->this_shared<VulkanInst>();

        m_asInstances_ram[idx] = *(inst->data());

        m_blasMeshDescriptions_ram[idx] = inst->scene()->as()->this_shared<BottomLevelAccelerationStructure>()->m_meshDescriptions.getBuffer()->getBufferDeviceAddress();

        idx++;
    }
    m_asInstances = m_asInstances_ram;
    m_blasMeshDescriptions = m_blasMeshDescriptions_ram;

    VkAccelerationStructureGeometryKHR accelerationStructureGeometry{};
    accelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    accelerationStructureGeometry.geometry = {};
    accelerationStructureGeometry.geometry.instances = {};
    accelerationStructureGeometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    accelerationStructureGeometry.geometry.instances.arrayOfPointers = VK_FALSE;
    accelerationStructureGeometry.geometry.instances.data = {};
    accelerationStructureGeometry.geometry.instances.data.deviceAddress = m_asInstances.getBuffer()->getBufferDeviceAddress();
    accelerationStructureGeometrys.push_back(accelerationStructureGeometry);

    VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
    accelerationStructureBuildRangeInfo.firstVertex = 0;
    accelerationStructureBuildRangeInfo.primitiveOffset = 0;
    accelerationStructureBuildRangeInfo.primitiveCount = geometries.size();
    accelerationStructureBuildRangeInfo.transformOffset = 0;
    accelerationStructureBuildRangeInfos.push_back(accelerationStructureBuildRangeInfo);

    createAccelerationStructure(accelerationStructureGeometrys, accelerationStructureBuildRangeInfos);
}

TopLevelAccelerationStructure::~TopLevelAccelerationStructure()
{

}

} // namespace rmagine
