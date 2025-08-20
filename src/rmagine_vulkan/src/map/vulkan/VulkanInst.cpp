#include "rmagine/map/vulkan/VulkanInst.hpp"

#include "rmagine/map/vulkan/VulkanScene.hpp"
#include "rmagine/map/vulkan/VulkanInst.hpp"



namespace rmagine
{

VulkanInst::VulkanInst() : Base(),
    instance(1, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR),
    instance_ram(1)
{
    instance_ram[0] = {};
    instance_ram[0].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance_ram[0].mask = 0xFF;
    instance_ram[0].transform = {{{1.0, 0.0, 0.0, 0.0},
                                  {0.0, 1.0, 0.0, 0.0},
                                  {0.0, 0.0, 1.0, 0.0}}};


    accelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    accelerationStructureGeometry.geometry = {};

    accelerationStructureGeometry.geometry.instances = {};
    accelerationStructureGeometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    accelerationStructureGeometry.geometry.instances.arrayOfPointers = VK_FALSE;
    accelerationStructureGeometry.geometry.instances.data = {};
    accelerationStructureGeometry.geometry.instances.data.deviceAddress = instance.getBuffer()->getBufferDeviceAddress();


    accelerationStructureBuildRangeInfo.firstVertex = 0;
    accelerationStructureBuildRangeInfo.primitiveOffset = 0;
    accelerationStructureBuildRangeInfo.primitiveCount = 0; // count not yet known
    accelerationStructureBuildRangeInfo.transformOffset = 0;
}

VulkanInst::~VulkanInst()
{
    
}

void VulkanInst::set(VulkanScenePtr scene)
{
    if(scene->type() != VulkanSceneType::GEOMETRIES)
    {
        throw std::runtime_error("can only instanciate a scene containing meshes, not one containing other instances.");
    }

    m_scene = scene;
    scene->addParent(this_shared<VulkanInst>());
    instance_ram[0].accelerationStructureReference = m_scene->as()->getDeviceAddress();
}

VulkanScenePtr VulkanInst::scene() const
{
    return m_scene;
}

void VulkanInst::apply()
{
    Matrix4x4 M = matrix();
    instance_ram[0].transform = {{{M(0,0), M(0,1), M(0,2), M(0,3)},
                                  {M(1,0), M(1,1), M(1,2), M(1,3)},
                                  {M(2,0), M(2,1), M(2,2), M(2,3)}}};
    m_changed = true;
}

void VulkanInst::commit()
{
    if(m_scene)
    {
        instance = instance_ram;

        accelerationStructureBuildRangeInfo.primitiveCount = m_scene->numOfChildNodes();
    }
}

unsigned int VulkanInst::depth() const 
{
    if(m_scene)
    {
        return m_scene->depth();
    }
    else
    {
        return 0;
    }
}

void VulkanInst::setId(unsigned int id)
{
    instance_ram[0].instanceCustomIndex = id;
}

unsigned int VulkanInst::id() const
{
    return instance_ram[0].instanceCustomIndex;
}

void VulkanInst::disable()
{
    instance_ram[0].mask = 0x00;
}

void VulkanInst::enable()
{
    instance_ram[0].mask = 0xFF;
}

} // namespace rmagine
