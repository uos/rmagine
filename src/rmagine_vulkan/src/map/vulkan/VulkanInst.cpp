#include "rmagine/map/vulkan/VulkanInst.hpp"

#include "rmagine/map/vulkan/VulkanScene.hpp"
#include "rmagine/map/vulkan/VulkanInst.hpp"



namespace rmagine
{

VulkanInst::VulkanInst() : Base(),
    m_data(new VkAccelerationStructureInstanceKHR)
{
    *m_data = {};
    m_data->flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    m_data->mask = 0xFF;
    m_data->transform = {{{1.0, 0.0, 0.0, 0.0},
                          {0.0, 1.0, 0.0, 0.0},
                          {0.0, 0.0, 1.0, 0.0}}};
}

VulkanInst::~VulkanInst()
{
    delete m_data;
}

void VulkanInst::set(VulkanScenePtr scene)
{
    //TODO: pretty sure the tree can only have depth one & you cannot instatiate a scene containing other instances
    if(scene->type() != VulkanSceneType::GEOMETRIES)
    {
        throw std::runtime_error("[VulkanInst::set()] ERROR - can only instanciate a scene containing meshes, not one containing other instances.");
    }

    m_scene = scene;
    scene->addParent(this_shared<VulkanInst>());
    m_data->accelerationStructureReference = m_scene->as()->getDeviceAddress();
}

VulkanScenePtr VulkanInst::scene() const
{
    return m_scene;
}

void VulkanInst::apply()
{
    Matrix4x4 M = matrix();
    m_data->transform = {{{M(0,0), M(0,1), M(0,2), M(0,3)},
                          {M(1,0), M(1,1), M(1,2), M(1,3)},
                          {M(2,0), M(2,1), M(2,2), M(2,3)}}};
    m_changed = true;
}

void VulkanInst::commit()
{
    if(m_scene)
    {
        // nothing to do here currently
        // is here just in case
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
    m_data->instanceCustomIndex = id;
}

unsigned int VulkanInst::id() const
{
    return m_data->instanceCustomIndex;
}

void VulkanInst::disable()
{
    m_data->mask = 0x00;
}

void VulkanInst::enable()
{
    m_data->mask = 0xFF;
}

const VkAccelerationStructureInstanceKHR* VulkanInst::data() const
{
    return m_data;
}

} // namespace rmagine
