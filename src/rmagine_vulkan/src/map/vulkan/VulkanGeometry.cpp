#include "rmagine/map/vulkan/VulkanGeometry.hpp"

#include "rmagine/map/vulkan/VulkanScene.hpp"
#include "rmagine/map/vulkan/VulkanInst.hpp"



namespace rmagine
{

void VulkanGeometry::cleanupParents()
{
    for(auto it = m_parents.cbegin(); it != m_parents.cend();)
    {
        if (it->expired())
        {
            m_parents.erase(it++);    // or "it = m.erase(it)" since C++11
        } else {
            ++it;
        }
    }
}

bool VulkanGeometry::removeParent(VulkanScenePtr parent)
{
    auto it = m_parents.find(parent);

    if(it != m_parents.end())
    {
        m_parents.erase(it);
        return true;
    }

    return false;
}

bool VulkanGeometry::hasParent(VulkanScenePtr parent) const
{
    return m_parents.find(parent) != m_parents.end();
}

std::unordered_set<VulkanScenePtr> VulkanGeometry::parents() const
{
    std::unordered_set<VulkanScenePtr> ret;

    for(VulkanSceneWPtr elem : m_parents)
    {
        if(VulkanScenePtr tmp = elem.lock())
        {
            ret.insert(tmp);
        }
    }
    
    return ret;
}

void VulkanGeometry::addParent(VulkanScenePtr parent)
{
    m_parents.insert(parent);
}

// VulkanScenePtr VulkanGeometry::makeScene()
// {
//     VulkanScenePtr geom_scene = std::make_shared<VulkanScene>();
//     geom_scene->add(this_shared<VulkanGeometry>());
//     return geom_scene;
// }

// VulkanInstPtr VulkanGeometry::instantiate()
// {
//     VulkanScenePtr geom_scene = makeScene();
//     geom_scene->commit();

//     VulkanInstPtr geom_inst = std::make_shared<VulkanInst>();
//     geom_inst->set(geom_scene);

//     return geom_inst;
// }

} // namespace rmagine
