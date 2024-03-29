#include "rmagine/map/optix/OptixGeometry.hpp"
#include <rmagine/math/linalg.h>

#include "rmagine/map/optix/OptixAccelerationStructure.hpp"
#include "rmagine/map/optix/OptixScene.hpp"
#include "rmagine/map/optix/OptixInst.hpp"

namespace rmagine
{

OptixGeometry::OptixGeometry(OptixContextPtr context)
:OptixEntity(context)
,OptixTransformable()
{
    // std::cout << "[OptixGeometry::OptixGeometry()] constructed." << std::endl;
}

OptixGeometry::~OptixGeometry()
{
    // std::cout << "[OptixGeometry::~OptixGeometry()] destroyed." << std::endl;
}

void OptixGeometry::cleanupParents()
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

bool OptixGeometry::removeParent(OptixScenePtr parent)
{
    auto it = m_parents.find(parent);

    if(it != m_parents.end())
    {
        m_parents.erase(it);
        return true;
    }

    return false;
}

bool OptixGeometry::hasParent(OptixScenePtr parent) const
{
    return m_parents.find(parent) != m_parents.end();
}

std::unordered_set<OptixScenePtr> OptixGeometry::parents() const
{
    std::unordered_set<OptixScenePtr> ret;

    for(OptixSceneWPtr elem : m_parents)
    {
        if(OptixScenePtr tmp = elem.lock())
        {
            ret.insert(tmp);
        }
    }
    
    return ret;
}

void OptixGeometry::addParent(OptixScenePtr parent)
{
    m_parents.insert(parent);
}

OptixScenePtr OptixGeometry::makeScene()
{
    OptixScenePtr geom_scene = std::make_shared<OptixScene>(m_ctx);
    geom_scene->add(this_shared<OptixGeometry>());
    return geom_scene;
}

OptixInstPtr OptixGeometry::instantiate()
{
    OptixScenePtr geom_scene = makeScene();
    geom_scene->commit();

    OptixInstPtr geom_inst = std::make_shared<OptixInst>(m_ctx);
    geom_inst->set(geom_scene);

    return geom_inst;
}

} // namespace rmagine