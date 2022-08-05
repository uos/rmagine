#include "rmagine/map/optix/OptixGeometry.hpp"
#include <rmagine/math/linalg.h>

#include "rmagine/map/optix/OptixAccelerationStructure.hpp"

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
    if(m_as)
    {
        cudaFree( reinterpret_cast<void*>( m_as->buffer ) );
    }
    // std::cout << "[OptixGeometry::~OptixGeometry()] destroyed." << std::endl;
}

OptixAccelerationStructurePtr OptixGeometry::acc()
{
    return m_as;
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

std::unordered_set<OptixInstPtr> OptixGeometry::parents() const
{
    std::unordered_set<OptixInstPtr> ret;

    for(OptixInstWPtr elem : m_parents)
    {
        if(auto tmp = elem.lock())
        {
            ret.insert(tmp);
        }
    }
    
    return ret;
}

void OptixGeometry::addParent(OptixInstPtr parent)
{
    m_parents.insert(parent);
}

} // namespace rmagine