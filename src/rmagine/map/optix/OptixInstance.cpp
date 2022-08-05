#include "rmagine/map/optix/OptixInstance.hpp"

namespace rmagine
{

OptixInstance::OptixInstance(OptixGeometryPtr geom, OptixContextPtr context)
:Base(context)
,m_geom(geom)
{
    std::cout << "[OptixInstance::OptixInstance()] constructed." << std::endl;
}

OptixInstance::~OptixInstance()
{
    std::cout << "[OptixInstance::~OptixInstance()] destroyed." << std::endl;
}

void OptixInstance::apply()
{
    Matrix4x4 M = matrix();
    m_instance.transform[ 0] = M(0,0); // Rxx
    m_instance.transform[ 1] = M(0,1); // Rxy
    m_instance.transform[ 2] = M(0,2); // Rxz
    m_instance.transform[ 3] = M(0,3); // tx
    m_instance.transform[ 4] = M(1,0); // Ryx
    m_instance.transform[ 5] = M(1,1); // Ryy
    m_instance.transform[ 6] = M(1,2); // Ryz
    m_instance.transform[ 7] = M(1,3); // ty 
    m_instance.transform[ 8] = M(2,0); // Rzx
    m_instance.transform[ 9] = M(2,1); // Rzy
    m_instance.transform[10] = M(2,2); // Rzz
    m_instance.transform[11] = M(2,3); // tz
}

void OptixInstance::commit()
{
    
}

} // namespace rmagine