#include "rmagine/map/optix/OptixInstance.hpp"

namespace rmagine
{

OptixInst::OptixInst(OptixGeometryPtr geom, OptixContextPtr context)
:Base(context)
,m_geom(geom)
{
    m_data.sbtOffset = 0;
    m_data.visibilityMask = 255;
    m_data.flags = OPTIX_INSTANCE_FLAG_NONE;
    m_data.traversableHandle = geom->handle()->handle;
    std::cout << "[OptixInst::OptixInst()] constructed." << std::endl;
}

OptixInst::~OptixInst()
{
    std::cout << "[OptixInst::~OptixInst()] destroyed." << std::endl;
}

void OptixInst::apply()
{
    Matrix4x4 M = matrix();
    m_data.transform[ 0] = M(0,0); // Rxx
    m_data.transform[ 1] = M(0,1); // Rxy
    m_data.transform[ 2] = M(0,2); // Rxz
    m_data.transform[ 3] = M(0,3); // tx
    m_data.transform[ 4] = M(1,0); // Ryx
    m_data.transform[ 5] = M(1,1); // Ryy
    m_data.transform[ 6] = M(1,2); // Ryz
    m_data.transform[ 7] = M(1,3); // ty 
    m_data.transform[ 8] = M(2,0); // Rzx
    m_data.transform[ 9] = M(2,1); // Rzy
    m_data.transform[10] = M(2,2); // Rzz
    m_data.transform[11] = M(2,3); // tz
}

void OptixInst::setId(unsigned int id)
{
    m_data.instanceId = id;
}

unsigned int OptixInst::id() const
{
    return m_data.instanceId;
}

OptixInstance OptixInst::data() const
{
    return m_data;
}

void OptixInst::commit()
{
    


}

} // namespace rmagine