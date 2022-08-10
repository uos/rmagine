#include "rmagine/map/optix/OptixInst.hpp"

#include "rmagine/map/optix/OptixAccelerationStructure.hpp"
#include "rmagine/map/optix/OptixGeometry.hpp"

#include <rmagine/util/optix/OptixDebug.hpp>

namespace rmagine
{

OptixInst::OptixInst(OptixContextPtr context)
:OptixEntity(context)
{
    m_data.sbtOffset = 0;
    m_data.visibilityMask = 255;
    m_data.flags = OPTIX_INSTANCE_FLAG_NONE;
}

OptixInst::OptixInst(
    OptixGeometryPtr geom)
:OptixInst(geom->context())
{
    // coused weak ptr error in gazebo 'std::bad_weak_ptr'
    // maybe the template function this_shared<OptixInst>() does not work in constructors?
    setGeometry(geom);
}

OptixInst::~OptixInst()
{
    if(m_data_gpu)
    {
        cudaFree( reinterpret_cast<void*>( m_data_gpu ) );
    }
    m_geom->cleanupParents();
}

void OptixInst::setGeometry(OptixGeometryPtr geom)
{
    m_geom = geom;
    geom->addParent(this_shared<OptixInst>());
    m_data.traversableHandle = geom->acc()->handle;
}

OptixGeometryPtr OptixInst::geometry() const
{
    return m_geom;
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

    if(m_data_gpu)
    {
        // was committed before
    } else {
        // first alloc
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>( &m_data_gpu ),
            sizeof(OptixInstance)
        ));
    }

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(m_data_gpu),
        &m_data,
        sizeof(OptixInstance),
        cudaMemcpyHostToDevice
    ));

    m_changed = true;
}

void OptixInst::setId(unsigned int id)
{
    m_data.instanceId = id;
}

unsigned int OptixInst::id() const
{
    return m_data.instanceId;
}

void OptixInst::disable()
{
    m_data.visibilityMask = 0;
}

void OptixInst::enable()
{
    m_data.visibilityMask = 255;
}

OptixInstance OptixInst::data() const
{
    return m_data;
}

CUdeviceptr OptixInst::data_gpu() const
{
    return m_data_gpu;
}

} // namespace rmagine