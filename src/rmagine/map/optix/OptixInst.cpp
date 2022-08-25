#include "rmagine/map/optix/OptixInst.hpp"

#include <optix_types.h>

#include "rmagine/map/optix/OptixAccelerationStructure.hpp"
#include "rmagine/map/optix/OptixGeometry.hpp"

#include <rmagine/util/optix/OptixDebug.hpp>

namespace rmagine
{

OptixInst::OptixInst(OptixContextPtr context)
:Base(context)
,m_data(new OptixInstance)
{
    m_data->sbtOffset = 0;
    m_data->visibilityMask = 255;
    m_data->flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
}

OptixInst::~OptixInst()
{
    delete m_data;

    if(sbt_data.scene)
    {
        cudaFree(sbt_data.scene);
    }

    if(m_scene)
    {
        m_scene->cleanupParents();
    }
}

void OptixInst::set(OptixScenePtr scene)
{
    m_scene = scene;
    scene->addParent(this_shared<OptixInst>());
    m_data->traversableHandle = scene->as()->handle;
}

OptixScenePtr OptixInst::scene() const
{
    return m_scene;
}

void OptixInst::apply()
{
    Matrix4x4 M = matrix();
    m_data->transform[ 0] = M(0,0); // Rxx
    m_data->transform[ 1] = M(0,1); // Rxy
    m_data->transform[ 2] = M(0,2); // Rxz
    m_data->transform[ 3] = M(0,3); // tx
    m_data->transform[ 4] = M(1,0); // Ryx
    m_data->transform[ 5] = M(1,1); // Ryy
    m_data->transform[ 6] = M(1,2); // Ryz
    m_data->transform[ 7] = M(1,3); // ty 
    m_data->transform[ 8] = M(2,0); // Rzx
    m_data->transform[ 9] = M(2,1); // Rzy
    m_data->transform[10] = M(2,2); // Rzz
    m_data->transform[11] = M(2,3); // tz

    m_changed = true;
}

void OptixInst::commit()
{
    if(m_scene)
    {
        if(!sbt_data.scene)
        {
            CUDA_CHECK( cudaMalloc(&sbt_data.scene, sizeof(OptixSceneSBT) ) );
        }

        CUDA_CHECK( cudaMemcpy(
            sbt_data.scene, 
            &m_scene->sbt_data, 
            sizeof(OptixSceneSBT), cudaMemcpyHostToDevice) );
    }
}

unsigned int OptixInst::depth() const 
{
    if(m_scene)
    {
        return m_scene->depth();
    } else {
        return 0;
    }
}

void OptixInst::setId(unsigned int id)
{
    m_data->instanceId = id;
}

unsigned int OptixInst::id() const
{
    return m_data->instanceId;
}

void OptixInst::disable()
{
    m_data->visibilityMask = 0;
}

void OptixInst::enable()
{
    m_data->visibilityMask = 255;
}

const OptixInstance* OptixInst::data() const
{
    return m_data;
}

} // namespace rmagine