#include "rmagine/map/OptixMap.hpp"
#include "rmagine/util/optix/OptixDebug.hpp"
#include "rmagine/types/MemoryCuda.hpp"
#include "rmagine/util/GenericAlign.hpp"

#include "rmagine/math/assimp_conversions.h"

#include <optix.h>
#include <optix_stubs.h>


#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <iomanip>

#include <map>

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

namespace rmagine {

OptixMap::OptixMap(
    OptixContextPtr optix_ctx)
:OptixEntity(optix_ctx)
{
    
}

OptixMap::OptixMap(OptixScenePtr scene)
:OptixEntity(scene->context())
,m_scene(scene)
{
    
}

OptixMap::~OptixMap()
{

}

void OptixMap::setScene(OptixScenePtr scene)
{
    setContext(scene->context());
    m_scene = scene;
}

OptixScenePtr OptixMap::scene() const
{
    return m_scene;
}

} // namespace mamcl