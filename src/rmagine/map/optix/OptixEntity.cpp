#include "rmagine/map/optix/OptixEntity.hpp"

namespace rmagine
{

OptixEntity::OptixEntity(OptixContextPtr context_)
:m_ctx(context_)
{
    
}

void OptixEntity::setContext(OptixContextPtr context)
{
    m_ctx = context;
}

} // namespace rmagine