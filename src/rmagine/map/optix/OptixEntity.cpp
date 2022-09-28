#include "rmagine/map/optix/OptixEntity.hpp"

namespace rmagine
{

OptixEntity::OptixEntity(OptixContextPtr context_)
{
    setContext(context_);
}

void OptixEntity::setContext(OptixContextPtr context)
{
    m_ctx = context;
    m_stream = m_ctx->getCudaContext()->createStream();
}

} // namespace rmagine