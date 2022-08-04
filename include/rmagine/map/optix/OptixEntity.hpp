#ifndef RMAGINE_MAP_OPTIX_ENTITY_HPP
#define RMAGINE_MAP_OPTIX_ENTITY_HPP

#include <rmagine/util/optix/OptixContext.hpp>

namespace rmagine
{

class OptixEntity
{
public:
    OptixEntity(OptixContextPtr context_ = optix_default_context());

    inline OptixContextPtr context() const
    {
        return m_ctx;
    }
protected:
    OptixContextPtr m_ctx;
};

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_ENTITY_HPP