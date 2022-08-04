#ifndef RMAGINE_MAP_OPTIX_ENTITY_HPP
#define RMAGINE_MAP_OPTIX_ENTITY_HPP

#include <rmagine/util/optix/OptixContext.hpp>

namespace rmagine
{

class OptixEntity
: public std::enable_shared_from_this<OptixEntity>
{
public:
    OptixEntity(OptixContextPtr context_ = optix_default_context());

    virtual ~OptixEntity() {};

    inline OptixContextPtr context() const
    {
        return m_ctx;
    }

    template<typename T>
    inline std::shared_ptr<T> this_shared()
    {
        return std::dynamic_pointer_cast<T>(shared_from_this());
    }

protected:
    OptixContextPtr m_ctx;
};

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_ENTITY_HPP