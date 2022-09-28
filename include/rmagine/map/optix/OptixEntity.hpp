#ifndef RMAGINE_MAP_OPTIX_ENTITY_HPP
#define RMAGINE_MAP_OPTIX_ENTITY_HPP

#include <rmagine/util/optix/OptixContext.hpp>
#include <rmagine/util/cuda/CudaStream.hpp>

namespace rmagine
{

class OptixEntity
: public std::enable_shared_from_this<OptixEntity>
{
public:
    OptixEntity(OptixContextPtr context_ = optix_default_context());

    virtual ~OptixEntity() {};

    std::string name;

    inline OptixContextPtr context() const
    {
        return m_ctx;
    }

    inline CudaStreamPtr stream() const 
    {
        return m_stream;
    }

    void setContext(OptixContextPtr context);

    template<typename T>
    inline std::shared_ptr<T> this_shared()
    {
        return std::dynamic_pointer_cast<T>(shared_from_this());
    }

protected:
    OptixContextPtr m_ctx;
    CudaStreamPtr m_stream;
};

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_ENTITY_HPP