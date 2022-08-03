#ifndef RMAGINE_MAP_OPTIX_GEOMETRY_HPP
#define RMAGINE_MAP_OPTIX_GEOMETRY_HPP

#include <memory>
#include <rmagine/math/types.h>
#include <rmagine/util/optix/OptixContext.hpp>
#include "OptixAccelerationStructure.hpp"

namespace rmagine
{

class OptixGeometry 
{
public:
    OptixGeometry(OptixContextPtr context = optix_default_context());

    virtual ~OptixGeometry();

    OptixAccelerationStructurePtr handle();

    void setTransform(const Transform& T);
    Transform transform() const;

    /**
     * @brief Set the Transform object. matrix must not contain scale.
     * Otherwise call setTransformAndScale
     * 
     * @param T 
     */
    void setTransform(const Matrix4x4& T);
    void setTransformAndScale(const Matrix4x4& T);

    void setScale(const Vector3& S);
    Vector3 scale() const;

    /**
     * @brief Obtain composed matrix
     * 
     * @return Matrix4x4 
     */
    Matrix4x4 matrix() const;

    virtual void apply();
    virtual void commit() = 0;

protected:
    Transform m_T;
    Vector3 m_S;
    OptixAccelerationStructurePtr m_as;

    OptixContextPtr m_ctx;
};

using OptixGeometryPtr = std::shared_ptr<OptixGeometry>;

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_GEOMETRY_HPP

