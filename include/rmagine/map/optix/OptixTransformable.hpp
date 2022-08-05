#ifndef RMAGINE_MAP_OPTIX_TRANSFORMABLE_HPP
#define RMAGINE_MAP_OPTIX_TRANSFORMABLE_HPP

#include <rmagine/math/types.h>
#include <memory>

namespace rmagine
{

class OptixTransformable
{
public:
    OptixTransformable();
    virtual ~OptixTransformable() {}

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

    bool changed() const;

    /**
     * @brief Finished setting transformation. Apply to data if necessary
     * 
     */
    virtual void apply() = 0;

    bool m_changed;
protected:
    Transform m_T;
    Vector3 m_S;
    
};

using OptixTransformablePtr = std::shared_ptr<OptixTransformable>;

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_TRANSFORMABLE_HPP