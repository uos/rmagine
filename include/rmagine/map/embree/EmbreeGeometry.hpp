#ifndef RMAGINE_MAP_EMBREE_GEOMETRY_HPP
#define RMAGINE_MAP_EMBREE_GEOMETRY_HPP

#include <memory>

#include <embree3/rtcore.h>

#include "EmbreeDevice.hpp"
#include "embree_types.h"
#include <rmagine/math/types.h>

namespace rmagine
{

void decompose(const Matrix4x4& M, Transform& T, Vector3& scale);

class EmbreeGeometry
: public std::enable_shared_from_this<EmbreeGeometry>
{
public:
    EmbreeGeometry(EmbreeDevicePtr device = embree_default_device());

    virtual ~EmbreeGeometry();

    // embree fields
    RTCGeometry handle() const;


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
     * @brief Apply all transformation changes to data if required
     * 
     */
    virtual void apply() {};

    void disable();
    
    void enable();

    void release();

    virtual void commit();

    EmbreeSceneWPtr parent;
    unsigned int id;
    std::string name;

protected:
    EmbreeDevicePtr m_device;
    RTCGeometry m_handle;

    Transform m_T;
    Vector3 m_S;
};

} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_GEOMETRY_HPP