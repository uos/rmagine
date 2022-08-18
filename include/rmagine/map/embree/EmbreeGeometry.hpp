#ifndef RMAGINE_MAP_EMBREE_GEOMETRY_HPP
#define RMAGINE_MAP_EMBREE_GEOMETRY_HPP

#include <memory>

#include <embree3/rtcore.h>

#include "EmbreeDevice.hpp"
#include "embree_types.h"

#include <rmagine/math/types.h>
#include <rmagine/math/linalg.h>

namespace rmagine
{

class EmbreeGeometry
: public std::enable_shared_from_this<EmbreeGeometry>
{
public:
    EmbreeGeometry(EmbreeDevicePtr device = embree_default_device());

    virtual ~EmbreeGeometry();

    // embree fields
    void setQuality(RTCBuildQuality quality);
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
     * @brief Obtain composed matrix
     * 
     * @return Matrix4x4 
     */
    Matrix4x4 matrix() const;

    /**
     * @brief Apply all transformation changes to data if required
     * 
     */
    virtual void apply() {};

    void disable();
    
    void enable();

    void release();

    virtual void commit();

    virtual EmbreeGeometryType type() const = 0;

    EmbreeScenePtr makeScene();

    EmbreeInstancePtr instantiate();

    void cleanupParents();
    

    std::unordered_map<EmbreeSceneWPtr, unsigned int> ids();
    std::unordered_map<EmbreeSceneWPtr, unsigned int> ids() const;

    /**
     * @brief Get unique (per scene) ID
     * 
     * @param scene  scene the object was attached to
     * @return unsigned int  returns geometry id
     */
    unsigned int id(EmbreeScenePtr scene) const;

    std::unordered_set<EmbreeSceneWPtr> parents;
    std::string name;


protected:

    bool anyParentCommittedOnce() const;

    EmbreeDevicePtr m_device;
    RTCGeometry m_handle;

    Transform m_T;
    Vector3 m_S;
};

} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_GEOMETRY_HPP