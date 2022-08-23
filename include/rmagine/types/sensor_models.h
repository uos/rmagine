#ifndef RMAGINE_TYPES_SENSOR_MODELS_H
#define RMAGINE_TYPES_SENSOR_MODELS_H

#include <rmagine/math/types.h>
#include <cstdint>
#include <math.h>

#include <rmagine/types/SharedFunctions.hpp>
#include <rmagine/types/Memory.hpp>

namespace rmagine
{

struct Interval {
    float min;
    float max;

    RMAGINE_INLINE_FUNCTION
    float invalidValue() const
    {
        return max + 1.0;
    }

    RMAGINE_INLINE_FUNCTION
    bool inside(const float& value) const 
    {
        return (value >= min && value <= max);
    } 
};

struct DiscreteInterval
{
    // minimum
    float min;
    // increment
    float inc;
    // total number of discrete values in this interval
    uint32_t size;

    RMAGINE_INLINE_FUNCTION
    float max() const
    {
        // return value at last array entry
        return min + static_cast<float>(size - 1) * inc;
    }

    RMAGINE_INLINE_FUNCTION
    float getValue(uint32_t id) const
    {
        return min + static_cast<float>(id) * inc;
    }

    RMAGINE_INLINE_FUNCTION
    float operator[](uint32_t id) const 
    {
        return getValue(id);
    }

    RMAGINE_INLINE_FUNCTION
    bool inside(const float& value) const 
    {
        return (value >= min && value <= max());
    }

    static float IncFromMinMaxSize(float min, float max, uint32_t size)
    {
        return (max - min) / ( static_cast<float>(size - 1) );
    }
};

struct Rectangle {
    Vector2 min;
    Vector2 max;
};

struct Box {
    Vector min;
    Vector max;
};

struct SphericalModel 
{
    static constexpr char name[] = "Sphere";

    // PHI: vertical, y-rot, pitch, polar angle, height
    DiscreteInterval phi;
    // THETA: horizontal, z-rot, yaw, azimuth, width
    DiscreteInterval theta;
    // RANGE: range
    Interval range; // range is valid if <= range_max && >= range_min

    RMAGINE_INLINE_FUNCTION
    uint32_t getWidth() const
    {
        return theta.size;
    }

    RMAGINE_INLINE_FUNCTION
    uint32_t getHeight() const
    {
        return phi.size;
    }

    RMAGINE_INLINE_FUNCTION
    uint32_t size() const
    {
        return getWidth() * getHeight();
    }

    RMAGINE_INLINE_FUNCTION
    float getPhi(uint32_t phi_id) const
    {
        return phi.getValue(phi_id);
    }

    RMAGINE_INLINE_FUNCTION
    float getTheta(uint32_t theta_id) const
    {
        return theta.getValue(theta_id);
    }

    RMAGINE_INLINE_FUNCTION
    Vector getDirection(uint32_t phi_id, uint32_t theta_id) const
    {
        const float phi_ = getPhi(phi_id);
        const float theta_ = getTheta(theta_id);
        return {cosf(phi_) * cosf(theta_), cosf(phi_) * sinf(theta_), sinf(phi_)};
    }

    RMAGINE_INLINE_FUNCTION
    Vector getOrigin(uint32_t phi_id, uint32_t theta_id) const 
    {
        return {0.0, 0.0, 0.0};
    }

    RMAGINE_INLINE_FUNCTION
    uint32_t getBufferId(uint32_t phi_id, uint32_t theta_id) const 
    {
        return phi_id * theta.size + theta_id;
    }
};

using LiDARModel = SphericalModel;

struct PinholeModel {
    static constexpr char name[] = "Pinhole";

    uint32_t width;
    uint32_t height;

    // maximum and minimum allowed range
    Interval range;

    // Focal length fx and fy
    float f[2];
    // Center cx and cy
    float c[2];

    RMAGINE_INLINE_FUNCTION
    uint32_t getWidth() const
    {
        return width;
    }

    RMAGINE_INLINE_FUNCTION
    uint32_t getHeight() const
    {
        return height;
    }

    RMAGINE_INLINE_FUNCTION
    uint32_t size() const
    {
        return getWidth() * getHeight();
    }

    RMAGINE_INLINE_FUNCTION
    Vector getDirectionOptical(uint32_t vid, uint32_t hid) const
    {
        // pX = fx * X + cx
        // pY = fy * Y + cy
        // X = (pX - cx) / fx
        const float pX = (static_cast<float>(hid) - c[0]) / f[0];
        const float pY = (static_cast<float>(vid) - c[1]) / f[1];
        
        const Vector dir_optical = {pX, pY, 1.0};
        return dir_optical.normalized();
    }

    RMAGINE_INLINE_FUNCTION
    Vector getDirection(uint32_t vid, uint32_t hid) const
    {
        const Vector dir_optical = getDirectionOptical(vid, hid);
        //  z -> x
        // -y -> z
        // -x -> y
        return {dir_optical.z, -dir_optical.x, -dir_optical.y};
    }

    RMAGINE_INLINE_FUNCTION
    Vector getOrigin(uint32_t phi_id, uint32_t theta_id) const 
    {
        return {0.0, 0.0, 0.0};
    }

    RMAGINE_INLINE_FUNCTION
    uint32_t getBufferId(uint32_t vid, uint32_t hid) const 
    {
        return vid * width + hid;
    }

};
// Distortion? Fisheye / radial-tangential ? 

using CameraModel = PinholeModel;
using DepthCameraModel = PinholeModel;

// TODO: distortion
struct RadialTangentialDistortion {
    // TODO
};

struct FisheyeDistortion {

};


struct CylindricModel {
    static constexpr char name[] = "Cylinder";
    // TODO
    
};

template<typename MemT>
struct O1DnModel_ {
    static constexpr char name[] = "O1Dn";

    uint32_t width;
    uint32_t height;

    // maximum and minimum allowed range
    Interval range;

    Vector orig;
    Memory<Vector, MemT> dirs;

    RMAGINE_INLINE_FUNCTION
    uint32_t getWidth() const 
    {
        return width;
    }

    RMAGINE_INLINE_FUNCTION
    uint32_t getHeight() const 
    {
        return height;
    }

    RMAGINE_INLINE_FUNCTION
    uint32_t size() const 
    {
        return getWidth() * getHeight();
    }

    RMAGINE_INLINE_FUNCTION
    uint32_t getBufferId(uint32_t vid, uint32_t hid) const 
    {
        return vid * getWidth() + hid;
    }

    RMAGINE_INLINE_FUNCTION
    Vector getOrigin(uint32_t vid, uint32_t hid) const 
    {
        return orig;
    }

    RMAGINE_INLINE_FUNCTION
    Vector getDirection(uint32_t vid, uint32_t hid) const 
    {
        return dirs[getBufferId(vid, hid)];
    }
};

using O1DnModel = O1DnModel_<RAM>;

template<typename MemT>
struct OnDnModel_ {
    static constexpr char name[] = "OnDn";

    uint32_t width;
    uint32_t height;

    // maximum and minimum allowed range
    Interval range;

    Memory<Vector, MemT> origs;
    Memory<Vector, MemT> dirs;


    RMAGINE_INLINE_FUNCTION
    uint32_t getWidth() const 
    {
        return width;
    }

    RMAGINE_INLINE_FUNCTION
    uint32_t getHeight() const 
    {
        return height;
    }

    RMAGINE_INLINE_FUNCTION
    uint32_t size() const 
    {
        return getWidth() * getHeight();
    }


    RMAGINE_INLINE_FUNCTION
    uint32_t getBufferId(uint32_t vid, uint32_t hid) const 
    {
        return vid * getWidth() + hid;
    }

    RMAGINE_INLINE_FUNCTION
    Vector getOrigin(uint32_t vid, uint32_t hid) const 
    {
        return origs[getBufferId(vid, hid)];
    }

    RMAGINE_INLINE_FUNCTION
    Vector getDirection(uint32_t vid, uint32_t hid) const 
    {
        return dirs[getBufferId(vid, hid)];
    }

};

using OnDnModel = OnDnModel_<RAM>;


} // namespace rmagine

#endif // RMAGINE_TYPES_SENSOR_MODELS_H