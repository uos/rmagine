#ifndef IMAGINE_TYPES_SENSOR_MODELS_H
#define IMAGINE_TYPES_SENSOR_MODELS_H

#include <imagine/math/types.h>
#include <cstdint>
#include <math.h>

#include <imagine/types/SharedFunctions.hpp>
#include <imagine/types/Memory.hpp>

namespace imagine
{

struct Interval {
    float min;
    float max;

    IMAGINE_INLINE_FUNCTION
    float invalidValue() const
    {
        return max + 1.0;
    }

    IMAGINE_INLINE_FUNCTION
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

    IMAGINE_INLINE_FUNCTION
    float max() const
    {
        // return value at last array entry
        return min + static_cast<float>(size - 1) * inc;
    }

    IMAGINE_INLINE_FUNCTION
    float getValue(uint32_t id) const
    {
        return min + static_cast<float>(id) * inc;
    }

    IMAGINE_INLINE_FUNCTION
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

    IMAGINE_INLINE_FUNCTION
    uint32_t getWidth() const
    {
        return theta.size;
    }

    IMAGINE_INLINE_FUNCTION
    uint32_t getHeight() const
    {
        return phi.size;
    }

    IMAGINE_INLINE_FUNCTION
    uint32_t size() const
    {
        return getWidth() * getHeight();
    }

    IMAGINE_INLINE_FUNCTION
    float getPhi(uint32_t phi_id) const
    {
        return phi.getValue(phi_id);
    }

    IMAGINE_INLINE_FUNCTION
    float getTheta(uint32_t theta_id) const
    {
        return theta.getValue(theta_id);
    }

    IMAGINE_INLINE_FUNCTION
    Vector getRay(uint32_t phi_id, uint32_t theta_id) const
    {
        const float phi_ = getPhi(phi_id);
        const float theta_ = getTheta(theta_id);
        return {cosf(phi_) * cosf(theta_), cosf(phi_) * sinf(theta_), sinf(phi_)};
    }

    IMAGINE_INLINE_FUNCTION
    Vector getOrigin(uint32_t phi_id, uint32_t theta_id) const 
    {
        return {0.0, 0.0, 0.0};
    }

    IMAGINE_INLINE_FUNCTION
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

    IMAGINE_INLINE_FUNCTION
    uint32_t getWidth() const
    {
        return width;
    }

    IMAGINE_INLINE_FUNCTION
    uint32_t getHeight() const
    {
        return height;
    }

    IMAGINE_INLINE_FUNCTION
    uint32_t size() const
    {
        return getWidth() * getHeight();
    }

    IMAGINE_INLINE_FUNCTION
    Vector getRayOptical(uint32_t vid, uint32_t hid) const
    {
        // pX = fx * X + cx
        // pY = fy * Y + cy
        // X = (pX - cx) / fx
        const float pX = (static_cast<float>(hid) - c[0]) / f[0];
        const float pY = (static_cast<float>(vid) - c[1]) / f[1];
        
        const Vector dir_optical = {pX, pY, 1.0};
        return dir_optical.normalized();
    }

    IMAGINE_INLINE_FUNCTION
    Vector getRay(uint32_t vid, uint32_t hid) const
    {
        const Vector dir_optical = getRayOptical(vid, hid);
        //  z -> x
        // -y -> z
        // -x -> y
        return {dir_optical.z, -dir_optical.x, -dir_optical.y};
    }

    IMAGINE_INLINE_FUNCTION
    Vector getOrigin(uint32_t phi_id, uint32_t theta_id) const 
    {
        return {0.0, 0.0, 0.0};
    }

    IMAGINE_INLINE_FUNCTION
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
    uint32_t width;
    uint32_t height;

    // maximum and minimum allowed range
    Interval range;

    Vector orig;
    Memory<Vector, MemT> rays;

    IMAGINE_INLINE_FUNCTION
    uint32_t getWidth() const 
    {
        return width;
    }

    IMAGINE_INLINE_FUNCTION
    uint32_t getHeight() const 
    {
        return height;
    }

    IMAGINE_INLINE_FUNCTION
    uint32_t size() const 
    {
        return getWidth() * getHeight();
    }

    IMAGINE_INLINE_FUNCTION
    uint32_t getBufferId(uint32_t vid, uint32_t hid) const 
    {
        return vid * getWidth() + hid;
    }

    IMAGINE_INLINE_FUNCTION
    Vector getOrigin(uint32_t vid, uint32_t hid) const 
    {
        return orig;
    }

    IMAGINE_INLINE_FUNCTION
    Vector getRay(uint32_t vid, uint32_t hid) const 
    {
        return rays[getBufferId(vid, hid)];
    }
};

using O1DnModel = O1DnModel_<RAM>;

template<typename MemT>
struct OnDnModel_ {
    uint32_t width;
    uint32_t height;

    // maximum and minimum allowed range
    Interval range;

    Memory<Vector, MemT> orig;
    Memory<Vector, MemT> rays;


    IMAGINE_INLINE_FUNCTION
    uint32_t getWidth() const 
    {
        return width;
    }

    IMAGINE_INLINE_FUNCTION
    uint32_t getHeight() const 
    {
        return height;
    }

    IMAGINE_INLINE_FUNCTION
    uint32_t size() const 
    {
        return getWidth() * getHeight();
    }


    IMAGINE_INLINE_FUNCTION
    uint32_t getBufferId(uint32_t vid, uint32_t hid) const 
    {
        return vid * getWidth() + hid;
    }

    IMAGINE_INLINE_FUNCTION
    Vector getOrigin(uint32_t vid, uint32_t hid) const 
    {
        return orig[getBufferId(vid, hid)];
    }

    IMAGINE_INLINE_FUNCTION
    Vector getRay(uint32_t vid, uint32_t hid) const 
    {
        return rays[getBufferId(vid, hid)];
    }

};

using OnDnModel = OnDnModel_<RAM>;

} // namespace imagine

#endif // IMAGINE_TYPES_SENSOR_MODELS_H