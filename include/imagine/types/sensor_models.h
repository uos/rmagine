#ifndef IMAGINE_TYPES_SENSOR_MODELS_H
#define IMAGINE_TYPES_SENSOR_MODELS_H

#include "types.h"
#include <cstdint>
#include <math.h>

namespace imagine
{

struct SphericalModel 
{
    // PHI: vertical, y-rot, pitch, polar angle 
    DiscreteInterval phi;
    // THETA: horizontal, z-rot, yaw, azimuth
    DiscreteInterval theta;
    // RANGE: range
    Interval range; // range is valid if <= range_max && >= range_min

    #ifdef __CUDA_ARCH__
    __inline__ __host__ __device__
    #else 
    inline 
    #endif    
    Vector getRay(uint32_t phi_id, uint32_t theta_id) const
    {
        const float phi_ = phi.min + static_cast<float>(phi_id) * phi.step;
        const float theta_ = theta.min + static_cast<float>(theta_id) * theta.step;
        // return {0.0, 0.0, 0.0};
        return {cosf(phi_) * cosf(theta_), cosf(phi_) * sinf(theta_), sinf(phi_)};
    }
};

using LiDARModel = SphericalModel;

struct CylindricModel {

};

struct PinholeModel {
    uint32_t width;
    uint32_t height;

    // maximum and minimum allowed range
    Interval range;

    // Focal length fx and fy
    float f[2];
    // Center cx and cy
    float c[2];
};
// Distortion? Fisheye / radial-tangential ? 

using CameraModel = PinholeModel;
using DepthCameraModel = PinholeModel;

struct RadialTangentialDistortion {
    // TODO
};

struct FisheyeDistortion {

};

} // namespace imagine

#endif // IMAGINE_TYPES_SENSOR_MODELS_H