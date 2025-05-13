/*
 * Copyright (c) 2022, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file
 * 
 * @brief Sensor models
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */


#ifndef RMAGINE_TYPES_SENSOR_MODELS_H
#define RMAGINE_TYPES_SENSOR_MODELS_H

#include <rmagine/math/types.h>
#include <cstdint>
#include <math.h>

#include <rmagine/types/shared_functions.h>
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
    uint32_t getSize() const 
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

    RMAGINE_INLINE_FUNCTION
    Vector2u getPixelCoord(uint32_t buffer_id) const
    {
        return {buffer_id % theta.size, buffer_id / theta.size};
    }

    // slice horizontal line. vertical is not currently not possible because of memory layout
    template<typename DataT, typename MemT>
    MemoryView<DataT, MemT> getRow(const MemoryView<DataT, MemT>& mem, uint32_t vid) const 
    {
        return mem.slice(vid * getWidth(), (vid+1) * getWidth());
    }

    // for RAM we can access single elements of a buffer
    template<typename DataT>
    DataT& getPixelValue(MemoryView<DataT, RAM>& mem, uint32_t vid, uint32_t hid) const 
    {
        return mem[getBufferId(vid, hid)];
    }

    template<typename DataT>
    DataT getPixelValue(const MemoryView<DataT, RAM>& mem, uint32_t vid, uint32_t hid) const 
    {
        return mem[getBufferId(vid, hid)];
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
    uint32_t getSize() const 
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
        return dir_optical.normalize();
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

    RMAGINE_INLINE_FUNCTION
    Vector2u getPixelCoord(uint32_t buffer_id) const
    {
        return {buffer_id % width, buffer_id / width};
    }

    // slice horizontal line. vertical is not currently not possible because of memory layout
    template<typename DataT, typename MemT>
    MemoryView<DataT, MemT> getRow(const MemoryView<DataT, MemT>& mem, uint32_t vid) const 
    {
        return mem.slice(vid * getWidth(), (vid+1) * getWidth());
    }

    // for RAM we can access single elements of a buffer
    template<typename DataT>
    DataT& getPixelValue(MemoryView<DataT, RAM>& mem, uint32_t vid, uint32_t hid) const 
    {
        return mem[getBufferId(vid, hid)];
    }

    template<typename DataT>
    DataT getPixelValue(const MemoryView<DataT, RAM>& mem, uint32_t vid, uint32_t hid) const 
    {
        return mem[getBufferId(vid, hid)];
    }
};
// Distortion? Fisheye / radial-tangential ? 

using CameraModel = PinholeModel;
using DepthCameraModel = PinholeModel;

// TODO: distortion
// struct RadialTangentialDistortion {
//     // TODO
// };

// struct FisheyeDistortion {

// };


// struct CylindricModel {
//     static constexpr char name[] = "Cylinder";
//     // TODO
    
// };

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
    uint32_t getSize() const 
    {
        return getWidth() * getHeight();
    }

    RMAGINE_INLINE_FUNCTION
    uint32_t size() const 
    {
        return getWidth() * getHeight();
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

    RMAGINE_INLINE_FUNCTION
    uint32_t getBufferId(uint32_t vid, uint32_t hid) const 
    {
        return vid * getWidth() + hid;
    }

    RMAGINE_INLINE_FUNCTION
    Vector2u getPixelCoord(uint32_t buffer_id) const
    {
        return {buffer_id % width, buffer_id / width};
    }

    // slice horizontal line. vertical is not currently not possible because of memory layout
    template<typename DataT, typename MemT_>
    MemoryView<DataT, MemT_> getRow(const MemoryView<DataT, MemT_>& mem, uint32_t vid) const 
    {
        return mem.slice(vid * getWidth(), (vid+1) * getWidth());
    }

    // for RAM we can access single elements of a buffer
    template<typename DataT>
    DataT& getPixelValue(MemoryView<DataT, RAM>& mem, uint32_t vid, uint32_t hid) const 
    {
        return mem[getBufferId(vid, hid)];
    }

    template<typename DataT>
    DataT getPixelValue(const MemoryView<DataT, RAM>& mem, uint32_t vid, uint32_t hid) const 
    {
        return mem[getBufferId(vid, hid)];
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
  uint32_t getSize() const 
  {
    return getWidth() * getHeight();
  }

  RMAGINE_INLINE_FUNCTION
  uint32_t size() const 
  {
    return getWidth() * getHeight();
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

  RMAGINE_INLINE_FUNCTION
  uint32_t getBufferId(uint32_t vid, uint32_t hid) const 
  {
    return vid * getWidth() + hid;
  }

  RMAGINE_INLINE_FUNCTION
  Vector2u getPixelCoord(uint32_t buffer_id) const
  {
    return {buffer_id % width, buffer_id / width};
  }

  // slice horizontal line. vertical is not currently not possible because of memory layout
  template<typename DataT, typename MemT_>
  MemoryView<DataT, MemT_> getRow(MemoryView<DataT, MemT_>& mem, uint32_t vid) const 
  {
    return mem.slice(vid * getWidth(), (vid+1) * getWidth());
  }

  // for CPU we can access single elements of a buffer
  template<typename DataT>
  DataT& getPixelValue(MemoryView<DataT, RAM>& mem, uint32_t vid, uint32_t hid) const 
  {
    return mem[getBufferId(vid, hid)];
  }

  template<typename DataT>
  DataT getPixelValue(const MemoryView<DataT, RAM>& mem, uint32_t vid, uint32_t hid) const 
  {
    return mem[getBufferId(vid, hid)];
  }
};

using OnDnModel = OnDnModel_<RAM>;


template<typename ModelT, typename DataT, typename MemT>
MemoryView<DataT, MemT> slice( 
    const MemoryView<DataT, MemT>& mem,
    const ModelT& model,
    const uint32_t pose_id)
{
  return mem.slice(model.getSize() * pose_id, model.getSize() * (pose_id + 1));
}

/**
 * Use this to mark a class that requires a certain sensor model to operate
 * This is useful when you have a base interface and want to check if 
 * the implementation requires a certain sensor model:
 * 
 * @code
 * if(auto model_setter = std::dynamic_pointer_cast<rm::ModelSetter<rm::O1DnModel> >(base_class_ptr))
 * {
 *   // RCC required sensor model
 *   model_setter->setModel(sensor_model_);
 * } 
 * @endcode
 * 
 */
template<typename ModelT>
class ModelSetter
{
public:
  virtual void setModel(const ModelT&) = 0;
};

} // namespace rmagine

#endif // RMAGINE_TYPES_SENSOR_MODELS_H