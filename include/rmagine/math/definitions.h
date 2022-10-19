#ifndef RMAGINE_MATH_DEFINITIONS_OLD_H
#define RMAGINE_MATH_DEFINITIONS_OLD_H

#include <math.h>
#include <float.h>
#include <stdint.h>

namespace rmagine
{

#define __UINT_MAX__ (__INT_MAX__ * 2U + 1U)

#define DEG_TO_RAD      0.017453292519943295
#define DEG_TO_RAD_F    0.017453292519943295f
#define RAD_TO_DEG      57.29577951308232
#define RAD_TO_DEG_F    57.29577951308232f

// Forward declarations
struct Vector2;
struct Vector3;
struct EulerAngles;
struct Quaternion;
struct Transform;
struct Matrix3x3;
struct Matrix4x4;
struct AABB;

} // namespace rmagine

#endif // RMAGINE_MATH_DEFINITIONS_H