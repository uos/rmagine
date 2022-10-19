#ifndef RMAGINE_MATH_DEFINITIONS_H
#define RMAGINE_MATH_DEFINITIONS_H

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


template<typename DataT>
struct Vector2_;

template<typename DataT>
struct Vector3_;

template<typename DataT>
struct EulerAngles_;

template<typename DataT>
struct Quaternion_;

template<typename DataT>
struct Transform_;

template<typename DataT, unsigned int Rows, unsigned int Cols>
struct Matrix_;

template<typename DataT>
struct AABB_;


using Vector2f = Vector2_<float>;
using Vector3f = Vector3_<float>;
using Matrix2x2f = Matrix_<float, 2, 2>;
using Matrix3x3f = Matrix_<float, 3, 3>;
using Matrix4x4f = Matrix_<float, 4, 4>;

using Vector2d = Vector2_<double>;
using Vector3d = Vector3_<double>;
using Matrix2x2d = Matrix_<double, 2, 2>;
using Matrix3x3d = Matrix_<double, 3, 3>;
using Matrix4x4d = Matrix_<double, 4, 4>;


#define DEFAULT_FP_PRECISION 32

using DefaultFloatType = float;

// default types
using Vector3 = Vector3_<DefaultFloatType>;
using Vector2 = Vector2_<DefaultFloatType>;
using Matrix2x2 = Matrix_<DefaultFloatType, 2, 2>;
using Matrix3x3 = Matrix_<DefaultFloatType, 3, 3>;
using Matrix4x4 = Matrix_<DefaultFloatType, 4, 4>;
using Quaternion = Quaternion_<DefaultFloatType>;
using EulerAngles = EulerAngles_<DefaultFloatType>;
using Transform = Transform_<DefaultFloatType>;
using AABB = AABB_<DefaultFloatType>;

// aliases
using Vector = Vector3;
using Point = Vector;



// struct Vector2;
// struct Vector3;
// struct EulerAngles;
// struct Quaternion;
// struct Transform;
// struct Matrix3x3;
// struct Matrix4x4;
// struct AABB;

} // namespace rmagine

#endif // RMAGINE_MATH_DEFINITIONS_H