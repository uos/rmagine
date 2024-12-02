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

template<typename DataT>
struct Gaussian1D_;

template<typename DataT>
struct Gaussian2D_;

template<typename DataT>
struct Gaussian3D_;

template<typename DataT>
struct CrossStatistics_;


using Vector2f = Vector2_<float>;
using Vector2u = Vector2_<uint32_t>;
using Vector2i = Vector2_<int32_t>;
using Vector3f = Vector3_<float>;
using Matrix2x2f = Matrix_<float, 2, 2>;
using Matrix3x3f = Matrix_<float, 3, 3>;
using Matrix4x4f = Matrix_<float, 4, 4>;
using Quaternionf = Quaternion_<float>;
using EulerAnglesf = EulerAngles_<float>;
using Transformf = Transform_<float>;
using AABBf = AABB_<float>;
using Gaussian1Df = Gaussian1D_<float>;
using Gaussian2Df = Gaussian2D_<float>;
using Gaussian3Df = Gaussian3D_<float>;
using CrossStatisticsf = CrossStatistics_<float>;

using Vector2d = Vector2_<double>;
using Vector3d = Vector3_<double>;
using Matrix2x2d = Matrix_<double, 2, 2>;
using Matrix3x3d = Matrix_<double, 3, 3>;
using Matrix4x4d = Matrix_<double, 4, 4>;
using Quaterniond = Quaternion_<double>;
using EulerAnglesd = EulerAngles_<double>;
using Transformd = Transform_<double>;
using AABBd = AABB_<double>;
using Gaussian1Dd = Gaussian1D_<double>;
using Gaussian2Dd = Gaussian2D_<double>;
using Gaussian3Dd = Gaussian3D_<double>;
using CrossStatisticsd = CrossStatistics_<double>;

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
using Gaussian1D = Gaussian1D_<DefaultFloatType>;
using Gaussian2D = Gaussian2D_<DefaultFloatType>;
using Gaussian3D = Gaussian3D_<DefaultFloatType>;
using CrossStatistics = CrossStatistics_<DefaultFloatType>;

// aliases
using Vector = Vector3;
using Point = Vector;

// @amock TODO: how to define a pixel? unsigned or signed?
// - projection operations can result in negative pixels
// using Pixel = Vector2u;
// using Pixel = Vector2i;


} // namespace rmagine

#endif // RMAGINE_MATH_DEFINITIONS_H