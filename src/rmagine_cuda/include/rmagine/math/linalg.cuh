#ifndef RMAGINE_MATH_LINALG_CUH
#define RMAGINE_MATH_LINALG_CUH

#include <rmagine/math/types.h>

namespace rmagine
{

namespace cuda
{

/**
 * @brief Composes Rotational, translational and scale parts 
 * into one 4x4 affine transformation matrix
 * 
 * @param T   transform object consisting of translational and rotational parts
 * @param scale   scale vector
 * @return Matrix4x4   composed 4x4 transformation matrix
 */
__device__
Matrix4x4 compose(const Transform& T, const Vector3& scale);

__device__
Matrix4x4 compose(const Transform& T, const Matrix3x3& S);

/**
 * @brief linear inter- or extrapolate between A and B with a factor
 * 
 * Examples:
 * - if fac=0.0 it is exactly A
 * - if fac=0.5 it is exactly between A and B
 * - if fac=1.0 it is exactly B
 * - if fac=2.0 it it goes on a (B-A) length from B (extrapolation)
*/
__device__
Quaternion polate(const Quaternion& A, const Quaternion& B, float fac);

/**
 * @brief linear inter- or extrapolate between A and B with a factor
 * 
 * Examples:
 * - if fac=0.0 it is exactly A
 * - if fac=0.5 it is exactly between A and B
 * - if fac=1.0 it is exactly B
 * - if fac=2.0 it it goes on a (B-A) length from B (extrapolation)
*/
__device__
Transform polate(const Transform& A, const Transform& B, float fac);

/**
 * @brief SVD that can be used for both CPU and GPU (Cuda kernels)
 *
 */
__device__
void svd(
    const Matrix3x3& A,
    Matrix3x3& U,
    Matrix3x3& W,
    Matrix3x3& V
);

__device__
void svd(
    const Matrix3x3& A, 
    Matrix3x3& U,
    Vector3& w,
    Matrix3x3& V
);

/**
 * @brief computes the optimal transformation according to Umeyama's algorithm 
 * 
 * Note: sometimes referred to as Kabsch/Umeyama
 * 
 * @param n_meas: if == 0: Resulting Transform is set to identity. Otherwise the standard Umeyama algorithm is performed
 * 
 */
__device__
Transform umeyama_transform(
    const Vector3& d,
    const Vector3& m,
    const Matrix3x3& C,
    const unsigned int n_meas = 1);
/**
 * @brief computes the optimal transformation according to Umeyama's algorithm 
 * 
 * Note: sometimes referred to as Kabsch/Umeyama
 * 
 * @param n_meas: if == 0: Resulting Transform is set to identity. Otherwise the standard Umeyama algorithm is performed
 * 
 */
__device__
Transform umeyama_transform(
    const CrossStatistics& stats);

} // namespace cuda

} // namespace rmagine

#endif // RMAGINE_MATH_LINALG_CUH