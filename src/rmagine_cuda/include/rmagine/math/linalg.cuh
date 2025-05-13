#ifndef RMAGINE_MATH_LINALG_CUH
#define RMAGINE_MATH_LINALG_CUH

#include <rmagine/math/types.h>

namespace rmagine
{

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

} // namespace rmagine

#endif // RMAGINE_MATH_LINALG_CUH