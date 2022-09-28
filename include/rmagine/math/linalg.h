#ifndef RMAGINE_MATH_LINALG_H
#define RMAGINE_MATH_LINALG_H

#include "types.h"

namespace rmagine
{

/**
 * @brief Composes Rotational, translational and scale parts 
 * into one 4x4 affine transformation matrix
 * 
 * @param T   transform object consisting of translational and rotational parts
 * @param scale   scale vector
 * @return Matrix4x4   composed 4x4 transformation matrix
 */
Matrix4x4 compose(const Transform& T, const Vector3& scale);

Matrix4x4 compose(const Transform& T, const Matrix3x3& S);

/**
 * @brief Decomposes Affine Transformation Matrix 
 * (including Scale) into its Translation, Rotation and Scale components
 * 
 * @param M 4x4 affine transformation matrix
 * @param T transform object consisting of translational and rotational parts
 * @param S 3x3 scale matrix
 */
void decompose(const Matrix4x4& M, Transform& T, Matrix3x3& S);

/**
 * @brief Decomposes Affine Transformation Matrix 
 * (including Scale) into its Translation, Rotation and Scale components
 * 
 * @param M 4x4 affine transformation matrix
 * @param T transform object consisting of translational and rotational parts
 * @param scale scale vector
 */
void decompose(const Matrix4x4& M, Transform& T, Vector3& scale);

} // namespace rmagine

#endif // RMAGINE_MATH_MATH_LINALG_H