#ifndef RMAGINE_MATH_LINALG_H
#define RMAGINE_MATH_LINALG_H

#include "types.h"

namespace rmagine
{

Matrix4x4 compose(const Transform& T, const Vector3& scale);


void decompose(const Matrix4x4& M, Transform& T, Matrix3x3& S);
void decompose(const Matrix4x4& M, Transform& T, Vector3& scale);



} // namespace rmagine

#endif // RMAGINE_MATH_MATH_LINALG_H