#ifndef RMAGINE_MATH_MATH_H
#define RMAGINE_MATH_MATH_H

#include <rmagine/math/types.h>
// self include ?!
#include <math.h>

#include <rmagine/types/SharedFunctions.hpp>

namespace rmagine
{

static RMAGINE_INLINE_FUNCTION
void set_identity(Quaternion& q)
{
    q.x = 0.0;
    q.y = 0.0;
    q.z = 0.0;
    q.w = 1.0;
}

static RMAGINE_INLINE_FUNCTION
void set_identity(Matrix3x3& M)
{
    M[0][0] = 1.0;
    M[0][1] = 0.0;
    M[0][2] = 0.0;

    M[1][0] = 0.0;
    M[1][1] = 1.0;
    M[1][2] = 0.0;

    M[2][0] = 0.0;
    M[2][1] = 0.0;
    M[2][2] = 1.0;
}

static RMAGINE_INLINE_FUNCTION
void set_identity(Matrix4x4& M)
{
    M[0][0] = 1.0;
    M[0][1] = 0.0;
    M[0][2] = 0.0;
    M[0][3] = 0.0;

    M[1][0] = 0.0;
    M[1][1] = 1.0;
    M[1][2] = 0.0;
    M[1][3] = 0.0;

    M[2][0] = 0.0;
    M[2][1] = 0.0;
    M[2][2] = 1.0;
    M[2][3] = 0.0;

    M[3][0] = 0.0;
    M[3][1] = 0.0;
    M[3][2] = 0.0;
    M[3][3] = 1.0;
}

} // namespace image

#endif // RMAGINE_MATH_MATH_H