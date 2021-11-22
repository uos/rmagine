#ifndef IMAGINE_MATH_MATH_H
#define IMAGINE_MATH_MATH_H

#include <imagine/math/types.h>
// self include ?!
#include <math.h>

#include <imagine/types/SharedFunctions.hpp>

namespace imagine
{

/**
 * @brief Apply Matrix4x4 to a vector
 * 
 * @param q 
 * @param p 
 * @return Vector 
 */
static IMAGINE_INLINE_FUNCTION
Vector mult(const Matrix4x4& M, const Vector& p)
{
    return {
        M[0][0] * p.x + M[0][1] * p.y + M[0][2] * p.z + M[0][3], 
        M[1][0] * p.x + M[1][1] * p.y + M[1][2] * p.z + M[1][3], 
        M[2][0] * p.x + M[2][1] * p.y + M[2][2] * p.z + M[2][3]
    };
}

static IMAGINE_INLINE_FUNCTION
Vector operator*(const Matrix4x4& M, const Vector& p)
{
    return mult(M, p);
}

static IMAGINE_INLINE_FUNCTION
Matrix3x3 transpose(const Matrix3x3& M)
{
    Matrix3x3 ret;
    
    ret[0][0] = M[0][0];
    ret[0][1] = M[1][0];
    ret[0][2] = M[2][0];

    ret[1][0] = M[0][1];
    ret[1][1] = M[1][1];
    ret[1][2] = M[2][1];

    ret[2][0] = M[0][2];
    ret[2][1] = M[1][2];
    ret[2][2] = M[2][2];

    return ret;
}

static IMAGINE_INLINE_FUNCTION
Matrix4x4 transpose(const Matrix4x4& M)
{
    Matrix4x4 ret;
    
    ret[0][0] = M[0][0];
    ret[0][1] = M[1][0];
    ret[0][2] = M[2][0];
    ret[0][3] = M[3][0];

    ret[1][0] = M[0][1];
    ret[1][1] = M[1][1];
    ret[1][2] = M[2][1];
    ret[1][3] = M[3][1];

    ret[2][0] = M[0][2];
    ret[2][1] = M[1][2];
    ret[2][2] = M[2][2];
    ret[2][3] = M[2][3];

    ret[3][0] = M[0][3];
    ret[3][1] = M[1][3];
    ret[3][2] = M[2][3];
    ret[3][3] = M[2][3];

    return ret;
}

static IMAGINE_INLINE_FUNCTION
void set_identity(Quaternion& q)
{
    q.x = 0.0;
    q.y = 0.0;
    q.z = 0.0;
    q.w = 1.0;
}

static IMAGINE_INLINE_FUNCTION
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

static IMAGINE_INLINE_FUNCTION
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

#endif // IMAGINE_MATH_MATH_H