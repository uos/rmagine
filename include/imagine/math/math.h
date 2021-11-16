#ifndef IMAGINE_MATH_MATH_H
#define IMAGINE_MATH_MATH_H

#include <imagine/types/types.h>
// self include ?!
#include <math.h>

#ifdef __CUDA_ARCH__
#define IMAGINE_HOST_DEVICE __inline__ __host__ __device__ 
#else
#define IMAGINE_HOST_DEVICE inline
#endif

namespace imagine
{

/**
 * @brief Invert a Quaternion
 * 
 * @param q 
 * @return Quaternion 
 */
static IMAGINE_HOST_DEVICE
Quaternion inv(const Quaternion& q)
{
    return {-q.x, -q.y, -q.z, q.w};
}

static IMAGINE_HOST_DEVICE
Quaternion operator~(const Quaternion& q)
{
    return inv(q);
}

/**
 * @brief Multiply two quaternions
 * 
 * @param q1 
 * @param q2 
 * @return Quaternion 
 */
static IMAGINE_HOST_DEVICE
Quaternion mult(const Quaternion& q1, const Quaternion& q2)
{
    return {q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y,
            q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x,
            q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w,
            q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z};
}

static IMAGINE_HOST_DEVICE
Quaternion operator*(const Quaternion& q1, const Quaternion& q2)
{
    return mult(q1, q2);
}

/**
 * @brief Rotate a vector with a quaternion
 * 
 * @param q 
 * @param p 
 * @return Vector 
 */
static IMAGINE_HOST_DEVICE
Vector mult(const Quaternion& q, const Vector& p)
{
    const Quaternion P{p.x, p.y, p.z, 0.0};
    const Quaternion PT = mult(mult(q, P), inv(q));
    return {PT.x, PT.y, PT.z};
}

static IMAGINE_HOST_DEVICE
Vector operator*(const Quaternion& q, const Vector& p)
{
    return mult(q,p);
}

/**
 * @brief Apply Matrix3x3 to a vector
 * 
 * @param q 
 * @param p 
 * @return Vector 
 */
static IMAGINE_HOST_DEVICE
Vector mult(const Matrix3x3& M, const Vector& p)
{
    return {
        M[0][0] * p.x + M[0][1] * p.y + M[0][2] * p.z, 
        M[1][0] * p.x + M[1][1] * p.y + M[1][2] * p.z, 
        M[2][0] * p.x + M[2][1] * p.y + M[2][2] * p.z
    };
}

static IMAGINE_HOST_DEVICE
Vector operator*(const Matrix3x3& M, const Vector& p)
{
    return mult(M,p);
}

/**
 * @brief Scale a vector by a scalar
 * 
 * @param v 
 * @param s 
 * @return Vector 
 */
static IMAGINE_HOST_DEVICE
Vector mult(const Vector& v, const float& s)
{
    return {v.x * s, v.y * s, v.z * s};
}

static IMAGINE_HOST_DEVICE
Vector operator*(const Vector& v, const float& s)
{
    return mult(v,s);
}

/**
 * @brief Scale a vector by a scalar inplace
 * 
 * @param v 
 * @param s 
 */
static IMAGINE_HOST_DEVICE
void multInplace(Vector& v, const float& s)
{
    v.x *= s;
    v.y *= s;
    v.z *= s;
}

/**
 * @brief Divide a vector by a scalar
 * 
 * @param v 
 * @param s 
 * @return Vector 
 */
static IMAGINE_HOST_DEVICE
Vector div(const Vector& v, const float& s)
{
    return {v.x / s, v.y / s, v.z / s};
}

static IMAGINE_HOST_DEVICE
Vector operator/(const Vector& v, const float& s)
{
    return div(v,s);
}

/**
 * @brief Divide a vector by a scalar
 * 
 * @param v 
 * @param s 
 * @return Vector 
 */
static IMAGINE_HOST_DEVICE
void divInplace(Vector& v, const float& s)
{
    v.x /= s;
    v.y /= s;
    v.z /= s;
}

/**
 * @brief Add two vectors
 * 
 * @param v1 
 * @param v2 
 * @return Vector 
 */
static IMAGINE_HOST_DEVICE
Vector add(const Vector& v1, const Vector& v2)
{
    return {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
}

static IMAGINE_HOST_DEVICE
Vector operator+(const Vector& v1, const Vector& v2)
{
    return add(v1,v2);
}

/**
 * @brief Subtract two vectors
 * 
 * @param v1 
 * @param v2 
 * @return Vector 
 */
static IMAGINE_HOST_DEVICE
Vector sub(const Vector& v1, const Vector& v2)
{
    return {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};
}

static IMAGINE_HOST_DEVICE
Vector operator-(const Vector& v1, const Vector& v2)
{
    return sub(v1,v2);
}

/**
 * @brief Transform of type T1*T2 = T3
 * 
 * @param R1 
 * @param t1 
 * @param R2 
 * @param t2 
 */
static IMAGINE_HOST_DEVICE
void transform(
    const Quaternion& R1, const Vector& t1,
    const Quaternion& R2, const Vector& t2, 
    Quaternion& R3, Vector& t3)
{
    // P_ = R1 * (R2 * P + t2) + t1;
    t3 = mult(R1, t2);
    R3 = mult(R1, R2);
    t3.x += t1.x;
    t3.y += t1.y;
    t3.z += t1.z;
}


/**
 * @brief Transform of type T2 = T1*T2
 * 
 * @param R1 
 * @param t1 
 * @param R2 
 * @param t2 
 */
static IMAGINE_HOST_DEVICE
void transformInplace(
    const Quaternion& R1, const Vector& t1, 
    Quaternion& R2, Vector& t2)
{
    // P_ = R1 * (R2 * P + t2) + t1;
    t2 = mult(R1, t2);
    R2 = mult(R1, R2);
    t2.x += t1.x;
    t2.y += t1.y;
    t2.z += t1.z;
}

/**
 * @brief Chain two transformations
 * 
 * @param T1 
 * @param T2 
 * @return Transform 
 */
static IMAGINE_HOST_DEVICE
Transform mult(const Transform& T1, const Transform& T2)
{
    Transform Tres;
    transform(T1.R, T1.t, T2.R, T2.t, Tres.R, Tres.t);
    return Tres;
}

static IMAGINE_HOST_DEVICE
Transform operator*(const Transform& T1, const Transform& T2)
{
    return mult(T1, T2);
}

/**
 * @brief Apply Transformation to a vector
 * 
 * @param T
 * @param v 
 * @return Vector 
 */
static IMAGINE_HOST_DEVICE
Vector mult(const Transform& T, const Vector& v)
{
    return add(mult(T.R, v), T.t);
}

static IMAGINE_HOST_DEVICE
Vector operator*(const Transform& T, const Vector& v)
{
    return mult(T, v);
}

static IMAGINE_HOST_DEVICE
Vector cross(const Vector& a, const Vector& b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

static IMAGINE_HOST_DEVICE
float dot(const Vector& a, const Vector& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static IMAGINE_HOST_DEVICE
float operator*(const Vector& a, const Vector& b)
{
    return dot(a, b);
}

static IMAGINE_HOST_DEVICE
float l2normSquared(const Vector& a)
{
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

static IMAGINE_HOST_DEVICE
float l2norm(const Vector& a)
{
    return sqrt(l2normSquared(a));
}

static IMAGINE_HOST_DEVICE
Vector normalized(const Vector& a)
{
    float norm = l2norm(a);
    return div(a, norm);
}

static IMAGINE_HOST_DEVICE
void normalize(Vector& a)
{
    const float norm = l2norm(a);
    divInplace(a, norm);
}

static IMAGINE_HOST_DEVICE
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

} // namespace image

#endif // IMAGINE_MATH_MATH_H