#ifndef RMAGINE_MATH_TYPES_H
#define RMAGINE_MATH_TYPES_H

#include <math.h>
#include <stdint.h>

#include <rmagine/types/SharedFunctions.hpp>

namespace rmagine
{

// Forward declarations
struct Vector2;
struct Vector3;
struct EulerAngles;
struct Quaternion;
struct Transform;
struct Matrix3x3;
struct Matrix4x4;

/**
 * @brief Vector2 class with functions
 * 
 */
struct Vector2 {
    // DATA
    float x;
    float y;

    // FUNCTIONS
    RMAGINE_INLINE_FUNCTION
    Vector2 add(const Vector2& b) const;

    RMAGINE_INLINE_FUNCTION
    void addInplace(const Vector2& b);

    RMAGINE_INLINE_FUNCTION
    Vector2 sub(const Vector2& b) const;

    RMAGINE_INLINE_FUNCTION
    void subInplace(const Vector2& b);

    RMAGINE_INLINE_FUNCTION
    Vector2 negation() const;

    RMAGINE_INLINE_FUNCTION
    void negate();

    RMAGINE_INLINE_FUNCTION
    float dot(const Vector2& b) const;

    /**
     * @brief product
     */
    RMAGINE_INLINE_FUNCTION
    float mult(const Vector2& b) const;

    RMAGINE_INLINE_FUNCTION
    Vector2 mult(const float& s) const;    

    RMAGINE_INLINE_FUNCTION
    void multInplace(const float& s);

    RMAGINE_INLINE_FUNCTION
    Vector2 div(const float& s) const;

    RMAGINE_INLINE_FUNCTION
    void divInplace(const float& s);

    RMAGINE_INLINE_FUNCTION
    float l2normSquared() const;

    RMAGINE_INLINE_FUNCTION
    float l2norm() const;

    RMAGINE_INLINE_FUNCTION
    float sum() const;

    RMAGINE_INLINE_FUNCTION
    float prod() const;

    RMAGINE_INLINE_FUNCTION
    float l1norm() const;

    RMAGINE_INLINE_FUNCTION
    void setZeros();

    // OPERATORS
    RMAGINE_INLINE_FUNCTION
    Vector2 operator+(const Vector2& b) const
    {
        return add(b);
    }

    RMAGINE_INLINE_FUNCTION
    Vector2& operator+=(const Vector2& b)
    {
        addInplace(b);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Vector2 operator-(const Vector2& b) const
    {
        return sub(b);
    }

    RMAGINE_INLINE_FUNCTION
    Vector2& operator-=(const Vector2& b)
    {
        subInplace(b);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Vector2 operator-() const
    {
        return negation();
    }

    RMAGINE_INLINE_FUNCTION
    float operator*(const Vector2& b) const
    {
        return mult(b);
    }

    RMAGINE_INLINE_FUNCTION
    Vector2 operator*(const float& s) const 
    {
        return mult(s);
    }

    RMAGINE_INLINE_FUNCTION
    Vector2& operator*=(const float& s) 
    {
        multInplace(s);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Vector2 operator/(const float& s) const 
    {
        return div(s);
    }

    RMAGINE_INLINE_FUNCTION
    Vector2& operator/=(const float& s) 
    {
        divInplace(s);
        return *this;
    }
};

/**
 * @brief Vector3 type
 * 
 */
struct Vector3 {
    float x;
    float y;
    float z;

    // FUNCTIONS
    RMAGINE_INLINE_FUNCTION
    Vector3 add(const Vector3& b) const;
    
    RMAGINE_INLINE_FUNCTION
    void addInplace(const Vector3& b);

    RMAGINE_INLINE_FUNCTION
    Vector3 sub(const Vector3& b) const;

    RMAGINE_INLINE_FUNCTION
    Vector3 negation() const;

    RMAGINE_INLINE_FUNCTION
    void negate();

    RMAGINE_INLINE_FUNCTION
    void subInplace(const Vector3& b);

    RMAGINE_INLINE_FUNCTION
    float dot(const Vector3& b) const;

    RMAGINE_INLINE_FUNCTION
    Vector3 cross(const Vector3& b) const;

    RMAGINE_INLINE_FUNCTION
    float mult(const Vector3& b) const;

    RMAGINE_INLINE_FUNCTION
    Vector3 mult(const float& s) const;

    RMAGINE_INLINE_FUNCTION
    void multInplace(const float& s);

    RMAGINE_INLINE_FUNCTION
    Vector3 div(const float& s) const;

    RMAGINE_INLINE_FUNCTION
    void divInplace(const float& s);

    RMAGINE_INLINE_FUNCTION
    float l2normSquared() const;

    RMAGINE_INLINE_FUNCTION
    float l2norm() const; 

    RMAGINE_INLINE_FUNCTION
    float sum() const;

    RMAGINE_INLINE_FUNCTION
    float prod() const;
    
    RMAGINE_INLINE_FUNCTION
    float l1norm() const;

    RMAGINE_INLINE_FUNCTION
    Vector3 normalized() const;

    RMAGINE_INLINE_FUNCTION
    void normalize();

    RMAGINE_INLINE_FUNCTION
    void setZeros();

    // OPERATORS
    RMAGINE_INLINE_FUNCTION
    Vector3 operator+(const Vector3& b) const
    {
        return add(b);
    }

    RMAGINE_INLINE_FUNCTION
    Vector3& operator+=(const Vector3& b)
    {
        addInplace(b);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Vector3 operator-(const Vector3& b) const
    {
        return sub(b);
    }

    RMAGINE_INLINE_FUNCTION
    Vector3& operator-=(const Vector3& b)
    {
        subInplace(b);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Vector3 operator-() const
    {
        return negation();
    }

    RMAGINE_INLINE_FUNCTION
    float operator*(const Vector3& b) const
    {
        return mult(b);
    }

    RMAGINE_INLINE_FUNCTION
    Vector3 operator*(const float& s) const 
    {
        return mult(s);
    }

    RMAGINE_INLINE_FUNCTION
    Vector3& operator*=(const float& s) 
    {
        multInplace(s);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Vector3 operator/(const float& s) const 
    {
        return div(s);
    }

    RMAGINE_INLINE_FUNCTION
    Vector3 operator/=(const float& s) 
    {
        divInplace(s);
        return *this;
    }
};

using Vector = Vector3;
using Point = Vector3;

/**
 * @brief EulerAngles type
 * 
 */
struct EulerAngles
{
    float roll; // x-axis
    float pitch; // y-axis
    float yaw; // z-axis

    // Functions
    RMAGINE_INLINE_FUNCTION
    void setIdentity();

    RMAGINE_INLINE_FUNCTION
    void set(const Quaternion& q);

    RMAGINE_INLINE_FUNCTION
    void set(const Matrix3x3& M);

    RMAGINE_INLINE_FUNCTION
    Vector3 mult(const Vector3& v) const;

    // Operators
    RMAGINE_INLINE_FUNCTION
    void operator=(const Quaternion& q)
    {
        set(q);
    }

    RMAGINE_INLINE_FUNCTION
    void operator=(const Matrix3x3& M)
    {
        set(M);
    }

    RMAGINE_INLINE_FUNCTION
    Vector3 operator*(const Vector3& v) const 
    {
        return mult(v);
    }
};

/**
 * @brief Quaternion type
 * 
 */
struct Quaternion
{
    // DATA
    float x;
    float y;
    float z;
    float w;

    RMAGINE_INLINE_FUNCTION
    void setIdentity();

    /**
     * @brief Invert this Quaternion
     * 
     * @return Quaternion 
     */
    RMAGINE_INLINE_FUNCTION
    Quaternion inv() const;

    RMAGINE_INLINE_FUNCTION
    void invInplace();

    /**
     * @brief Multiply quaternion
     * 
     * @param q2 
     * @return Quaternion 
     */
    RMAGINE_INLINE_FUNCTION
    Quaternion mult(const Quaternion& q2) const;

    RMAGINE_INLINE_FUNCTION
    void multInplace(const Quaternion& q2);

    /**
     * @brief Rotate a vector with this quaternion
     * 
     * @param p 
     * @return Vector 
     */
    RMAGINE_INLINE_FUNCTION
    Vector3 mult(const Vector3& p) const;

    RMAGINE_INLINE_FUNCTION
    float l2normSquared() const;

    RMAGINE_INLINE_FUNCTION
    float l2norm() const;

    RMAGINE_INLINE_FUNCTION
    void normalize();

    RMAGINE_INLINE_FUNCTION
    void set(const Matrix3x3& M);

    RMAGINE_INLINE_FUNCTION
    void set(const EulerAngles& e);

    // TODO: Quatenrion from rotation around an axis v by an angle a
    // RMAGINE_INLINE_FUNCTION
    // void set(const Vector3& v, float a);

    // OPERATORS
    RMAGINE_INLINE_FUNCTION
    Quaternion operator~() const 
    {
        return inv();
    }

    RMAGINE_INLINE_FUNCTION
    Quaternion operator*(const Quaternion& q2) const 
    {
        return mult(q2);
    }

    RMAGINE_INLINE_FUNCTION
    Vector3 operator*(const Vector3& p) const
    {
        return mult(p);
    }

    RMAGINE_INLINE_FUNCTION
    Quaternion& operator*=(const Quaternion& q2) 
    {
        multInplace(q2);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    void operator=(const Matrix3x3& M)
    {
        set(M);
    }

    RMAGINE_INLINE_FUNCTION
    void operator=(const EulerAngles& e)
    {
        set(e);
    }
};

/**
 * @brief Transform type
 * 
 * Consists of rotational part represented as @link rmagine::Quaternion Quaternion @endlink 
 * and a translational part represented as @link rmagine::Vector3 Vector3 @endlink  
 * 
 * Additionally it contains a timestamp uint32_t
 * 
 */
struct Transform {
    // DATA
    Quaternion R;
    Vector t;
    uint32_t stamp;

    // FUNCTIONS
    RMAGINE_INLINE_FUNCTION
    void setIdentity();

    RMAGINE_INLINE_FUNCTION
    void set(const Matrix4x4& M);

    RMAGINE_INLINE_FUNCTION
    Transform inv() const;

    /**
     * @brief Transform of type T3 = this*T2
     * 
     * @param T2 Other transform
     */
    RMAGINE_INLINE_FUNCTION
    Transform mult(const Transform& T2) const;

    /**
     * @brief Transform of type this = this * T2
     * 
     * @param T2 Other transform
     */
    RMAGINE_INLINE_FUNCTION
    void multInplace(const Transform& T2);

    RMAGINE_INLINE_FUNCTION
    Vector3 mult(const Vector3& v) const;

    // OPERATORS
    RMAGINE_INLINE_FUNCTION
    void operator=(const Matrix4x4& M)
    {
        set(M);
    }

    RMAGINE_INLINE_FUNCTION
    Transform operator~() const
    {
        return inv();
    }

    RMAGINE_INLINE_FUNCTION
    Transform operator*(const Transform& T2) const 
    {
        return mult(T2);
    }

    RMAGINE_INLINE_FUNCTION
    Transform& operator*=(const Transform& T2) 
    {
        multInplace(T2);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Vector3 operator*(const Vector3& v) const
    {
        return mult(v);
    }
};

/**
 * @brief Matrix3x3 class
 * 
 * Same order than Eigen default -> Can be reinterpret-casted or mapped.
 * 
 */
struct Matrix3x3 {
    // DATA
    float data[3][3];
    
    // ACCESS
    RMAGINE_INLINE_FUNCTION
    float& at(unsigned int i, unsigned int j);

    RMAGINE_INLINE_FUNCTION
    float at(unsigned int i, unsigned int j) const;

    RMAGINE_INLINE_FUNCTION
    float& operator()(unsigned int i, unsigned int j);

    RMAGINE_INLINE_FUNCTION
    float operator()(unsigned int i, unsigned int j) const;

    RMAGINE_INLINE_FUNCTION
    float* operator[](const unsigned int i);

    RMAGINE_INLINE_FUNCTION
    const float* operator[](const unsigned int i) const;

    // FUNCTIONS
    RMAGINE_INLINE_FUNCTION
    void setIdentity();

    RMAGINE_INLINE_FUNCTION
    void setZeros();

    RMAGINE_INLINE_FUNCTION
    void setOnes();

    RMAGINE_INLINE_FUNCTION
    void set(const Quaternion& q);

    RMAGINE_INLINE_FUNCTION
    void set(const EulerAngles& e);

    RMAGINE_INLINE_FUNCTION
    Matrix3x3 transpose() const;

    RMAGINE_INLINE_FUNCTION
    Matrix3x3 T() const;

    RMAGINE_INLINE_FUNCTION
    void transposeInplace();

    RMAGINE_INLINE_FUNCTION
    float trace() const;

    RMAGINE_INLINE_FUNCTION
    float det() const;

    RMAGINE_INLINE_FUNCTION
    Matrix3x3 inv() const;

    /**
     * @brief Assuming Matrix3x3 to be a rotation matrix. then M.inv = M.transpose
     * 
     * @return Matrix3x3 
     */
    RMAGINE_INLINE_FUNCTION
    Matrix3x3 invRigid() const;

    RMAGINE_INLINE_FUNCTION
    Matrix3x3 mult(const float& s) const;

    RMAGINE_INLINE_FUNCTION
    void multInplace(const float& s);

    RMAGINE_INLINE_FUNCTION
    Vector mult(const Vector& p) const;

    RMAGINE_INLINE_FUNCTION
    Matrix3x3 mult(const Matrix3x3& M) const;

    // OPERATORS
    RMAGINE_INLINE_FUNCTION
    Matrix3x3 operator*(const float& s) const
    {
        return mult(s);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix3x3& operator*=(const float& s)
    {
        multInplace(s);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Vector operator*(const Vector& p) const
    {
        return mult(p);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix3x3 operator*(const Matrix3x3& M) const 
    {
        return mult(M);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix3x3 operator~() const
    {
        return inv();
    }

    RMAGINE_INLINE_FUNCTION
    void operator=(const Quaternion& q)
    {
        set(q);
    }

    RMAGINE_INLINE_FUNCTION
    void operator=(const EulerAngles& e)
    {
        set(e);
    }
};

/**
 * @brief Matrix4x4 type.
 * 
 * Same order as Eigen-default -> can be reinterpret-casted or mapped.
 * 
 */
struct Matrix4x4 {
    float data[4][4];

    RMAGINE_INLINE_FUNCTION
    float& at(unsigned int i, unsigned int j);

    RMAGINE_INLINE_FUNCTION
    float at(unsigned int i, unsigned int j) const;

    RMAGINE_INLINE_FUNCTION
    float& operator()(unsigned int i, unsigned int j);

    RMAGINE_INLINE_FUNCTION
    float operator()(unsigned int i, unsigned int j) const;

    RMAGINE_INLINE_FUNCTION
    float* operator[](const unsigned int i);

    RMAGINE_INLINE_FUNCTION
    const float* operator[](const unsigned int i) const;

    // FUNCTIONS
    RMAGINE_INLINE_FUNCTION
    void setIdentity();

    RMAGINE_INLINE_FUNCTION
    void setZeros();

    RMAGINE_INLINE_FUNCTION
    void setOnes();

    RMAGINE_INLINE_FUNCTION
    void set(const Transform& T);

    RMAGINE_INLINE_FUNCTION
    Matrix3x3 rotation() const;

    RMAGINE_INLINE_FUNCTION
    void setRotation(const Matrix3x3& R);

    RMAGINE_INLINE_FUNCTION
    void setRotation(const Quaternion& q);

    RMAGINE_INLINE_FUNCTION
    void setRotation(const EulerAngles& e);

    RMAGINE_INLINE_FUNCTION
    Vector translation() const;

    RMAGINE_INLINE_FUNCTION
    void setTranslation(const Vector& t);

    RMAGINE_INLINE_FUNCTION
    Matrix4x4 transpose() const;

    RMAGINE_INLINE_FUNCTION
    Matrix4x4 T() const;

    RMAGINE_INLINE_FUNCTION
    float trace() const;

    RMAGINE_INLINE_FUNCTION
    float det() const;

    RMAGINE_INLINE_FUNCTION
    Matrix4x4 inv() const ;

    /**
     * @brief Assuming Matrix4x4 to be rigid transformation. Then: (R|t)^(-1) = (R^T| -R^T t)
     * 
     * @return Matrix4x4 
     */
    RMAGINE_INLINE_FUNCTION
    Matrix4x4 invRigid();

    RMAGINE_INLINE_FUNCTION
    Matrix4x4 mult(const float& s) const;

    RMAGINE_INLINE_FUNCTION
    void multInplace(const float& s);

    RMAGINE_INLINE_FUNCTION
    Vector mult(const Vector& v) const;

    RMAGINE_INLINE_FUNCTION
    Matrix4x4 mult(const Matrix4x4& M) const;

    // OPERATORS
    RMAGINE_INLINE_FUNCTION
    void operator=(const Transform& T)
    {
        set(T);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix4x4 operator*(const float& s) const
    {
        return mult(s);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix4x4& operator*=(const float& s)
    {
        multInplace(s);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Vector operator*(const Vector& v) const 
    {
        return mult(v);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix4x4 operator*(const Matrix4x4& M) const 
    {
        return mult(M);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix4x4 operator~() const 
    {
        return inv();
    }
};


//////////////////////////////
///// INLINE IMPLEMENTATIONS
///////////////////////////////


/////////////////////
///// Vector2 ///////
/////////////////////

RMAGINE_INLINE_FUNCTION
Vector2 Vector2::add(const Vector2& b) const
{
    return {x + b.x, y + b.y};
}

RMAGINE_INLINE_FUNCTION
void Vector2::addInplace(const Vector2& b)
{
    x += b.x;
    y += b.y;
}

RMAGINE_INLINE_FUNCTION
Vector2 Vector2::sub(const Vector2& b) const
{
    return {x - b.x, y - b.y};
}

RMAGINE_INLINE_FUNCTION
void Vector2::subInplace(const Vector2& b)
{
    x -= b.x;
    y -= b.y;
}

RMAGINE_INLINE_FUNCTION
Vector2 Vector2::negation() const
{
    return {-x, -y};
}

RMAGINE_INLINE_FUNCTION
void Vector2::negate()
{
    x = -x;
    y = -y;
}

RMAGINE_INLINE_FUNCTION
float Vector2::dot(const Vector2& b) const 
{
    return x * b.x + y * b.y; 
}

RMAGINE_INLINE_FUNCTION
float Vector2::mult(const Vector2& b) const
{
    return dot(b);
}

RMAGINE_INLINE_FUNCTION
Vector2 Vector2::mult(const float& s) const 
{
    return {x * s, y * s};
}

RMAGINE_INLINE_FUNCTION
void Vector2::multInplace(const float& s) 
{
    x *= s;
    y *= s;
}

RMAGINE_INLINE_FUNCTION
Vector2 Vector2::div(const float& s) const 
{
    return {x / s, y / s};
}

RMAGINE_INLINE_FUNCTION
void Vector2::divInplace(const float& s) 
{
    x /= s;
    y /= s;
}

RMAGINE_INLINE_FUNCTION
float Vector2::l2normSquared() const
{
    return x*x + y*y;
}

RMAGINE_INLINE_FUNCTION
float Vector2::l2norm() const 
{
    return sqrtf(l2normSquared());
}

RMAGINE_INLINE_FUNCTION
float Vector2::sum() const 
{
    return x + y;
}

RMAGINE_INLINE_FUNCTION
float Vector2::prod() const 
{
    return x * y;
}

RMAGINE_INLINE_FUNCTION
float Vector2::l1norm() const 
{
    return fabs(x) + fabs(y);
}

RMAGINE_INLINE_FUNCTION
void Vector2::setZeros()
{
    x = 0.0;
    y = 0.0;
}

/////////////////////
///// Vector3 ///////
/////////////////////

RMAGINE_INLINE_FUNCTION
Vector3 Vector3::add(const Vector3& b) const
{
    return {x + b.x, y + b.y, z + b.z};
}

RMAGINE_INLINE_FUNCTION
void Vector3::addInplace(const Vector3& b)
{
    x += b.x;
    y += b.y;
    z += b.z;
}

RMAGINE_INLINE_FUNCTION
Vector3 Vector3::sub(const Vector3& b) const
{
    return {x - b.x, y - b.y, z - b.z};
}

RMAGINE_INLINE_FUNCTION
Vector3 Vector3::negation() const
{
    return {-x, -y, -z};
}

RMAGINE_INLINE_FUNCTION
void Vector3::negate() 
{
    x = -x;
    y = -y;
    z = -z;
}

RMAGINE_INLINE_FUNCTION
void Vector3::subInplace(const Vector3& b)
{
    x -= b.x;
    y -= b.y;
    z -= b.z;
}

RMAGINE_INLINE_FUNCTION
float Vector3::dot(const Vector3& b) const 
{
    return x * b.x + y * b.y + z * b.z;
}

RMAGINE_INLINE_FUNCTION
Vector3 Vector3::cross(const Vector3& b) const
{
    return {
        y * b.z - z * b.y,
        z * b.x - x * b.z,
        x * b.y - y * b.x
    };
}

RMAGINE_INLINE_FUNCTION
float Vector3::mult(const Vector3& b) const
{
    return dot(b);
}

RMAGINE_INLINE_FUNCTION
Vector3 Vector3::mult(const float& s) const 
{
    return {x * s, y * s, z * s};
}

RMAGINE_INLINE_FUNCTION
void Vector3::multInplace(const float& s) 
{
    x *= s;
    y *= s;
    z *= s;
}

RMAGINE_INLINE_FUNCTION
Vector3 Vector3::div(const float& s) const 
{
    return {x / s, y / s, z / s};
}

RMAGINE_INLINE_FUNCTION
void Vector3::divInplace(const float& s) 
{
    x /= s;
    y /= s;
    z /= s;
}

RMAGINE_INLINE_FUNCTION
float Vector3::l2normSquared() const
{
    return x*x + y*y + z*z;
}

RMAGINE_INLINE_FUNCTION
float Vector3::l2norm() const 
{
    return sqrtf(l2normSquared());
}

RMAGINE_INLINE_FUNCTION
float Vector3::sum() const 
{
    return x + y + z;
}

RMAGINE_INLINE_FUNCTION
float Vector3::prod() const 
{
    return x * y * z;
}

RMAGINE_INLINE_FUNCTION
float Vector3::l1norm() const 
{
    return fabs(x) + fabs(y) + fabs(z);
}

RMAGINE_INLINE_FUNCTION
Vector3 Vector3::normalized() const 
{
    return div(l2norm());
}

RMAGINE_INLINE_FUNCTION
void Vector3::normalize() 
{
    divInplace(l2norm());
}

RMAGINE_INLINE_FUNCTION
void Vector3::setZeros()
{
    x = 0.0;
    y = 0.0;
    z = 0.0;
}


/////////////////////
//// EulerAngles ////
/////////////////////

RMAGINE_INLINE_FUNCTION
void EulerAngles::setIdentity()
{
    roll = 0.0;
    pitch = 0.0;
    yaw = 0.0;
}

RMAGINE_INLINE_FUNCTION
void EulerAngles::set(const Quaternion& q)
{
    // TODO: check
    // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    // checked once

    // roll (x-axis)
    const float sinr_cosp = 2.0f * (q.w * q.x + q.y * q.z);
    const float cosr_cosp = 1.0f - 2.0f * (q.x * q.x + q.y * q.y);
    // pitch (y-axis)
    const float sinp = 2.0f * (q.w * q.y - q.z * q.x);
    // yaw (z-axis)
    const float siny_cosp = 2.0f * (q.w * q.z + q.x * q.y);
    const float cosy_cosp = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);


    // roll (x-axis)
    roll = atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis)
    if (fabs(sinp) >= 1.0f)
    {
        pitch = copysignf(M_PI / 2, sinp); // use 90 degrees if out of range
    } else {
        pitch = asinf(sinp);
    }

    // yaw (z-axis)
    yaw = atan2f(siny_cosp, cosy_cosp);
}

RMAGINE_INLINE_FUNCTION
void EulerAngles::set(const Matrix3x3& M)
{
    // extracted from knowledge of Matrix3x3::set(EulerAngles)
    // plus EulerAngles::set(Quaternion)
    // TODO: check. tested once: correct
    
    // roll (x-axis)
    const float sinr_cosp = -M(1,2);
    const float cosr_cosp = M(2,2);
    
    // pitch (y-axis)
    const float sinp = M(0,2);

    // yaw (z-axis)
    const float siny_cosp = -M(0,1);
    const float cosy_cosp = M(0,0);

    // roll (x-axis)
    roll = atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis)
    if (fabs(sinp) >= 1.0f)
    {
        pitch = copysignf(M_PI / 2, sinp); // use 90 degrees if out of range
    } else {
        pitch = asinf(sinp);
    }

    // yaw (z-axis)
    yaw = atan2f(siny_cosp, cosy_cosp);
}

RMAGINE_INLINE_FUNCTION
Vector3 EulerAngles::mult(const Vector3& v) const
{
    Quaternion q;
    q = *this;
    return q * v;
}

////////////////////
//// Quaternion ////
////////////////////

RMAGINE_INLINE_FUNCTION
void Quaternion::setIdentity()
{
    x = 0.0;
    y = 0.0;
    z = 0.0;
    w = 1.0;
}

RMAGINE_INLINE_FUNCTION
Quaternion Quaternion::inv() const 
{
    return {-x, -y, -z, w};
}

RMAGINE_INLINE_FUNCTION
void Quaternion::invInplace()
{
    x = -x;
    y = -y;
    z = -z;
}

RMAGINE_INLINE_FUNCTION
Quaternion Quaternion::mult(const Quaternion& q2) const 
{
    return {w*q2.x + x*q2.w + y*q2.z - z*q2.y,
            w*q2.y - x*q2.z + y*q2.w + z*q2.x,
            w*q2.z + x*q2.y - y*q2.x + z*q2.w,
            w*q2.w - x*q2.x - y*q2.y - z*q2.z};
}

RMAGINE_INLINE_FUNCTION
void Quaternion::multInplace(const Quaternion& q2) 
{
    const Quaternion tmp = mult(q2);
    x = tmp.x;
    y = tmp.y;
    z = tmp.z;
    w = tmp.w;
}

RMAGINE_INLINE_FUNCTION
Vector3 Quaternion::mult(const Vector3& p) const
{
    const Quaternion P{p.x, p.y, p.z, 0.0};
    const Quaternion PT = this->mult(P).mult(inv());
    return {PT.x, PT.y, PT.z};
}

RMAGINE_INLINE_FUNCTION
float Quaternion::l2normSquared() const 
{
    return w * w + x * x + y * y + z * z;
}

RMAGINE_INLINE_FUNCTION
float Quaternion::l2norm() const 
{
    return sqrtf(l2normSquared());
}

RMAGINE_INLINE_FUNCTION
void Quaternion::normalize()
{
    const float d = l2norm();
    x /= d;
    y /= d;
    z /= d;
    w /= d;
}

RMAGINE_INLINE_FUNCTION
void Quaternion::set(const Matrix3x3& M)
{
    // https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    // TODO: test
    // 1. test: correct
    float tr = M.trace();

    if (tr > 0) { 
        const float S = sqrtf(tr + 1.0) * 2; // S=4*qw 
        w = 0.25f * S;
        x = (M(2,1) - M(1,2)) / S;
        y = (M(0,2) - M(2,0)) / S; 
        z = (M(1,0) - M(0,1)) / S; 
    } else if ((M(0,0) > M(1,1)) && (M(0,0) > M(2,2))) { 
        const float S = sqrtf(1.0 + M(0,0) - M(1,1) - M(2,2)) * 2.0f; // S=4*qx 
        w = (M(2,1) - M(1,2)) / S;
        x = 0.25f * S;
        y = (M(0,1) + M(1,0)) / S; 
        z = (M(0,2) + M(2,0)) / S; 
    } else if (M(1,1) > M(2,2) ) { 
        const float S = sqrtf(1.0 + M(1,1) - M(0,0) - M(2,2)) * 2.0f; // S=4*qy
        w = (M(0,2) - M(2,0)) / S;
        x = (M(0,1) + M(1,0)) / S; 
        y = 0.25f * S;
        z = (M(1,2) + M(2,1)) / S; 
    } else { 
        const float S = sqrtf(1.0 + M(2,2) - M(0,0) - M(1,1)) * 2.0f; // S=4*qz
        w = (M(1,0) - M(0,1)) / S;
        x = (M(0,2) + M(2,0)) / S;
        y = (M(1,2) + M(2,1)) / S;
        z = 0.25f * S;
    }
}

RMAGINE_INLINE_FUNCTION
void Quaternion::set(const EulerAngles& e)
{
    // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    // TODO: check, 
    // 1. test: correct
    const float cr = cosf(e.roll / 2.0f);
    const float sr = sinf(e.roll / 2.0f);
    const float cp = cosf(e.pitch / 2.0f);
    const float sp = sinf(e.pitch / 2.0f);
    const float cy = cosf(e.yaw / 2.0f);
    const float sy = sinf(e.yaw / 2.0f);

    w = cr * cp * cy + sr * sp * sy;
    x = sr * cp * cy - cr * sp * sy;
    y = cr * sp * cy + sr * cp * sy;
    z = cr * cp * sy - sr * sp * cy;
}

////////////////////
//// Transform /////
////////////////////

RMAGINE_INLINE_FUNCTION
void Transform::setIdentity()
{
    R.setIdentity();
    t.setZeros();
}

RMAGINE_INLINE_FUNCTION
void Transform::set(const Matrix4x4& M)
{
    R = M.rotation();
    t = M.translation();
}

RMAGINE_INLINE_FUNCTION
Transform Transform::inv() const
{
    Transform Tinv;
    Tinv.R = ~R;
    Tinv.t = -(Tinv.R * t);
    return Tinv;
}

RMAGINE_INLINE_FUNCTION
Transform Transform::mult(const Transform& T2) const
{
    // P_ = R1 * (R2 * P + t2) + t1;
    Transform T3;
    T3.t = R * T2.t;
    T3.R = R * T2.R;
    T3.t += t;
    return T3;
}

RMAGINE_INLINE_FUNCTION
void Transform::multInplace(const Transform& T2)
{
    // P_ = R1 * (R2 * P + t2) + t1;
    // P_ = R1 * R2 * P + R1 * t2 + t1
    // =>
    // t_ = R1 * t2 + t1
    // R_ = R1 * R2
    t = R * T2.t + t;
    R = R * T2.R;
}

RMAGINE_INLINE_FUNCTION
Vector3 Transform::mult(const Vector3& v) const
{
    return R * v + t;
}


////////////////////
//// Matrix3x3 /////
////////////////////

RMAGINE_INLINE_FUNCTION
float& Matrix3x3::at(unsigned int i, unsigned int j)
{
    return data[j][i];
}

RMAGINE_INLINE_FUNCTION
float Matrix3x3::at(unsigned int i, unsigned int j) const
{
    return data[j][i];
}   

RMAGINE_INLINE_FUNCTION
float& Matrix3x3::operator()(unsigned int i, unsigned int j)
{
    return at(i,j);
}

RMAGINE_INLINE_FUNCTION
float Matrix3x3::operator()(unsigned int i, unsigned int j) const
{
    return at(i,j);
}

RMAGINE_INLINE_FUNCTION
float* Matrix3x3::operator[](const unsigned int i) 
{
    return data[i];
};

RMAGINE_INLINE_FUNCTION
const float* Matrix3x3::operator[](const unsigned int i) const 
{
    return data[i];
};

RMAGINE_INLINE_FUNCTION
void Matrix3x3::setIdentity()
{
    at(0,0) = 1.0f;
    at(0,1) = 0.0f;
    at(0,2) = 0.0f;
    at(1,0) = 0.0f;
    at(1,1) = 1.0f;
    at(1,2) = 0.0f;
    at(2,0) = 0.0f;
    at(2,1) = 0.0f;
    at(2,2) = 1.0f;
}

RMAGINE_INLINE_FUNCTION
void Matrix3x3::setZeros()
{
    at(0,0) = 0.0f;
    at(0,1) = 0.0f;
    at(0,2) = 0.0f;
    at(1,0) = 0.0f;
    at(1,1) = 0.0f;
    at(1,2) = 0.0f;
    at(2,0) = 0.0f;
    at(2,1) = 0.0f;
    at(2,2) = 0.0f;
}

RMAGINE_INLINE_FUNCTION
void Matrix3x3::setOnes()
{
    at(0,0) = 1.0f;
    at(0,1) = 1.0f;
    at(0,2) = 1.0f;
    at(1,0) = 1.0f;
    at(1,1) = 1.0f;
    at(1,2) = 1.0f;
    at(2,0) = 1.0f;
    at(2,1) = 1.0f;
    at(2,2) = 1.0f;
}

RMAGINE_INLINE_FUNCTION
void Matrix3x3::set(const Quaternion& q)
{
    // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    
    // inhomogeneous expression
    // only for unit quaternions

    at(0,0) = 2.0f * (q.w * q.w + q.x * q.x) - 1.0f;
    at(0,1) = 2.0f * (q.x * q.y - q.w * q.z);
    at(0,2) = 2.0f * (q.x * q.z + q.w * q.y);
    at(1,0) = 2.0f * (q.x * q.y + q.w * q.z);
    at(1,1) = 2.0f * (q.w * q.w + q.y * q.y) - 1.0f;
    at(1,2) = 2.0f * (q.y * q.z - q.w * q.x);
    at(2,0) = 2.0f * (q.x * q.z - q.w * q.y);
    at(2,1) = 2.0f * (q.y * q.z + q.w * q.x);
    at(2,2) = 2.0f * (q.w * q.w + q.z * q.z) - 1.0f;

    // TODO:
    // homogeneous expession


    // TESTED
}

RMAGINE_INLINE_FUNCTION
void Matrix3x3::set(const EulerAngles& e)
{
    // Wrong?
    // TODO check
    // 1. test: correct

    const float cA = cosf(e.roll);
    const float sA = sinf(e.roll);
    const float cB = cosf(e.pitch);
    const float sB = sinf(e.pitch);
    const float cC = cosf(e.yaw);
    const float sC = sinf(e.yaw);

    at(0,0) =  cB * cC;
    at(0,1) = -cB * sC;
    at(0,2) =  sB;
   
    at(1,0) =  sA * sB * cC + cA * sC;
    at(1,1) = -sA * sB * sC + cA * cC;
    at(1,2) = -sA * cB;
    
    at(2,0) = -cA * sB * cC + sA * sC;
    at(2,1) =  cA * sB * sC + sA * cC;
    at(2,2) =  cA * cB;
}

RMAGINE_INLINE_FUNCTION
Matrix3x3 Matrix3x3::transpose() const 
{
    Matrix3x3 ret;

    ret(0,0) = at(0,0);
    ret(0,1) = at(1,0);
    ret(0,2) = at(2,0);
    
    ret(1,0) = at(0,1);
    ret(1,1) = at(1,1);
    ret(1,2) = at(2,1);

    ret(2,0) = at(0,2);
    ret(2,1) = at(1,2);
    ret(2,2) = at(2,2);

    return ret;
}

RMAGINE_INLINE_FUNCTION
Matrix3x3 Matrix3x3::T() const 
{
    return transpose();
}

RMAGINE_INLINE_FUNCTION
void Matrix3x3::transposeInplace()
{
    // use only one float as additional memory
    float swap_mem;
    // can we do this without additional memory?

    swap_mem = at(0,1);
    at(0,1) = at(1,0);
    at(1,0) = swap_mem;

    swap_mem = at(0,2);
    at(0,2) = at(2,0);
    at(2,0) = swap_mem;

    swap_mem = at(1,2);
    at(1,2) = at(2,1);
    at(2,1) = swap_mem;
}

RMAGINE_INLINE_FUNCTION
float Matrix3x3::trace() const
{
    return at(0, 0) + at(1, 1) + at(2, 2);
}

RMAGINE_INLINE_FUNCTION
float Matrix3x3::det() const
{
    return  at(0, 0) * (at(1, 1) * at(2, 2) - at(2, 1) * at(1, 2)) -
            at(0, 1) * (at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0)) +
            at(0, 2) * (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0));
}

RMAGINE_INLINE_FUNCTION
Matrix3x3 Matrix3x3::inv() const
{
    Matrix3x3 ret;

    const float invdet = 1 / det();

    ret(0, 0) = (at(1, 1) * at(2, 2) - at(2, 1) * at(1, 2)) * invdet;
    ret(0, 1) = (at(0, 2) * at(2, 1) - at(0, 1) * at(2, 2)) * invdet;
    ret(0, 2) = (at(0, 1) * at(1, 2) - at(0, 2) * at(1, 1)) * invdet;
    ret(1, 0) = (at(1, 2) * at(2, 0) - at(1, 0) * at(2, 2)) * invdet;
    ret(1, 1) = (at(0, 0) * at(2, 2) - at(0, 2) * at(2, 0)) * invdet;
    ret(1, 2) = (at(1, 0) * at(0, 2) - at(0, 0) * at(1, 2)) * invdet;
    ret(2, 0) = (at(1, 0) * at(2, 1) - at(2, 0) * at(1, 1)) * invdet;
    ret(2, 1) = (at(2, 0) * at(0, 1) - at(0, 0) * at(2, 1)) * invdet;
    ret(2, 2) = (at(0, 0) * at(1, 1) - at(1, 0) * at(0, 1)) * invdet;

    return ret;
}

/**
    * @brief Assuming Matrix3x3 to be a rotation matrix. then M.inv = M.transpose
    * 
    * @return Matrix3x3 
    */
RMAGINE_INLINE_FUNCTION
Matrix3x3 Matrix3x3::invRigid() const 
{
    return T();
}

RMAGINE_INLINE_FUNCTION
Matrix3x3 Matrix3x3::mult(const float& s) const
{
    Matrix3x3 ret;
    ret(0,0) = at(0,0) * s;
    ret(0,1) = at(0,1) * s;
    ret(0,2) = at(0,2) * s;
    ret(1,0) = at(1,0) * s;
    ret(1,1) = at(1,1) * s;
    ret(1,2) = at(1,2) * s;
    ret(2,0) = at(2,0) * s;
    ret(2,1) = at(2,1) * s;
    ret(2,2) = at(2,2) * s;
    return ret;
}

RMAGINE_INLINE_FUNCTION
void Matrix3x3::multInplace(const float& s)
{
    at(0,0) *= s;
    at(0,1) *= s;
    at(0,2) *= s;
    at(1,0) *= s;
    at(1,1) *= s;
    at(1,2) *= s;
    at(2,0) *= s;
    at(2,1) *= s;
    at(2,2) *= s;
}

RMAGINE_INLINE_FUNCTION
Vector Matrix3x3::mult(const Vector& p) const
{
    return {
        at(0,0) * p.x + at(0,1) * p.y + at(0,2) * p.z, 
        at(1,0) * p.x + at(1,1) * p.y + at(1,2) * p.z, 
        at(2,0) * p.x + at(2,1) * p.y + at(2,2) * p.z
    };
}

RMAGINE_INLINE_FUNCTION
Matrix3x3 Matrix3x3::mult(const Matrix3x3& M) const
{
    Matrix3x3 res;
    res.setZeros();
    for (unsigned int row = 0; row < 3; row++) {
        for (unsigned int col = 0; col < 3; col++) {
            for (unsigned int inner = 0; inner < 3; inner++) {
                res(row,col) += at(row, inner) * M(inner, col);
            }
        }
    }
    return res;
}

////////////////////
//// Matrix4x4 /////
////////////////////

RMAGINE_INLINE_FUNCTION
float& Matrix4x4::at(unsigned int i, unsigned int j)
{
    return data[j][i];
}

RMAGINE_INLINE_FUNCTION
float Matrix4x4::at(unsigned int i, unsigned int j) const
{
    return data[j][i];
}   

RMAGINE_INLINE_FUNCTION
float& Matrix4x4::operator()(unsigned int i, unsigned int j)
{
    return at(i,j);
}

RMAGINE_INLINE_FUNCTION
float Matrix4x4::operator()(unsigned int i, unsigned int j) const
{
    return at(i,j);
}

RMAGINE_INLINE_FUNCTION
float* Matrix4x4::operator[](const unsigned int i) 
{
    return data[i];
};

RMAGINE_INLINE_FUNCTION
const float* Matrix4x4::operator[](const unsigned int i) const 
{
    return data[i];
};

// FUNCTIONS
RMAGINE_INLINE_FUNCTION
void Matrix4x4::setIdentity()
{
    at(0,0) = 1.0;
    at(0,1) = 0.0;
    at(0,2) = 0.0;
    at(0,3) = 0.0;
    at(1,0) = 0.0;
    at(1,1) = 1.0;
    at(1,2) = 0.0;
    at(1,3) = 0.0;
    at(2,0) = 0.0;
    at(2,1) = 0.0;
    at(2,2) = 1.0;
    at(2,3) = 0.0;
    at(3,0) = 0.0;
    at(3,1) = 0.0;
    at(3,2) = 0.0;
    at(3,3) = 1.0;
}

RMAGINE_INLINE_FUNCTION
void Matrix4x4::setZeros()
{
    at(0,0) = 0.0;
    at(0,1) = 0.0;
    at(0,2) = 0.0;
    at(0,3) = 0.0;
    at(1,0) = 0.0;
    at(1,1) = 0.0;
    at(1,2) = 0.0;
    at(1,3) = 0.0;
    at(2,0) = 0.0;
    at(2,1) = 0.0;
    at(2,2) = 0.0;
    at(2,3) = 0.0;
    at(3,0) = 0.0;
    at(3,1) = 0.0;
    at(3,2) = 0.0;
    at(3,3) = 0.0;
}

RMAGINE_INLINE_FUNCTION
void Matrix4x4::setOnes()
{
    at(0,0) = 1.0;
    at(0,1) = 1.0;
    at(0,2) = 1.0;
    at(0,3) = 1.0;
    at(1,0) = 1.0;
    at(1,1) = 1.0;
    at(1,2) = 1.0;
    at(1,3) = 1.0;
    at(2,0) = 1.0;
    at(2,1) = 1.0;
    at(2,2) = 1.0;
    at(2,3) = 1.0;
    at(3,0) = 1.0;
    at(3,1) = 1.0;
    at(3,2) = 1.0;
    at(3,3) = 1.0;
}

RMAGINE_INLINE_FUNCTION
void Matrix4x4::set(const Transform& T)
{
    setIdentity();
    setRotation(T.R);
    setTranslation(T.t);
}

RMAGINE_INLINE_FUNCTION
Matrix3x3 Matrix4x4::rotation() const
{
    Matrix3x3 R;
    R(0,0) = at(0,0);
    R(0,1) = at(0,1);
    R(0,2) = at(0,2);
    R(1,0) = at(1,0);
    R(1,1) = at(1,1);
    R(1,2) = at(1,2);
    R(2,0) = at(2,0);
    R(2,1) = at(2,1);
    R(2,2) = at(2,2);
    return R;
}

RMAGINE_INLINE_FUNCTION
void Matrix4x4::setRotation(const Matrix3x3& R)
{
    at(0,0) = R(0,0);
    at(0,1) = R(0,1);
    at(0,2) = R(0,2);
    at(1,0) = R(1,0);
    at(1,1) = R(1,1);
    at(1,2) = R(1,2);
    at(2,0) = R(2,0);
    at(2,1) = R(2,1);
    at(2,2) = R(2,2);
}

RMAGINE_INLINE_FUNCTION
void Matrix4x4::setRotation(const Quaternion& q)
{
    Matrix3x3 R;
    R = q;
    setRotation(R);
}

RMAGINE_INLINE_FUNCTION
void Matrix4x4::setRotation(const EulerAngles& e)
{
    Matrix3x3 R;
    R = e;
    setRotation(R);
}

RMAGINE_INLINE_FUNCTION
Vector Matrix4x4::translation() const
{
    return {at(0,3), at(1,3), at(2,3)};
}

RMAGINE_INLINE_FUNCTION
void Matrix4x4::setTranslation(const Vector& t)
{
    at(0,3) = t.x;
    at(1,3) = t.y;
    at(2,3) = t.z;
}

RMAGINE_INLINE_FUNCTION
Matrix4x4 Matrix4x4::transpose() const 
{
    Matrix4x4 ret;

    ret(0,0) = at(0,0);
    ret(0,1) = at(1,0);
    ret(0,2) = at(2,0);
    ret(0,3) = at(3,0);
    
    ret(1,0) = at(0,1);
    ret(1,1) = at(1,1);
    ret(1,2) = at(2,1);
    ret(1,3) = at(3,1);

    ret(2,0) = at(0,2);
    ret(2,1) = at(1,2);
    ret(2,2) = at(2,2);
    ret(2,3) = at(3,2);

    ret(3,0) = at(0,3);
    ret(3,1) = at(1,3);
    ret(3,2) = at(2,3);
    ret(3,3) = at(3,3);

    return ret;
}

RMAGINE_INLINE_FUNCTION
Matrix4x4 Matrix4x4::T() const 
{
    return transpose();
}

RMAGINE_INLINE_FUNCTION
float Matrix4x4::trace() const
{
    return at(0,0) + at(1,1) + at(2,2) + at(3,3);
}

RMAGINE_INLINE_FUNCTION
float Matrix4x4::det() const 
{
    // TODO: check
    const float A2323 = at(2,2) * at(3,3) - at(2,3) * at(3,2);
    const float A1323 = at(2,1) * at(3,3) - at(2,3) * at(3,1);
    const float A1223 = at(2,1) * at(3,2) - at(2,2) * at(3,1);
    const float A0323 = at(2,0) * at(3,3) - at(2,3) * at(3,0);
    const float A0223 = at(2,0) * at(3,2) - at(2,2) * at(3,0);
    const float A0123 = at(2,0) * at(3,1) - at(2,1) * at(3,0);
    const float A2313 = at(1,2) * at(3,3) - at(1,3) * at(3,2);
    const float A1313 = at(1,1) * at(3,3) - at(1,3) * at(3,1);
    const float A1213 = at(1,1) * at(3,2) - at(1,2) * at(3,1);
    const float A2312 = at(1,2) * at(2,3) - at(1,3) * at(2,2);
    const float A1312 = at(1,1) * at(2,3) - at(1,3) * at(2,1);
    const float A1212 = at(1,1) * at(2,2) - at(1,2) * at(2,1);
    const float A0313 = at(1,0) * at(3,3) - at(1,3) * at(3,0);
    const float A0213 = at(1,0) * at(3,2) - at(1,2) * at(3,0);
    const float A0312 = at(1,0) * at(2,3) - at(1,3) * at(2,0);
    const float A0212 = at(1,0) * at(2,2) - at(1,2) * at(2,0);
    const float A0113 = at(1,0) * at(3,1) - at(1,1) * at(3,0);
    const float A0112 = at(1,0) * at(2,1) - at(1,1) * at(2,0);

    return  at(0,0) * ( at(1,1) * A2323 - at(1,2) * A1323 + at(1,3) * A1223 ) 
            - at(0,1) * ( at(1,0) * A2323 - at(1,2) * A0323 + at(1,3) * A0223 ) 
            + at(0,2) * ( at(1,0) * A1323 - at(1,1) * A0323 + at(1,3) * A0123 ) 
            - at(0,3) * ( at(1,0) * A1223 - at(1,1) * A0223 + at(1,2) * A0123 );;
}

RMAGINE_INLINE_FUNCTION
Matrix4x4 Matrix4x4::inv() const 
{
    // https://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
    // answer of willnode at Jun 8 '17 at 23:09

    const float A2323 = at(2,2) * at(3,3) - at(2,3) * at(3,2);
    const float A1323 = at(2,1) * at(3,3) - at(2,3) * at(3,1);
    const float A1223 = at(2,1) * at(3,2) - at(2,2) * at(3,1);
    const float A0323 = at(2,0) * at(3,3) - at(2,3) * at(3,0);
    const float A0223 = at(2,0) * at(3,2) - at(2,2) * at(3,0);
    const float A0123 = at(2,0) * at(3,1) - at(2,1) * at(3,0);
    const float A2313 = at(1,2) * at(3,3) - at(1,3) * at(3,2);
    const float A1313 = at(1,1) * at(3,3) - at(1,3) * at(3,1);
    const float A1213 = at(1,1) * at(3,2) - at(1,2) * at(3,1);
    const float A2312 = at(1,2) * at(2,3) - at(1,3) * at(2,2);
    const float A1312 = at(1,1) * at(2,3) - at(1,3) * at(2,1);
    const float A1212 = at(1,1) * at(2,2) - at(1,2) * at(2,1);
    const float A0313 = at(1,0) * at(3,3) - at(1,3) * at(3,0);
    const float A0213 = at(1,0) * at(3,2) - at(1,2) * at(3,0);
    const float A0312 = at(1,0) * at(2,3) - at(1,3) * at(2,0);
    const float A0212 = at(1,0) * at(2,2) - at(1,2) * at(2,0);
    const float A0113 = at(1,0) * at(3,1) - at(1,1) * at(3,0);
    const float A0112 = at(1,0) * at(2,1) - at(1,1) * at(2,0);

    float det_  = at(0,0) * ( at(1,1) * A2323 - at(1,2) * A1323 + at(1,3) * A1223 ) 
                - at(0,1) * ( at(1,0) * A2323 - at(1,2) * A0323 + at(1,3) * A0223 ) 
                + at(0,2) * ( at(1,0) * A1323 - at(1,1) * A0323 + at(1,3) * A0123 ) 
                - at(0,3) * ( at(1,0) * A1223 - at(1,1) * A0223 + at(1,2) * A0123 ) ;

    // inv det
    det_ = 1.0f / det_;

    Matrix4x4 ret;
    ret(0,0) = det_ *   ( at(1,1) * A2323 - at(1,2) * A1323 + at(1,3) * A1223 );
    ret(0,1) = det_ * - ( at(0,1) * A2323 - at(0,2) * A1323 + at(0,3) * A1223 );
    ret(0,2) = det_ *   ( at(0,1) * A2313 - at(0,2) * A1313 + at(0,3) * A1213 );
    ret(0,3) = det_ * - ( at(0,1) * A2312 - at(0,2) * A1312 + at(0,3) * A1212 );
    ret(1,0) = det_ * - ( at(1,0) * A2323 - at(1,2) * A0323 + at(1,3) * A0223 );
    ret(1,1) = det_ *   ( at(0,0) * A2323 - at(0,2) * A0323 + at(0,3) * A0223 );
    ret(1,2) = det_ * - ( at(0,0) * A2313 - at(0,2) * A0313 + at(0,3) * A0213 );
    ret(1,3) = det_ *   ( at(0,0) * A2312 - at(0,2) * A0312 + at(0,3) * A0212 );
    ret(2,0) = det_ *   ( at(1,0) * A1323 - at(1,1) * A0323 + at(1,3) * A0123 );
    ret(2,1) = det_ * - ( at(0,0) * A1323 - at(0,1) * A0323 + at(0,3) * A0123 );
    ret(2,2) = det_ *   ( at(0,0) * A1313 - at(0,1) * A0313 + at(0,3) * A0113 );
    ret(2,3) = det_ * - ( at(0,0) * A1312 - at(0,1) * A0312 + at(0,3) * A0112 );
    ret(3,0) = det_ * - ( at(1,0) * A1223 - at(1,1) * A0223 + at(1,2) * A0123 );
    ret(3,1) = det_ *   ( at(0,0) * A1223 - at(0,1) * A0223 + at(0,2) * A0123 );
    ret(3,2) = det_ * - ( at(0,0) * A1213 - at(0,1) * A0213 + at(0,2) * A0113 );
    ret(3,3) = det_ *   ( at(0,0) * A1212 - at(0,1) * A0212 + at(0,2) * A0112 );

    return ret;
}

RMAGINE_INLINE_FUNCTION
Matrix4x4 Matrix4x4::invRigid()
{
    Matrix4x4 ret;
    ret.setIdentity();

    Matrix3x3 R = rotation();
    Vector t = translation();

    R.transposeInplace();
    ret.setRotation(R);
    ret.setTranslation(- (R * t) );

    return ret;
}

RMAGINE_INLINE_FUNCTION
Matrix4x4 Matrix4x4::mult(const float& s) const
{
    Matrix4x4 ret;

    ret(0,0) = at(0,0) * s;
    ret(0,1) = at(0,1) * s;
    ret(0,2) = at(0,2) * s;
    ret(0,3) = at(0,3) * s;

    ret(1,0) = at(1,0) * s;
    ret(1,1) = at(1,1) * s;
    ret(1,2) = at(1,2) * s;
    ret(1,3) = at(1,3) * s;

    ret(2,0) = at(2,0) * s;
    ret(2,1) = at(2,1) * s;
    ret(2,2) = at(2,2) * s;
    ret(2,3) = at(2,3) * s;

    ret(3,0) = at(3,0) * s;
    ret(3,1) = at(3,1) * s;
    ret(3,2) = at(3,2) * s;
    ret(3,3) = at(3,3) * s;

    return ret;
}

RMAGINE_INLINE_FUNCTION
void Matrix4x4::multInplace(const float& s)
{
    at(0,0) *= s;
    at(0,1) *= s;
    at(0,2) *= s;
    at(0,3) *= s;

    at(1,0) *= s;
    at(1,1) *= s;
    at(1,2) *= s;
    at(1,3) *= s;

    at(2,0) *= s;
    at(2,1) *= s;
    at(2,2) *= s;
    at(2,3) *= s;

    at(3,0) *= s;
    at(3,1) *= s;
    at(3,2) *= s;
    at(3,3) *= s;
}

RMAGINE_INLINE_FUNCTION
Vector Matrix4x4::mult(const Vector& v) const
{
    return {
        at(0,0) * v.x + at(0,1) * v.y + at(0,2) * v.z + at(0,3),
        at(1,0) * v.x + at(1,1) * v.y + at(1,2) * v.z + at(1,3),
        at(2,0) * v.x + at(2,1) * v.y + at(2,2) * v.z + at(2,3)
    };
}

RMAGINE_INLINE_FUNCTION
Matrix4x4 Matrix4x4::mult(const Matrix4x4& M) const 
{
    Matrix4x4 res;
    res.setZeros();

    for (unsigned int row = 0; row < 4; row++) {
        for (unsigned int col = 0; col < 4; col++) {
            for (unsigned int inner = 0; inner < 4; inner++) {
                res(row,col) += at(row,inner) * M(inner,col);
            }
        }
    }

    return res;
}

// Static Functions

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

} // namespace rmagine 

#endif // RMAGINE_MATH_TYPES_H