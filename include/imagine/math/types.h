#ifndef IMAGINE_MATH_TYPES_H
#define IMAGINE_MATH_TYPES_H

#include <math.h>
#include <stdint.h>

#include <imagine/types/SharedFunctions.hpp>

namespace imagine
{

// Forward declarations
struct Vector2;
struct Vector3;
struct Quaternion;
struct Matrix3x3;
struct Matrix4x4;
struct Transform;

/**
 * @brief Vector2 class with functions
 * 
 */
struct Vector2 {
    // DATA
    float x;
    float y;

    // FUNCTIONS
    IMAGINE_INLINE_FUNCTION
    Vector2 add(const Vector2& b) const
    {
        return {x + b.x, y + b.y};
    }

    IMAGINE_INLINE_FUNCTION
    void addInplace(const Vector2& b)
    {
        x += b.x;
        y += b.y;
    }

    IMAGINE_INLINE_FUNCTION
    Vector2 sub(const Vector2& b) const
    {
        return {x - b.x, y - b.y};
    }

    IMAGINE_INLINE_FUNCTION
    void subInplace(const Vector2& b)
    {
        x -= b.x;
        y -= b.y;
    }

    /**
     * @brief Multiply quaternion
     * 
     * @param q2 
     * @return Quaternion 
     */
    IMAGINE_INLINE_FUNCTION
    float dot(const Vector2& b) const 
    {
        return x * b.x + y * b.y; 
    }

    /**
     * @brief product
     */
    IMAGINE_INLINE_FUNCTION
    float mult(const Vector2& b) const
    {
        return dot(b);
    }

    IMAGINE_INLINE_FUNCTION
    Vector2 mult(const float& s) const 
    {
        return {x * s, y * s};
    }

    IMAGINE_INLINE_FUNCTION
    void multInplace(const float& s) 
    {
        x *= s;
        y *= s;
    }

    IMAGINE_INLINE_FUNCTION
    Vector2 div(const float& s) const 
    {
        return {x / s, y / s};
    }

    IMAGINE_INLINE_FUNCTION
    void divInplace(const float& s) 
    {
        x /= s;
        y /= s;
    }

    IMAGINE_INLINE_FUNCTION
    float l2normSquared() const
    {
        return x*x + y*y;
    }

    IMAGINE_INLINE_FUNCTION
    float l2norm() const 
    {
        return sqrtf(l2normSquared());
    }

    IMAGINE_INLINE_FUNCTION
    float sum() const 
    {
        return x + y;
    }

    IMAGINE_INLINE_FUNCTION
    float prod() const 
    {
        return x * y;
    }

    IMAGINE_INLINE_FUNCTION
    float l1norm() const 
    {
        return fabs(x) + fabs(y);
    }

    IMAGINE_INLINE_FUNCTION
    void setZeros()
    {
        x = 0.0;
        y = 0.0;
    }

    // OPERATORS
    IMAGINE_INLINE_FUNCTION
    Vector2 operator+(const Vector2& b) const
    {
        return add(b);
    }

    IMAGINE_INLINE_FUNCTION
    void operator+=(const Vector2& b)
    {
        addInplace(b);
    }

    IMAGINE_INLINE_FUNCTION
    Vector2 operator-(const Vector2& b) const
    {
        return sub(b);
    }

    IMAGINE_INLINE_FUNCTION
    void operator-=(const Vector2& b)
    {
        subInplace(b);
    }

    IMAGINE_INLINE_FUNCTION
    float operator*(const Vector2& b) const
    {
        return mult(b);
    }

    IMAGINE_INLINE_FUNCTION
    Vector2 operator*(const float& s) const 
    {
        return mult(s);
    }

    IMAGINE_INLINE_FUNCTION
    void operator*=(const float& s) 
    {
        multInplace(s);
    }

    IMAGINE_INLINE_FUNCTION
    Vector2 operator/(const float& s) const 
    {
        return div(s);
    }

    IMAGINE_INLINE_FUNCTION
    void operator/=(const float& s) 
    {
        divInplace(s);
    }
};

struct Vector3 {
    float x;
    float y;
    float z;

    // FUNCTIONS
    IMAGINE_INLINE_FUNCTION
    Vector3 add(const Vector3& b) const
    {
        return {x + b.x, y + b.y, z + b.z};
    }

    IMAGINE_INLINE_FUNCTION
    void addInplace(const Vector3& b)
    {
        x += b.x;
        y += b.y;
        z += b.z;
    }

    IMAGINE_INLINE_FUNCTION
    Vector3 sub(const Vector3& b) const
    {
        return {x - b.x, y - b.y, z - b.z};
    }

    IMAGINE_INLINE_FUNCTION
    Vector3 negation() const
    {
        return {-x, -y, -z};
    }

    IMAGINE_INLINE_FUNCTION
    void negate() 
    {
        x = -x;
        y = -y;
        z = -z;
    }

    IMAGINE_INLINE_FUNCTION
    void subInplace(const Vector3& b)
    {
        x -= b.x;
        y -= b.y;
        z -= b.z;
    }

    /**
     * @brief Multiply quaternion
     * 
     * @param q2 
     * @return Quaternion 
     */
    IMAGINE_INLINE_FUNCTION
    float dot(const Vector3& b) const 
    {
        return x * b.x + y * b.y + z * b.z;
    }

    /**
     * @brief product
     */
    IMAGINE_INLINE_FUNCTION
    float mult(const Vector3& b) const
    {
        return dot(b);
    }

    IMAGINE_INLINE_FUNCTION
    Vector3 mult(const float& s) const 
    {
        return {x * s, y * s, z * s};
    }

    IMAGINE_INLINE_FUNCTION
    void multInplace(const float& s) 
    {
        x *= s;
        y *= s;
        z *= s;
    }

    IMAGINE_INLINE_FUNCTION
    Vector3 div(const float& s) const 
    {
        return {x / s, y / s, z / s};
    }

    IMAGINE_INLINE_FUNCTION
    void divInplace(const float& s) 
    {
        x /= s;
        y /= s;
        z /= s;
    }

    IMAGINE_INLINE_FUNCTION
    float l2normSquared() const
    {
        return x*x + y*y + z*z;
    }

    IMAGINE_INLINE_FUNCTION
    float l2norm() const 
    {
        return sqrtf(l2normSquared());
    }

    IMAGINE_INLINE_FUNCTION
    float sum() const 
    {
        return x + y + z;
    }

    IMAGINE_INLINE_FUNCTION
    float prod() const 
    {
        return x * y * z;
    }

    IMAGINE_INLINE_FUNCTION
    float l1norm() const 
    {
        return fabs(x) + fabs(y) + fabs(z);
    }

    IMAGINE_INLINE_FUNCTION
    Vector3 normalized() const 
    {
        return div(l2norm());
    }

    IMAGINE_INLINE_FUNCTION
    void normalize() 
    {
        divInplace(l2norm());
    }

    IMAGINE_INLINE_FUNCTION
    void setZeros()
    {
        x = 0.0;
        y = 0.0;
        z = 0.0;
    }

    IMAGINE_INLINE_FUNCTION
    Vector3 cross(const Vector3& b) const
    {
        return {
            y * b.z - z * b.y,
            z * b.x - x * b.z,
            x * b.y - y * b.x
        };
    }

    // OPERATORS
    IMAGINE_INLINE_FUNCTION
    Vector3 operator+(const Vector3& b) const
    {
        return add(b);
    }

    IMAGINE_INLINE_FUNCTION
    void operator+=(const Vector3& b)
    {
        addInplace(b);
    }

    IMAGINE_INLINE_FUNCTION
    Vector3 operator-(const Vector3& b) const
    {
        return sub(b);
    }

    IMAGINE_INLINE_FUNCTION
    void operator-=(const Vector3& b)
    {
        subInplace(b);
    }

    IMAGINE_INLINE_FUNCTION
    Vector3 operator-() const
    {
        return negation();
    }

    IMAGINE_INLINE_FUNCTION
    float operator*(const Vector3& b) const
    {
        return mult(b);
    }

    IMAGINE_INLINE_FUNCTION
    Vector3 operator*(const float& s) const 
    {
        return mult(s);
    }

    IMAGINE_INLINE_FUNCTION
    void operator*=(const float& s) 
    {
        multInplace(s);
    }

    IMAGINE_INLINE_FUNCTION
    Vector3 operator/(const float& s) const 
    {
        return div(s);
    }

    IMAGINE_INLINE_FUNCTION
    void operator/=(const float& s) 
    {
        divInplace(s);
    }
};

using Vector = Vector3;
using Point = Vector3;

struct Quaternion
{
    // DATA
    float x;
    float y;
    float z;
    float w;

    /**
     * @brief Invert a Quaternion
     * 
     * @param q 
     * @return Quaternion 
     */
    IMAGINE_INLINE_FUNCTION
    Quaternion inv() const 
    {
        return {-x, -y, -z, w};
    }

    IMAGINE_INLINE_FUNCTION
    void invInplace()
    {
        x = -x;
        y = -y;
        z = -z;
    }

    /**
     * @brief Multiply quaternion
     * 
     * @param q2 
     * @return Quaternion 
     */
    IMAGINE_INLINE_FUNCTION
    Quaternion mult(const Quaternion& q2) const 
    {
        return {w*q2.x + x*q2.w + y*q2.z - z*q2.y,
                w*q2.y - x*q2.z + y*q2.w + z*q2.x,
                w*q2.z + x*q2.y - y*q2.x + z*q2.w,
                w*q2.w - x*q2.x - y*q2.y - z*q2.z};
    }

    IMAGINE_INLINE_FUNCTION
    void multInplace(const Quaternion& q2) 
    {
        const Quaternion tmp = mult(q2);
        x = tmp.x;
        y = tmp.y;
        z = tmp.z;
        w = tmp.w;
    }

    /**
     * @brief Rotate a vector with a quaternion
     * 
     * @param q 
     * @param p 
     * @return Vector 
     */
    IMAGINE_INLINE_FUNCTION
    Vector3 mult(const Vector3& p) const
    {
        const Quaternion P{p.x, p.y, p.z, 0.0};
        const Quaternion PT = this->mult(P).mult(inv());
        return {PT.x, PT.y, PT.z};
    }

    IMAGINE_INLINE_FUNCTION
    void setIdentity()
    {
        x = 0.0;
        y = 0.0;
        z = 0.0;
        w = 1.0;
    }

    // OPERATORS
    IMAGINE_INLINE_FUNCTION
    Quaternion operator~() const 
    {
        return inv();
    }

    IMAGINE_INLINE_FUNCTION
    Quaternion operator*(const Quaternion& q2) const 
    {
        return mult(q2);
    }

    IMAGINE_INLINE_FUNCTION
    Vector3 operator*(const Vector3& p) const
    {
        return mult(p);
    }

    IMAGINE_INLINE_FUNCTION
    void operator*=(const Quaternion& q2) 
    {
        multInplace(q2);
    }
};

// 16*4 Byte Transform struct 
struct Transform {
    // DATA
    Quaternion R;
    Vector t;
    uint32_t stamp;

    // FUNCTIONS
    IMAGINE_INLINE_FUNCTION
    Transform inv() const
    {
        Transform Tinv;
        Tinv.R = ~R;
        Tinv.t = -(Tinv.R * t);
        return Tinv;
    }

    /**
     * @brief Transform of type T3 = this*T2
     * 
     * @param T2 Other transform
     */
    IMAGINE_INLINE_FUNCTION
    Transform mult(const Transform& T2) const
    {
        // P_ = R1 * (R2 * P + t2) + t1;
        Transform T3;
        T3.t = R * T2.t;
        T3.R = R * T2.R;
        T3.t += t;
        return T3;
    }

    /**
     * @brief Transform of type this = this * T2
     * 
     * @param T2 Other transform
     */
    IMAGINE_INLINE_FUNCTION
    void multInplace(const Transform& T2)
    {
        // P_ = R1 * (R2 * P + t2) + t1;
        // P_ = R1 * R2 * P + R1 * t2 + t1
        // =>
        // t_ = R1 * t2 + t1
        // R_ = R1 * R2
        t = R * T2.t + t;
        R = R * T2.R;
    }

    IMAGINE_INLINE_FUNCTION
    Vector3 mult(const Vector3& v) const
    {
        return R * v + t;
    }

    IMAGINE_INLINE_FUNCTION
    void setIdentity()
    {
        R.setIdentity();
        t.setZeros();
    }

    // OPERATORS
    IMAGINE_INLINE_FUNCTION
    Transform operator~() const
    {
        return inv();
    }

    IMAGINE_INLINE_FUNCTION
    Transform operator*(const Transform& T2) const 
    {
        return mult(T2);
    }

    IMAGINE_INLINE_FUNCTION
    void operator*=(const Transform& T2) 
    {
        multInplace(T2);
    }

    IMAGINE_INLINE_FUNCTION
    Vector3 operator*(const Vector3& v) const
    {
        return mult(v);
    }

};

// TODOs: 
// - check if Eigen::Matrix3f raw data is same
// - check if math is correct
struct Matrix3x3 {
    // DATA
    float data[3][3];
    
    // ACCESS
    IMAGINE_INLINE_FUNCTION
    float& at(unsigned int i, unsigned int j)
    {
        return data[i][j];
    }

    IMAGINE_INLINE_FUNCTION
    float at(unsigned int i, unsigned int j) const
    {
        return data[i][j];
    }   

    IMAGINE_INLINE_FUNCTION
    float& operator()(unsigned int i, unsigned int j)
    {
        return at(i,j);
    }

    IMAGINE_INLINE_FUNCTION
    float operator()(unsigned int i, unsigned int j) const
    {
        return at(i,j);
    }

    IMAGINE_INLINE_FUNCTION
    float* operator[](const unsigned int i) {
        return data[i];
    };

    IMAGINE_INLINE_FUNCTION
    const float* operator[](const unsigned int i) const {
        return data[i];
    };

    // FUNCTIONS
    IMAGINE_INLINE_FUNCTION
    Matrix3x3 transpose() const 
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

    IMAGINE_INLINE_FUNCTION
    Matrix3x3 T() const 
    {
        return transpose();
    }

    IMAGINE_INLINE_FUNCTION
    void transposeInplace()
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

    IMAGINE_INLINE_FUNCTION
    float det() const
    {
        return  at(0, 0) * (at(1, 1) * at(2, 2) - at(2, 1) * at(1, 2)) -
                at(0, 1) * (at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0)) +
                at(0, 2) * (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0));
    }

    IMAGINE_INLINE_FUNCTION
    Matrix3x3 inv() const
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

    IMAGINE_INLINE_FUNCTION
    Vector mult(const Vector& p) const
    {
        return {
            at(0,0) * p.x + at(0,1) * p.y + at(0,2) * p.z, 
            at(1,0) * p.x + at(1,1) * p.y + at(1,2) * p.z, 
            at(2,0) * p.x + at(2,1) * p.y + at(2,2) * p.z
        };
    }

    IMAGINE_INLINE_FUNCTION
    Matrix3x3 mult(const Matrix3x3& p) const
    {
        Matrix3x3 res;
        // TODO

        return res;
    }

    // OPERATORS
    IMAGINE_INLINE_FUNCTION
    Vector operator*(const Vector& p) const
    {
        return mult(p);
    }

    IMAGINE_INLINE_FUNCTION
    Matrix3x3 operator~() const
    {
        return inv();
    }
};

struct Matrix4x4 {
    float data[4][4];

    // ACCESS
    IMAGINE_INLINE_FUNCTION
    float& at(unsigned int i, unsigned int j)
    {
        return data[i][j];
    }

    IMAGINE_INLINE_FUNCTION
    float at(unsigned int i, unsigned int j) const
    {
        return data[i][j];
    }   

    IMAGINE_INLINE_FUNCTION
    float& operator()(unsigned int i, unsigned int j)
    {
        return at(i,j);
    }

    IMAGINE_INLINE_FUNCTION
    float operator()(unsigned int i, unsigned int j) const
    {
        return at(i,j);
    }

    IMAGINE_INLINE_FUNCTION
    float* operator[](const unsigned int i) {
        return data[i];
    };

    IMAGINE_INLINE_FUNCTION
    const float* operator[](const unsigned int i) const {
        return data[i];
    };

    // FUNCTIONS


    // OPERATORS

};

} // namespace imagine 

#endif // IMAGINE_MATH_TYPES_H