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
    void setIdentity()
    {
        at(0,0) = 1.0;
        at(0,1) = 0.0;
        at(0,2) = 0.0;
        at(1,0) = 0.0;
        at(1,1) = 1.0;
        at(1,2) = 0.0;
        at(2,0) = 0.0;
        at(2,1) = 0.0;
        at(2,2) = 1.0;
    }

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

    /**
     * @brief Assuming Matrix3x3 to be a rotation matrix. then M.inv = M.transpose
     * 
     * @return Matrix3x3 
     */
    IMAGINE_INLINE_FUNCTION
    Matrix3x3 invRigid() const 
    {
        return T();
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
    Matrix3x3 mult(const Matrix3x3& M) const
    {
        Matrix3x3 res;
        // TODO

        for (unsigned int row = 0; row < 3; row++) {
            for (unsigned int col = 0; col < 3; col++) {
                for (unsigned int inner = 0; inner < 3; inner++) {
                    res(row,col) += at(row,inner) * M(inner,col);
                }
            }
        }

        return res;
    }

    // OPERATORS
    IMAGINE_INLINE_FUNCTION
    Vector operator*(const Vector& p) const
    {
        return mult(p);
    }

    IMAGINE_INLINE_FUNCTION
    Matrix3x3 operator*(const Matrix3x3& M) const 
    {
        return mult(M);
    }

    IMAGINE_INLINE_FUNCTION
    Matrix3x3 operator~() const
    {
        return inv();
    }
};

struct Matrix4x4 {
    float data[4][4];

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
    void setIdentity()
    {
        data[0][0] = 1.0;
        data[0][1] = 0.0;
        data[0][2] = 0.0;
        data[0][3] = 0.0;
        data[1][0] = 0.0;
        data[1][1] = 1.0;
        data[1][2] = 0.0;
        data[1][3] = 0.0;
        data[2][0] = 0.0;
        data[2][1] = 0.0;
        data[2][2] = 1.0;
        data[2][3] = 0.0;
        data[3][0] = 0.0;
        data[3][1] = 0.0;
        data[3][2] = 0.0;
        data[3][3] = 1.0;
    }

    IMAGINE_INLINE_FUNCTION
    Matrix4x4 transpose() const 
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

    IMAGINE_INLINE_FUNCTION
    Matrix4x4 T() const 
    {
        return transpose();
    }

    IMAGINE_INLINE_FUNCTION
    float det() const 
    {
        const float A2323 = data[2][2] * data[3][3] - data[2][3] * data[3][2];
        const float A1323 = data[2][1] * data[3][3] - data[2][3] * data[3][1];
        const float A1223 = data[2][1] * data[3][2] - data[2][2] * data[3][1];
        const float A0323 = data[2][0] * data[3][3] - data[2][3] * data[3][0];
        const float A0223 = data[2][0] * data[3][2] - data[2][2] * data[3][0];
        const float A0123 = data[2][0] * data[3][1] - data[2][1] * data[3][0];
        const float A2313 = data[1][2] * data[3][3] - data[1][3] * data[3][2];
        const float A1313 = data[1][1] * data[3][3] - data[1][3] * data[3][1];
        const float A1213 = data[1][1] * data[3][2] - data[1][2] * data[3][1];
        const float A2312 = data[1][2] * data[2][3] - data[1][3] * data[2][2];
        const float A1312 = data[1][1] * data[2][3] - data[1][3] * data[2][1];
        const float A1212 = data[1][1] * data[2][2] - data[1][2] * data[2][1];
        const float A0313 = data[1][0] * data[3][3] - data[1][3] * data[3][0];
        const float A0213 = data[1][0] * data[3][2] - data[1][2] * data[3][0];
        const float A0312 = data[1][0] * data[2][3] - data[1][3] * data[2][0];
        const float A0212 = data[1][0] * data[2][2] - data[1][2] * data[2][0];
        const float A0113 = data[1][0] * data[3][1] - data[1][1] * data[3][0];
        const float A0112 = data[1][0] * data[2][1] - data[1][1] * data[2][0];

        const float invDet  = data[0][0] * ( data[1][1] * A2323 - data[1][2] * A1323 + data[1][3] * A1223 ) 
                            - data[0][1] * ( data[1][0] * A2323 - data[1][2] * A0323 + data[1][3] * A0223 ) 
                            + data[0][2] * ( data[1][0] * A1323 - data[1][1] * A0323 + data[1][3] * A0123 ) 
                            - data[0][3] * ( data[1][0] * A1223 - data[1][1] * A0223 + data[1][2] * A0123 );

        return 1 / invDet;
    }

    IMAGINE_INLINE_FUNCTION
    Matrix4x4 inv() const 
    {
        // https://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
        // answer of willnode at Jun 8 '17 at 23:09
    
        const float A2323 = data[2][2] * data[3][3] - data[2][3] * data[3][2];
        const float A1323 = data[2][1] * data[3][3] - data[2][3] * data[3][1];
        const float A1223 = data[2][1] * data[3][2] - data[2][2] * data[3][1];
        const float A0323 = data[2][0] * data[3][3] - data[2][3] * data[3][0];
        const float A0223 = data[2][0] * data[3][2] - data[2][2] * data[3][0];
        const float A0123 = data[2][0] * data[3][1] - data[2][1] * data[3][0];
        const float A2313 = data[1][2] * data[3][3] - data[1][3] * data[3][2];
        const float A1313 = data[1][1] * data[3][3] - data[1][3] * data[3][1];
        const float A1213 = data[1][1] * data[3][2] - data[1][2] * data[3][1];
        const float A2312 = data[1][2] * data[2][3] - data[1][3] * data[2][2];
        const float A1312 = data[1][1] * data[2][3] - data[1][3] * data[2][1];
        const float A1212 = data[1][1] * data[2][2] - data[1][2] * data[2][1];
        const float A0313 = data[1][0] * data[3][3] - data[1][3] * data[3][0];
        const float A0213 = data[1][0] * data[3][2] - data[1][2] * data[3][0];
        const float A0312 = data[1][0] * data[2][3] - data[1][3] * data[2][0];
        const float A0212 = data[1][0] * data[2][2] - data[1][2] * data[2][0];
        const float A0113 = data[1][0] * data[3][1] - data[1][1] * data[3][0];
        const float A0112 = data[1][0] * data[2][1] - data[1][1] * data[2][0];

        float det   = data[0][0] * ( data[1][1] * A2323 - data[1][2] * A1323 + data[1][3] * A1223 ) 
                    - data[0][1] * ( data[1][0] * A2323 - data[1][2] * A0323 + data[1][3] * A0223 ) 
                    + data[0][2] * ( data[1][0] * A1323 - data[1][1] * A0323 + data[1][3] * A0123 ) 
                    - data[0][3] * ( data[1][0] * A1223 - data[1][1] * A0223 + data[1][2] * A0123 ) ;

        det = 1 / det;

        Matrix4x4 ret;
        ret(0,0) = det *   ( data[1][1] * A2323 - data[1][2] * A1323 + data[1][3] * A1223 );
        ret(0,1) = det * - ( data[0][1] * A2323 - data[0][2] * A1323 + data[0][3] * A1223 );
        ret(0,2) = det *   ( data[0][1] * A2313 - data[0][2] * A1313 + data[0][3] * A1213 );
        ret(0,3) = det * - ( data[0][1] * A2312 - data[0][2] * A1312 + data[0][3] * A1212 );
        ret(1,0) = det * - ( data[1][0] * A2323 - data[1][2] * A0323 + data[1][3] * A0223 );
        ret(1,1) = det *   ( data[0][0] * A2323 - data[0][2] * A0323 + data[0][3] * A0223 );
        ret(1,2) = det * - ( data[0][0] * A2313 - data[0][2] * A0313 + data[0][3] * A0213 );
        ret(1,3) = det *   ( data[0][0] * A2312 - data[0][2] * A0312 + data[0][3] * A0212 );
        ret(2,0) = det *   ( data[1][0] * A1323 - data[1][1] * A0323 + data[1][3] * A0123 );
        ret(2,1) = det * - ( data[0][0] * A1323 - data[0][1] * A0323 + data[0][3] * A0123 );
        ret(2,2) = det *   ( data[0][0] * A1313 - data[0][1] * A0313 + data[0][3] * A0113 );
        ret(2,3) = det * - ( data[0][0] * A1312 - data[0][1] * A0312 + data[0][3] * A0112 );
        ret(3,0) = det * - ( data[1][0] * A1223 - data[1][1] * A0223 + data[1][2] * A0123 );
        ret(3,1) = det *   ( data[0][0] * A1223 - data[0][1] * A0223 + data[0][2] * A0123 );
        ret(3,2) = det * - ( data[0][0] * A1213 - data[0][1] * A0213 + data[0][2] * A0113 );
        ret(3,3) = det *   ( data[0][0] * A1212 - data[0][1] * A0212 + data[0][2] * A0112 );

        return ret;
    }

    IMAGINE_INLINE_FUNCTION
    Matrix3x3 rotation() const
    {
        Matrix3x3 R;
        R(0,0) = data[0][0];
        R(0,1) = data[0][1];
        R(0,2) = data[0][2];
        R(1,0) = data[1][0];
        R(1,1) = data[1][1];
        R(1,2) = data[1][2];
        R(2,0) = data[2][0];
        R(2,1) = data[2][1];
        R(2,2) = data[2][2];
        return R;
    }

    IMAGINE_INLINE_FUNCTION
    void setRotation(const Matrix3x3& R)
    {
        data[0][0] = R(0,0);
        data[0][1] = R(0,1);
        data[0][2] = R(0,2);
        data[1][0] = R(1,0);
        data[1][1] = R(1,1);
        data[1][2] = R(1,2);
        data[2][0] = R(2,0);
        data[2][1] = R(2,1);
        data[2][2] = R(2,2);
    }

    IMAGINE_INLINE_FUNCTION
    Vector translation() const
    {
        return {data[0][3], data[1][3], data[2][3]};
    }

    IMAGINE_INLINE_FUNCTION
    void setTranslation(const Vector& t)
    {
        data[0][3] = t.x;
        data[1][3] = t.y;
        data[2][3] = t.z;
    }

    /**
     * @brief Assuming Matrix4x4 to be rigid transformation. Then: (R|t)^(-1) = (R^T| -R^T t)
     * 
     * @return Matrix4x4 
     */
    IMAGINE_INLINE_FUNCTION
    Matrix4x4 invRigid()
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

    IMAGINE_INLINE_FUNCTION
    Vector mult(const Vector& v) const
    {
        return {
            at(0,0) * v.x + at(0,1) * v.y + at(0,2) * v.z + at(0,3),
            at(1,0) * v.x + at(1,1) * v.y + at(1,2) * v.z + at(1,3),
            at(2,0) * v.x + at(2,1) * v.y + at(2,2) * v.z + at(2,3)
        };
    }

    IMAGINE_INLINE_FUNCTION
    Matrix4x4 mult(const Matrix4x4& M) const 
    {
        Matrix4x4 res;

        for (unsigned int row = 0; row < 4; row++) {
            for (unsigned int col = 0; col < 4; col++) {
                for (unsigned int inner = 0; inner < 4; inner++) {
                    res(row,col) += at(row,inner) * M(inner,col);
                }
            }
        }

        return res;
    }

    // OPERATORS
    IMAGINE_INLINE_FUNCTION
    Vector operator*(const Vector& v) const 
    {
        return mult(v);
    }

    IMAGINE_INLINE_FUNCTION
    Matrix4x4 operator*(const Matrix4x4& M) const 
    {
        return mult(M);
    }

    IMAGINE_INLINE_FUNCTION
    Matrix4x4 operator~() const 
    {
        return inv();
    }
};

} // namespace imagine 

#endif // IMAGINE_MATH_TYPES_H