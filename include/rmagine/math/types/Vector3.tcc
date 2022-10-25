#include "Vector3.hpp"
#include <math.h>

namespace rmagine
{

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Vector3_<DataT> Vector3_<DataT>::add(const Vector3_<DataT>& b) const
{
    return {x + b.x, y + b.y, z + b.z};
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
volatile Vector3_<DataT> Vector3_<DataT>::add(volatile Vector3_<DataT>& b) const volatile
{
    return {x + b.x, y + b.y, z + b.z};
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Vector3_<DataT>::addInplace(const Vector3_<DataT>& b)
{
    x += b.x;
    y += b.y;
    z += b.z;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Vector3_<DataT>::addInplace(volatile Vector3_<DataT>& b) volatile
{
    x += b.x;
    y += b.y;
    z += b.z;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Vector3_<DataT> Vector3_<DataT>::sub(const Vector3_<DataT>& b) const
{
    return {x - b.x, y - b.y, z - b.z};
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Vector3_<DataT> Vector3_<DataT>::negate() const
{
    return {-x, -y, -z};
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Vector3_<DataT>::negateInplace() 
{
    x = -x;
    y = -y;
    z = -z;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Vector3_<DataT>::subInplace(const Vector3_<DataT>& b)
{
    x -= b.x;
    y -= b.y;
    z -= b.z;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
DataT Vector3_<DataT>::dot(const Vector3_<DataT>& b) const 
{
    return x * b.x + y * b.y + z * b.z;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Vector3_<DataT> Vector3_<DataT>::cross(const Vector3_<DataT>& b) const
{
    return {
        y * b.z - z * b.y,
        z * b.x - x * b.z,
        x * b.y - y * b.x
    };
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
DataT Vector3_<DataT>::mult(const Vector3_<DataT>& b) const
{
    return dot(b);
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Vector3_<DataT> Vector3_<DataT>::multEwise(const Vector3_<DataT>& b) const
{
    return {x * b.x, y * b.y, z * b.z};
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Matrix_<DataT, 3, 3> Vector3_<DataT>::multT(const Vector3_<DataT>& b) const
{
    Matrix_<DataT, 3, 3> C;
    C(0,0) = x * b.x;
    C(1,0) = y * b.x;
    C(2,0) = z * b.x;
    C(0,1) = x * b.y;
    C(1,1) = y * b.y;
    C(2,1) = z * b.y;
    C(0,2) = x * b.z;
    C(1,2) = y * b.z;
    C(2,2) = z * b.z;
    return C;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Vector3_<DataT> Vector3_<DataT>::mult(const DataT& s) const 
{
    return {x * s, y * s, z * s};
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
volatile Vector3_<DataT> Vector3_<DataT>::mult(const DataT& s) const volatile 
{
    return {x * s, y * s, z * s};
}


template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Vector3_<DataT>::multInplace(const DataT& s) 
{
    x *= s;
    y *= s;
    z *= s;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Vector3_<DataT> Vector3_<DataT>::div(const DataT& s) const 
{
    return {x / s, y / s, z / s};
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Vector3_<DataT>::divInplace(const DataT& s) 
{
    x /= s;
    y /= s;
    z /= s;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
DataT Vector3_<DataT>::l2normSquared() const
{
    return x*x + y*y + z*z;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
DataT Vector3_<DataT>::l2norm() const 
{
    return sqrtf(l2normSquared());
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
DataT Vector3_<DataT>::sum() const 
{
    return x + y + z;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
DataT Vector3_<DataT>::prod() const 
{
    return x * y * z;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
DataT Vector3_<DataT>::l1norm() const 
{
    return fabs(x) + fabs(y) + fabs(z);
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Vector3_<DataT> Vector3_<DataT>::normalize() const 
{
    return div(l2norm());
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Vector3_<DataT>::normalizeInplace() 
{
    divInplace(l2norm());
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Vector3_<DataT>::setZeros()
{
    x = static_cast<DataT>(0);
    y = static_cast<DataT>(0);
    z = static_cast<DataT>(0);
}

template<typename DataT>
template<typename ConvT>
RMAGINE_INLINE_FUNCTION
Vector3_<ConvT> Vector3_<DataT>::cast() const
{
    return {
        static_cast<ConvT>(x),
        static_cast<ConvT>(y),
        static_cast<ConvT>(z)
    };
}

} // namespace rmagine
