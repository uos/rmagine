#include "Vector2.hpp"

namespace rmagine
{

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Vector2_<DataT> Vector2_<DataT>::add(const Vector2_<DataT>& b) const
{
    return {x + b.x, y + b.y};
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Vector2_<DataT>::addInplace(const Vector2_<DataT>& b)
{
    x += b.x;
    y += b.y;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Vector2_<DataT> Vector2_<DataT>::sub(const Vector2_<DataT>& b) const
{
    return {x - b.x, y - b.y};
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Vector2_<DataT>::subInplace(const Vector2_<DataT>& b)
{
    x -= b.x;
    y -= b.y;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Vector2_<DataT> Vector2_<DataT>::negate() const
{
    return {-x, -y};
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Vector2_<DataT>::negateInplace()
{
    x = -x;
    y = -y;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
DataT Vector2_<DataT>::dot(const Vector2_<DataT>& b) const 
{
    return x * b.x + y * b.y; 
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
DataT Vector2_<DataT>::mult(const Vector2_<DataT>& b) const
{
    return dot(b);
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Vector2_<DataT> Vector2_<DataT>::mult(const DataT& s) const 
{
    return {x * s, y * s};
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Vector2_<DataT>::multInplace(const DataT& s) 
{
    x *= s;
    y *= s;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Vector2_<DataT> Vector2_<DataT>::div(const DataT& s) const 
{
    return {x / s, y / s};
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Vector2_<DataT>::divInplace(const DataT& s) 
{
    x /= s;
    y /= s;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Vector2_<DataT> Vector2_<DataT>::to(const Vector2_<DataT>& o) const
{
  return o.sub(*this);
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
DataT Vector2_<DataT>::l2normSquared() const
{
    return x*x + y*y;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
DataT Vector2_<DataT>::l2norm() const 
{
    return sqrtf(l2normSquared());
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
DataT Vector2_<DataT>::sum() const 
{
    return x + y;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
DataT Vector2_<DataT>::prod() const 
{
    return x * y;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
DataT Vector2_<DataT>::l1norm() const 
{
    return fabs(x) + fabs(y);
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Vector2_<DataT>::setZeros()
{
    x = 0.0;
    y = 0.0;
}

template<typename DataT>
template<typename ConvT>
RMAGINE_INLINE_FUNCTION
Vector2_<ConvT> Vector2_<DataT>::cast() const
{
    return {
        static_cast<ConvT>(x),
        static_cast<ConvT>(y)
    };
}

} // namespace rmagine