#include "AABB.hpp"

namespace rmagine
{

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Vector3_<DataT> AABB_<DataT>::size() const
{
    return max - min;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
DataT AABB_<DataT>::volume() const
{
    const Vector3_<DataT> _size = size();
    DataT _volume = _size.l2norm();

    if(_size.x < 0.f || _size.y < 0.f || _size.z < 0.f)
    {
        // compute volume and add minus to signalize wrong
        _volume = -_volume;
    }

    return _volume;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void AABB_<DataT>::init()
{
    min.x = FLT_MAX;
    min.y = FLT_MAX;
    min.z = FLT_MAX;
    max.x = -FLT_MAX;
    max.y = -FLT_MAX;
    max.z = -FLT_MAX;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void AABB_<DataT>::expand(const Vector3_<DataT>& p)
{
    min.x = fminf(min.x, p.x);
    min.y = fminf(min.y, p.y);
    min.z = fminf(min.z, p.z);
    max.x = fmaxf(max.x, p.x);
    max.y = fmaxf(max.y, p.y);
    max.z = fmaxf(max.z, p.z);
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void AABB_<DataT>::expand(const AABB_<DataT>& o)
{
    // assuming AABBs to be initialized
    min.x = fminf(min.x, o.min.x);
    min.y = fminf(min.y, o.min.y);
    min.z = fminf(min.z, o.min.z);
    max.x = fmaxf(max.x, o.max.x);
    max.y = fmaxf(max.y, o.max.y);
    max.z = fmaxf(max.z, o.max.z);
}

template<typename DataT>
template<typename ConvT>
RMAGINE_INLINE_FUNCTION
AABB_<ConvT> AABB_<DataT>::cast() const
{
    return {
        min.template cast<ConvT>(),
        max.template cast<ConvT>()
    };
}

} // namespace rmagine