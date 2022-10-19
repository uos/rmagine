#ifndef RMAGINE_MATH_AABB_HPP
#define RMAGINE_MATH_AABB_HPP

#include "definitions.h"
#include <rmagine/types/shared_functions.h>

namespace rmagine
{

template<typename DataT>
struct AABB_
{
    Vector3_<DataT> min;
    Vector3_<DataT> max;

    RMAGINE_INLINE_FUNCTION
    DataT volume() const;

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> size() const;

    RMAGINE_INLINE_FUNCTION
    void init();

    RMAGINE_INLINE_FUNCTION
    void expand(const Vector3_<DataT>& p);

    RMAGINE_INLINE_FUNCTION
    void expand(const AABB_<DataT>& o);
};

} // namespace rmagine

#include "AABB.tcc"

#endif // RMAGINE_MATH_AABB_HPP