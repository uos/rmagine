#ifndef RMAGINE_MATH_VECTOR_HPP
#define RMAGINE_MATH_VECTOR_HPP

#include "definitions.h"

namespace rmagine
{

template<typename DataT>
struct Vector2_
{
    DataT x;
    DataT y;
};

template<typename DataT>
struct Vector3_
{
    DataT x;
    DataT y;
    DataT z;
};

} // namespace rmagine

#endif // RMAGINE_MATH_VECTOR_HPP