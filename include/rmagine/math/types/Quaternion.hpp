#ifndef RMAGINE_MATH_QUATERNION_HPP
#define RMAGINE_MATH_QUATERNION_HPP

#include "definitions.h"
#include <rmagine/types/shared_functions.h>

namespace rmagine
{

template<typename DataT>
struct Quaternion_
{
    DataT x;
    DataT y;
    DataT z;
    DataT w;
};

} // namespace rmagine

#include "Quaternion.tcc"

#endif // RMAGINE_MATH_QUATERNION_HPP