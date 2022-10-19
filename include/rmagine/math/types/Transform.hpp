#ifndef RMAGINE_MATH_TRANSFORM_HPP
#define RMAGINE_MATH_TRANSFORM_HPP

#include "definitions.h"
#include <rmagine/types/shared_functions.h>

namespace rmagine
{

/**
 * @brief Transform type
 * 
 * Consists of rotational part represented as @link rmagine::Quaternion Quaternion @endlink 
 * and a translational part represented as @link rmagine::Vector3 Vector3 @endlink  
 * 
 * Additionally it contains a timestamp uint32_t
 * 
 */
template<typename DataT>
struct Transform_
{
    // DATA
    Quaternion_<DataT> R;
    Vector3_<DataT> t;
    uint32_t stamp;
};

} // namespace rmagine

#include "Transform.tcc"

#endif // RMAGINE_MATH_TRANSFORM_HPP