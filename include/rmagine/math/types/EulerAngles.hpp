#ifndef RMAGINE_MATH_EULER_ANGLES_HPP
#define RMAGINE_MATH_EULER_ANGLES_HPP

#include "definitions.h"
#include <rmagine/types/shared_functions.h>

namespace rmagine
{

/**
 * @brief EulerAngles type
 * 
 */
template<typename DataT>
struct EulerAngles_ 
{
    float roll;     // x-axis
    float pitch;    // y-axis
    float yaw;      // z-axis

    // RMAGINE_INLINE_FUNCTION
    // void setIdentity();
};

} // rmagine

#include "EulerAngles.tcc"

#endif // RMAGINE_MATH_EULER_ANGLES_HPPs