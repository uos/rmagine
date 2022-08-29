#ifndef RMAGINE_TYPES_SENSORS_H
#define RMAGINE_TYPES_SENSORS_H

#include "sensor_models.h"

namespace rmagine
{

/**
 * @brief Velodyne VLP-16 with 900 horizontal scan points
 * 
 * @return SphericalModel 
 */
SphericalModel vlp16_900();

SphericalModel vlp16_360();

/**
 * @brief VLP-16 900
 * 
 * @return SphericalModel 
 */
SphericalModel example_spherical();





} // namespace rmagine

#endif // RMAGINE_TYPES_SENSORS_H