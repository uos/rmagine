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





SphericalModel example_spherical();

PinholeModel example_pinhole();

O1DnModel example_o1dn();

OnDnModel example_ondn();



} // namespace rmagine

#endif // RMAGINE_TYPES_SENSORS_H