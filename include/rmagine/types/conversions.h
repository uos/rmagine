#ifndef RMAGINE_TYPES_CONVERSIONS_H
#define RMAGINE_TYPES_CONVERSIONS_H

#include <rmagine/math/types.h>
#include <cstdint>
#include <math.h>

#include <rmagine/types/Memory.hpp>

#include "sensor_models.h"

namespace rmagine
{


void convert(const SphericalModel& in, O1DnModel& out);
void convert(const SphericalModel& in, OnDnModel& out);

void convert(const PinholeModel& in, OnDnModel& out, bool optical = false);
void convert(const PinholeModel& in, O1DnModel& out, bool optical = false);


} // rmagine


#endif // RMAGINE_TYPES_CONVERSIONS_H