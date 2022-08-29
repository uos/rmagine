#include "rmagine/types/sensors.h"

namespace rmagine
{

SphericalModel vlp16_900()
{
    SphericalModel model;
    model.theta.min = -M_PI;
    model.theta.inc = (360.0 / 900.0) * DEG_TO_RAD_F;
    model.theta.size = 900;

    model.phi.min = -15.0 * DEG_TO_RAD_F;
    model.phi.inc = 2.0 * DEG_TO_RAD_F;
    model.phi.size = 16;
    
    model.range.min = 0.5;
    model.range.max = 130.0;
    return model;
}

SphericalModel vlp16_360()
{
    SphericalModel model;
    model.theta.min = -M_PI;
    model.theta.inc = 1.0 * DEG_TO_RAD_F;
    model.theta.size = 360;

    model.phi.min = -15.0 * DEG_TO_RAD_F;
    model.phi.inc = 2.0 * DEG_TO_RAD_F;
    model.phi.size = 16;
    
    model.range.min = 0.5;
    model.range.max = 130.0;
    return model;
}


SphericalModel example_spherical()
{
    SphericalModel model;
    model.theta.min = -M_PI;
    model.theta.inc = 1.0 * DEG_TO_RAD_F;
    model.theta.size = 360;

    model.phi.min = -30.0 * DEG_TO_RAD_F;
    model.phi.inc = 1.0 * DEG_TO_RAD_F;
    model.phi.size = 60;
    
    model.range.min = 0.5;
    model.range.max = 130.0;
    return model;
}

} // namespace rmagine