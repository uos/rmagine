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
    model.theta.inc = 0.4 * M_PI / 180.0;
    model.theta.size = 900;
    
    model.phi.min = -15.0 * M_PI / 180.0;
    model.phi.inc = 2.0 * M_PI / 180.0;
    model.phi.size = 16;
    
    model.range.min = 0.0;
    model.range.max = 100.0;

    return model;
}

PinholeModel example_pinhole()
{
    PinholeModel model;
    model.width = 200;
    model.height = 150;
    model.c[0] = 100.0; // ~ half of width
    model.c[1] = 75.0; // ~ half of height
    model.f[0] = 100.0;
    model.f[1] = 100.0;
    model.range.min = 0.0;
    model.range.max = 100.0;
    return model;
}

O1DnModel example_o1dn()
{
    O1DnModel model;

    model.orig.x = 0.0;
    model.orig.y = 0.0;
    model.orig.z = 0.0;

    model.width = 200;
    model.height = 1;

    model.dirs.resize(model.width * model.height);

    float step_size = 0.05;

    for(int i=0; i<200; i++)
    {
        float y = - static_cast<float>(i - 100) * step_size;
        float x = cos(y) * 2.0 + 2.0;
        float z = -1.0;

        model.dirs[i].x = x;
        model.dirs[i].y = y;
        model.dirs[i].z = z;

        model.dirs[i].normalizeInplace();
    }

    model.range.min = 0.0;
    model.range.max = 100.0;

    return model;
}

OnDnModel example_ondn()
{
    OnDnModel model;

    model.width = 200;
    model.height = 1;

    model.dirs.resize(model.width * model.height);
    model.origs.resize(model.width * model.height);

    float step_size = 0.05;

    for(int i=0; i<200; i++)
    {
        float percent = static_cast<float>(i) / static_cast<float>(200);
        float step = - static_cast<float>(i - 100) * step_size;
        float y = sin(step);
        float x = cos(step);

        model.origs[i].x = 0.0;
        model.origs[i].y = y * percent;
        model.origs[i].z = x * percent;

        model.dirs[i].x = 1.0;
        model.dirs[i].y = 0.0;
        model.dirs[i].z = 0.0;
    }

    model.range.min = 0.0;
    model.range.max = 100.0;

    return model;

}

} // namespace rmagine