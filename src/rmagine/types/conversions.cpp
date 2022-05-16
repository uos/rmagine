#include "rmagine/types/conversions.h"

namespace rmagine
{

void convert(const SphericalModel& in, O1DnModel& out)
{
    out.dirs.resize(in.size());
    out.orig = {0.0, 0.0, 0.0};
    out.width = in.getWidth();
    out.height = in.getHeight();

    out.range.min = in.range.min;
    out.range.max = in.range.max;

    for(size_t vid=0; vid<in.getHeight(); vid++)
    {
        for(size_t hid=0; hid<in.getWidth(); hid++)
        {
            const unsigned int loc_id = out.getBufferId(vid, hid);
            out.dirs[loc_id] = in.getDirection(vid, hid);
        }
    }
}

void convert(const SphericalModel& in, OnDnModel& out)
{
    out.dirs.resize(in.size());
    out.origs.resize(in.size());
    out.width = in.getWidth();
    out.height = in.getHeight();

    out.range.min = in.range.min;
    out.range.max = in.range.max;

    for(size_t vid=0; vid<in.getHeight(); vid++)
    {
        for(size_t hid=0; hid<in.getWidth(); hid++)
        {
            const unsigned int loc_id = out.getBufferId(vid, hid);
            out.dirs[loc_id] = in.getDirection(vid, hid);
            out.origs[loc_id] = in.getDirection(vid, hid);
        }
    }
}

void convert(const PinholeModel& in, O1DnModel& out, bool optical)
{
    out.dirs.resize(in.size());
    out.orig = {0.0, 0.0, 0.0};
    out.width = in.getWidth();
    out.height = in.getHeight();

    out.range.min = in.range.min;
    out.range.max = in.range.max;

    for(size_t vid=0; vid<in.getHeight(); vid++)
    {
        for(size_t hid=0; hid<in.getWidth(); hid++)
        {
            const unsigned int loc_id = out.getBufferId(vid, hid);
            if(optical)
            {
                out.dirs[loc_id] = in.getDirectionOptical(vid, hid);
            } else {
                out.dirs[loc_id] = in.getDirection(vid, hid);
            }
        }
    }
}

void convert(const PinholeModel& in, OnDnModel& out, bool optical)
{
    out.dirs.resize(in.size());
    out.origs.resize(in.size());
    out.width = in.getWidth();
    out.height = in.getHeight();

    out.range.min = in.range.min;
    out.range.max = in.range.max;

    for(size_t vid=0; vid<in.getHeight(); vid++)
    {
        for(size_t hid=0; hid<in.getWidth(); hid++)
        {
            const unsigned int loc_id = out.getBufferId(vid, hid);
            out.dirs[loc_id] = in.getDirection(vid, hid);
            if(optical)
            {
                out.dirs[loc_id] = in.getDirectionOptical(vid, hid);
            } else {
                out.dirs[loc_id] = in.getDirection(vid, hid);
            }
        }
    }
}

} // namespace rmagine