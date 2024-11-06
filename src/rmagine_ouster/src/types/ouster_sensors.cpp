#include "rmagine/types/ouster_sensors.h"

#include <json/json.h>
#include <fstream>

namespace rmagine {

O1DnModel o1dn_from_ouster_meta_file(std::string filename)
{
    O1DnModel model;

    std::ifstream ouster_file(filename, std::ifstream::binary);

    Json::Value ouster_config;
    ouster_file >> ouster_config;

    std::string lidar_mode = ouster_config["lidar_mode"].asString();

    size_t x_id = lidar_mode.find("x");
    unsigned int lidar_width = std::stoi(lidar_mode.substr(0, x_id));
    unsigned int lidar_frequency = std::stoi(lidar_mode.substr(x_id+1));

    std::vector<float> beam_altitude_angles;
    for(auto val : ouster_config["beam_altitude_angles"])
    {
        beam_altitude_angles.push_back(val.asFloat());
    }

    unsigned int lidar_height = beam_altitude_angles.size();

    std::vector<float> beam_azimuth_angles;
    for(auto val : ouster_config["beam_azimuth_angles"])
    {
        beam_azimuth_angles.push_back(val.asFloat());
    }

    const float lidar_origin_to_beam_origin_mm = ouster_config["lidar_origin_to_beam_origin_mm"].asFloat();

    // std::cout << "Loading Ouster Lidar: " << lidar_width << "x" << lidar_height << ". " << lidar_frequency << "hz" << std::endl;

    model.width = lidar_width;
    model.height = lidar_height;
    model.dirs.resize(lidar_width * lidar_height);
    
    const float hor_start_angle = 0.0;
    const float hor_angle_inc = (2.0 * M_PI) / static_cast<float>(model.getWidth());

    const float ver_start_angle = 0.0;

    

    model.orig = {0.0, 0.0, 0.0};

    model.range.min = 0.1;
    model.range.max = 80.0;

    for(unsigned int vid = 0; vid < model.getHeight(); vid++)
    {
        // vertical angle
        const float phi = beam_altitude_angles[vid] * DEG_TO_RAD_F;

        for(unsigned int hid = 0; hid < model.getWidth(); hid++)
        {
            // horizontal angle
            const float theta_offset = beam_azimuth_angles[vid] * DEG_TO_RAD_F;
            const float theta = hor_start_angle + hor_angle_inc * static_cast<float>(hid) + theta_offset;

            // compute direction: polar to cartesian
            Vector dir{
                cosf(phi) * cosf(theta), 
                cosf(phi) * sinf(theta), 
                sinf(phi)};

            const unsigned int buf_id = model.getBufferId(vid, hid);
            model.dirs[buf_id] = dir;
        }
    }

    return model;
}

} // namespace rmagine
