#include "rmagine/types/ouster_sensors.h"

#include <jsoncpp/json/json.h>
#include <fstream>

namespace rmagine {

O1DnModel o1dn_from_ouster_meta_file(std::string filename)
{
  O1DnModel ret;

  std::ifstream ouster_file(filename, std::ifstream::binary);

  Json::Value ouster_config;
  ouster_file >> ouster_config;

  std::cout << ouster_config << std::endl;


  return ret;
}

} // namespace rmagine