#ifndef RMAGINE_MAP_MAP_HPP
#define RMAGINE_MAP_MAP_HPP

#include <unordered_map>
#include <string>
#include "Map.hpp"

namespace rmagine
{

using MapMap = std::unordered_map<std::string, MapPtr>;

using MapMapPtr = std::shared_ptr<MapMap>;

} // namespace rmagine

#endif // RMAGINE_MAP_MAP_HPP
